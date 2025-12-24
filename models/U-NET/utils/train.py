import os
import torch
import numpy as np
import torch.nn as nn
import argparse
import json
import dataclasses
import segmentation_models_pytorch as smp

from src.dataset import fetch_loaders
from config.config import load_config
from src.metrics import metrics, update_metrics, agg_metrics
from src.losses_binary import BinaryDiceLoss, DiceBCELoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.transforms import v2 as T

def save_model(model, optimizer, epoch, loss, path):
    """Saves the model state, optimizer state, epoch, and loss to the given path."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Model saved to {path}")

def initialize_model(config):
    # model initialization
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        decoder_use_batchnorm=config.model_opts.batch_norm,
        in_channels=config.model_opts.inchannels, # RGB + SWIR1 channels
        classes=config.model_opts.classes,    # glacier vs non-glacier
    )

    '''# need to create a new conv layer for the new input channels
    old_conv = model.encoder.conv1

    new_conv = nn.Conv2d(
        in_channels=config.model_opts.inchannels, 
        out_channels=old_conv.out_channels, 
        kernel_size=old_conv.kernel_size, 
        stride=old_conv.stride, 
        padding=old_conv.padding, 
        bias=(old_conv.bias is not None)
    )

    # Copy ImageNet weights for the first 3 channels (RGB)
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight
        # Initialize the additional channels (4-6) as the mean of the first three
        if config.model_opts.inchannels > 3:
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            for i in range(3, config.model_opts.inchannels):
                new_conv.weight[:, i:i+1, :, :] = mean_weight
                
        # Copy bias if it exists
        if old_conv.bias is not None:
            new_conv.bias = old_conv.bias

    # Replace the model's first conv layer with the new one
    model.encoder.conv1 = new_conv'''
'''
    # define loss function and optimizer
    if config.loss_opts.type == "DiceLoss":
        criterion = BinaryDiceLoss(smooth=config.loss_opts.smooth)
    elif config.loss_opts.type == "CE_DiceLoss":
        criterion = DiceBCELoss(weights_ce=config.loss_opts.weights_ce, 
                                weights_dice=config.loss_opts.weights_dice, 
                                smooth=config.loss_opts.smooth)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.model_opts.lr, weight_decay=config.model_opts.weight_decay)
    return model, criterion, optimizer'''
    
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device, metrics_opts):
    model.train()
    running_loss = 0.0
    metrics_accum = {}

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device).float()

        optimizer.zero_grad()
        logits = model(inputs)
        targets = targets.unsqueeze(1)  # add channel dimension

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        metrics_batch = metrics(logits, targets, dataclasses.asdict(metrics_opts))

        # accumulate metrics
        update_metrics(metrics_accum, metrics_batch)
    
    agg_metrics(metrics_accum)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, metrics_accum

def validate_one_epoch(model, dataloader, criterion, device, metrics_opts):
    model.eval()
    running_loss = 0.0
    metrics_accum = {}

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            targets = targets.unsqueeze(1).float()  # add channel dimension

            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            metrics_batch = metrics(outputs, targets, dataclasses.asdict(metrics_opts))
            # accumulate metrics
            update_metrics(metrics_accum, metrics_batch)
        
    agg_metrics(metrics_accum)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, metrics_accum

def train_model(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize model, criterion, optimizer
    print("Initializing model...")
    model, criterion, optimizer = initialize_model(config)
    model.to(device)

    # transforms 
    transforms = T.Compose([
        T.RandomResizedCrop(size=(128, 128), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.ToDtype(torch.float32, scale=True),
    ])

    # fetch data loaders
    print("Fetching data loaders...")
    train_loader, val_loader = fetch_loaders(
        npz_path=config.paths.training.dataset_path,
        mode='train',
        val_size=config.training_opts.val_size,
        transform=transforms,
        batch_size=config.training_opts.batch_size,
        train_shuffle=True
    )

    best_val_loss = float('inf')
    
    args = dict(
        run_name=config.training_opts.run_name,
        num_epochs=config.training_opts.num_epochs,
        learning_rate=config.model_opts.lr, 
        weight_decay=config.model_opts.weight_decay,
        batch_size=config.training_opts.batch_size
    )

    # Setup logging
    log_dir = f"{config.paths.training.save_path}/runs/{config.training_opts.run_name}/logs/"
    out_dir = f"{config.paths.training.save_path}/runs/{config.training_opts.run_name}/models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)
    writer.add_text("Arguments", json.dumps(args))
    writer.add_text("Configuration Parameters", json.dumps(dataclasses.asdict(config)))
    
    #mask_names = config.log_opts.mask_names

    for epoch in tqdm(range(config.training_opts.num_epochs), desc="Training Epochs", smoothing=0.9):

        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, config.metrics_opts)
        val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, device, config.metrics_opts)

        print(f"Epoch {epoch+1}/{config.training_opts.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        # Log losses and metrics
        writer.add_scalar("Loss/Train", train_loss, epoch+1)
        writer.add_scalar("Loss/Val", val_loss, epoch+1)

        for k, v in train_metrics.items():
            writer.add_scalar(f"Metrics/Train/{k}", v.item(), epoch+1)
        for k, v in val_metrics.items():
            writer.add_scalar(f"Metrics/Val/{k}", v.item(), epoch+1)

        # Save the model if validation loss has decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(out_dir, f"best_model_epoch_{epoch+1}.pth")
            save_model(model, optimizer, epoch+1, val_loss, save_path)

    # print best validation loss
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("Training complete")
    writer.close()

if __name__ == "__main__":
    
    # Load configuration
    parser = argparse.ArgumentParser(description="Train U-NET model")
    parser.add_argument('--config', type=str, default="train.yaml", help='Path to the config file')
    args = parser.parse_args()

    config = load_config(args.config)

    '''device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # Load preprocessed data from .npz files
    train_npz_path = os.path.join(config.paths.training.dataset_path, 'glacier_train.npz')
    val_npz_path = os.path.join(config.paths.training.dataset_path, 'glacier_val.npz')
    
    train_data = np.load(train_npz_path)
    val_data = np.load(val_npz_path)

    # Extract images and masks from the npz files
    x_train = train_data['X']  # Shape: (N, C, H, W)
    y_train = train_data['Y']  # Shape: (N,)
    x_val = val_data['X']  # Shape: (N, C, H, W)
    y_val = val_data['Y']  # Shape: (N,)

    print(f"Loaded training data: {x_train.shape}, labels: {y_train.shape}")
    print(f"Loaded validation data: {x_val.shape}, labels: {y_val.shape}")

    # Create data loaders from numpy arrays
    train_loader, val_loader = fetch_loaders(x_train, y_train, x_val, y_val, config.training_opts.batch_size)
    
    # model initialization 
    model, criterion, optimizer = initialize_model(config)
    model.to(device)'''

    train_model(config)