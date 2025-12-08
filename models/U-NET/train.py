import os
import torch
import torch.nn as nn
import argparse
import json
import segmentation_models_pytorch as smp

from src.dataset import makeDataloader
from config.config import load_config
from src.metrics import metrics, update_metrics, agg_metrics
from src.losses import DiceLoss, CE_DiceLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        decoder_use_batchnorm=config.model_opts.batch_norm,
        in_channels=config.model_opts.in_channels, # RGB + NIR channels
        classes=config.model_opts.classes,    # glacier vs non-glacier
    )

    # need to create a new conv layer for the new input channels
    old_conv = model.encoder.conv1

    first_conv = nn.Conv2d(
        in_channels=config.model_opts.in_channels, 
        out_channels=old_conv.out_channels, 
        kernel_size=old_conv.kernel_size, 
        stride=old_conv.stride, 
        padding=old_conv.padding, 
        bias=(old_conv.bias is not None)
    )

    # Copy ImageNet weights for the first 3 channels
    first_conv.weight.data[:, :3, :, :] = old_conv.weight.data
    # Initialize the 4th channel as the mean of the first three
    first_conv.weight.data[:, 3:4, :, :] = old_conv.weight.data.mean(dim=1, keepdim=True)

    # Replace the model's first conv layer with the new one
    model.encoder.conv1 = first_conv

    # define loss function and optimizer
    if config.loss_opts.type == "DiceLoss":
        criterion = DiceLoss(smooth=config.loss_opts.smooth)
    elif config.loss_opts.type == "CE_DiceLoss":
        criterion = CE_DiceLoss(weights_ce=config.loss_opts.weights_ce, 
                                weights_dice=config.loss_opts.weights_dice, 
                                smooth=config.loss_opts.smooth)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    return model, criterion, optimizer

def train_one_epoch(model, dataloader, criterion, optimizer, device, metrics_opts):
    model.train()
    running_loss = 0.0
    metrics_accum = {}

    for inputs, targets in tqdm(dataloader, desc="Training Batches", smoothing=0.4):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        metrics_batch = metrics(outputs, targets, metrics_opts)

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
        for inputs, targets in tqdm(dataloader, desc="Validation Batches", smoothing=0.4):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            metrics_batch = metrics(outputs, targets, metrics_opts)
            # accumulate metrics
            update_metrics(metrics_accum, metrics_batch)
        
    agg_metrics(metrics_accum)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, metrics_accum

def train_model(model, train_loader, val_loader, criterion, optimizer, device, config):
    best_val_loss = float('inf')
    
    args = dict(
        run_name=config.training_opts.run_name,
        num_epochs=config.training_opts.num_epochs,
        learning_rate=config.model_opts.lr, 
        weight_decay=config.model_opts.weights_decay,
        batch_size=config.training_opts.batch_size
    )

    # Setup logging
    log_dir = f"{config.paths.training.dataset_path}/runs/{config.training_opts.run_name}/logs/"
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)
    writer.add_text("Arguments", json.dumps(args))
    writer.add_text("Configuration Parameters", json.dumps(config))
    out_dir = f"{config.paths.training.dataset_path}/runs/{config.training_opts.run_name}/models/"
    #mask_names = config.log_opts.mask_names

    for epoch in tqdm(range(config.training_opts.num_epochs), desc="Training Epochs", smoothing=0.4):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv = os.path.join(config.paths.training.dataset_path, config.paths.training.train_csv)
    val_csv = os.path.join(config.paths.training.dataset_path, config.paths.training.val_csv)
    train_img_dir = os.path.join(config.paths.training.dataset_path, config.paths.training.train_img_dir)
    val_img_dir = os.path.join(config.paths.training.dataset_path, config.paths.training.val_img_dir)

    # Data loaders
    train_loader = makeDataloader(train_csv, train_img_dir, config.training_opts.batch_size, shuffle=True)
    val_loader = makeDataloader(val_csv, val_img_dir, config.training_opts.batch_size, shuffle=False)
    
    # model initialization 
    model, criterion, optimizer = initialize_model(config)
    model.to(device)

    train_model(model, train_loader, val_loader, criterion, optimizer, device, config)