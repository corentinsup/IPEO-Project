# Utilities for the training loop
from tqdm.auto import tqdm
import torch

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Runs one epoch of training.

    Args:
        model (torch.nn.Module): The model to train.
        loader (torch.utils.data.DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the training on ("cuda" or "cpu").
    Returns:
        float: Average loss over the epoch.
    """
    
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(images) # /!\ (B, 2, H, W) -> 1 channel per class
        
        loss = criterion(outputs, masks)

        # Backward
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Average loss
    return running_loss / len(loader)


def validate_one_epoch(model, loader, criterion, iou_metric, device):
    """Runs one epoch of validation. Computes both loss and IoU metric.

    Args:
        model (torch.nn.Module): The model to validate.
        loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        criterion (torch.nn.Module): Loss function.
        iou_metric (torchmetrics.JaccardIndex): IoU metric function.
        device (torch.device): Device to run the validation on ("cuda" or "cpu").
    Returns:
        tuple: Average loss and IoU over the epoch.
    """
    
    model.eval()
    running_loss = 0.0
    
    # reset IoU
    iou_metric.reset()
    
    pbar = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            # Forward
            outputs = model(images)  # /!\ (B, 2, H, W) -> 1 channel per class
            
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            # Update IoU metric
            iou_metric.update(outputs, masks)
            
    # Average loss
    avg_loss = running_loss / len(loader)
    # Compute IoU
    avg_iou = iou_metric.compute().item()
    return avg_loss, avg_iou