# Util file to run an inference
from tqdm.auto import tqdm
import torch

def run_eval(model, loader, criterion=None, iou_metric=None, return_preds=False, device="cpu"):
    """Runs inference on the provided DataLoader. Optionally computes loss and IoU metric if provided.

    Args:
        model (torch.nn.Module): The model to run inference with.
        loader (torch.utils.data.DataLoader): DataLoader for the inference data.
        criterion (torch.nn.Module, optional): Loss function. Defaults to None.
        iou_metric (torchmetrics.JaccardIndex, optional): IoU metric function. Defaults to None.
        return_preds (bool, optional): Whether to return predictions, masks, and images. Defaults to False.
        device (torch.device, optional): Device to run the inference on ("cuda" or "cpu"). Defaults to "cpu".
    Returns:
        tuple: Average loss and IoU over the dataset if criterion and iou_metric are provided, else None.
    """
    
    model.eval()
    running_loss = 0.0
    
    # Stockage optionnel des résultats pour visualisation
    saved_preds = []
    saved_masks = []
    saved_imgs = []

    if iou_metric is not None:
        iou_metric.reset()
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Inference"):
            images, masks = images.to(device), masks.to(device)
            
            # Forward
            outputs = model(images)
            
            # Loss
            if criterion is not None:
                loss = criterion(outputs, masks)
                running_loss += loss.item()
            
            # Argmax to get the class
            preds = torch.argmax(outputs, dim=1)

            # Metric
            if iou_metric is not None:
                iou_metric.update(preds, masks)
            
            # If we want to retrieve images for later display
            if return_preds:
                # Move everything to CPU to avoid saturating VRAM
                saved_preds.append(preds.cpu())
                # Also keep the original masks/images for comparison
                saved_masks.append(masks.cpu())
                saved_imgs.append(images.cpu())
    
    avg_loss = running_loss / len(loader) if criterion is not None else None
    avg_iou = iou_metric.compute().item() if iou_metric is not None else None
    
    if return_preds:
        # Concatenate everything into one big tensor per category
        return avg_loss, avg_iou, torch.cat(saved_imgs), torch.cat(saved_masks), torch.cat(saved_preds)
    
    return avg_loss, avg_iou