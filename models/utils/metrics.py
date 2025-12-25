# Utility file with function for the metrics
import segmentation_models_pytorch as smp
from torchmetrics import JaccardIndex
import torch.nn as nn

def get_dice_loss(mode="multiclass", ignore_index=255):
    """Get Dice loss from segmentation_models_pytorch library.
    
    Args:
        ignore_index (int): Class index to ignore during loss computation.
    Returns:
        smp.losses.DiceLoss: Dice loss function.
    """
    return smp.losses.DiceLoss(
        mode=mode,  # adapt to binary or multiclass
        from_logits=True,
        ignore_index=ignore_index    
    )
    
    
def get_iou_metric(mode="multiclass", ignore_index=255):
    """Get IoU metric from torchmetrics library. 
    This function is updated each batch with .update() and the final result is obtained with compute().
    
    Args:
        ignore_index (int): Class index to ignore during metric computation.
    
    Returns:
        torchmetrics.JaccardIndex: IoU metric function.
    """
    return JaccardIndex(task=mode, num_classes=2, ignore_index=ignore_index).to("cuda")

def get_combined_loss(factor_BCE, factor_DICE, ignore_index=255):
    """Get a combined loss function that sums Dice loss and Cross Entropy loss.
    
    Args:
        factor_BCE (float): Weight for the Cross Entropy loss.
        factor_DICE (float): Weight for the Dice loss.
        ignore_index (int): Class index to ignore during loss computation.
    Returns:
        function: Combined loss function.
    """
    
    assert factor_BCE + factor_DICE == 1.0, "The sum of factor_BCE and factor_DICE must be 1.0"

    dice_loss_fn = get_dice_loss(mode = "multiclass", ignore_index=ignore_index)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    # else:  # U-NET
    #     dice_loss_fn = get_dice_loss(mode = "binary", ignore_index=ignore_index)
    #     ce_loss_fn = nn.BCEWithLogitsLoss()
    
    def combined_loss(preds, targets):
        loss_dice = dice_loss_fn(preds, targets)
        loss_ce = ce_loss_fn(preds, targets)
        return factor_DICE * loss_dice + factor_BCE * loss_ce
    
    return combined_loss