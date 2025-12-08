import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, true):
        # only for binary segmentation so no need to one-hot encode
        # apply softmax to get probabilities
        probs = F.softmax(pred, dim=1)

        # Select foreground probability (class 1) only glacier class
        probs_fg = probs[:, 1, :, :] # shape (B, H, W)
        true_fg = (true == 1).float() 
        
        # flatten tensors
        probs_fg = probs_fg.view(probs_fg.size(0), -1)
        true_fg = true_fg.view(true_fg.size(0), -1)

        # computation of dice loss
        intersection = (probs_fg * true_fg).sum(dim=1)
        union = probs_fg.sum(dim=1) + true_fg.sum(dim=1)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice.mean()
        return loss
    
class CE_DiceLoss(nn.Module):
    def __init__(self, weights_ce=None, weights_dice=None, smooth=1):
        super(CE_DiceLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.weights_ce = weights_ce
        self.weights_dice = weights_dice
        
    def forward(self, pred, true):
        ce = self.weights_ce * self.ce_loss(pred, true) if self.weights_ce is not None else self.ce_loss(pred, true)
        dice = self.weights_dice * self.dice_loss(pred, true) if self.weights_dice is not None else self.dice_loss(pred, true)
        return ce + dice
