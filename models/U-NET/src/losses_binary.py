import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, true):
        # only for binary segmentation so no need to one-hot encode
        # apply sigmoid to get probabilities
        probs = torch.sigmoid(pred)
        
        # flatten tensors
        probs_fg = probs.view(probs.size(0), -1)
        true_fg = true.view(true.size(0), -1)

        # computation of dice loss
        intersection = (probs_fg * true_fg).sum(dim=1)
        union = probs_fg.sum(dim=1) + true_fg.sum(dim=1)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice.mean()
        return loss
    
class DiceBCELoss(nn.Module):
    def __init__(self, weights_ce=None, weights_dice=None, smooth=1):
        super(DiceBCELoss, self).__init__()
        self.ce_loss = nn.BCEWithLogitsLoss()                                   # binary cross-entropy loss, apply a sigmoid inside
        self.dice_loss = BinaryDiceLoss(smooth=smooth)                          # dice loss 
        self.weights_ce = weights_ce if weights_ce is not None else 1.0
        self.weights_dice = weights_dice if weights_dice is not None else 1.0
        
    def forward(self, pred, true):
        ce = self.weights_ce * self.ce_loss(pred, true)
        dice = self.weights_dice * self.dice_loss(pred, true)
        return ce + dice
