import torch
import torch.nn as nn
import numpy as np
from monai.metrics import compute_hausdorff_distance

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0): 
        super().__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice_score

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    def forward(self, pred_logits, target):
        bce = self.bce_loss(pred_logits, target)
        pred_probs = torch.sigmoid(pred_logits)
        dice = self.dice_loss(pred_probs, target)
        return self.bce_weight * bce + self.dice_weight * dice

def calculate_dice_score(pred, target, smooth=1.0):
    pred_flat, target_flat = pred.view(-1), target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)).item()

def calculate_accuracy(pred, target):
    return ((pred > 0.5).float() == target).float().mean().item()

def get_hd95(pred, target):
    try:
        if pred.shape[0] > 0:
            hd_val = compute_hausdorff_distance(pred, target, include_background=True, percentile=95)
            if not np.isnan(hd_val.mean()):
                return hd_val.mean().item()
    except:
        pass
    return 0.0