import torch
import torch.nn.functional as F
from dataset.helper import loss_weights_freq

"""
pred: (N, C, H, W) - logits before softmax
gt:   (N, C, H, W) - ground truth in one-hot encoding
"""
def dice_loss(pred, gt, class_weight, eps=1e-5):
    channels = pred.shape[1]
    pred_soft = F.softmax(pred, dim=1)  # (N, C, H, W)

    intersection = torch.sum(pred_soft * gt, dim=(0, 2, 3))  # (C,)
    union = torch.sum(pred_soft + gt, dim=(0, 2, 3))         # (C,)
    dice = (2. * intersection + eps) / (union + eps)         # (C,)
    weighted_loss = (1 - dice) * class_weight
    return weighted_loss.sum()

def ce_loss(pred, gt, class_weight):
    gt_indices = torch.argmax(gt, dim=1)  # (N, H, W)
    return F.cross_entropy(pred, gt_indices, weight=class_weight)

def dice_ce_loss(pred, gt, dice_weight=0.3, name = 'Brats'):
    class_weight = loss_weights_freq[f'{name}_log_inv'].to(pred.device)
    dice = dice_loss(pred, gt, class_weight)
    ce = ce_loss(pred, gt, class_weight) 
    return dice_weight * dice + (1 - dice_weight) * ce