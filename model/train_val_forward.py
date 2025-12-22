import torch
from torch import nn
import torch.nn.functional as F


def simple_train_forward(model: nn.Module, gt=None, image=None, **kwargs):
    assert gt is not None and image is not None
    return model(gt, image, **kwargs)


def simple_val_forward(model: nn.Module, gt=None, image=None, **kwargs):
    time_ensemble = kwargs.pop('time_ensemble') if 'time_ensemble' in kwargs else False
    gt_sizes = kwargs.pop('gt_sizes') if time_ensemble else None
    
    # device = next(model.parameters()).device
    # image = image.to(device)
    # if gt is not None:
    #     gt = gt.to(device)
    # if 'extra_cond' in kwargs and kwargs['extra_cond'] is not None:
    #     kwargs['extra_cond'] = kwargs['extra_cond'].to(device)
    
    pred = model.sample(image, **kwargs)

    if not time_ensemble:
        pred = F.softmax(pred, dim=1)

    if time_ensemble:
        preds = torch.stack(model.history, dim=1).detach().cpu()  # (N, T, C, H, W)
        pred = torch.mean(preds, dim=1)     # (N, C, H, W)

        def process(i, p, gt_size):
            '''
            Args:
                i: index in batch
                p: shape (C, H, W) — logits
                gt_size: (C, H_orig, W_orig) — original spatial size
            Returns:
                probs: (C, H_orig, W_orig) — softmax-normalized
            '''
            logits = F.interpolate(p.unsqueeze(0), size=gt_size[-2:], mode='bilinear', align_corners=False).squeeze(0)  # (C, H_orig, W_orig)
            probs = F.softmax(logits, dim=0)

            ps = F.interpolate(preds[i], size=gt_size[-2:], mode='bilinear', align_corners=False)  # (T, C, H_orig, W_orig)
            ps_soft = F.softmax(ps, dim=1)
            avg_ps = ps_soft.mean(dim=0)            # (C, H_orig, W_orig)
            confidence = avg_ps.max(dim=0).values   # (H_orig, W_orig)

            # mask = (confidence > 0.5).float()       # (H_orig, W_orig)
            # probs = probs * mask.unsqueeze(0)       
            # probs = probs / (probs.sum(dim=0, keepdim=True) + 1e-8) # (C, H_orig, W_orig)
            return probs

        pred = torch.stack(
            [process(index, p, gt_size) for index, (p, gt_size) in enumerate(zip(pred, gt_sizes))],
            dim = 0
        )
    return {
        "image": image, # (N, 1, H, W) in [-1, 1]
        "pred" : pred,  # (N, C, H/H_orig, H/W_orig) in softmax
        "gt"   : gt,    # (N, C, H, W) in one-hot
    }