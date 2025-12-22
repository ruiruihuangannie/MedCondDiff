import os
import numpy as np
import gzip
import torch
from tqdm import tqdm
from PIL import Image


class Eval:
    def __init__(self, gt_root, pred_root, num_channels, verbose=True, log_with=None, image_root=None):
        self.num_channels = num_channels
        self.verbose = verbose
        self.log_with = log_with

        self.gt_fns = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.pt.gz')]
        self.pred_fns = [os.path.join(pred_root, f) for f in os.listdir(pred_root) if f.endswith('.png')]
        self.image_root = image_root
        self.gt_fns.sort()
        self.pred_fns.sort()
        assert len(self.gt_fns) == len(self.pred_fns), "Number of gt and pred files must be the same, gt_fns: {}, pred_fns: {}".format(len(self.gt_fns), len(self.pred_fns))
        
        self.value_to_class = {0: 0, 64: 1, 128: 2, 191: 3, 255: 4}
        self.gts = self._load_gt()
        self.preds = self._load_pred(self.gts[0].shape)

    def _calc_accuracy(self):
        total, correct = 0, 0
        for pred, gt in tqdm(zip(self.preds, self.gts), total=len(self.preds), desc="Calculating Accuracy", disable=not self.verbose):
            correct += np.sum(pred == gt)
            total += gt.size
        return correct / total if total > 0 else 0.0

    def _calc_iou(self):
        intersection_per_class = np.zeros(self.num_channels)
        union_per_class = np.zeros(self.num_channels)
        for pred, gt in tqdm(zip(self.preds, self.gts), total=len(self.preds), desc="Calculating IoU", disable=not self.verbose):
            for c in range(self.num_channels):
                pred_mask = (pred == c)
                gt_mask = (gt == c)
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                intersection_per_class[c] += intersection
                union_per_class[c] += union
        eps = 1e-6
        iou_per_class = intersection_per_class / (union_per_class + eps)
        return np.mean(iou_per_class)

    def _calc_f1(self):
        tp_lst = np.zeros(self.num_channels)
        fp_lst = np.zeros(self.num_channels)
        fn_lst = np.zeros(self.num_channels)
        for pred, gt in tqdm(zip(self.preds, self.gts), total=len(self.preds), desc="Calculating F1 scores", disable=not self.verbose):
            for k in range(self.num_channels):
                tp_lst[k] += np.logical_and(pred == k, gt == k).sum()
                fp_lst[k] += np.logical_and(pred == k, gt != k).sum()
                fn_lst[k] += np.logical_and(pred != k, gt == k).sum()
        prc_denom = tp_lst + fp_lst
        prc_per_class = np.divide(tp_lst, prc_denom, out=np.zeros_like(tp_lst), where=prc_denom!=0)

        rcl_denom = tp_lst + fn_lst
        rcl_per_class = np.divide(tp_lst, rcl_denom, out=np.zeros_like(tp_lst), where=rcl_denom!=0)
        
        f1_denom = prc_per_class + rcl_per_class
        f1_per_class = np.divide(2 * (prc_per_class * rcl_per_class), f1_denom, out=np.zeros_like(tp_lst), where=f1_denom!=0)
        
        return {
            'precision': np.mean(prc_per_class),
            'recall': np.mean(rcl_per_class),
            'dice': np.mean(f1_per_class)
        }
    
    def _load_gt(self):
        gt_list = []
        for idx in tqdm(range(len(self.gt_fns)), total=len(self.gt_fns), desc="Loading GT", disable=not self.verbose):
            with gzip.open(self.gt_fns[idx], "rb") as f:
                gt = torch.load(f)
            gt_arr = torch.argmax(gt, dim=0).cpu().numpy()
            gt_list.append(gt_arr)
        return gt_list

    def _load_pred(self, gt_shape):
        pred_list = []
        for idx in tqdm(range(len(self.pred_fns)), total=len(self.pred_fns), desc="Loading Pred", disable=not self.verbose):
            img = np.array(Image.open(self.pred_fns[idx]).convert('L'))
            if img.shape != gt_shape:
                img = np.array(Image.fromarray(img).resize(gt_shape[::-1], Image.NEAREST))
            pred_class = np.vectorize(self.value_to_class.get)(img)
            pred_list.append(pred_class)
        return pred_list
    
    def _calc_hi(self):
        tp_lst = np.zeros(self.num_channels)
        fp_lst = np.zeros(self.num_channels)
        fn_lst = np.zeros(self.num_channels)
        for pred, gt in tqdm(zip(self.preds, self.gts), total=len(self.preds), desc="Calculating HI scores", disable=not self.verbose):
            for k in range(self.num_channels):
                tp_lst[k] += np.logical_and(pred == k, gt == k).sum()
                fp_lst[k] += np.logical_and(pred == k, gt != k).sum()
        eps = 1e-8
        pred_lst = tp_lst + fp_lst
        hi_per_class = fp_lst/(pred_lst+eps)
        
        return np.mean(hi_per_class)

    def evaluate(self):
        self._wandb_logging()
        return {
            'acc': self._calc_accuracy(),
            'mIoU': self._calc_iou(),
            'f1-scores': self._calc_f1(),
            'hi-index': self._calc_hi()
        }

    def _wandb_logging(self):
        if self.log_with is not None and self.image_root is not None:
            import wandb

            image_fns = [os.path.join(self.image_root, f) for f in os.listdir(self.image_root) if f.endswith('.png')]
            image_fns.sort()
            assert len(self.gt_fns) == len(image_fns)
            for gt_arr, pred_fn, image_fn in tqdm(zip(self.gts, self.pred_fns, image_fns), total=len(self.gts), desc="Logging to WandB", disable=not self.verbose):
                gt = np.round((gt_arr * 255 / (self.num_channels - 1))).astype(np.uint8)
                pred = Image.open(pred_fn).convert('L')
                image = Image.open(image_fn).convert('L')
                wandb.log({
                    'Test': [
                        wandb.Image(image, caption=f"Input Image:{os.path.basename(image_fn)}"),
                        wandb.Image(pred, caption="Prediction"),
                        wandb.Image(gt, caption="Ground Truth"),    
                    ]
                })