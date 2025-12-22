import glob
import os
from collections import defaultdict
from pathlib import Path
import random
import re
import math
import numpy as np
import torch
from tqdm import tqdm
import wandb
from accelerate import Accelerator
from utils.logger_utils import create_url_shortcut_of_wandb, create_logger_of_wandb
from utils.train_utils import SmoothedValue, set_random_seed
from utils.import_utils import fill_args_from_dict
from utils.convert_utils import onehot_to_greyscale, neg1pos1_to_greyscale, softmax_to_greyscale

from model.train_val_forward import simple_train_forward, simple_val_forward
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from PIL import Image


def get_random_subset_loader(dataset):
    indices = random.sample(range(len(dataset)), 1)
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=1, shuffle=False)

def run_on_seed(func):
    def wrapper(*args, **kwargs):
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        set_random_seed(0)
        res = func(*args, **kwargs)
        set_random_seed(seed)
        return res
    return wrapper


class Trainer(object):
    def __init__(
            self,
            model,
            train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader = None,
            train_forward_fn=simple_train_forward,
            val_forward_fn=simple_val_forward,
            gradient_accumulate_every=1,
            optimizer=None, scheduler=None,
            train_num_epoch=100,
            results_folder='./results',
            amp=False,
            fp16=False,
            split_batches=True,
            log_with='wandb',
            cfg=None,
    ):
        super().__init__()
        """
            Initialize the accelerator.
        """
        from accelerate import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no',
            log_with='wandb' if log_with else None,
            gradient_accumulation_steps=gradient_accumulate_every,
            kwargs_handlers=[ddp_kwargs]
        )
        project_name = getattr(cfg, "project_name", 'ResidualDiffsuion-v7')
        self.accelerator.init_trackers(project_name, config=OmegaConf.to_container(cfg, resolve=True))
        create_url_shortcut_of_wandb(accelerator=self.accelerator)
        self.logger = create_logger_of_wandb(accelerator=self.accelerator, rank=not self.accelerator.is_main_process)
        self.accelerator.native_amp = amp
        """
            Initialize the model and parameters.
        """
        self.model = model
        self.train_forward_fn = train_forward_fn
        self.val_forward_fn = val_forward_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.gradient_accumulate_every = gradient_accumulate_every
        # calculate training steps
        self.train_num_epoch = train_num_epoch
        # optimizer
        self.opt = optimizer

        if self.accelerator.is_main_process:
            # save results in wandb folder if results_folder is not specified
            self.results_folder = Path(results_folder if results_folder
                                       else os.path.join(self.accelerator.get_tracker('wandb', unwrap=True).dir, "../"))
            self.results_folder.mkdir(exist_ok=True)
        """
            Initialize the data loader.
        """
        self.cur_epoch = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt, self.scheduler, self.train_loader, self.test_loader \
            = self.accelerator.prepare(self.model, self.opt, scheduler, self.train_loader, self.test_loader)

    def save(self, epoch, max_to_keep=10):
        """
        Delete the old checkpoints to save disk space.
        """
        if not self.accelerator.is_local_main_process:
            return
        ckpt_files = glob.glob(os.path.join(self.results_folder, 'model-[0-9]*.pt'))
        # keep the last n-1 checkpoints
        ckpt_files = [f for f in ckpt_files if re.match(r'.*model-\d+\.pt$', os.path.basename(f))]
        ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))
        ckpt_files_to_delete = ckpt_files[:-max_to_keep]
        for ckpt_file in ckpt_files_to_delete:
            os.remove(ckpt_file)
        data = {
            'epoch': self.cur_epoch,
            'model': self.accelerator.get_state_dict(self.model),
            'scaler': self.accelerator.scaler.state_dict() if (self.accelerator.scaler is not None) else None
        }

        save_name = str(self.results_folder / f'model-{epoch}.pt')
        last_save_name = str(self.results_folder / f'model-{epoch}-last.pt')

        # if save file exists, rename it to last_save_name
        if os.path.exists(save_name):
            os.remove(last_save_name) if os.path.exists(last_save_name) else None
            os.rename(save_name, last_save_name)

        torch.save(data, save_name)

    def load(self, resume_path: str = None, pretrained_path: str = None):
        accelerator = self.accelerator
        device = accelerator.device

        if resume_path is not None:
            data = torch.load(resume_path, map_location=device)

            self.cur_epoch = data['epoch']
            if (self.accelerator.scaler is not None and data['scaler'] is not None):
                self.accelerator.scaler.load_state_dict(data['scaler'])

        elif pretrained_path is not None:
            data = torch.load(pretrained_path, map_location=device)
        else:
            raise ValueError('Must specify either milestone or path')
        if self.scheduler is not None:
            # step scheduler to the last epoch
            for _ in range(self.cur_epoch):
                self.scheduler.step()
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'], strict=False)

    @torch.inference_mode()
    @run_on_seed
    def val_time_ensemble(self, model, test_data_loader, accelerator, thresholding=False, save_to=None):
        """
        validation function
        """
        global _best_iou
        if '_best_iou' not in globals():
            _best_iou = 0.0

        model.eval()
        model = accelerator.unwrap_model(model)
        device = model.device

        total_intersection_per_class = None
        total_union_per_class = None
        num_classes = 0
        total_cross_entropy = 0.0
        total_samples = 0

        if len(test_data_loader) == 0:
            return 0.0, _best_iou

        for data in tqdm(test_data_loader, disable=not accelerator.is_main_process):
            image, gt_raw, name = data['image'], data['gt_raw'], data['name']
            image = image.to(device)
            
            if num_classes == 0:
                num_classes = gt_raw[0].shape[1]
                total_intersection_per_class = torch.zeros(num_classes, device=device, dtype=torch.long)
                total_union_per_class = torch.zeros(num_classes, device=device, dtype=torch.long)
                
            pred_ensem = self.val_forward_fn(model, 
                                             image=image, 
                                             time_ensemble=True,
                                             gt_sizes=[g.shape for g in gt_raw], 
                                             verbose=False)
            pred_ensem = pred_ensem["pred"]     # (N, C, H_orig, W_orig) in softmax

            # save to disk
            if save_to is not None:
                eps = 1e-8
                for pred, image_name, gt in zip(pred_ensem, name, gt_raw):
                    pred_g = softmax_to_greyscale(pred)
                    img = Image.fromarray(pred_g, mode='L')
                    img.save(os.path.join(save_to, image_name))
                    total_cross_entropy += (-torch.sum(gt * torch.log(pred.to('cpu') + eps))).item()
                    total_samples += (gt.shape[1] * gt.shape[2])

            for pred, gt in zip(pred_ensem, gt_raw):
                pred_labels = pred.to(device).argmax(dim=0)
                gt_labels = gt.to(device).argmax(dim=0)
                for c in range(1, num_classes):
                    pred_mask = (pred_labels == c)
                    gt_mask = (gt_labels == c)
                    total_intersection_per_class[c] += (pred_mask & gt_mask).sum()
                    total_union_per_class[c] += (pred_mask | gt_mask).sum()

        accelerator.wait_for_everyone()

        if save_to is not None:
            total_cross_entropy_tensor = torch.tensor(total_cross_entropy, device=device)
            total_samples_tensor = torch.tensor(total_samples, device=device)
            total_cross_entropy_tensor = accelerator.reduce(total_cross_entropy_tensor, reduction="sum")
            total_samples_tensor = accelerator.reduce(total_samples_tensor, reduction="sum")
            mean_ce_loss = (total_cross_entropy_tensor.item() / total_samples_tensor.item()) if total_samples_tensor.item() > 0 else 0.0
            self.logger.info(f"Mean Cross-Entropy: {mean_ce_loss}")

        if total_intersection_per_class is None: # for empty dataloader
             return 0.0, _best_iou
        
        total_intersection_per_class = accelerator.reduce(total_intersection_per_class, reduction="sum")
        total_union_per_class = accelerator.reduce(total_union_per_class, reduction="sum")

        # Exclude background class (c=0)
        iou_per_class = []
        for c in range(1, num_classes):
            intersection = total_intersection_per_class[c].float()
            union = total_union_per_class[c].float()
            if union > 0:
                iou_per_class.append(intersection / (union + 1e-8))

        mean_iou = torch.tensor(iou_per_class).mean().item() if iou_per_class else 0.0
        
        self.logger.info(f"iou_per_class: {iou_per_class}")
        _best_iou = max(_best_iou, mean_iou)
        return mean_iou, _best_iou

    def train(self):
        accelerator = self.accelerator
        for epoch in range(self.cur_epoch, self.train_num_epoch):
            self.cur_epoch = epoch
            # Train
            self.model.train()
            loss_sm = SmoothedValue(window_size=10)
            with tqdm(total=len(self.train_loader), disable=not accelerator.is_main_process) as pbar:
                for data in self.train_loader:
                    with accelerator.autocast(), accelerator.accumulate(self.model):
                        loss = fill_args_from_dict(self.train_forward_fn, data)(model=self.model)
                        accelerator.backward(loss)
                        accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.opt.step()
                        self.opt.zero_grad()
                    loss_sm.update(loss.item())
                    pbar.set_description(
                        f'Epoch:{epoch}/{self.train_num_epoch} loss: {loss_sm.avg:.4f}({loss_sm.global_avg:.4f})')
                    self.accelerator.log({'loss': loss_sm.avg, 'lr': self.opt.param_groups[0]['lr']})
                    pbar.update()

            if self.scheduler is not None:
                self.scheduler.step()

            accelerator.wait_for_everyone()
            loss_sm_gather = accelerator.gather(torch.tensor([loss_sm.global_avg]).to(accelerator.device))
            loss_sm_avg = loss_sm_gather.mean().item()
            self.logger.info(f'Epoch:{epoch}/{self.train_num_epoch} loss: {loss_sm_avg:.4f}')
            self.save(self.cur_epoch)

            # Val
            self.model.eval()
            if (epoch + 1) % 1 == 0 or (epoch >= self.train_num_epoch * 0.7):
                iou, best_iou = self.val_time_ensemble(self.model, self.test_loader, accelerator)
                self.logger.info(f'Epoch:{epoch}/{self.train_num_epoch} iou: {iou:.4f}({best_iou:.4f})')
                accelerator.log({'iou': iou, 'best_iou': best_iou})
                if iou == best_iou:
                    self.save("best")

            # Visualize
            with torch.inference_mode():
                if accelerator.is_main_process:
                    model = self.accelerator.unwrap_model(self.model)
                    device = accelerator.device
                    for tracker in accelerator.trackers:
                        if tracker.name == "wandb":
                            data = next(iter(get_random_subset_loader(self.test_loader.dataset)))
                            for k, v in data.items():
                                if isinstance(v, torch.Tensor):
                                    data[k] = v.to(device)
                                elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                                    data[k] = [t.to(device) for t in v]
                            out = fill_args_from_dict(self.val_forward_fn, data)(model=model, verbose=False)
                            
                            img = neg1pos1_to_greyscale(out['image'][0].squeeze()) # (H, W) in [-1, 1] -> (H, W) in [0, 255]
                            pred = softmax_to_greyscale(out['pred'][0]) # (C, H, W) in softmax to (H, W) in greyscale [0.5, 0.73]
                            gt = onehot_to_greyscale(out['gt'][0]) # (C, H, W) in one-hot to (H, W) in greyscale

                            tracker.log({
                                'validation': [
                                    wandb.Image(img, caption="Img"),
                                    wandb.Image(pred, caption="Pred"), 
                                    wandb.Image(gt, caption=data["name"][0])
                                ]
                            })

            accelerator.wait_for_everyone()
        self.logger.info('training complete')
        accelerator.end_training()