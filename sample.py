from utils import init_env

import argparse

import torch
from utils.collate_utils import collate
from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str
from utils.init_utils import add_args
# from utils.train_utils import set_random_seed
from torch.utils.data import DataLoader
from utils.trainer import Trainer

import sys
import os
from pathlib import Path

# set_random_seed(7)


def get_loader(cfg):
    test_dataset = instantiate_from_config(cfg.test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate  
    )
    return test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--results_folder', type=str, default='./results')
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_sample_steps', type=int, default=10)
    parser.add_argument('--log_with', type=str, default=None)

    cfg = add_args(parser)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cfg.num_sample_steps is not None:
        cfg.diffusion_model.params.num_sample_steps = cfg.num_sample_steps

    cond_uvit = instantiate_from_config(cfg.cond_uvit, conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass))
    model = recurse_instantiate_from_config(cfg.model, unet=cond_uvit)
    diffusion_model = instantiate_from_config(cfg.diffusion_model, model=model)

    test_loader = get_loader(cfg)
    test_name = cfg.test_dataset.name

    optimizer = instantiate_from_config(cfg.optimizer, params=model.parameters())

    trainer = Trainer(
        diffusion_model,
        train_loader=None, test_loader=None,
        val_forward_fn=get_obj_from_str(cfg.val_forward_fn),
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        results_folder=cfg.results_folder,
        optimizer=optimizer,
        train_num_epoch=cfg.num_epoch,
        amp=cfg.fp16,
        log_with=cfg.log_with,
        cfg=cfg,
    )

    trainer.load(pretrained_path=cfg.checkpoint)
    
    test_loader = trainer.accelerator.prepare(test_loader)

    trainer.model.eval()
    gt_root = cfg.test_dataset.params.gt_root
    image_root = cfg.test_dataset.params.image_root
    os.makedirs(cfg.results_folder, exist_ok=True)

    iou, _ = trainer.val_time_ensemble(
        model=trainer.model,
        test_data_loader=test_loader,
        accelerator=trainer.accelerator,
        thresholding=False, 
        save_to=cfg.results_folder)
    trainer.accelerator.wait_for_everyone()
    trainer.accelerator.print(f'{cfg.test_dataset.name} mIoU: {iou}')

    if trainer.accelerator.is_main_process:
        from utils.eval import Eval
        evaluator = Eval(
            gt_root = gt_root,
            pred_root = cfg.results_folder,
            num_channels=cfg.num_channels,
            log_with=cfg.log_with,
            image_root=image_root,
        )
        results = evaluator.evaluate()
        trainer.accelerator.print(results)
        with open(os.path.join(cfg.results_folder, 'results.txt'), 'w') as f:
            f.write(str(results))

    trainer.accelerator.wait_for_everyone()