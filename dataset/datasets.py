import os
import sys
from glob import glob
from typing import Dict, Tuple
import gzip

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.init_env import resolve_path
from dataset.helper import calculate_class_weights, calculate_sample_weights

class dataset(Dataset):
    def __init__(self, image_root, gt_root, size, split, name = 'Brats'):
        """
        General dataset class for both training and testing.
        Args:
            image_root (str): Path to image directory
            gt_root (str): Path to ground truth directory  
            size (int): Size to resize images
            split (str): 'train', 'val', or 'test' mode
        """
        super().__init__()
        self.resize = size
        self.split = split
        self.index = 0
        
        image_root = resolve_path(image_root)
        gt_root = resolve_path(gt_root)
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.png') or f.endswith('.pt.gz')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.pt.gz')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        assert len(self.images) == len(self.gts), f"Mismatch: {len(self.images)} images vs {len(self.gts)} ground truth files"

        if self.split == 'train':
            # class_weights = calculate_class_weights(self.gts)
            # self.sample_weights = calculate_sample_weights(self.gts, class_weights)
            self.sample_weights = torch.load(f"config/{name}/sample_weights.pt")
            print(f'Finish loading pre-calculated sample weights')

        # set transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),                      # Converts to [0, 1]
            transforms.Lambda(lambda x: x * 2.0 - 1.0)  # Normalize to [-1, 1]
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize), interpolation=transforms.InterpolationMode.NEAREST),
        ])
            
        self.dataset_size = len(self.images)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx: int):
        
        raw_img = self.images[idx]
        if raw_img.endswith('.pt.gz'):
            with gzip.open(raw_img, "rb") as f:
                image = torch.load(f)
                image = transforms.functional.to_pil_image(image)
        else:
            image = self.binary_loader(raw_img)

        with gzip.open(self.gts[idx], "rb") as f:
            gt = torch.load(f)

        data = {}
        data['image']  = self.img_transform(image) # (H, W) in [-1, 1]
        if self.split in ['val', 'test']:
            data['gt_raw'] = gt                        # (C, H_orig, W_orig) in one-hot
        data['gt']     = self.gt_transform(gt)     # (C, H, W) in one-hot
        data['name']   = self.images[idx].split('/')[-1]       
        return data

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    
    def _load_data(self):
        """Load data method for compatibility with test_dataset interface"""

        assert self.split == 'test', "load_data() is only available in test mode"

        raw_img = self.images[idx]
        if raw_img.endswith('.pt.gz'):
            with gzip.open(raw_img, "rb") as f:
                image = torch.load(f)
        else:
            image = self.binary_loader(raw_img)
            
        with gzip.open(self.gts[self.index], "rb") as f:
            gt_raw = torch.load(f)

        image  = self.img_transform(image)
        gt     = self.gt_transform(gt_raw)
        name   = self.images[self.index].split('/')[-1]
        
        self.index += 1
        self.index = self.index % self.dataset_size

        return image, gt_raw, gt, name

    def __iter__(self):
        """Iterator for compatibility with test_dataset interface"""

        assert self.split == 'test', "Iterator is only available in test mode"
        for i in range(self.dataset_size):
            yield self._load_data()