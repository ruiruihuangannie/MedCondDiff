import os
import sys

import numpy as np
from PIL import Image
import torch
import nibabel as nib
from tqdm import tqdm
import gzip

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.init_env import resolve_path

class Slicer:
    """
    Slice 3D medical images (e.g., .nii.gz files) into 2D slices and organize them into train/test/validation sets.
      
    Args:
        image_root (str): Path to directory containing 3D image files
        gt_root (str): Path to directory containing 3D ground truth/mask files  
        split (list): Train/validation/test split ratios as [train, val, test] summing to 10
        num_classes (int): Number of classes in the ground truth
        format (str): File extension to process (default: '.nii.gz')
        verbose (bool): Whether to print progress information
        
    Raises:
        ValueError: If split ratios don't sum to 10 or wrong number of splits provided
        AssertionError: If number of image and ground truth files don't match
    """
    def __init__(self, image_root, gt_root, num_classes, split = [8, 1, 1], format = '.nii.gz', verbose = True):
        self.image_root = resolve_path(image_root)
        self.gt_root = resolve_path(gt_root)
        self.format = format
        self.num_classes = int(num_classes)
        self.verbose = verbose
        self.image_slice_dir = os.path.dirname(self.image_root)
        self.gt_slice_dir = os.path.dirname(self.gt_root)

        if self.verbose:
            print(f"[Info] Loading images from: {self.image_root}")
            print(f"[Info] Loading ground truth from: {self.gt_root}")

        self.raw_3d_images = [os.path.join(self.image_root, f) for f in os.listdir(self.image_root) if f.endswith(self.format)]
        self.raw_3d_gts = [os.path.join(self.gt_root, f) for f in os.listdir(self.gt_root) if f.endswith(self.format)]
        self.raw_3d_images.sort()
        self.raw_3d_gts.sort()

        assert len(self.raw_3d_images) == len(self.raw_3d_gts), f"Mismatch: {len(self.raw_3d_images)} images vs {len(self.raw_3d_gts)} ground truth files"
        size_3d = len(self.raw_3d_images)
      
        if len(split) != 3 or sum(split) != 10:
            raise ValueError("Split must be a list of 3 numbers that sum to 10 (e.g., [8, 1, 1])")

        train = int(size_3d / 10.0 * split[0])
        val  = int(size_3d / 10.0 * split[1])
        test   = size_3d - train - val
        self.split = [train, val, test]

    def extract(self):
        if len(self.raw_3d_images) > 0:
            if self.verbose:
                print(f"[Info] Extracting slices from image {self.format} files...")
            self._extract_images()
            if self.verbose:
                print(f"[Info] Extracting slices from ground truth {self.format} files...")
            self._extract_gts()

    def _extract_images(self):
        """
        Extract slices from 3D/4D nii.gz files and save as PNG (for 2D) or .pt.gz (for 3D/4D) images.
        Handles both single-channel (grayscale) and multi-channel images.
        """
        self.tr_folder = os.path.join(self.image_slice_dir, "TrainImage")
        self.vl_folder = os.path.join(self.image_slice_dir, "ValImage")
        self.ts_folder = os.path.join(self.image_slice_dir, "TestImage")

        os.makedirs(self.tr_folder, exist_ok=True)
        os.makedirs(self.vl_folder, exist_ok=True)
        os.makedirs(self.ts_folder, exist_ok=True)

        for idx, fn in tqdm(enumerate(self.raw_3d_images), desc='Slicing images', total=len(self.raw_3d_images)):
            img_nib = nib.load(fn)
            img_data = img_nib.get_fdata()
            base_name = os.path.splitext(os.path.splitext(os.path.basename(fn))[0])[0]  # Remove .nii.gz

            # Determine target folder
            if idx < self.split[0]:
                target_folder = self.tr_folder
            elif idx < self.split[0] + self.split[1]:
                target_folder = self.vl_folder
            else:
                target_folder = self.ts_folder

            # 3D image (single channel): shape (H, W, D)
            if img_data.ndim == 3:
                p_low, p_high = np.percentile(img_data, (0.5, 99.5))
                for i in range(img_data.shape[2]):
                    img_slice = img_data[:, :, i]
                    img_slice = np.clip(img_slice, p_low, p_high)
                    img_slice = ((img_slice - p_low) / (p_high - p_low)) * 255.0
                    img_slice = img_slice.astype(np.uint8)
                    img_slice_path = os.path.join(target_folder, f"{base_name}_slice_{i:03d}.png")
                    Image.fromarray(img_slice, mode="L").save(img_slice_path)

            # 4D image (multi-channel or 3D volume per channel): shape (H, W, D, C)
            elif img_data.ndim == 4:
                # Save each slice (across D) as a tensor of shape (C, H, W)
                for i in range(img_data.shape[2]):
                    pt_slice = img_data[:, :, i, :]  # shape (H, W, C)
                    pt_slice_tensor = []
                    for j in range(pt_slice.shape[2]):
                        cur = pt_slice[:, :, j]
                        p_low, p_high = np.percentile(cur, (0.5, 99.5))
                        cur = np.clip(cur, p_low, p_high)
                        if p_high == 0:
                            cur = np.zeros_like(cur, dtype=np.uint8)
                        else:
                            cur = (((cur - p_low) / (p_high - p_low)) * 255.0).astype(np.uint8)
                        pt_slice_tensor.append(torch.from_numpy(cur))
                    # Stack to (C, H, W)
                    pt_slice_tensor = torch.stack(pt_slice_tensor, dim=0)
                    pt_slice_path = os.path.join(target_folder, f"{base_name}_slice_{i:03d}.pt.gz")
                    with gzip.open(pt_slice_path, "wb") as f:
                        torch.save(pt_slice_tensor, f)
            else:
                raise ValueError(f"Unsupported image ndim: {img_data.ndim} for file {fn}")

    def _extract_gts(self):
        """ Extract 2D slices from nii.gz files and save as PNG images. """

        self.tr_folder = os.path.join(self.gt_slice_dir, "TrainGT")
        self.vl_folder = os.path.join(self.gt_slice_dir, "ValGT")
        self.ts_folder = os.path.join(self.gt_slice_dir, "TestGT")

        os.makedirs(self.tr_folder, exist_ok=True)
        os.makedirs(self.vl_folder, exist_ok=True)
        os.makedirs(self.ts_folder, exist_ok=True)

        for idx, fn in tqdm(enumerate(self.raw_3d_gts), desc='Slicing ground truth', total=len(self.raw_3d_gts)):
            gt_nib = nib.load(fn)
            gt_data = gt_nib.get_fdata()
            
            if idx < self.split[0]:
                target_folder = self.tr_folder
            elif idx < self.split[0] + self.split[1]:
                target_folder = self.vl_folder
            else:
                target_folder = self.ts_folder

            for i in range(gt_data.shape[2]):
                gt_slice = gt_data[:, :, i]
                base_name = os.path.splitext(os.path.splitext(os.path.basename(fn))[0])[0]  # Remove .nii.gz
                gt_slice_path = os.path.join(target_folder, f"{base_name}_slice_{i:03d}.pt.gz")
                
                gt_slice = gt_slice.astype(np.uint8)
                one_hot = torch.zeros((self.num_classes, gt_slice.shape[0], gt_slice.shape[1]))
                for i in range(self.num_classes):
                    one_hot[i][gt_slice == i] = 1

                with gzip.open(gt_slice_path, "wb") as f:
                    torch.save(one_hot, f)

def test_gt_slicer(fn):
    with gzip.open(fn, "rb") as f:
        gt_file = torch.load(f)
    composite_array = np.zeros((gt_file.shape[1], gt_file.shape[2]), dtype=np.uint8)
    for i in range(gt_file.shape[0]):
        channel_data = gt_file[i].numpy()
        composite_array[channel_data == 1] = i * 80
    composite_img = Image.fromarray(composite_array.astype(np.uint8))
    output_path = fn.replace('.pt.gz', '.png')
    composite_img.save(output_path)

def test_img_slicer(fn):
    with gzip.open(fn, "rb") as f:
        img_file = torch.load(f)
    for i in range(img_file.shape[0]):
        composite_img = Image.fromarray(img_file[i].numpy().astype(np.uint8))
        output_path = fn.replace('.pt.gz', f'_{i}.png')
        composite_img.save(output_path)

if __name__ == '__main__':
    if len(sys.argv) == 3:
        # python slicer.py /path/to/ground_truth_file.pt
        if sys.argv[2] == 'gt':
            test_gt_slicer(sys.argv[1])
        elif sys.argv[2] == 'img':
            test_img_slicer(sys.argv[1])

    elif len(sys.argv) == 4:
        # python slicer.py /path/to/images /path/to/ground_truth num_classes (include background)
        slicer = Slicer(sys.argv[1], sys.argv[2], sys.argv[3])
        slicer.extract()
    else:
        print("Usage: python slicer.py <image_root> <gt_root> <num_classes>")