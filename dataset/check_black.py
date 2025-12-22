import torch
import os
import sys
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import gzip

def check_pt(dir, save_path):
    image_dir = Path(dir)
    pt_files = list(image_dir.rglob("*.pt.gz"))
    black_files = []
    for file_path in tqdm(pt_files, desc="Checking .pt images"):
        try:
            with gzip.open(file_path, "rb") as f:
                tensor = torch.load(f)
            if torch.all(tensor[0] == 1):
                black_files.append(str(file_path))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    with open(save_path, "w") as f:
        for file in black_files:
            f.write(file + "\n")
    return len(black_files), len(pt_files)

def check_png(dir, save_path):
    image_dir = Path(dir)
    png_files = list(image_dir.rglob("*.png"))
    black_files = []
    for file_path in tqdm(png_files, desc="Checking .png images"):
        image = Image.open(file_path).convert('L')
        image_tensor = torch.from_numpy(np.array(image))
        if torch.all(image_tensor == 0):
            black_files.append(str(file_path))

    with open(save_path, "w") as f:
        for file in black_files:
            f.write(file + "\n")
    return len(black_files), len(png_files)

if __name__ == "__main__":
    if sys.argv[3] == 'pt':
        o1, o2 = check_pt(sys.argv[1], sys.argv[2])
    elif sys.argv[3] == 'png':
        o1, o2 = check_png(sys.argv[1], sys.argv[2])
    else:
        raise ValueError("Invalid input, the third argument should be 'pt' or 'png'")

    print(f"Found {o1} black images out of {o2} total.")