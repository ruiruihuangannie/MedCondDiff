import torch
import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import math
import gzip

# Set device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_weights_freq = {
    'ACTK_inv': torch.tensor([0.0016, 0.0337, 0.1435, 0.2156, 0.6057]),
    'ACTK_log_inv': torch.tensor([0.0372, 0.1627, 0.2353, 0.2560, 0.3088]),
    'ACTK': torch.tensor([0.2, 0.2, 0.2, 0.2, 0.5]),
    'AMOS_inv': torch.tensor([2.0603e-05, 6.4349e-03, 9.2543e-03, 8.8369e-03, 4.4735e-02, 8.2542e-02, 1.0153e-03, 3.7108e-03, 1.1293e-02, 1.9747e-02, 1.8237e-02, 3.9609e-01, 3.3904e-01, 2.3168e-02, 1.0575e-02, 2.5295e-02]),
    'AMOS_log_inv': torch.tensor([0.0095, 0.0562, 0.0594, 0.0590, 0.0733, 0.0786, 0.0401, 0.0514, 0.0612, 0.0661, 0.0654, 0.0924, 0.0910, 0.0675, 0.0606, 0.0683]),
    'MSD_inv': torch.tensor([1.7318e-04, 8.6757e-02, 9.1307e-01]),
    'MSD_log_inv': torch.tensor([0.0448, 0.4017, 0.5535]),
    'AMOS-Small_inv': torch.tensor([1.6979e-05, 5.5548e-03, 1.0395e-02, 5.8678e-03, 5.0450e-02, 9.2521e-02, 1.0463e-03, 4.1703e-03, 1.2513e-02, 1.9170e-02, 2.0475e-02, 3.9523e-01, 3.1871e-01, 2.3099e-02, 1.2445e-02, 2.8328e-02]),
    'AMOS-Small_log_inv': torch.tensor([0.0054, 0.0538, 0.0599, 0.0543, 0.0753, 0.0812, 0.0376, 0.0510, 0.0617, 0.0658, 0.0665, 0.0954, 0.0933, 0.0677, 0.0616, 0.0696]),
    'Brats_inv': torch.tensor([0.0009, 0.1219, 0.4300, 0.4472]),
    'Brats_log_inv': torch.tensor([0.0388, 0.2732, 0.3429, 0.3451])
}

def calculate_class_weights(pt_files, lg=False):
    cnt, c, h, w = None, None, None, None

    flag = True
    for pt_file in tqdm(pt_files, desc="Loading tensors"):
        with gzip.open(pt_file, "rb") as f:
            tensor = torch.load(f, map_location=device)
        if flag:
            c, h, w = tensor.shape
            cnt = torch.zeros(c, dtype=torch.int64, device=device)
            flag = False

        for i in range(c):
            cnt[i] += (tensor[i] == 1).sum()

    class_weights = torch.zeros(c, dtype=torch.float32, device=device)
    total_pixels = h * w * len(pt_files)
    for i, count in enumerate(cnt):
        count = count.item() if isinstance(count, torch.Tensor) else count
        if lg:
            class_weights[i] = math.log(1 + total_pixels / count)
        else:
            class_weights[i] = total_pixels / count
    class_weights = class_weights / class_weights.sum()
    return class_weights

def calculate_sample_weights(gt_files, class_weights, name):
    class_weights = class_weights.to(device)
    sample_weights = []
    for gt_path in tqdm(gt_files, desc="Calculating sample weights"):
        with gzip.open(gt_path, "rb") as f:
            gt = torch.load(f, map_location=device)
        class_indices = torch.unique(torch.argmax(gt, dim=0))
        weights = class_weights[class_indices]
        sample_weights.append(torch.max(weights).item())
    torch.save(sample_weights, f"config/{name}/sample_weights.pt")
    return torch.tensor(sample_weights, dtype=torch.double, device=device)

if __name__ == "__main__":
    path, name = sys.argv[1], sys.argv[2]
    pt_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pt.gz')]
    weights = calculate_class_weights(pt_files)
    print(f'class weights: {weights}')
    log_weights = calculate_class_weights(pt_files, lg=True)
    print(f'log class weights: {log_weights}')
    sample_weights = calculate_sample_weights(pt_files, weights, name)
    print(f'pt_files: {len(pt_files)} sample weights: {sample_weights.shape}')