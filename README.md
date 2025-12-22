# MedCondDiff

This repository is the official implementation of [MedCondDiff: Lightweight, Robust, Semantically Guided Diffusion for Medical Image Segmentation](https://arxiv.org/abs/2512.00350). Our implementation is based on the PyTorch implementation of [DDPM](https://arxiv.org/abs/2006.11239).

<!-- We provide pretrained weights and inference results in the releases. -->

## Requirements
- Python 3.10
- CUDA 11.8 (or compatible)
- PyTorch 2.0+

To install requirements:

```bash
pip install -r requirements.txt
```

Or use the conda environment:

```bash
conda env create -f environment.yml
conda activate <env_name>
```

## Dataset
Use the [AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K) or its dataset variants.

For space efficiency, datasets should be placed in the `dataset/` directory with ground truths in `.pt.gz` format and predictions as `.png`.

## Run

To train the model, run:

```bash
accelerate launch train.py --config config/ACTK/MedCondDiff_352x352.yaml --num_epoch=100 --batch_size=32 --gradient_accumulate_every=1
```

To evaluate a model on datasets like ACTK:

```bash
accelerate launch sample.py --config config/ACTK/MedCondDiff_352x352.yaml --checkpoint wandb/model-11.pt --results_folder ./test/
```

The evaluation uses the `Eval` class in `utils/eval.py` to compute accuracy, mIoU, precision, recall, and Dice score.

Pretrained model weights are available in the releases.

## Citation
```
@misc{huang2025medconddifflightweightrobustsemantically,
      title={MedCondDiff: Lightweight, Robust, Semantically Guided Diffusion for Medical Image Segmentation}, 
      author={Ruirui Huang and Jiacheng Li},
      year={2025},
      eprint={2512.00350},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2512.00350}, 
}
```