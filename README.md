# Task 1 PyTorch Baseline

This package provides a simple multi-modal baseline for Task 1 blockage
prediction using RGB-D video frames and radio E2 features. It aligns video,
radio, and annotations by timestamp and predicts the blockage state at
`t + dt` (default `dt = 0.142s`).

## Setup
Create a conda environment and install dependencies:

```bash
conda env create -f baselines/task1_pytorch/environment.yml
conda activate task1-baseline
pip install -e baselines/task1_pytorch
```

Note: Dependencies are managed via the conda environment files; there is no
`requirements.txt` to install.

GPU setup (CUDA):

```bash
conda env create -f baselines/task1_pytorch/environment.gpu.yml
conda activate task1-baseline
pip install -e baselines/task1_pytorch
```

Training will automatically use CUDA if available, or you can force it with
`--device cuda`.

## Quick start
Train on exp1..exp4 and validate on exp5:

```bash
python -m task1_baseline.train \
  --dataset-root dataset \
  --train-scenarios exp1,exp2,exp3,exp4 \
  --val-scenarios exp5 \
  --video-mode rgbd \
  --image-size 128 \
  --epochs 15 \
  --batch-size 16
```

Leave-one-scenario-out preset (hold out exp5):

```bash
python -m task1_baseline.train \
  --dataset-root dataset \
  --preset loso \
  --holdout-scenario exp5 \
  --video-mode rgbd \
  --image-size 128 \
  --epochs 15 \
  --batch-size 16
```

Pretrained visual backbone (useful for small datasets):

```bash
python -m task1_baseline.train \
  --dataset-root dataset \
  --backbone resnet18 \
  --pretrained \
  --freeze-backbone \
  --video-mode rgbd \
  --epochs 10
```

Note: `--pretrained` downloads torchvision weights if not cached. If you are
offline, pre-download the weights once or copy them into your Torch cache.

Evaluate a saved checkpoint:

```bash
python -m task1_baseline.eval \
  --dataset-root dataset \
  --scenarios exp5 \
  --checkpoint runs/task1_baseline/best.pt
```

Write per-frame predictions:

```bash
python -m task1_baseline.predict \
  --dataset-root dataset \
  --scenarios exp5 \
  --checkpoint runs/task1_baseline/best.pt \
  --output runs/task1_baseline/exp5_predictions.csv
```

## Model overview
The Task 1 baseline uses a two-branch network to fuse video and radio:
- Visual branch: either a lightweight 3-layer CNN (stride-2 conv + BN + ReLU + global
  average pool) or a ResNet18 backbone. The first conv is adapted for RGB-D (4 ch),
  RGB (3 ch), or disparity-only (1 ch) inputs.
- Radio branch: a 2-layer MLP (Linear -> ReLU -> Dropout -> Linear -> ReLU) that
  embeds the E2 feature vector.
- Fusion: concatenate visual and radio features, then classify with a small MLP
  into 3 blockage classes (no/partial/full).
- Normalization: E2 features are standardized using train-set mean/std. Class
  weights can be auto-computed for cross-entropy via `--class-weights auto`.

## Performance
Best validation metrics observed in the current tests: val loss 0.1486, acc 0.973, macro F1 0.785

## Notes
- `dataset/index.csv` is used to locate frames, radio, and annotations.
- Alignment uses nearest-neighbor matching with configurable tolerances.
- `--video-mode none` enables a radio-only baseline.
