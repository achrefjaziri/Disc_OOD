# 

This repository contains the implementation of DISC (Diffusion-Based Statistical Characterization) from our AISTATS Paper "Beyond Binary Out-of-Distribution Detection: Characterizing
Distributional Shifts with Multi-Statistic Diffusion Trajectories" 


## Usage
To replicate the main results 

```bash
python scripts/compute_features.py --config configs/imagenet_disc.yaml
python scripts/run_evaluation.py --config configs/imagenet_disc.yaml
python scripts/make_tables.py --config configs/imagenet_disc.yaml
```

For quick checks, feature extraction also supports:

```bash
python scripts/compute_features.py --config configs/imagenet_disc.yaml --smoke_test
python scripts/compute_features.py --config configs/imagenet_disc.yaml --datasets imagenet_id cifar10 --max_samples 32
```

## What DISC Computes

For each input image and selected diffusion timestep, the feature extractor can
record:

- pixel reconstruction statistics: MSE, SSIM, LPIPS
- texture/frequency divergences: LBP, HOG, intensity, and optional DTCWT metrics
- optional local complexity from activation sign changes around noisy samples

Feature files are saved per dataset as compressed NumPy archives so evaluation
can be rerun without recomputing diffusion trajectories.
Final archives are written to `outputs/features/*_disc_features.npz`; partial
run shards are kept under `outputs/features/_shards/` and merged at the end.

## Data

All paths are configured in `configs/imagenet_disc.yaml`. The default config uses
environment-variable placeholders rather than local absolute paths:

- `IMAGENET_VAL_ROOT`
- `IMAGENET_A_ROOT`
- `IMAGENET_O_ROOT`
- `IMAGENET_C_ROOT`
- `IMAGENET64_CKPT`
- `TORCHVISION_DATA_ROOT`

ImageNet-A, ImageNet-O, and ImageNet-C are expected as image folders. CIFAR-10
and MNIST are loaded through `torchvision`.


## Evaluation

The evaluation script reports:

1. Binary ID-vs-OOD AUROC with `IsolationForest` fitted on ID DISC features.
2. Multi-OOD clustering accuracy with `KMeans` on OOD feature vectors.
3. Supervised OOD-family classification accuracy with sklearn `MLPClassifier`.

Results are written under `outputs/results/` by default as
`binary_auroc.csv`, `clustering_accuracy.csv`, `supervised_accuracy.csv`, and
`summary.json`.
