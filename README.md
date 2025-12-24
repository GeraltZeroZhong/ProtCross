# ProtCross

ProtCross: Bridging the PDB-AlphaFold Gap for Robust Binding Site Prediction on Protein Point Clouds.

## Overview
ProtCross is a research codebase for binding site prediction on protein point clouds with domain adaptation between experimentally determined PDB structures (source domain) and AlphaFold2 (AF2) structures (target domain). The pipeline builds point-cloud inputs with residue-level features (ESM embeddings + 3D coordinates), trains a PointNet++ backbone with a segmentation head, and optionally applies a domain discriminator with confidence-aware (pLDDT) weighting.

The project uses Hydra for configuration, PyTorch Lightning for training loops, and Torch Geometric for point-cloud data handling.

## Features
- **Point-cloud binding site segmentation** using a PointNet++ backbone and a segmentation head.
- **Domain adaptation (DANN-style)** with a gradient reversal layer to align PDB and AF2 feature distributions.
- **Confidence-aware weighting** based on AF2 pLDDT scores.
- **ESM feature extraction + PCA reduction** for residue embeddings.
- **Automated benchmarking scripts** for ablations and repeated experiments.
- **Hydra config overrides** to quickly toggle model variants and training settings.

## Installation
1. Create the conda environment (CUDA-enabled if available):

   ```bash
   conda env create -f environment.yml
   conda activate evopoint_da
   ```

2. (Optional) If you are using a CPU-only machine, adjust `environment.yml` to remove CUDA-specific packages or install compatible CPU wheels for PyTorch/Torch Geometric.

## Usage
### 1) Prepare data
Raw and processed data are expected in the `data/` folder :

```
./data/
  raw_pdb/          # Raw PDB or CIF files (you provide)
  raw_af2/          # AlphaFold2 PDB files
  processed_pdb/    # Preprocessed .pt files (source domain)
  processed_af2/    # Preprocessed .pt files (target domain)
```

**Download AF2 structures based on raw .pdb files**
```bash
python scripts/get_af2.py
```

**Preprocess ESM + PCA features**
Fit PCA (typically on PDB/source data):
```bash
python scripts/preprocess_esm.py \
  --data_dir data/raw_pdb \
  --output_dir data/processed_pdb \
  --fit_pca \
  --pca_model_path pca_128.pkl
```

Process AF2 data with the fitted PCA:
```bash
python scripts/preprocess_esm.py \
  --data_dir data/raw_af2 \
  --output_dir data/processed_af2 \
  --pca_model_path pca_128.pkl \
  --is_af2
```

** Map labels from PDB to AF2**
```bash
python scripts/map_labels.py
```

### 2) Train
```bash
python train.py
```

Hydra overrides can be used to change behavior, for example:
```bash
python train.py model.use_esm=False model.use_da=False model.feature_dim=128
```

### 3) Test
```bash
python test.py
```

### 4) Benchmark / repeated experiments
```bash
python run_benchmark.py
python run_multiseed_benchmark.py
python run_repeated_experiments.py
```

## Configuration
Hydra config files live in `configs/`:
- `configs/train.yaml` sets defaults and global paths.
- `configs/data/protein_seg.yaml` defines data locations and loader settings.
- `configs/model/da_module.yaml` controls model hyperparameters and feature flags.
- `configs/trainer/default.yaml` configures PyTorch Lightning Trainer settings.

You can override any config at the command line using `key=value` syntax. Examples:
```bash
# Change dataset paths
python train.py data.data_dir_pdb=/abs/path/to/processed_pdb data.data_dir_af2=/abs/path/to/processed_af2

# Disable domain adaptation
python train.py model.use_da=False

# Reduce epochs for quick experiments
python train.py trainer.max_epochs=10
```

## Folder Structure
```
.
├── configs/
│   ├── data/
│   │   └── protein_seg.yaml
│   ├── model/
│   │   └── da_module.yaml
│   ├── trainer/
│   │   └── default.yaml
│   └── train.yaml
├── scripts/
│   ├── get_af2.py
│   ├── map_labels.py
│   ├── plot_metrics.py
│   └── preprocess_esm.py
├── src/
│   └── evopoint_da/
│       ├── data/
│       │   ├── components.py
│       │   ├── datamodule.py
│       │   └── dataset.py
│       └── models/
│           ├── backbones/
│           │   └── pointnet2.py
│           ├── heads/
│           │   └── classifier.py
│           └── module.py
├── train.py
├── test.py
├── run_benchmark.py
├── run_evaluate_only.py
├── run_multiseed_benchmark.py
├── run_repeated_experiments.py
└── environment.yml
```
