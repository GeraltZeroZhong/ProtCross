# ProtCross

ProtCross is a domain-adaptive protein point-cloud learning framework for binding-site prediction across experimentally solved **PDB** structures and predicted **AlphaFold2 (AF2)** structures.

The codebase combines:
- residue-level structural geometry (Cα coordinates),
- language-model residue embeddings (ESM-C), and
- confidence-aware domain adaptation (pLDDT-weighted DANN)

to improve robustness when transferring from PDB (source domain) to AF2 (target domain).

---

## Table of Contents
- [1. Project Overview](#1-project-overview)
- [2. Repository Layout](#2-repository-layout)
- [3. Installation](#3-installation)
  - [3.1 System Requirements](#31-system-requirements)
  - [3.2 Create Environment](#32-create-environment)
  - [3.3 ESM-C Model Weights (Required)](#33-esm-c-model-weights-required)
  - [3.4 Verify Installation](#34-verify-installation)
- [4. Data Preparation](#4-data-preparation)
- [5. Usage](#5-usage)
  - [5.1 Preprocess Source (PDB) with PCA Fit](#51-preprocess-source-pdb-with-pca-fit)
  - [5.2 Preprocess Target (AF2) with Shared PCA](#52-preprocess-target-af2-with-shared-pca)
  - [5.3 Map Labels from PDB to AF2](#53-map-labels-from-pdb-to-af2)
  - [5.4 Train](#54-train)
  - [5.5 Evaluate / Test](#55-evaluate--test)
  - [5.6 Run Multi-seed Benchmark](#56-run-multi-seed-benchmark)
  - [5.7 Single-structure Prediction](#57-single-structure-prediction)
- [6. Configuration Guide (Hydra)](#6-configuration-guide-hydra)
- [7. Troubleshooting](#7-troubleshooting)
- [8. License](#8-license)

---

## 1. Project Overview

### Core capabilities
- **Binding-site segmentation on protein point clouds** with PointNet++.
- **Domain adaptation (DANN-style)** via gradient reversal and domain discriminator.
- **AF2 confidence-aware weighting** based on pLDDT.
- **ESM-C embeddings + PCA reduction** for residue features.
- **Hydra-driven experiment control** with easy command-line overrides.

### Primary stack
- PyTorch + PyTorch Lightning
- Torch Geometric
- Hydra
- ESM (EvolutionaryScale)

---

## 2. Repository Layout

```text
ProtCross/
├── configs/
│   ├── data/protein_seg.yaml
│   ├── model/da_module.yaml
│   ├── trainer/default.yaml
│   └── train.yaml
├── data/
│   ├── raw_pdb/
│   ├── raw_af2/
│   ├── processed_pdb/
│   └── processed_af2/
├── scripts/
│   ├── preprocess_esm.py
│   ├── get_af2.py
│   ├── map_labels.py
│   └── ...
├── src/evopoint_da/
│   ├── data/
│   └── models/
├── train.py
├── test_adaptive.py
├── run_multiseed_benchmark.py
├── run_Predict_ProtCross.py
└── environment.yml
```

---

## 3. Installation

### 3.1 System Requirements

- Linux (recommended) or WSL2
- Python 3.10
- Conda (Miniconda or Anaconda)
- NVIDIA GPU + CUDA 12.1 (recommended for training and ESM-C preprocessing)

> CPU-only runs are possible for debugging/small tests but will be significantly slower.

### 3.2 Create Environment

```bash
conda env create -f environment.yml
conda activate evopoint_da
```

The provided environment includes:
- `pytorch==2.3.0`
- `pytorch-cuda==12.1`
- Torch Geometric and companion packages
- `esm>=3.1.0` for ESM-C APIs

#### CPU-only notes
If you do not have a CUDA-capable GPU:
1. Remove/replace `pytorch-cuda=12.1` in `environment.yml`.
2. Install CPU-compatible PyTorch/Torch Geometric wheels.
3. Run preprocessing/training with `--device cpu` or CPU trainer settings.

### 3.3 ESM-C Model Weights (Required)

ProtCross currently expects a **local ESM-C checkpoint path** for `--model_name` in preprocessing.

Example:
```bash
python scripts/preprocess_esm.py \
  --data_dir data/raw_pdb \
  --output_dir data/processed_pdb \
  --fit_pca \
  --model_name /absolute/path/to/esmc_600m_weights.pth \
  --pca_model_path pca_esmc_128.pkl
```

Important details:
- `--model_name` is treated as a local file path in current code.
- The script truncates sequences to length 1022 for ESM-C context compatibility.
- Fit PCA once on source (PDB), then reuse the same PCA file for AF2.

### 3.4 Verify Installation

```bash
python -c "import torch; import torch_geometric; import pytorch_lightning; import hydra; import esm; print('OK')"
```

---

## 4. Data Preparation

Expected layout:

```text
data/
├── raw_pdb/          # input PDB/CIF structures (source)
├── raw_af2/          # input AF2 PDB structures (target)
├── processed_pdb/    # generated .pt files for source
└── processed_af2/    # generated .pt files for target
```

Optional AF2 retrieval helper:

```bash
python scripts/get_af2.py
```

---

## 5. Usage

### 5.1 Preprocess Source (PDB) with PCA Fit

```bash
python scripts/preprocess_esm.py \
  --data_dir data/raw_pdb \
  --output_dir data/processed_pdb \
  --fit_pca \
  --model_name /absolute/path/to/esmc_600m_weights.pth \
  --pca_model_path pca_esmc_128.pkl \
  --pca_dim 128
```

### 5.2 Preprocess Target (AF2) with Shared PCA

```bash
python scripts/preprocess_esm.py \
  --data_dir data/raw_af2 \
  --output_dir data/processed_af2 \
  --model_name /absolute/path/to/esmc_600m_weights.pth \
  --pca_model_path pca_esmc_128.pkl \
  --is_af2
```

### 5.3 Map Labels from PDB to AF2

```bash
python scripts/map_labels.py
```

### 5.4 Train

Default training:

```bash
python train.py
```

Common Hydra overrides:

```bash
# Disable domain adaptation
python train.py model.use_da=False

# Disable ESM features
python train.py model.use_esm=False

# Short debugging run
python train.py trainer.max_epochs=5

# Custom data directories
python train.py \
  data.data_dir_pdb=/abs/path/to/processed_pdb \
  data.data_dir_af2=/abs/path/to/processed_af2
```

### 5.5 Evaluate / Test

```bash
python test_adaptive.py
```

Additional analysis scripts are available (e.g., `scripts/eval_run.py`) for task-specific reporting.

### 5.6 Run Multi-seed Benchmark

```bash
python run_multiseed_benchmark.py
```

### 5.7 Single-structure Prediction

Use the prediction helper to run inference on one PDB structure and optionally write per-residue probabilities to the B-factor column of a new PDB file:

```bash
python run_Predict_ProtCross.py \
  --pdb_file examples/AF-P00734-F1-model_v6-example.pdb \
  --ckpt_path saved_weights/D_1/last.ckpt \
  --esm_weights esmc_weights/esmc_600m_2024_12_v0.pth \
  --pca_path data/pca_esmc_128.pkl \
  --output_pdb examples/prediction_result.pdb \
  --threshold 0.5
```

---

## 6. Configuration Guide (Hydra)

Main configuration files:
- `configs/train.yaml`: global defaults and run-level settings.
- `configs/data/protein_seg.yaml`: data module paths and loading parameters.
- `configs/model/da_module.yaml`: architecture and adaptation hyperparameters.
- `configs/trainer/default.yaml`: PyTorch Lightning trainer options.

Hydra override syntax:

```bash
python train.py key1=value1 key2=value2
```

Tip: keep all experiment commands in shell scripts to ensure reproducibility.

---

## 7. Troubleshooting

- **`FileNotFoundError` for ESM-C weights**
  - Ensure `--model_name` points to an existing local `.pth` checkpoint file.
- **Torch Geometric install issues**
  - Verify that your torch version and wheel index URL match the environment (torch 2.3.0 + cu121).
- **OOM during preprocessing/training**
  - Reduce batch size, use shorter runs, or switch to a smaller subset first.
- **No files processed**
  - Confirm input extensions are `.pdb`/`.cif` and paths are correct.

---

## 8. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
