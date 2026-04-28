# ProtCross

ProtCross is a domain-adaptive protein point-cloud learning framework for binding-site prediction across experimentally solved **PDB** structures and predicted **AlphaFold2 (AF2)** structures. The model accepts structures from AlphaFold and can write per-residue binding probabilities to the **B-factor column** of a new PDB output file.

**Published paper (JCIM):** Zhong, S., & Jiang, Y. (2026). ProtCross: Bridging the PDB-AlphaFold Gap for Binding Site Prediction with Protein Point Clouds. Journal of chemical information and modeling, 66(7), 3688-3701. https://doi.org/10.1021/acs.jcim.5c03224

The codebase combines:
- residue-level structural geometry (C-alpha coordinates),
- language-model residue embeddings (ESM-C), and
- confidence-aware domain adaptation (pLDDT-weighted DANN)

to improve robustness when transferring from PDB (source domain) to AF2 (target domain).

---

## Quick Start

Use this path when you only want to predict binding sites for one structure.

```bash
pip install "protcross[predict]"
protcross setup-assets
protcross predict input.pdb --output input.protcross.pdb
```

`protcross setup-assets` installs runtime assets into `~/.cache/protcross/assets/v0.1.2` by default:

```text
protcross-0.1.2-binding-moad-final.ckpt    # recommended release checkpoint
pca_esmc_128_binding_moad_0.1.2.pkl        # matching PCA reducer
esmc_600m_2024_12_v0.pth                   # ESM-C weights from Hugging Face
```

PyPI packages ship code only. The checkpoint and PCA file must be attached to the `v0.1.2` GitHub release with the exact filenames above, or supplied with custom URLs:

```bash
protcross setup-assets \
  --checkpoint-url https://example.org/protcross-0.1.2-binding-moad-final.ckpt \
  --pca-url https://example.org/pca_esmc_128_binding_moad_0.1.2.pkl
```

Use the `0.1.2` release checkpoint for practical binding-site prediction and when reporting ProtCross as a benchmark method. If ProtCross is used as a benchmark, report the probability threshold used for the final predictions.

To reproduce the original ProtCross paper results, use the `0.1.1` checkpoint `best-epoch=59.ckpt` and its matching `pca_esmc_128.pkl`. The original paper experiments were based on the PDBbind v2020 refined set.

If your system already has ESM-C weights, skip that large download and pass the path at prediction time:

```bash
protcross setup-assets --skip-esm
protcross predict input.pdb \
  --esm-weights /absolute/path/to/esmc_600m_2024_12_v0.pth \
  --output input.protcross.pdb
```

## Table of Contents
- [Quick Start](#quick-start)
- [1. Project Overview](#1-project-overview)
- [2. Installation](#2-installation)
  - [2.1 System Requirements](#21-system-requirements)
  - [2.2 Create Environment](#22-create-environment)
  - [2.3 Runtime Assets](#23-runtime-assets)
  - [2.4 Verify Installation](#24-verify-installation)
- [3. Usage](#3-usage)
  - [3.1 Apply ProtCross (Inference with Existing Model)](#31-apply-protcross-inference-with-existing-model)
    - [3.1.1 Single-structure Prediction](#311-single-structure-prediction)
    - [3.1.2 Batch Prediction (Multiple Structures)](#312-batch-prediction-multiple-structures)
  - [3.2 Reproduce ProtCross (Training Pipeline)](#32-reproduce-protcross-training-pipeline)
    - [3.2.1 Data Preparation](#321-data-preparation)
    - [3.2.2 Preprocess Source (PDB) with PCA Fit](#322-preprocess-source-pdb-with-pca-fit)
    - [3.2.3 Preprocess Target (AF2) with Shared PCA](#323-preprocess-target-af2-with-shared-pca)
    - [3.2.4 Map Labels from PDB to AF2](#324-map-labels-from-pdb-to-af2)
    - [3.2.5 Train](#325-train)
    - [3.2.6 Evaluate / Test](#326-evaluate--test)
    - [3.2.7 Run Multi-seed Benchmark](#327-run-multi-seed-benchmark)
- [4. Configuration Guide (Hydra)](#4-configuration-guide-hydra)
- [5. Repository Layout](#5-repository-layout)
- [6. Troubleshooting](#6-troubleshooting)
- [7. Changelog](#7-changelog)
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

## 2. Installation

### 2.1 System Requirements

- Linux (recommended) or WSL2
- Python 3.10
- Conda (Miniconda or Anaconda)
- NVIDIA GPU + CUDA 12.1 (recommended for training and ESM-C preprocessing)

> CPU-only runs are possible for debugging/small tests but will be significantly slower.

### 2.2 Create Environment

For development or full paper reproduction:

```bash
conda env create -f environment.yml
conda activate evopoint_da
pip install -e ".[test,esm]"
```

For the lightweight prediction interface from PyPI:

```bash
pip install "protcross[predict]"
protcross setup-assets
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

### 2.3 Runtime Assets

ProtCross separates code and large runtime assets. PyPI distributions include the Python package and command-line tools, while the pretrained checkpoint, PCA reducer, and ESM-C weights are downloaded after installation.

Recommended setup:

```bash
protcross setup-assets
```

By default this downloads:
- ESM-C 600M weights from https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12
- `protcross-0.1.2-binding-moad-final.ckpt` from the ProtCross `v0.1.2` GitHub release
- `pca_esmc_128_binding_moad_0.1.2.pkl` from the ProtCross `v0.1.2` GitHub release

The default install location is `~/.cache/protcross/assets/v0.1.2`. You can override it with `PROTCROSS_ASSETS_DIR` or `--output-dir`:

```bash
PROTCROSS_ASSETS_DIR=/data/protcross-assets protcross setup-assets
protcross setup-assets --output-dir /data/protcross-assets
```

After setup, prediction can discover assets automatically:

```bash
protcross predict input.pdb --output input.protcross.pdb
```

For source checkouts or custom releases, explicit paths are still supported:

```bash
protcross predict input.pdb \
  --checkpoint checkpoints/protcross-0.1.2-binding-moad-final.ckpt \
  --esm-weights /absolute/path/to/esmc_600m_2024_12_v0.pth \
  --pca data/pca_esmc_128_binding_moad_0.1.2.pkl \
  --output input.protcross.pdb
```

#### What is ESM-C?
**ESM-C** is EvolutionaryScale's protein language model family for extracting residue-level sequence representations. In ProtCross, ESM-C embeddings are used as per-residue features.

Recommended checkpoint for this project:
- **ESM-C 600M (2024-12)**: https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12

The ESM-C model repository uses a custom non-commercial license. Review the model terms before downloading or redistributing derived assets.

#### Manual ESM-C download fallback

If the automatic downloader is unavailable in your environment, you can download the model weights from Hugging Face in either of the following ways.

**Option A - Git LFS clone**
```bash
# 1) Install Git LFS once (if needed)
git lfs install

# 2) Clone the model repository
git clone https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12
```

**Option B - Hugging Face CLI**
```bash
# 1) Install CLI
pip install -U "huggingface_hub[cli]"

# 2) Download repository files to a local directory
huggingface-cli download EvolutionaryScale/esmc-600m-2024-12 \
  --local-dir ./esmc-600m-2024-12
```

After downloading, locate `data/weights/esmc_600m_2024_12_v0.pth` and pass its absolute path to `--model_name` / `--esm_weights`.

Example:
```bash
python scripts/preprocess_esm.py \
  --data_dir data/raw_pdb \
  --output_dir data/processed_pdb \
  --fit_pca \
  --model_name /absolute/path/to/esmc_600m_2024_12_v0.pth \
  --pca_model_path pca_esmc_128.pkl
```

Important details:
- `--model_name` is treated as a local file path in current code.
- The script truncates sequences to length 1022 for ESM-C context compatibility.

### 2.4 Verify Installation

```bash
python -c "import torch; import torch_geometric; import pytorch_lightning; import hydra; import esm; print('OK')"
pytest -q
```

---

## 3. Usage

This section is split into two workflows:
- **Apply ProtCross**: use the released checkpoint for inference.
- **Reproduce ProtCross**: rebuild datasets/features and retrain/evaluate the model.

For most users, the `0.1.2` release checkpoint is the recommended model. It was trained as a release model for external generalization evaluation using Binding MOAD-derived source structures and matched AF2 target structures. If you use ProtCross as a benchmark, use the release checkpoint and report the threshold used for evaluation.

For reproducing the original ProtCross paper, use the `0.1.1` checkpoint `best-epoch=59.ckpt` and matching `pca_esmc_128.pkl`; those assets correspond to the original PDBbind v2020 refined-set experiments.

### 3.1 Apply ProtCross (Inference with Existing Model)

### 3.1.1 Single-structure Prediction

You can directly run inference on one PDB structure and write per-residue probabilities to the B-factor column of a new PDB file.

The recommended 0.1.2 path for PyPI users is:

```bash
protcross setup-assets
protcross predict examples/6fhu.pdb --output examples/6fhu.pred.pdb
```

Add a score table or a custom threshold when needed:

```bash
protcross predict examples/6fhu.pdb \
  --output examples/6fhu.pred.pdb \
  --scores-tsv examples/6fhu.scores.tsv \
  --threshold 0.5
```

You can also keep model assets in an explicit directory:

```bash
protcross predict examples/6fhu.pdb \
  --assets-dir /path/to/protcross-assets \
  --output examples/6fhu.pred.pdb
```

The asset directory should contain `protcross-0.1.2-binding-moad-final.ckpt`, `esmc_600m_2024_12_v0.pth`, and `pca_esmc_128_binding_moad_0.1.2.pkl`. Older `v0.1.1` asset directories with `best-epoch=59.ckpt` and `pca_esmc_128.pkl` are still accepted for paper reproduction. Alternatively, set `PROTCROSS_CHECKPOINT`, `PROTCROSS_ESM_WEIGHTS`, and `PROTCROSS_PCA`.

The standalone entry point is equivalent:

```bash
protcross-predict examples/6fhu.pdb --output examples/6fhu.pred.pdb
```

The legacy 0.1.0 command is still supported:

```bash
python run_Predict_ProtCross.py \
  --pdb_file examples/6fhu.pdb \
  --ckpt_path checkpoints/protcross-0.1.2-binding-moad-final.ckpt \
  --esm_weights /absolute/path/to/esmc_600m_2024_12_v0.pth \
  --pca_path data/pca_esmc_128_binding_moad_0.1.2.pkl \
  --output_pdb examples/6fhu.pred.pdb
```

Python API:

```python
from evopoint_da.inference import ProtCrossPredictor, predict_pdb

result = predict_pdb("examples/6fhu.pdb", output_pdb="examples/6fhu.pred.pdb")
print(result.format_summary())

predictor = ProtCrossPredictor.from_default_assets()
result = predictor.predict("examples/6fhu.pdb", output_pdb="examples/6fhu.pred.pdb")

result = predict_pdb(
    "examples/6fhu.pdb",
    ckpt_path="checkpoints/protcross-0.1.2-binding-moad-final.ckpt",
    esm_weights="/absolute/path/to/esmc_600m_2024_12_v0.pth",
    pca_path="data/pca_esmc_128_binding_moad_0.1.2.pkl",
    output_pdb="examples/6fhu.pred.pdb",
)
print(result.format_summary())
```

### 3.1.2 Batch Prediction (Multiple Structures)

`protcross-predict` predicts one structure each run. For batch inference, iterate over PDB files in a shell loop:

```bash
protcross setup-assets
mkdir -p batch_outputs
for pdb in /path/to/pdb_dir/*.pdb; do
  base="$(basename "${pdb}" .pdb)"
  protcross-predict "${pdb}" \
    --output "batch_outputs/${base}_pred.pdb" \
    --threshold 0.5
done
```

### 3.2 Reproduce ProtCross (Training Pipeline)

### 3.2.1 Data Preparation

Expected layout:

```text
data/
|--- raw_pdb/          # input PDB/CIF structures (source)
|--- raw_af2/          # input AF2 PDB structures (target)
|--- processed_pdb/    # generated .pt files for source
`--- processed_af2/    # generated .pt files for target
```

Optional AF2 retrieval helper:

```bash
protcross-download-af2 \
  --raw-pdb-dir data/raw_pdb \
  --output-dir data/raw_af2 \
  --mapping-file pdb_uniprot_mapping.json
```

### 3.2.2 Preprocess Source (PDB) with PCA Fit

```bash
protcross-preprocess \
  --data_dir data/raw_pdb \
  --output_dir data/processed_pdb \
  --fit_pca \
  --model_name ~/.cache/protcross/assets/v0.1.2/esmc_600m_2024_12_v0.pth \
  --pca_model_path pca_esmc_128.pkl \
  --pca_dim 128
```

### 3.2.3 Preprocess Target (AF2) with Shared PCA

```bash
protcross-preprocess \
  --data_dir data/raw_af2 \
  --output_dir data/processed_af2 \
  --model_name ~/.cache/protcross/assets/v0.1.2/esmc_600m_2024_12_v0.pth \
  --pca_model_path pca_esmc_128.pkl \
  --is_af2
```

### 3.2.4 Map Labels from PDB to AF2

```bash
protcross-map-labels \
  --processed-pdb-dir data/processed_pdb \
  --processed-af2-dir data/processed_af2 \
  --raw-pdb-dir data/raw_pdb \
  --raw-af2-dir data/raw_af2 \
  --mapping-file pdb_uniprot_mapping.json
```

### 3.2.5 Train

Default training:

```bash
python train.py
```

Equivalent installed CLI:

```bash
protcross-train
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

### 3.2.6 Evaluate / Test

```bash
python test_adaptive.py ckpt_path=checkpoints/protcross-0.1.2-binding-moad-final.ckpt
```

Additional analysis scripts are available (e.g., `scripts/eval_run.py`) for task-specific reporting.

### 3.2.7 Run Multi-seed Benchmark

```bash
python run_multiseed_benchmark.py
```

---

## 4. Configuration Guide (Hydra)

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

## 5. Repository Layout

```text
ProtCross/
|--- configs/
|   |--- data/protein_seg.yaml
|   |--- model/da_module.yaml
|   |--- trainer/default.yaml
|   `--- train.yaml
|--- data/
|   |--- raw_pdb/
|   |--- raw_af2/
|   |--- processed_pdb/
|   `--- processed_af2/
|--- scripts/
|   |--- preprocess_esm.py      # compatibility wrapper
|   |--- get_af2.py             # compatibility wrapper
|   |--- map_labels.py          # compatibility wrapper
|   `--- ...                    # one-off analysis and plotting utilities
|--- src/evopoint_da/
|   |--- cli/                   # installed command entry points
|   |--- data/
|   |--- experiments/           # reproduction benchmark workflows
|   |--- evaluation/
|   |--- inference/             # lightweight predictor API
|   `--- models/
|--- train.py
|--- test_adaptive.py
|--- run_multiseed_benchmark.py
|--- run_Predict_ProtCross.py
`--- environment.yml
```

---

## 6. Troubleshooting

- **`FileNotFoundError` for ESM-C weights**
  - Run `protcross setup-assets`, or ensure `--esm-weights` / `--model_name` points to an existing local `.pth` checkpoint file.
- **`protcross setup-assets` cannot find GitHub release assets**
  - Attach `protcross-0.1.2-binding-moad-final.ckpt` and `pca_esmc_128_binding_moad_0.1.2.pkl` to the `v0.1.2` GitHub release, or pass `--checkpoint-url` and `--pca-url`.
- **Torch Geometric install issues**
  - Verify that your torch version and wheel index URL match the environment (torch 2.3.0 + cu121).
- **OOM during preprocessing/training**
  - Reduce batch size, use shorter runs, or switch to a smaller subset first.

---

## 7. Changelog

### 0.1.2

Release checkpoint update for external prediction and benchmark use.

- Adds the `0.1.2` Binding MOAD-trained release checkpoint and matching PCA reducer as the default runtime assets. This release checkpoint was trained on Binding MOAD-derived labels from 41,409 PDB source structures spanning 20,387 unique ligand IDs, with 8,953 matched AF2 target structures. 
- Filters common crystallization additives, salts, ions, and caps from default ligand-adjacent residue labeling.

### 0.1.1

Engineering-focused reproducibility release.

- Adds installable package metadata and console commands: `protcross-predict`, `protcross-preprocess`, and `protcross-train`.
- Adds a unified `protcross` CLI plus `protcross-setup-assets`, `protcross-download-af2`, and `protcross-map-labels`.
- Adds one-command runtime asset setup for the ESM-C weights, released checkpoint, and PCA reducer.
- Splits prediction into a lightweight API (`evopoint_da.inference`) that accepts one PDB/mmCIF and writes binding probabilities to the B-factor column.
- Keeps 0.1.0 entry points (`run_Predict_ProtCross.py`, `train.py`, `scripts/preprocess_esm.py`, `scripts/get_af2.py`, `scripts/map_labels.py`) as compatibility wrappers.
- Splits shared structure parsing, ESM-C feature extraction, PCA reduction, AF2 downloads, label mapping, metrics, experiments, and domain weighting into reusable modules.
- Fixes the standard training DataModule so target-domain AF2 batches can be supplied to domain adaptation without requiring positive labels.
- Adds pytest smoke/unit tests, including checkpoint CPU forward on a published processed sample.

### 0.1.0

Initial public release of ProtCross.

- Provides the core ProtCross framework for domain-adaptive binding-site prediction across PDB and AlphaFold2 protein structures.
- Includes PointNet++-based protein point-cloud segmentation, ESM-C residue embeddings, PCA feature reduction, pLDDT-aware target weighting, and DANN-style domain adaptation.
- Documents the full training and evaluation workflow, including preprocessing, label mapping, Hydra configuration, testing, and multi-seed benchmarking.

---

## 8. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
