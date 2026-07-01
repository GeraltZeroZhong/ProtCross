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

### Recommended for new users: Desktop app

Download ProtCross Desktop from the GitHub release page:

- Windows 10/11 x64: [ProtCross Desktop 0.2.0 setup.exe](https://github.com/GeraltZeroZhong/ProtCross/releases/download/v0.2.0/ProtCross%20Desktop_0.2.0_x64-setup.exe)
- macOS 12+: [ProtCross Desktop 0.2.0 DMG](https://github.com/GeraltZeroZhong/ProtCross/releases/download/v0.2.0/ProtCross_Desktop_0.2.0_macos.dmg)
- Release page: [ProtCross v0.2.0](https://github.com/GeraltZeroZhong/ProtCross/releases/tag/v0.2.0)

These links become active when the `v0.2.0` desktop release is published. If a
direct file link is unavailable, open the release page and download the Windows
installer or macOS DMG from the Assets section.

Desktop setup flow:

1. Install and launch ProtCross Desktop.
2. Confirm the ESM-C license prompt. ESM-C weights are not bundled in the app.
3. In Setup, install the CPU backend or choose GPU/MPS acceleration if needed.
4. Download or import ESM-C weights when prompted.
5. Select a PDB/mmCIF file and run prediction. Results include the annotated
   structure, scores TSV, pockets JSON, and summary JSON.

### Command line

```bash
pip install "protcross[predict]"
protcross predict input.pdb --accept-esm-license
```

By default this writes a complete result package next to the input:
`input.protcross.pdb`, `input.protcross.scores.tsv`,
`input.protcross.pockets.json`, and `input.protcross.summary.json`. For mmCIF
input, the annotated structure output uses `.cif`.

On the first prediction run, ProtCross can install runtime assets into
`~/.cache/protcross/assets/v0.1.2` by default. Before downloading or using
ESM-C weights, review the upstream ESM-C model license and pass
`--accept-esm-license` once:

```text
protcross-0.1.2-binding-moad-final.ckpt    # recommended release checkpoint
pca_esmc_128_binding_moad_0.1.2.pkl        # matching PCA reducer
esmc_600m_2024_12_v0.pth                   # ESM-C weights from Hugging Face
```

PyPI packages ship code only. Use `protcross setup-assets` when you want to
pre-download assets, install a specific asset bundle, or provide custom URLs:

```bash
protcross setup-assets --accept-esm-license \
  --checkpoint-url https://example.org/protcross-0.1.2-binding-moad-final.ckpt \
  --pca-url https://example.org/pca_esmc_128_binding_moad_0.1.2.pkl
```

Use the `0.1.2` release checkpoint for practical binding-site prediction and when reporting ProtCross as a benchmark method. If ProtCross is used as a benchmark, report the probability threshold used for the final predictions.

Training a new release-style model uses the modern `protcross train`,
`protcross preprocess`, `protcross download-af2`, and `protcross map-labels`
commands. Paper-reproduction notes are documented in
`reproduction/legacy/README.md`.

CLI assets and Desktop assets are configured separately: CLI commands use the
ProtCross cache or explicit `--checkpoint/--esm-weights/--pca` paths, while
Desktop stores its manifest and user-selected assets under the app data folder.
Running `protcross setup-assets` does not by itself make the Desktop app ready.

If your system already has ESM-C weights, skip that large download and pass the path at prediction time:

```bash
protcross setup-assets --skip-esm
protcross predict input.pdb \
  --esm-weights /absolute/path/to/esmc_600m_2024_12_v0.pth \
  --out-dir protcross-results
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
  - [3.2 Retrain Release Workflow](#32-retrain-release-workflow)
    - [3.2.1 Data Preparation](#321-data-preparation)
    - [3.2.2 Preprocess Source (PDB) with PCA Fit](#322-preprocess-source-pdb-with-pca-fit)
    - [3.2.3 Preprocess Target (AF2) with Shared PCA](#323-preprocess-target-af2-with-shared-pca)
    - [3.2.4 Map Labels from PDB to AF2](#324-map-labels-from-pdb-to-af2)
    - [3.2.5 Train](#325-train)
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
- Python 3.10. The 0.1.3 wheel metadata is intentionally limited to Python
  3.10 because the release test matrix and PyTorch/Torch Geometric stack are
  locked there.
- Conda (Miniconda or Anaconda)
- NVIDIA GPU + CUDA 12.1 (recommended for training and ESM-C preprocessing)

> CPU-only runs are possible for debugging/small tests but will be significantly slower.

### 2.2 Create Environment

For development or training:

```bash
conda env create -f environment.yml
conda activate protcross
pip install -e ".[test,esm]"
```

For the lightweight prediction interface from PyPI:

```bash
pip install "protcross[predict]"
protcross predict input.pdb --accept-esm-license --output input.protcross.pdb
```

The provided environment includes:
- `pytorch==2.3.x`
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
protcross setup-assets --accept-esm-license
```

Manual setup is optional for standard prediction because `protcross predict`
can automatically download missing default assets. Downloading or using ESM-C weights
requires `--accept-esm-license` or `PROTCROSS_ACCEPT_ESM_LICENSE=1`. By default
asset setup downloads:
- ESM-C 600M weights from https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12
- `protcross-0.1.2-binding-moad-final.ckpt` from the ProtCross `v0.1.2` GitHub release
- `pca_esmc_128_binding_moad_0.1.2.pkl` from the ProtCross `v0.1.2` GitHub release

The default asset version is fixed by the installed package for reproducibility.
`default` and `latest` both mean the current packaged stable bundle, currently
`0.1.2`; they do not query a floating remote latest release. The concrete bundle
version is recorded in `protcross-assets.json`. The default install location is
`~/.cache/protcross/assets/v0.1.2`. You can override it with
`PROTCROSS_ASSETS_DIR` or `--output-dir`:

```bash
PROTCROSS_ASSETS_DIR=/data/protcross-assets protcross setup-assets --accept-esm-license
protcross setup-assets --accept-esm-license --output-dir /data/protcross-assets
```

After setup, prediction can discover assets automatically:

```bash
protcross predict input.pdb --accept-esm-license --out-dir protcross-results
```

To disable automatic asset setup during prediction, pass `--no-auto-assets` or
`--offline`. To repair a stale cache, pass `--refresh-assets`.

For source checkouts or custom releases, explicit paths are still supported:

```bash
protcross predict input.pdb \
  --checkpoint checkpoints/protcross-0.1.2-binding-moad-final.ckpt \
  --esm-weights /absolute/path/to/esmc_600m_2024_12_v0.pth \
  --pca data/pca_esmc_128_binding_moad_0.1.2.pkl \
  --accept-esm-license \
  --out-dir protcross-results
```

Explicit local checkpoint, PCA, and ESM-C files are treated as trusted local
inputs only when their SHA256 matches the selected release bundle. For custom
or experimental assets, pass `--trust-unverified-assets` after verifying the
files came from a source you trust; checkpoint, PCA pickle, and torch weight
loading can execute code through their underlying serialization formats.

#### What is ESM-C?
**ESM-C** is EvolutionaryScale's protein language model family for extracting residue-level sequence representations. In ProtCross, ESM-C embeddings are used as per-residue features.

Recommended checkpoint for this project:
- **ESM-C 600M (2024-12)**: https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12

The ESM-C model repository uses upstream model terms that are separate from
ProtCross's MIT code license. Review the model terms before downloading or
using ESM-C weights. ProtCross release checkpoint/PCA assets are distributed as
ProtCross runtime assets and do not bundle ESM-C model weights; review both
ProtCross and upstream model terms before redistributing custom assets derived
from ESM-C embeddings.

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

After downloading, locate `data/weights/esmc_600m_2024_12_v0.pth` and pass its absolute path to `--esm-weights`.

Example:
```bash
protcross preprocess \
  --data-dir data/raw_pdb \
  --output-dir data/processed_pdb \
  --fit-pca \
  --esm-weights /absolute/path/to/esmc_600m_2024_12_v0.pth \
  --pca artifacts/protcross-pca-128.pkl
```

Important details:
- `--esm-weights` is treated as a local file path.
- The script truncates sequences to length 1022 for ESM-C context compatibility.

### 2.4 Verify Installation

```bash
python -c "import torch; import torch_geometric; import pytorch_lightning; import hydra; import esm; print('OK')"
pytest -q
```

---

## 3. Usage

This section is split into two maintained workflows:
- **Apply 0.1.2 release model**: use the released checkpoint for inference.
- **Retrain release workflow**: rebuild datasets/features and train with modern CLI commands.

For most users, the `0.1.2` release checkpoint is the recommended model. It was trained as a release model for external generalization evaluation using Binding MOAD-derived source structures and matched AF2 target structures. If you use ProtCross as a benchmark, use the release checkpoint and report the threshold used for evaluation.

Paper-reproduction notes are kept under `reproduction/legacy/`.

### 3.1 Apply ProtCross (Inference with Existing Model)

### 3.1.1 Single-structure Prediction

You can directly run inference on one PDB/mmCIF structure and write a complete
prediction package.

The recommended 0.1.2 path for PyPI users is:

```bash
protcross predict examples/6fhu.pdb --accept-esm-license --out-dir examples/protcross-results
```

This writes:
- `6fhu.protcross.pdb`: annotated structure with scored protein-residue B-factors set to ProtCross probabilities.
- `6fhu.protcross.scores.tsv`: residue-level table for downstream scripts.
- `6fhu.protcross.pockets.json`: pocket center, residues, scores, residue count, and clustered pockets.
- `6fhu.protcross.summary.json`: machine-readable run summary.

Use `--summary-only` to restore the old behavior and print a summary without
creating default output files. Individual paths can still override the defaults:

```bash
protcross predict examples/6fhu.pdb \
  --accept-esm-license \
  --output examples/6fhu.protcross.pdb \
  --scores-tsv examples/6fhu.scores.tsv \
  --pocket-json examples/6fhu.pockets.json \
  --summary-json examples/6fhu.summary.json \
  --threshold 0.5
```

Prediction output semantics:
- `--threshold` uses a strict `probability > threshold` rule for binary calls, TSV `is_binding`, pocket selection, clustering, and summaries.
- The annotated structure B-factor column stores continuous ProtCross probabilities in the `0..1` range for scored protein residues. It is not an experimental B-factor. By default, unscored atoms in scored model(s) are written as `0.0` so downstream scripts do not mix ProtCross probabilities with original B-factors or pLDDT. Fully unscored models in multi-model structures keep their original values and are reported in warnings. Use `--unscored-bfactor-policy keep` only when you explicitly want to preserve original values for unscored atoms in scored model(s).
- Pocket coordinates are in the input structure coordinate frame, in Angstrom, using C-alpha atoms.
- `summary.top_pocket.center` and terminal `Pocket center` refer to the highest-ranked clustered pocket, not the aggregate of all selected residues. `aggregate_pocket.center` is reported separately for all selected residues.
- Clustered pockets are connected components from selected residues using the default 8 Angstrom C-alpha cutoff.
- CLI and Python API prediction fail on structures longer than `--max-len` by default. Pass `--allow-truncation` or `allow_truncation=True` to explicitly score only the leading ESM-C context window.

Extended TSV columns:

```text
residue_id, residue_key, residue_id_namespace, model_id, chain_id,
auth_asym_id, label_asym_id, residue_number, auth_seq_id, label_seq_id,
insertion_code, resname, one_letter_code, input_bfactor, probability,
is_binding, x, y, z, cluster_id, is_scored, rank_global, rank_within_chain
```

Output schema contract:
- `scores.tsv`: tab-separated UTF-8; probabilities are floats in `0..1`; coordinates are Angstrom C-alpha coordinates in input frame; nullable fields are empty strings.
- `pockets.json`: `schema_version` is `protcross-pocket-v1`; includes `asset_version`, threshold metadata, coordinate units, `residue_id_namespaces`, `aggregate_pocket`, and `clustered_pockets`. If no residue is selected, `aggregate_pocket` is `null` and `clustered_pockets` is empty.
- `summary.json`: `schema_version` is `protcross-summary-v1`; includes input path, device, asset version, threshold, cluster cutoff, scored/original residue counts, truncation flag, `unscored_bfactor_policy`, `top_pocket`, `aggregate_pocket`, top residues, and output paths.
- Empty-pocket example: `selected_residue_count: 0`, `top_pocket: null`, `aggregate_pocket: null`, `cluster_count: 0`.

You can also keep model assets in an explicit directory:

```bash
protcross predict examples/6fhu.pdb \
  --assets-dir /path/to/protcross-assets \
  --out-dir examples/protcross-results
```

The asset directory should contain `protcross-assets.json` plus
`protcross-0.1.2-binding-moad-final.ckpt`, `esmc_600m_2024_12_v0.pth`, and
`pca_esmc_128_binding_moad_0.1.2.pkl`. Alternatively, set
`PROTCROSS_CHECKPOINT`, `PROTCROSS_ESM_WEIGHTS`, and `PROTCROSS_PCA`.

Python API:

```python
from protcross.inference import ProtCrossPredictor, predict_pdb

# Missing default runtime assets are installed automatically unless
# auto_setup_assets=False is passed.
result = predict_pdb(
    "examples/6fhu.pdb",
    accept_esm_license=True,
    output_pdb="examples/6fhu.protcross.pdb",
    scores_tsv="examples/6fhu.protcross.scores.tsv",
    pocket_json="examples/6fhu.protcross.pockets.json",
    summary_json="examples/6fhu.protcross.summary.json",
)
print(result.format_summary())

predictor = ProtCrossPredictor.from_default_assets(
    embedding_cache_dir=".protcross-feature-cache",
    accept_esm_license=True,
)
result = predictor.predict("examples/6fhu.pdb", pocket_json="examples/6fhu.protcross.pockets.json")

result = predict_pdb(
    "examples/6fhu.pdb",
    ckpt_path="checkpoints/protcross-0.1.2-binding-moad-final.ckpt",
    esm_weights="/absolute/path/to/esmc_600m_2024_12_v0.pth",
    pca_path="data/pca_esmc_128_binding_moad_0.1.2.pkl",
    output_pdb="examples/6fhu.protcross.pdb",
)
print(result.format_summary())
```

### 3.1.2 Batch Prediction (Multiple Structures)

`protcross predict` predicts one structure each run. For a few files, iterate in
a shell loop and optionally reuse a feature cache across invocations:

```bash
protcross setup-assets --accept-esm-license
mkdir -p batch_outputs
for pdb in /path/to/pdb_dir/*.pdb; do
  base="$(basename "${pdb}" .pdb)"
  protcross predict "${pdb}" \
    --accept-esm-license \
    --out-dir batch_outputs \
    --embedding-cache-dir .protcross-feature-cache \
    --threshold 0.5
done
```

For larger batches, prefer the Python API so ESM, PCA, and the segmentation model
are loaded once:

```python
from pathlib import Path
from protcross.inference import ProtCrossPredictor

predictor = ProtCrossPredictor.from_default_assets(
    device="auto",
    embedding_cache_dir=".protcross-feature-cache",
    accept_esm_license=True,
)
results = predictor.predict_many(
    sorted(Path("/path/to/pdb_dir").glob("*.pdb")),
    threshold=0.5,
    allow_truncation=False,
)
```

### 3.2 Retrain Release Workflow

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
protcross download-af2 \
  --raw-pdb-dir data/raw_pdb \
  --output-dir data/raw_af2 \
  --mapping-file artifacts/pdb_uniprot_mapping.json
```

### 3.2.2 Preprocess Source (PDB) with PCA Fit

```bash
protcross preprocess \
  --data-dir data/raw_pdb \
  --output-dir data/processed_pdb \
  --fit-pca \
  --esm-weights ~/.cache/protcross/assets/v0.1.2/esmc_600m_2024_12_v0.pth \
  --pca artifacts/protcross-pca-128.pkl \
  --pca-dim 128
```

### 3.2.3 Preprocess Target (AF2) with Shared PCA

```bash
protcross preprocess \
  --data-dir data/raw_af2 \
  --output-dir data/processed_af2 \
  --esm-weights ~/.cache/protcross/assets/v0.1.2/esmc_600m_2024_12_v0.pth \
  --pca artifacts/protcross-pca-128.pkl \
  --is-af2
```

### 3.2.4 Map Labels from PDB to AF2

```bash
protcross map-labels \
  --processed-pdb-dir data/processed_pdb \
  --processed-af2-dir data/processed_af2 \
  --raw-pdb-dir data/raw_pdb \
  --raw-af2-dir data/raw_af2 \
  --mapping-file artifacts/pdb_uniprot_mapping.json
```

### 3.2.5 Train

Default training:

```bash
protcross train
```

Common Hydra overrides:

```bash
# Disable domain adaptation
protcross train model.use_da=False

# Disable ESM features
protcross train model.use_esm=False

# Short debugging run
protcross train trainer.max_epochs=5

# Custom data directories
protcross train \
  data.data_dir_pdb=/abs/path/to/processed_pdb \
  data.data_dir_af2=/abs/path/to/processed_af2
```

Evaluation helpers, multi-seed benchmark scripts, and paper-reproduction notes
are documented in `reproduction/legacy/README.md`.

---

## 4. Configuration Guide (Hydra)

Main configuration files:
- `configs/train.yaml`: global defaults and run-level settings.
- `configs/data/protein_seg.yaml`: data module paths and loading parameters.
- `configs/model/da_module.yaml`: architecture and adaptation hyperparameters.
- `configs/trainer/default.yaml`: PyTorch Lightning trainer options.

Hydra override syntax:

```bash
protcross train key1=value1 key2=value2
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
|--- reproduction/
|   `--- legacy/                # historical reproduction scripts and notes
|--- src/protcross/
|   |--- cli/                   # installed command entry points
|   |--- data/
|   |--- experiments/           # reproduction benchmark workflows
|   |--- evaluation/
|   |--- inference/             # lightweight predictor API
|   `--- models/
`--- environment.yml
```

---

## 6. Troubleshooting

- **`FileNotFoundError` for ESM-C weights**
  - Run `protcross setup-assets --accept-esm-license`, or ensure `--esm-weights` points to an existing local `.pth` checkpoint file.
- **ESM-C license confirmation is required**
  - Review the upstream ESM-C terms, then pass `--accept-esm-license` or set `PROTCROSS_ACCEPT_ESM_LICENSE=1` before downloading or using ESM-C weights.
- **Explicit custom assets are rejected**
  - Use the managed 0.1.2 assets when possible. For experimental local assets, pass `--trust-unverified-assets` only after verifying the files are trusted.
- **`protcross setup-assets` cannot find GitHub release assets**
  - Attach `protcross-0.1.2-binding-moad-final.ckpt` and `pca_esmc_128_binding_moad_0.1.2.pkl` to the `v0.1.2` GitHub release, or pass `--checkpoint-url` and `--pca-url`.
- **Torch Geometric install issues**
  - Verify that your torch version and wheel index URL match the environment (torch 2.3.x + cu121).
- **OOM during preprocessing/training**
  - Reduce batch size, use shorter runs, or switch to a smaller subset first.

---

## 7. Changelog

### 0.1.3

Preparation release for user-facing prediction outputs and package cleanup.

- Makes `protcross predict` write a default result package with annotated structure, extended scores TSV, pocket JSON, and summary JSON.
- Reports clustered pocket centers, residue lists, probabilities, residue counts, truncation metadata, asset version, and output files for docking/MD downstream workflows.
- Renames the canonical source package to `protcross` while keeping a deprecated import alias for 0.1.x compatibility.
- Moves historical paper-reproduction scripts and notes under `reproduction/legacy/`.
- Keeps the default runtime asset bundle fixed at the stable `0.1.2` checkpoint/PCA unless the user explicitly selects another asset version.

### 0.1.2

Release checkpoint update for external prediction and benchmark use.

- Adds the `0.1.2` Binding MOAD-trained release checkpoint and matching PCA reducer as the default runtime assets. This release checkpoint was trained on Binding MOAD-derived labels from 41,409 PDB source structures spanning 20,387 unique ligand IDs, with 8,953 matched AF2 target structures. 
- Filters common crystallization additives, salts, ions, and caps from default ligand-adjacent residue labeling.

Earlier release notes and archived command references are maintained in
`reproduction/legacy/README.md`.

---

## 8. License

ProtCross code is licensed under the MIT License. See [LICENSE](LICENSE) for
details. ESM-C weights are not part of the ProtCross code license and remain
governed by EvolutionaryScale's upstream model terms. The release
checkpoint/PCA assets do not include ESM-C weights; custom redistribution of
ESM-C-derived model assets should be reviewed against the applicable upstream
and ProtCross terms.
