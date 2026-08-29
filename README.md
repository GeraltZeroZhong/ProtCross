# ProtCross

[![PyPI](https://img.shields.io/pypi/v/protcross?label=PyPI&color=0f766e)](https://pypi.org/project/protcross/)
[![Windows Desktop](https://img.shields.io/badge/Windows-10%2F11%20x64-0078d4?logo=windows11&logoColor=white)](https://github.com/GeraltZeroZhong/ProtCross/releases)
[![macOS Desktop](https://img.shields.io/badge/macOS-12%2B%20Apple%20Silicon-111827?logo=apple&logoColor=white)](https://github.com/GeraltZeroZhong/ProtCross/releases)
[![Version](https://img.shields.io/badge/version-0.2.3-2563eb)](#version-history)
[![Python](https://img.shields.io/badge/python-3.10-3776ab)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-16a34a)](LICENSE)
[![Paper](https://img.shields.io/badge/DOI-10.1021%2Facs.jcim.5c03224-ca8a04)](https://doi.org/10.1021/acs.jcim.5c03224)

**Install:** [PyPI / CLI](https://pypi.org/project/protcross/) ·
[Windows x64 Desktop](https://github.com/GeraltZeroZhong/ProtCross/releases) ·
[macOS Apple Silicon Desktop](https://github.com/GeraltZeroZhong/ProtCross/releases)

ProtCross is a protein binding-site prediction tool for experimental and
predicted PDB/mmCIF structures, including AlphaFold models. It converts protein
coordinates into residue-level binding-site scores, ranked spatial clusters,
centroids, annotated structures, and provenance for structure triage, receptor
preparation, docking setup, and experimental prioritization. The method is
described in [*ProtCross: Bridging the PDB-AlphaFold Gap for Binding Site
Prediction with Protein Point Clouds*](https://doi.org/10.1021/acs.jcim.5c03224),
published in the *Journal of Chemical Information and Modeling*.

Predicted structures have expanded structural coverage, yet a single
unrefined model often requires site-centred assessment before downstream
atomistic modelling. Global fold accuracy and pLDDT alone do not establish a
ligand-compatible pocket: local side chains, loops, conformational state,
cofactors, ions, waters, and biological assembly can materially alter docking
poses and virtual-screening enrichment
([Holcomb et al., 2023](https://doi.org/10.1002/pro.4530);
[Karelina et al., 2023](https://doi.org/10.7554/eLife.89386);
[Lyu et al., 2024](https://doi.org/10.1126/science.adn6354)).

## Quick start

Choose the interface that matches your task:

| Goal | Start here |
| --- | --- |
| Run a local prediction from a terminal | Install the CLI below |
| Call ProtCross from a Python workflow | Install the CLI, then open [Python API](#python-api) |
| Use a guided interface and 3D viewer | Download [ProtCross Desktop](#desktop-application) |

ProtCross 0.2.3 requires Python 3.10. This first run installs the prediction
dependencies, prepares the managed model assets, checks a structure, and writes
one result package:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "protcross[predict]"

protcross setup-assets --accept-esm-license
protcross inspect input.pdb
protcross predict input.pdb --out-dir protcross-results
```

Review the [ESM-C model terms](https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement)
before recording acceptance. Initial asset setup downloads approximately
2.14 GiB; later predictions reuse the verified local cache.

The prediction command creates:

```text
protcross-results/
├── input.protcross.pdb
├── input.protcross.scores.tsv
├── input.protcross.pockets.json
└── input.protcross.summary.json
```

For a graphical workflow, install the Windows x64 or macOS Apple Silicon build
from [GitHub Releases](https://github.com/GeraltZeroZhong/ProtCross/releases),
then follow the three readiness steps in **Setup**. Desktop manages its own
runtime and assets and opens completed predictions in Mol*.

## Contents

- [Quick start](#quick-start)
- [Highlights](#highlights)
- [Installation](#installation)
- [Run predictions](#run-predictions)
- [Output package](#output-package)
- [Batch inference](#batch-inference)
- [Python API](#python-api)
- [Assets](#assets)
- [Desktop application](#desktop-application)
- [Model and inference pipeline](#model-and-inference-pipeline)
- [Training and development](#training-and-development)
- [Troubleshooting](#troubleshooting)
- [Version history](#version-history)
- [Citation](#citation)
- [License](#license)

## Highlights

- Inspect PDB, mmCIF, and AlphaFold coordinate files before model loading
- Score residues and rank spatial binding-site clusters with centroids
- Export annotated coordinates, a full residue table, cluster JSON, and run provenance
- Run on CPU, CUDA, or Apple MPS through the CLI, Python API, or local Desktop
- Reuse verified assets and reduced ESM/PCA features across repeated work
- Process structure collections with bounded ESM-C and PointNet++ microbatches
- Review persistent Desktop batches and regroup completed results interactively

## Installation

### Requirements

| Workflow | Requirements |
| --- | --- |
| CLI and Python inference | Python 3.10, PyTorch 2.3, local runtime assets |
| Desktop inference | Windows 10/11 x64 or macOS 12+ Apple Silicon |
| Training | Python 3.10, Conda, CUDA recommended |
| Desktop development | Node.js 20, Rust 1.88, Tauri 2 system packages |

ProtCross declares `python >=3.10,<3.11`. Create a dedicated Python 3.10
environment, choose the platform command below, and confirm the installed
version before preparing assets.

### Linux CPU

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  torch==2.3.1+cpu torchvision==0.18.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install "protcross[predict]"
protcross --version
```

### Linux CUDA 12.1

Install the CUDA wheel matching the PyTorch 2.3 runtime, then install ProtCross:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install "protcross[predict]"
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

Use the [PyTorch installation selector](https://pytorch.org/get-started/locally/)
when the host driver requires another PyTorch 2.3 wheel.

### macOS Apple Silicon

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "protcross[predict]"
protcross --version
```

PyTorch supplies Apple Silicon wheels through the Python package index.

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install `
  torch==2.3.1+cpu torchvision==0.18.1+cpu `
  --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install "protcross[predict]"
.\.venv\Scripts\protcross.exe --version
```

Activate the environment to use the shorter commands shown throughout this
README.

After installation, run `protcross setup-assets --accept-esm-license` once, or
open [Assets](#assets) to select another cache location or an existing ESM-C
file.

## Run predictions

### Inspect a structure

`protcross inspect` parses coordinate metadata without loading model assets.
The repository includes [`examples/6fhu.pdb`](examples/6fhu.pdb) for a first
check, so input preparation can be tested before the 2.14 GiB ESM-C download.

```bash
protcross inspect input.cif
protcross inspect input.cif --chain A
protcross inspect input.cif --json
```

Use the report to choose a chain, identify missing Cα atoms or modified
residues, and check whether a chain exceeds the 1,022-residue ESM-C context.
`--json` sends successful and failed inspections to stdout as machine-readable
JSON and uses the process exit code to signal success or failure.

```text
Input: examples/6fhu.pdb
Format: PDB
Models: 1
Chains: A
Scorable residues: 52
Longest chain context: 52 / 1022
Ready for prediction.
```

### Predict one structure

```bash
protcross predict input.pdb --out-dir results
```

Select one chain or an available accelerator when the task requires it:

```bash
protcross predict input.cif --chain A --out-dir results
protcross predict input.pdb --device auto --out-dir results
```

Common options:

| Option | Default | Function |
| --- | ---: | --- |
| `--chain ID` | all chains | Select one author chain ID |
| `--device` | `cpu` | Select `cpu`, `cuda`, `cuda:N`, `mps`, or `auto` |
| `--threshold` | `0.5` | Select residues with `score > threshold` |
| `--pocket-cluster-cutoff` | `8.0` | Set the Cα graph cutoff in Å |
| `--max-len` | `1022` | Set the per-chain ESM-C residue limit |
| `--allow-truncation` | disabled | Score the leading `max_len` residues of each long chain |
| `--embedding-cache-dir` | unset | Cache reduced ESM/PCA residue features |
| `--overwrite` | disabled | Replace an existing result package |
| `--offline` | disabled | Restrict asset resolution to local files |

The CLI writes four result files unless `--summary-only` is selected. Progress
messages go to stderr; the terminal summary goes to stdout. Use `--quiet` when
another process only needs the files and exit code.

Set explicit output paths when a workflow owns the file layout:

```bash
protcross predict input.cif \
  --chain A \
  --output results/input.protcross.cif \
  --scores-tsv results/input.protcross.scores.tsv \
  --pocket-json results/input.protcross.pockets.json \
  --summary-json results/input.protcross.summary.json
```

### Structure handling

| Input property | Processing rule |
| --- | --- |
| File format | `.pdb`, `.cif`, or `.mmcif` |
| Coordinate model | First model |
| Assembly | Coordinates supplied in the input file |
| Scorable residue | Standard polymer amino acid with a Cα atom |
| Geometry | One centered Cα point cloud across selected chains |
| Sequence context | Coordinate-observed sequence, embedded independently per chain |
| Chain selection | All scorable chains or one `--chain` value |
| Maximum context | 1,022 scorable residues per chain |
| Canonical order | Model, author chain, label/auth sequence position, insertion code |
| Alternate Cα | Bio.PDB-selected conformer |
| Modified residue | Omitted from geometry and sequence context |
| Coordinate break | Observed residues remain in the same per-chain ESM-C sequence |

The parser retains author and label identifiers from mmCIF, insertion codes,
residue names, input B-factors, raw Cα coordinates, and the source-file hash.

### Command reference

| Command | Purpose |
| --- | --- |
| `protcross inspect` | Validate and summarize a coordinate file |
| `protcross predict` | Run single-structure inference |
| `protcross setup-assets` | Install and verify runtime assets |
| `protcross preprocess` | Convert structures into training tensors |
| `protcross download-af2` | Download matched AlphaFold structures |
| `protcross map-labels` | Transfer PDB-derived labels to AF2 samples |
| `protcross train` | Launch the Hydra/Lightning training workflow |

```bash
protcross --help
protcross COMMAND --help
```

## Output package

### Files

| Task | File to use | Contents |
| --- | --- | --- |
| Color or share the scored structure | `input.protcross.pdb` or `.cif` | Input coordinates with residue scores in B-factor fields |
| Rank and filter every scored residue | `input.protcross.scores.tsv` | Identifiers, scores, calls, coordinates, cluster IDs, and ranks |
| Use predicted sites in a script | `input.protcross.pockets.json` | Thresholded residue clusters, members, centroids, and spatial statistics |
| Audit or reproduce a run | `input.protcross.summary.json` | Parameters, asset and input hashes, runtime, warnings, and top results |

PDB annotation preserves record order and updates B-factor columns on
`ATOM`/`HETATM` records. mmCIF annotation retains coordinate categories and
serializes an updated document. PDB B-factor values use two decimal places;
TSV and JSON retain full numeric precision. The default
`--unscored-bfactor-policy zero` assigns `0.0` to unscored atoms in the scored
model. The `keep` policy retains input values for unscored atoms. Additional
coordinate models retain their original values.

### Scores and clusters

| Stage | Definition |
| --- | --- |
| Residue score | `softmax(logits, dim=1)[:, 1]` |
| Binary call | `score > threshold` |
| Cluster graph | Selected Cα pairs with distance `<= cluster_cutoff` |
| Components | Single-linkage connected components, including singletons |
| Cluster order | Descending count/mean/maximum, then ascending canonical index |
| Cluster center | Score-weighted Cα centroid |

Start analysis with `model_score` and `rank` in the TSV. Higher scores indicate
stronger model support for the binding-site class. Scores are continuous model
outputs and are not independently calibrated probabilities. `probability` is a
schema compatibility alias for `model_score`.

The threshold controls binary calls and cluster membership; it does not change
the underlying scores. Empty selections produce zero clusters and null
aggregate/top-cluster entries. Selected chains share one geometry graph, so a
cluster can span a chain interface.

### Schemas

- `protcross-pocket-v2` for `pockets.json`
- `protcross-summary-v2` for `summary.json`
- extended TSV columns for PDB and mmCIF residue identifiers

The JSON package records the application version, scoring procedure, selected
asset bundle, asset hashes, input SHA256, threshold, clustering parameters,
device, precision, and effective microbatch size.

## Batch inference

Use one `ProtCrossPredictor` to process a directory of structures. ProtCross
0.2.3 preserves input order, keeps each structure in its own result directory,
and bounds ESM-C and PointNet++ microbatches by count and residue cost.

```python
from pathlib import Path

from protcross.inference import ProtCrossPredictor

structure_dir = Path("structures")
inputs = sorted(
    path
    for path in structure_dir.iterdir()
    if path.is_file() and path.suffix.lower() in {".pdb", ".cif", ".mmcif"}
)
if not inputs:
    raise FileNotFoundError(f"No PDB/mmCIF structures found in {structure_dir}")

output_dir = Path("batch-results")
output_dir.mkdir(parents=True, exist_ok=True)

predictor = ProtCrossPredictor.from_default_assets(
    device="auto",
    embedding_cache_dir=".protcross-feature-cache",
    accept_esm_license=True,
)

output_paths = []
for index, path in enumerate(inputs, start=1):
    result_dir = output_dir / f"{index:04d}-{path.stem}"
    result_dir.mkdir(parents=True, exist_ok=True)
    structure_suffix = ".cif" if path.suffix.lower() in {".cif", ".mmcif"} else ".pdb"
    output_paths.append(
        {
            "output_pdb": result_dir / f"{path.stem}.protcross{structure_suffix}",
            "scores_tsv": result_dir / f"{path.stem}.protcross.scores.tsv",
            "pocket_json": result_dir / f"{path.stem}.protcross.pockets.json",
            "summary_json": result_dir / f"{path.stem}.protcross.summary.json",
        }
    )

results = predictor.predict_many(
    inputs,
    output_paths=output_paths,
    batch_size=4,
    max_batch_residues=4096,
    max_batch_quadratic_cost=4 * 1022**2,
    return_exceptions=True,
)

for path, result in zip(inputs, results):
    if isinstance(result, Exception):
        print(f"FAILED {path}: {result}")
    else:
        print(f"DONE   {path}: {result.output_files['summary_json']}")
```

The per-input result directories keep files distinct when structures share a
stem or use different coordinate formats. `return_exceptions=True` lets the
remaining inputs finish and keeps each exception in its original list position.
Pass `chain_ids=[None, "A", ...]` to choose a chain independently for each
input; `None` selects all scorable chains. The list must follow `inputs` order.

### Scheduler controls

| Parameter | Default | Budget |
| --- | ---: | --- |
| `batch_size` | `4` | Maximum PointNet++ graphs per microbatch |
| `max_batch_residues` | `4096` | Maximum total graph nodes |
| `max_batch_quadratic_cost` | `4 * 1022**2` | Maximum `sum(n_i**2)` geometry cost |
| `feature_batch_size` | CPU: `min(batch_size, 2)`; accelerator: `batch_size` | Maximum ESM-C sequences per microbatch |
| `max_feature_padded_tokens` | `2048` | Maximum padded ESM-C token matrix |
| `return_exceptions` | `False` | Per-item exception collection |

Each structure remains one PointNet++ graph and each chain remains one ESM-C
context. Identical chain sequences share feature extraction within a
microbatch. Persistent cache entries include the sequence, PCA dimension,
context limit, cache schema, and ESM/PCA asset identity.

The residue and quadratic-cost settings bound both a microbatch and each
individual structure. Select one chain or raise an explicit limit when a graph
exceeds that budget. Accelerator memory errors trigger recursive microbatch
splitting; a single item that still exhausts memory is reported through the
selected exception mode. Desktop batch jobs use this API with groups of four.

With the default exception mode, the return type is
`list[PredictionResult]`. With `return_exceptions=True`, it is
`list[PredictionResult | Exception]`. Each `output_paths` entry accepts the four
writer names shown in the example.

## Python API

### Single-structure helper

Use `predict_pdb` for a script that scores one structure and writes a complete
result package:

```python
from pathlib import Path

from protcross.inference import predict_pdb

output_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)

result = predict_pdb(
    "examples/6fhu.pdb",
    device="cpu",
    accept_esm_license=True,
    output_pdb="results/6fhu.protcross.pdb",
    scores_tsv="results/6fhu.protcross.scores.tsv",
    pocket_json="results/6fhu.protcross.pockets.json",
    summary_json="results/6fhu.protcross.summary.json",
)

print(result.format_summary())
```

`predict_pdb` resolves and downloads missing managed assets by default. Set
`offline=True` for a local-cache-only run.

### Reusable predictor

Load `ProtCrossPredictor` once when a process will score several structures:

```python
from protcross.inference import ProtCrossPredictor

predictor = ProtCrossPredictor.from_default_assets(
    device="cpu",
    accept_esm_license=True,
)

result = predictor.predict("examples/6fhu.pdb", threshold=0.5)
records = result.to_records()
pockets = result.to_pocket_dict()
summary = result.to_summary_dict()
```

Use one predictor per device worker and serialize calls that share an instance.
Independent processes load independent model instances. Python writers publish
each output file through atomic replacement and keep the annotated structure in
the input coordinate format.

Structure inspection is also available from Python:

```python
from protcross.data import inspect_structure

inspection = inspect_structure("examples/6fhu.pdb")
```

### PredictionResult interface

| Member | Value |
| --- | --- |
| `scores` | Read-only NumPy residue-score array |
| `binding_residues` | Residues above the active threshold |
| `to_records()` | Extended residue dictionaries |
| `to_pocket_dict()` | Cluster schema payload |
| `to_summary_dict()` | Run summary payload |
| `format_summary()` | Terminal-oriented text summary |
| `write_pdb()` | Annotated coordinate output |
| `write_scores_tsv()` | Residue table output |
| `write_pocket_json()` | Cluster JSON output |
| `write_summary_json()` | Provenance JSON output |

## Assets

### Managed assets

Prediction uses three matched assets: a ProtCross checkpoint, its PCA reducer,
and ESM-C 600M weights. For most users, install the managed bundle once:

```bash
protcross setup-assets --accept-esm-license
```

The command verifies and installs these files under
`~/.cache/protcross/assets/v0.1.2`:

```text
protcross-0.1.2-binding-moad-final.ckpt
pca_esmc_128_binding_moad_0.1.2.pkl
esmc_600m_2024_12_v0.pth
protcross-assets.json
```

The ESM-C download is approximately 2.14 GiB. Interrupted transfers resume from
retained partial data. Setup verifies SHA256 hashes and publishes completed
files atomically. Later predictions reuse the manifest verification while file
size and modification time remain unchanged.

| Bundle | Checkpoint and PCA |
| --- | --- |
| `default`, `latest`, `0.1.2` | `protcross-0.1.2-binding-moad-final.ckpt` and `pca_esmc_128_binding_moad_0.1.2.pkl` |
| `0.1.1-paper` | `best-epoch=59.ckpt` and `pca_esmc_128.pkl` for the published PDBbind v2020 workflow |

Release compatibility:

| Interface | Version |
| --- | --- |
| Application and Desktop | `0.2.3` |
| Default checkpoint/PCA bundle | `0.1.2` |
| Paper reproduction bundle | `0.1.1-paper` |
| Pocket and summary schemas | `protcross-pocket-v2`, `protcross-summary-v2` |

`default` and `latest` resolve to the bundle pinned by the installed package.
Keep the checkpoint and PCA reducer from the same bundle.

Configure another managed directory with either interface:

```bash
PROTCROSS_ASSETS_DIR=/data/protcross-assets \
  protcross setup-assets --accept-esm-license

protcross setup-assets \
  --output-dir /data/protcross-assets \
  --accept-esm-license
```

Use `--refresh-assets` for a fresh download and verification. Use `--offline`
or `--no-auto-assets` to limit prediction to local files.

### Existing or custom assets

Reuse an existing ESM-C file with an absolute path:

```bash
protcross setup-assets --skip-esm --accept-esm-license
protcross predict input.pdb \
  --esm-weights /absolute/path/to/esmc_600m_2024_12_v0.pth \
  --accept-esm-license \
  --out-dir protcross-results
```

Explicit release assets are verified against the selected bundle's SHA256.
Custom experimental files use `--trust-unverified-assets`; output provenance
stores their real hashes and verification status.

```bash
protcross predict input.pdb \
  --checkpoint /trusted/custom/model.ckpt \
  --esm-weights /trusted/custom/esmc.pth \
  --pca /trusted/custom/reducer.pkl \
  --trust-unverified-assets \
  --accept-esm-license
```

Checkpoint, PCA, and PyTorch weight files can contain executable serialized
objects. Load them from controlled storage. ESM-C weights are distributed
through the EvolutionaryScale model repository.[^2]

CLI and Desktop assets use separate storage roots. Desktop records its selected
assets in the operating-system application-data directory.

## Desktop application

ProtCross Desktop combines a Tauri 2 shell, React interface, Mol* viewer, and
local Python sidecar. The sidecar binds to `127.0.0.1` on a dynamic port and
uses a per-session token for local API requests.

### Install

Download the matching release artifact and `SHA256SUMS.txt` from
[the v0.2.3 release](https://github.com/GeraltZeroZhong/ProtCross/releases/tag/v0.2.3):

```text
ProtCross_Desktop_0.2.3_x64-setup.exe
ProtCross_Desktop_0.2.3_macos-aarch64.dmg
```

The guided first-launch workflow installs a CPU runtime, records ESM-C term
acceptance, downloads or imports model assets, and validates readiness. Advanced
runtime options provide NVIDIA CUDA on Windows, Apple MPS on macOS, custom Conda
environments, and proxy configuration. Reserve approximately 5 GiB for the
runtime and ESM-C asset.

Run a first Desktop prediction in five steps:

1. Open **Setup**, install a backend, and validate it.
2. Review the ESM-C terms, then download or import the ESM-C weights.
3. Open **Predict**, select a local PDB/mmCIF file, and inspect it.
4. Select the chain scope and output directory; expand prediction settings when needed.
5. Open **Results** to inspect the 0–1 score color scale, residue clusters, and output package.

The interface follows the system appearance by default and also provides light
and dark modes. Keyboard focus indicators, reduced-motion handling, high-contrast
support, resizable layouts, semantic status messages, and compact-window reflow
are built into the frontend design system.

### Local architecture

```text
Tauri 2 shell
  -> React + Mol* frontend
  -> token-authenticated localhost API
  -> protcross_desktop Python sidecar
  -> ProtCross predictor and batch scheduler
```

Desktop batch jobs reuse one predictor and the input inspection reports. Each
staged file has its own chain selector, including all scorable chains and blank
chain IDs. The monitor exposes per-item status and full errors, supports
cancellation between microbatches, and can start a new queue containing only
failed or interrupted items.

For a batch run, open **Batch**, add the structures, review the deduplicated
list, choose one output root, and start the queue. Progress and per-item status
remain visible while the queue runs. Each input receives a unique subdirectory
with the four-file result package. Select any completed row to inspect it in
**Results**. Recent batch history is stored in the Desktop application-data
directory and restored on the next launch. Work that was active during a
restart appears as interrupted and can be retried; completed items and their
files remain available.

When the output field is empty, Desktop writes single predictions under its
application-data `outputs/<structure>/` directory and batch predictions under
`outputs/batch/<job-id>/`. The active platform path is displayed below the
output field.

The **Results** workspace colors scored residues by model score and gives
unscored residues a neutral gray color, including residues outside a selected
chain or a truncated sequence context. Adjust the displayed score cutoff and
Cα clustering distance to regroup the complete residue table immediately;
this updates the viewer and cluster inspector without running the model or
changing output files. Reopen a previous package by selecting its
`*.protcross.summary.json` file. Use **Diagnostics** to test the runtime, review
asset health, and export a sanitized support ZIP with bounded log excerpts.

## Model and inference pipeline

```mermaid
flowchart LR
    accTitle: ProtCross inference pipeline
    accDescr: Coordinate files are parsed into per-chain sequences and a shared C-alpha graph, embedded with ESM-C and PCA, scored by PointNet++, and serialized as annotated coordinates, scores TSV, pockets JSON, and summary JSON.

    coordinates["PDB or mmCIF"] --> parser["Structure parser"]
    parser --> sequence["Per-chain sequence"]
    parser --> geometry["Centered Cα graph"]
    sequence --> esmc["ESM-C 600M"]
    esmc --> pca["PCA 128"]
    pca --> pointnet["PointNet++"]
    geometry --> pointnet
    pointnet --> scores["Residue scores"]
    scores --> clusters["Threshold and cluster"]
    clusters --> outputs["Four-file result package"]
```

### Components

| Component | Configuration |
| --- | --- |
| ESM-C | 600M, hidden size 1,152, 36 layers, 18 attention heads |
| PCA | Paired reducer, 128 output dimensions |
| Set abstraction 1 | Sampling ratio `0.5`, radius `10 Å`, 64 neighbors |
| Set abstraction 2 | Sampling ratio `0.25`, radius `20 Å`, 64 neighbors |
| Set abstraction 3 | Sampling ratio `0.1`, radius `40 Å`, 64 neighbors |
| Feature propagation | Three `k=3` interpolation stages |
| Segmentation head | `128 -> 64 -> 32 -> 2`, dropout `0.5` |

The inference parser creates centered Cα geometry and per-chain sequence
chunks. ESM-C embeddings are reduced with the PCA asset paired to the selected
checkpoint. PointNet++ processes every input structure as an independent graph
and returns one two-class logit vector per residue.

The geometry backend uses pure-PyTorch farthest-point sampling, radius search,
and stable KNN interpolation. Radius neighborhoods retain the first 64 source
neighbors in canonical input order. Inference runs in FP32 and records the
execution mode in `summary.json`. Canonical ordering and neighbor selection are
deterministic; floating-point reductions remain device- and kernel-dependent.

### Training architecture

ProtCross uses a source-domain residue segmentation objective and adversarial
domain adaptation between PDB and matched AF2 structures. The target-domain
adversarial term supports pLDDT weighting. The maintained model configuration
uses `feature_dim=128`, `use_esm=true`, `use_da=true`, and `da_weight=0.2`.

Training labels are generated from standard-residue Cα atoms within `6 Å` of
eligible hetero-residue atoms. The parser applies a versioned residue-name
filter for waters, common crystallization additives, salts, ions, and terminal
caps.

## Training and development

### Repository layout

```text
src/protcross/
├── cli/             unified command-line entry points
├── data/            parsing, ESM, PCA, preprocessing, datasets
├── models/          PointNet++, segmentation, domain adaptation
├── inference/       prediction, batching, clustering, serialization
├── training/        Hydra and Lightning orchestration
├── evaluation/      metrics and adaptive evaluation
└── experiments/     benchmark and strategy-search workflows
configs/             Hydra training configuration
desktop/             Tauri, React, Python sidecar, packaging
tests/               core, inference, data, CLI, Desktop
reproduction/        archived paper-era workflows
examples/            example coordinate files
```

### Development environment

```bash
conda env create -f environment.yml
conda activate protcross
python -m pip install -e ".[dev,esm]"
```

### Maintained training workflow

Place source coordinate files in `data/raw_pdb`. `protcross download-af2`
resolves PDB-to-UniProt accessions, downloads matching AF2 structures, and
writes `artifacts/pdb_uniprot_mapping.json` for the later mapping stage.

```bash
protcross download-af2 \
  --raw-pdb-dir data/raw_pdb \
  --output-dir data/raw_af2 \
  --mapping-file artifacts/pdb_uniprot_mapping.json

protcross preprocess \
  --data-dir data/raw_pdb \
  --output-dir data/processed_pdb \
  --fit-pca \
  --esm-weights ~/.cache/protcross/assets/v0.1.2/esmc_600m_2024_12_v0.pth \
  --pca artifacts/protcross-pca-128.pkl \
  --pca-dim 128 \
  --accept-esm-license

protcross preprocess \
  --data-dir data/raw_af2 \
  --output-dir data/processed_af2 \
  --esm-weights ~/.cache/protcross/assets/v0.1.2/esmc_600m_2024_12_v0.pth \
  --pca artifacts/protcross-pca-128.pkl \
  --is-af2 \
  --accept-esm-license

protcross map-labels \
  --processed-pdb-dir data/processed_pdb \
  --processed-af2-dir data/processed_af2 \
  --raw-pdb-dir data/raw_pdb \
  --raw-af2-dir data/raw_af2 \
  --mapping-file artifacts/pdb_uniprot_mapping.json

protcross train
```

Preprocessing writes one `.pt` tensor package per structure and an atomic
`protcross-preprocess-manifest.json`. The manifest records completion state,
input hashes, generated outputs, failures, and skipped files. PCA fitting uses
the configured preprocessing seed. Training cache freshness uses `.pt` file
names, sizes, and modification times, so dataset startup does not reread every
tensor package solely to hash it.

Hydra configuration entry points:

| File | Scope |
| --- | --- |
| [`configs/train.yaml`](configs/train.yaml) | experiment seed and composed defaults |
| [`configs/data/protein_seg.yaml`](configs/data/protein_seg.yaml) | source/target datasets and loaders |
| [`configs/model/da_module.yaml`](configs/model/da_module.yaml) | segmentation and domain-adaptation model |
| [`configs/trainer/default.yaml`](configs/trainer/default.yaml) | epochs, precision, devices, logging |

Example overrides:

```bash
protcross train model.use_da=False
protcross train model.use_esm=False trainer.max_epochs=5
protcross train \
  data.data_dir_pdb=/abs/path/to/processed_pdb \
  data.data_dir_af2=/abs/path/to/processed_af2
```

### Paper workflow

```bash
protcross setup-assets \
  --asset-version 0.1.1-paper \
  --accept-esm-license

python reproduction/legacy/run_Predict_ProtCross.py \
  --pdb_file examples/6fhu.pdb \
  --asset-version 0.1.1-paper \
  --accept-esm-license
```

The [`reproduction/legacy/`](reproduction/legacy/) directory contains the archived PDBbind v2020
workflow and paper-era entry points.

### Tests

```bash
python -m pytest -q
ruff check src tests desktop/backend desktop/installer
(cd desktop/frontend && npm ci && npm run build)
python desktop/installer/validate_version_consistency.py
python -m build --wheel --outdir dist
```

Desktop backend tests run from the repository root:

```bash
python -m pytest -q tests/desktop
```

Desktop development requires the Tauri 2 platform prerequisites.[^3]

```bash
python -m pip install -e ".[predict]"
python -m pip install -e desktop/backend
cd desktop/frontend
npm ci
export PROTCROSS_DESKTOP_BACKEND_PATH="../backend"
export PROTCROSS_DESKTOP_PYTHON="python"
npm run tauri:dev
```

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| Unsupported Python version | Create a Python 3.10 environment and reinstall |
| ESM-C acceptance prompt | Run `protcross setup-assets --accept-esm-license` |
| Interrupted asset transfer | Repeat setup; the downloader resumes retained `.part` data |
| Asset verification failure | Run setup with `--refresh-assets` |
| Existing output path | Select another `--out-dir` or pass `--overwrite` |
| CUDA or MPS unavailable | Select `--device cpu` and verify with `protcross predict --help` |
| Structure parse failure | Run `protcross inspect input.pdb --json` and review the report |
| Chain exceeds 1,022 residues | Select a chain or pass `--allow-truncation` |
| Batch memory pressure | Reduce `batch_size`, residue budget, or padded-token budget |
| Desktop backend failure | Open **Diagnostics**, run the environment test, and reinstall the selected backend |
| Desktop asset failure | Re-run asset verification or import a local asset file from **Setup** |
| Desktop issue report | Export the diagnostic package from **Diagnostics** |

Report reproducible issues through the
[issue tracker](https://github.com/GeraltZeroZhong/ProtCross/issues). Include the
exact command, `summary.json`, platform, Python/PyTorch versions, and sanitized
diagnostics.

## Version history

### 0.2.3

- Reused verified asset manifests and metadata-based dataset signatures to cut
  repeated hashing and startup work; accelerated AF2 indexing, preprocessing,
  strategy search, and long-log assembly.
- Removed redundant prediction, CLI, data-loading, and output-rollback layers;
  retained bounded scheduling, input contracts, and atomic per-file outputs.
- Improved machine-readable inspection errors, chain guidance, Desktop error
  display and diagnostic exports, per-file batch chain selection, failed-item
  retry, restart-safe batch history, interactive result regrouping, and neutral
  rendering for unscored residues.

### 0.2.2

- Added bounded ESM-C and PointNet++ microbatching with graph, residue,
  quadratic-cost, and padded-token budgets; duplicate-sequence reuse; validated
  feature caching; recursive accelerator-OOM splitting; and ordered per-item
  error isolation.
- Accelerated deterministic geometry and inference parsing; corrected small-set
  split leakage, label-alignment statistics, AF2 mapping, and mmCIF residue
  identity; added finite-value gates, isolated feature-cache namespaces, and
  transactional dataset publication and atomic per-file result writes.
- Rebuilt ProtCross Desktop around responsive task workspaces, semantic OKLCH
  themes, accessible interaction states, score-aware Mol* rendering, structured
  diagnostics, persistent batch feedback, and result-package reopening.

### 0.2.1

- Added model-free structure inspection and Desktop input checks.
- Added resumable, locked, SHA256-verified asset setup.
- Added deterministic pure-PyTorch geometry and canonical residue ordering.
- Added v2 output schemas and record-preserving structure annotation.

### 0.2.0

- Added the Tauri/React Desktop application and Mol* visualization.
- Added local backend setup, batch queue foundations, and diagnostics.

### 0.1.3

- Added the four-file result package and unified `protcross` CLI.
- Archived paper-era wrappers under `reproduction/legacy/`.

### 0.1.2

- Added the Binding MOAD checkpoint and paired PCA asset bundle.
- Added the maintained structure preprocessing rules.

### 0.1.1 and 0.1.0

- Added the paper checkpoint/PCA, preprocessing pipeline, domain adaptation,
  evaluation components, and initial command-line workflow.

## Citation

If ProtCross contributes to a publication, cite:

```bibtex
@article{zhong2026protcross,
  title = {{ProtCross}: Bridging the {PDB}-{AlphaFold} Gap for Binding Site Prediction with Protein Point Clouds},
  author = {Zhong, Shuyu and Jiang, Yuying},
  journal = {Journal of Chemical Information and Modeling},
  year = {2026},
  volume = {66},
  number = {7},
  pages = {3688--3701},
  doi = {10.1021/acs.jcim.5c03224}
}
```

## License

ProtCross source code is distributed under the [MIT License](LICENSE).
ESM-C weights use EvolutionaryScale's model terms.[^1] ProtCross checkpoint and
PCA bundles are distributed separately from ESM-C weights.

[^1]: EvolutionaryScale. [Cambrian Non-Commercial License Agreement](https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement).

[^2]: EvolutionaryScale. [ESM-C 600M 2024-12 model repository](https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12).

[^3]: Tauri. [Prerequisites](https://v2.tauri.app/start/prerequisites/).
