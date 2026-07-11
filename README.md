# ProtCross

ProtCross runs locally on PDB or mmCIF coordinate files, including AlphaFold
models. It is a protein binding-site prediction tool that scores each scorable
standard amino-acid residue and exports annotated coordinates, a residue-score
table, predicted-residue clusters, and a reproducibility summary.

**Project citation:** Zhong, S., & Jiang, Y. (2026). “ProtCross: Bridging the
PDB-AlphaFold Gap for Binding Site Prediction with Protein Point Clouds.”
*Journal of Chemical Information and Modeling*, 66(7), 3688–3701.
[https://doi.org/10.1021/acs.jcim.5c03224](https://doi.org/10.1021/acs.jcim.5c03224)

## Quick start

### Desktop

Download the Windows x64 or macOS Apple Silicon testing build from
[GitHub Releases](https://github.com/GeraltZeroZhong/ProtCross/releases). Open
the app, install the recommended CPU backend, download the model weights after
accepting the ESM-C license, select a structure, and choose **Run prediction**.

### Command line

After following the platform-specific [installation instructions](#installation-and-assets),
run:

```bash
protcross inspect input.pdb
protcross setup-assets --accept-esm-license
protcross predict input.pdb --out-dir protcross-results
```

Review the [ESM-C license](https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement)
before accepting it. Results are written to `protcross-results/`; use
`input.protcross.scores.tsv` for residue-level scores.

## Contents

- [Quick start](#quick-start)
- [Inspect a structure](#inspect-a-structure)
- [Understand the results](#understand-the-results)
- [Scientific scope and limitations](#scientific-scope)
- [Installation and assets](#installation-and-assets)
- [Common workflows and Python API](#common-workflows-and-python-api)
- [Troubleshooting and help](#troubleshooting-and-help)
- [Maintainer guide](#maintainer-guide)
- [Version history](#version-history)
- [License, citation, and references](#license-citation-and-references)

## Inspect a structure

`protcross inspect` is read-only and needs no model assets. Start by inspecting
each new coordinate file:

```bash
protcross inspect input.cif
protcross inspect input.cif --chain A
protcross inspect input.cif --json
protcross inspect --help
```

Example:

```text
Input: examples/6fhu.pdb
Format: PDB
Models: 1 (ProtCross scores the first model)
Chains: A
Scorable residues: 52
Longest chain context: 52 / 1022
Assembly: supplied coordinates only (biological-assembly operators are not applied)
  Chain A: 52 scorable; 0 missing CA; 0 modified/non-standard; 0 coordinate break(s); 0 numbering gap(s)
Ready for prediction.
```

Use the report as follows:

| Finding | Recommended action |
| --- | --- |
| Multiple models | Confirm that scoring the first coordinate model is appropriate |
| Multiple chains | Confirm the supplied assembly, then keep all chains or select one with `--chain` |
| Missing `CA` or modified residues | Check whether omitted residues affect the biological question |
| Coordinate breaks or numbering gaps | Verify residue mapping against the source structure |
| Parser warning | Repair malformed or duplicate coordinates before relying on the result |
| Chain longer than 1,022 scorable residues | Split deliberately or opt into `--allow-truncation` |

## Understand the results

For `input.pdb`, the default package is:

| File | Purpose | Quantitative use |
| --- | --- | --- |
| `input.protcross.pdb` | Annotated copy with scores in B-factor fields | Visualization; PDB scores round to two decimals |
| `input.protcross.scores.tsv` | Per-residue identifiers, model scores, coordinates, calls, and ranks | Preferred residue-level table |
| `input.protcross.pockets.json` | Thresholded predicted-residue clusters and geometry | Machine-readable cluster analysis |
| `input.protcross.summary.json` | Run settings, warnings, assets, hashes, outputs, and top cluster | Provenance and run comparison |

For mmCIF input, the annotated structure uses `.cif`. The `pockets.json`,
`top_pocket`, and `probability` names are compatibility names: the contents are
predicted-residue clusters and uncalibrated model scores, not physical pockets
or probabilities.

### Score and threshold

- Treat the continuous `softmax(logits)[:, 1]` value as a residue-ranking score
  within a run. It is not independently calibrated: `0.8` does not mean an 80%
  binding probability, and score differences do not measure affinity.
- Keep the checkpoint, principal-component analysis (PCA) reducer,
  preprocessing, assembly, and software version fixed when comparing scores.
  The model does not quantify epistemic,
  conformational, or assay uncertainty.
- The default call is strictly `score > 0.5`; exactly `0.5` is not selected.
  This equal-logit threshold is a software default, not a biologically optimized
  cutoff. Changing it updates calls, clusters, and centroids without changing
  the continuous scores.

### Annotated structure

Every atom in a scored residue receives the same model score in the output
B-factor field. These values are neither experimental B-factors nor pLDDT.
PDB output preserves record order and text and changes only B-factor columns on
`ATOM` and `HETATM` lines; use TSV or JSON for full precision. mmCIF categories
are retained, but the text is reserialized. Annotated output keeps the input
format; convert PDB/mmCIF separately if another format is needed.

By default, unscored atoms in a scored model are written as `0.0`, so zero can
mean _unscored_ rather than confidently negative. Fully unscored additional
models retain their original values. `--unscored-bfactor-policy keep` preserves
unscored input values but mixes different meanings in one field.

### Output schemas

- `scores.tsv` uses `model_score` as the canonical field. `probability` is an
  equal-value compatibility alias.
- `pockets.json` uses schema `protcross-pocket-v2`.
- `summary.json` uses schema `protcross-summary-v2`.
- Both JSON files record score semantics, input identity, asset provenance,
  input interpretation, threshold, clustering, and truncation. `summary.json`
  additionally records runtime versions, warnings, and output paths.
- An empty result is valid:
  `selected_residue_count: 0`, `top_pocket: null`,
  `aggregate_pocket: null`, and `cluster_count: 0`.

<a id="scientific-scope"></a>

## Scientific scope and limitations

### Model task and architecture

ProtCross ranks residues by resemblance to the source-domain positive labels
defined below. Use it to rank residues or compare regions within a run while
keeping assets and settings fixed.

The model combines centered Cα point clouds with per-chain ESM-C embeddings
reduced to 128 dimensions by the paired PCA asset. Its PointNet++ spatial
aggregation radii are 10, 20, and 40 Å. Training used adversarial domain
adaptation from PDB to AlphaFold2 (AF2); normalized AF2 pLDDT weighted the
target-domain adversarial loss as `sin((π/2) × normalized pLDDT)`. Input
B-factors or pLDDT are not direct features of the inference segmentation head.

### Model and asset versions

| Component | Version or role |
| --- | --- |
| Software and output interface | `0.2.1` |
| Default checkpoint | `0.1.2`, Binding MOAD-selected structures with heuristic hetero-residue-proximity labels |
| Default PCA reducer | `0.1.2`, paired with the default checkpoint |
| ESM-C backbone | `esmc-600m-2024-12` |
| Published-paper assets | `0.1.1-paper`, PDBbind v2020 refined-set workflow |

The published paper evaluated the `0.1.1-paper` assets. ProtCross `0.2.1`
instead uses the default `0.1.2` checkpoint and paired PCA bundle; that model
was not evaluated in that paper workflow. Keep each checkpoint paired with the
PCA reducer from the same asset bundle.

### Exact input interpretation

| Input property | ProtCross behavior |
| --- | --- |
| Coordinate model | Scores the first model only |
| Assembly | Uses coordinates exactly as supplied; applies no biological-assembly or crystallographic symmetry operators |
| Scorable residues | The 20 standard polymer amino acids represented as `ATOM` residues and containing `CA` |
| Geometry | Centers Cα coordinates; all selected chains form one point cloud |
| Sequence | Builds each chain from coordinate-observed scorable residues, not `SEQRES` or `_entity_poly` alignment |
| ESM-C context | Embeds chains separately, then concatenates their reduced features |
| Chain selection | Uses all chains by default or one author chain ID with `--chain` |
| Ligands and non-protein atoms | Not inference features |
| Input B-factor or pLDDT | Retained as metadata but not passed to the prediction head |

### Release-training label definition

The default labels come from the generic structure parser at
`src/protcross/data/structure.py`, not directly from curated Binding MOAD
ligand-residue annotations. A protein residue receives label `1` when:

1. it is one of the 20 standard polymer amino acids and contains `CA`;
2. that `CA` lies within 6.0 Å of any atom in an eligible ligand-like residue;
3. the ligand-like residue is not excluded by the default residue-name filter.

This is a **protein Cα to ligand-residue atom** rule, not a
protein-heavy-atom to ligand-heavy-atom distance. Explicit hydrogens are not
removed before the neighbor search. A long side chain can contact a ligand
while its Cα remains beyond 6 Å and is labeled negative.

A neighboring residue is treated as ligand-like when its residue name is
outside the 20 standard amino acids or Bio.PDB represents it with a nonblank
hetero field. This broad rule can include cofactors, glycans, nucleic acids,
modified residues, and unfiltered crystallization components; it is not a
small-organic-ligand-only task.

Default exclusions:

| Category | Excluded residue names |
| --- | --- |
| Water-like | `HOH`, `WAT`, `H2O`, `DOD` |
| Additives, buffers, solvents | `GOL`, `EDO`, `PEG`, `PE4`, `PG4`, `PGE`, `DMS`, `ACT`, `ACY`, `FMT`, `MES`, `TRS` |
| Salt-like species | `SO4`, `PO4`, `NO3` |
| Monatomic ions | `NA`, `CL`, `K`, `MG`, `CA`, `ZN`, `MN`, `FE`, `CU`, `CO`, `NI`, `CD`, `HG`, `CS`, `RB`, `LI`, `F`, `BR`, `I` |
| Protein-like artifacts or caps | `MSE`, `ACE`, `NME` |

These exclusions reduce common crystallization-artifact labels but also mean
that a listed metal ion alone does not create a positive metal-binding label.
Residue-name filters are a practical preprocessing rule; their biological
relevance remains structure-specific.

### Binding MOAD relationship and release training

Binding MOAD metadata selected the release PDB cohort.[^2] Preprocessing then
passed each complete coordinate file through the generic rule above; it did not
restrict labels to the particular valid ligand instance or ligand ID listed by
Binding MOAD. The release cohort is therefore described as **Binding
MOAD-selected structures with heuristic hetero-residue-proximity labels**.

The cohort records 41,409 selected PDB structures, 95,212 metadata entries
marked as valid ligand instances, and 20,387 unique valid ligand IDs. The latter
two counts describe the cohort; they are not counts of chemical entities
isolated during label generation.

| Domain | Candidate or preprocessed | Loaded for release training | Role |
| --- | ---: | ---: | --- |
| PDB source | 41,409 | 40,360 | Supervised segmentation and source-domain discrimination |
| Matched AF2 target | 8,953 | 8,953 | Unlabeled, pLDDT-weighted target-domain discrimination |

Repository provenance is retained in
`data/binding_moad/binding_moad_summary.json` for cohort and download counts and
`data/binding_moad/train_binding_moad_0.1.2.log` for loaded-sample counts and
training completion.

The source loader required at least one positive label. Training used
`train_split=all`, `target_split=all`, `val_split=null`, ran for 150 epochs,
and saved the final checkpoint. No held-out validation or test set was used for
checkpoint selection, threshold selection, calibration, or an unbiased
performance estimate.

Bundled internal-comparison artifacts have substantial training overlap:

- AF2 overlap: 1,114 of 1,178 accessions also occur in target training
  (`94.6%`).
- PDB overlap: 1,465 of 1,594 structure IDs occur in the source cohort
  (`91.9%`).

They are useful regression and debugging references rather than an independent
external benchmark. For a new benchmark, remove overlap at least by PDB ID and
UniProt accession, consider excluding close homologs, and choose the threshold
on a separate validation set before evaluating the test set.

### Geometric postprocessing

Postprocessing uses Ångström coordinates in the supplied input frame:

| Step | Exact behavior |
| --- | --- |
| Residue selection | Keep residues with `score > threshold` |
| Clustering | Single-linkage connected components on selected Cα atoms |
| Default edge cutoff | Cα–Cα distance ≤ 8.0 Å |
| Cluster ranking | Descending residue count, descending mean score, descending maximum score, then earliest canonical residue index |
| Cluster center | Score-weighted centroid of selected Cα coordinates |
| Aggregate center | Score-weighted centroid across every selected residue |

Single linkage is transitive: a chain of locally adjacent residues can form one
cluster even when its endpoints are far apart. JSON reports the maximum
pairwise Cα distance and sets `diffuse_single_linkage_cluster=true` when the
diameter exceeds `max(2 × cluster_cutoff, 16 Å)`; the flag does not split or
reject the cluster. All selected chains share one graph, so a cluster can span
chains at an interface.

The reported center is only a score-weighted Cα centroid, not a validated
cavity or docking-box center. ProtCross does not model molecular surfaces,
pocket volume, atom-level sterics, channels, or ligand reachability. Use an
independent cavity or surface method before docking or simulation.

### Structural interpretation limits

| Situation | Behavior and consequence |
| --- | --- |
| Multiple coordinate models | Only the first is scored; ensemble variability is absent |
| Multiple chains | Chains are analyzed jointly unless `--chain` is used; results depend on whether the file is an asymmetric unit, biological assembly, or custom complex |
| Assembly operators | Not applied; assembly-created interfaces may be absent |
| Modified amino acids | `MSE`, `SEC`, `PYL`, phosphoresidues, and other nonstandard residues are skipped |
| Standard residue without `CA` | Skipped and not scored |
| Coordinate gap | Observed residues on either side are still concatenated for ESM-C sequence context |
| Author numbering gap | Reported as a mapping warning; it is not by itself proof of missing coordinates |
| Malformed or duplicate content | Bio.PDB parser construction warnings are surfaced for review |
| Alternate `CA` conformers | Bio.PDB's selected conformer is used |
| Rigid coordinate rotation | The current PointNet++ backbone is not rotation invariant; rotating the supplied frame can change scores |
| Chain longer than 1,022 scorable residues | Fails by default; `--allow-truncation` keeps only the leading 1,022 scorable residues of each long chain |
| Ligand present in query input | Ligand coordinates are not features, although a holo conformation can still affect protein geometry |

Insertion codes and mmCIF author/label identifiers are retained in extended
outputs. Verify residue mapping before combining ProtCross results with
sequence annotations.

### Recommended uses

- Rank candidate residues or regions for follow-up analysis.
- Compare predicted-residue clusters within one run using fixed settings.
- Generate hypotheses for standard-residue proteins resembling the training
  domain.
- Screen PDB or AF2 structures after checking assembly, chain, coverage, and
  truncation warnings.
- Supply candidates to independent conservation, cavity, docking,
  mutagenesis, or experimental workflows.

### Interpretation boundaries

ProtCross is designed to nominate residue-level hypotheses. Use separate
methods or evidence for:

- Ligand identity, pose, selectivity, kinetics, thermodynamics, or druggability.
- Site-class-specific predictions for metal, nucleic-acid, glycan,
  protein-interface, or covalent sites.
- Scores for modified residues or residues without Cα coordinates.
- Biological-assembly inference, conformational ensembles, induced fit, or
  dynamics.
- Stand-alone evidence for high-stakes decisions or an independent benchmark
  when evaluation structures overlap release training.

### Reproducibility

Default asset identities:

| Asset | SHA256 |
| --- | --- |
| `protcross-0.1.2-binding-moad-final.ckpt` | `ccb56884b21402a027bfae9d4779f38c8f534513d980a96d7cd78c9931748b65` |
| `pca_esmc_128_binding_moad_0.1.2.pkl` | `0f4e11806a622642c07dad539cec4216030220c1b5f3fc44c7926a2f6bca4d62` |
| `esmc_600m_2024_12_v0.pth` | `8ef856e1a237ee3f995442df997a962e70057faadecf38fc0c8561bd3c2f4324` |

ProtCross 0.2.1 fixes the geometry backend to deterministic pure-PyTorch
farthest-point sampling (FPS), radius neighborhoods, and k-nearest neighbors
(KNN), and uses stable chain/polymer ordering. The backbone is not
rigid-rotation invariant, so keep the supplied coordinate frame unchanged for
direct rerun comparisons. Exact bitwise equality across CPU, CUDA, and MPS is
not guaranteed.

For a reproducible report:

1. Save `summary.json` and `pockets.json`; together they record the software,
   runtime, asset hashes, input checksum, settings, and predictions.
2. Record structure source/version, supplied assembly, selected chain,
   threshold, cluster cutoff, truncation policy, and the overlap-removal
   procedure for benchmark studies.
3. Use checkpoint and PCA assets from the same bundle (`0.1.2` or
   `0.1.1-paper`).
4. Re-run `protcross inspect` after any structure-file change.

Implementation traceability:

- Label construction: `src/protcross/data/structure.py`
- Input inspection: `src/protcross/data/inspection.py`
- Preprocessing: `src/protcross/data/preprocess.py`
- Training data loading: `src/protcross/data/dataset.py`
- Domain-adaptive model: `src/protcross/models/module.py`
- Confidence weighting: `src/protcross/models/domain_weights.py`
- Point-cloud backbone: `src/protcross/models/backbones/pointnet2.py`
- Prediction and postprocessing: `src/protcross/inference/predictor.py`

## Installation and assets

This README describes the `0.2.1` source tree. Run `protcross --version` before
following version-specific commands, especially when another installation may
already be on `PATH`.

### Requirements

| Use | Requirements |
| --- | --- |
| CLI or Python prediction | Python 3.10; Windows x64, Linux x86-64/WSL2, or macOS Apple Silicon; CPU supported |
| Desktop prediction | Windows 10/11 x64 or macOS 12+ Apple Silicon |
| Development or training | Python 3.10; Conda recommended; NVIDIA CUDA strongly recommended for large preprocessing/training |
| Desktop development | Node.js 20, Rust 1.88, and the Tauri v2 operating-system prerequisites |

Conda is not required for ordinary CLI prediction. CPU is the supported and
recommended default for one-structure prediction; training and large batches
are substantially slower without a compatible accelerator.

### Desktop installation

Download the following files from [GitHub Releases](https://github.com/GeraltZeroZhong/ProtCross/releases)
when available:

```text
ProtCross_Desktop_0.2.1_x64-setup.exe
ProtCross_Desktop_0.2.1_macos-aarch64.dmg
SHA256SUMS.txt
```

The `0.2.1` Desktop artifacts are unsigned and unnotarized prerelease testing
builds, so Windows SmartScreen or macOS Gatekeeper may warn or block them. The
macOS build supports Apple Silicon only; there is no Intel build.
Before opening an installer, compare its SHA256 with the corresponding entry in
`SHA256SUMS.txt`: use `shasum -a 256 <file>` on macOS or
`Get-FileHash <file> -Algorithm SHA256` in PowerShell.

On first launch:

1. Install the recommended CPU backend.
2. Review and accept the ESM-C license.
3. Download the ESM-C weights, or import an existing copy.
4. Select a `.pdb`, `.cif`, or `.mmcif` structure and review the inspection.
5. Choose the chains to score, then run the prediction.

The app performs prediction locally. Its Desktop assets are separate from the
CLI cache described below.

For any CLI platform, download and extract the `0.2.1` source archive from
[GitHub Releases](https://github.com/GeraltZeroZhong/ProtCross/releases), then
open the extracted repository root.

### CLI installation on Linux

From the checked-out repository root, using Python 3.10:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  torch==2.3.1+cpu torchvision==0.18.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[predict]"
protcross --version
```

### CLI installation on macOS Apple Silicon

From the checked-out repository root, using Python 3.10:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[predict]"
protcross --version
```

Do not use the Linux CPU wheel index on macOS; PyTorch supplies the Apple
Silicon wheel through the normal Python package index.

### CLI installation on Windows PowerShell

From the checked-out repository root, using 64-bit Python 3.10:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install `
  torch==2.3.1+cpu torchvision==0.18.1+cpu `
  --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install ".[predict]"
.\.venv\Scripts\protcross.exe --version
```

The explicit `.exe` path works without activating the environment. Use the
same prefix for the Quick start commands, or activate `.venv` first.

### Managed runtime assets

The Python package contains code only; runtime assets are installed separately.
The default managed command is:

```bash
protcross setup-assets --accept-esm-license
```

The first setup downloads about 2.14 GiB of ESM-C weights and resumes a valid
partial download if interrupted.

It installs:

```text
protcross-0.1.2-binding-moad-final.ckpt
pca_esmc_128_binding_moad_0.1.2.pkl
esmc_600m_2024_12_v0.pth
protcross-assets.json
```

The default directory is `~/.cache/protcross/assets/v0.1.2`. `default` and
`latest` both mean the stable bundle pinned by the installed package, currently
`0.1.2`; they do not query a floating remote release. Override the location:

```bash
PROTCROSS_ASSETS_DIR=/data/protcross-assets \
  protcross setup-assets --accept-esm-license

protcross setup-assets \
  --output-dir /data/protcross-assets \
  --accept-esm-license
```

Use `--refresh-assets` to repair and reverify a managed cache. Use
`--no-auto-assets` or `--offline` to forbid automatic downloads during
prediction.

CLI and Desktop assets are configured separately. CLI uses the cache or
explicit command-line paths; Desktop stores its manifest and selected assets in
the operating-system app-data directory. Running CLI setup does not make the
Desktop app ready.

### Existing or custom assets

If ESM-C is already available:

```bash
protcross setup-assets --skip-esm --accept-esm-license
protcross predict input.pdb \
  --esm-weights /absolute/path/to/esmc_600m_2024_12_v0.pth \
  --accept-esm-license \
  --out-dir protcross-results
```

Explicit release assets are verified against the selected bundle's SHA256. For
custom experimental files, `--trust-unverified-assets` records the local-file
override alongside their real hashes and verification status in output JSON.
Checkpoint, PCA pickle, and PyTorch weight deserialization can execute code, so
load custom files only from a trusted source.

```bash
protcross predict input.pdb \
  --checkpoint /trusted/custom/model.ckpt \
  --esm-weights /trusted/custom/esmc.pth \
  --pca /trusted/custom/reducer.pkl \
  --trust-unverified-assets \
  --accept-esm-license
```

ESM-C weights remain governed by EvolutionaryScale's model terms, separate from
ProtCross's MIT code license.[^1] The official model repository provides Git
LFS and Hugging Face CLI download options.[^3] After a manual download, pass the
absolute file path with `--esm-weights`.

## Common workflows and Python API

### Single chain, explicit outputs, and overwrite

```bash
protcross predict input.cif \
  --chain A \
  --output results/input.protcross.cif \
  --scores-tsv results/input.protcross.scores.tsv \
  --pocket-json results/input.protcross.pockets.json \
  --summary-json results/input.protcross.summary.json \
  --threshold 0.5
```

ProtCross refuses to overwrite existing outputs unless `--overwrite` is
provided. See `protcross predict --help` for all options.

### Python API

```python
from protcross.inference import predict_pdb

result = predict_pdb(
    "input.pdb",
    device="cpu",
    accept_esm_license=True,
    output_pdb="results/input.protcross.pdb",
    scores_tsv="results/input.protcross.scores.tsv",
    pocket_json="results/input.protcross.pockets.json",
    summary_json="results/input.protcross.summary.json",
)
print(result.format_summary())
```

For repeated predictions, load one `ProtCrossPredictor` and give each structure
unique output paths. This example writes a batch safely while loading the
models only once:

```python
from pathlib import Path
from protcross.inference import ProtCrossPredictor
from uuid import uuid4

inputs = sorted(Path("structures").glob("*.pdb"))
output_dir = Path("batch-results") / f"run-{uuid4().hex}"
output_dir.mkdir(parents=True, exist_ok=False)

# Run `protcross setup-assets --accept-esm-license` first.
predictor = ProtCrossPredictor.from_default_assets(
    device="cpu",
    embedding_cache_dir=".protcross-feature-cache",
    accept_esm_license=True,
)

for structure in inputs:
    stem = structure.stem
    predictor.predict(
        structure,
        output_pdb=output_dir / f"{stem}.protcross.pdb",
        scores_tsv=output_dir / f"{stem}.protcross.scores.tsv",
        pocket_json=output_dir / f"{stem}.protcross.pockets.json",
        summary_json=output_dir / f"{stem}.protcross.summary.json",
    )
```

### Device selection

```bash
protcross predict input.pdb --device cpu
protcross predict input.pdb --device cuda
protcross predict input.pdb --device cuda:1
protcross predict input.pdb --device mps
protcross predict input.pdb --device auto
```

`auto` selects CUDA, then Apple MPS, then CPU. Use CPU as the comparison
baseline when device-dependent numerical differences matter.

## Troubleshooting and help

### Wrong Python version

The release wheel supports Python `>=3.10,<3.11`. Verify with
`python --version`; create a Python 3.10 environment rather than forcing the
package into 3.11 or newer.

### License confirmation required

Review the ESM-C terms, then run:

```bash
protcross setup-assets --accept-esm-license
```

If an older managed manifest lacks confirmation, repeat that command. Explicit
ESM paths may still require `--accept-esm-license` for the current run.

### Interrupted or incomplete asset download

Repeat the same setup command. ProtCross retains and resumes `.part` data.
Keep at least 2.4 GiB free on the asset target, and plan for roughly 5 GiB total
when the Python environment is installed on the same disk. Use
`--refresh-assets` when a completed file fails verification.

### Existing outputs

ProtCross refuses silent replacement. Select a new `--out-dir` or pass
`--overwrite` intentionally.

### CUDA or MPS unavailable

Run with `--device cpu`. Use `--device auto` only when automatic acceleration
selection is desired. MPS is experimental.

### Structure rejected

Run `protcross inspect input.pdb --json`. Review chain choice, missing Cα,
modified residues, parser warnings, and the 1,022-residue context limit before
using `--allow-truncation`.

### Command help

```bash
protcross --help
protcross COMMAND --help
```

Report reproducible bugs at the
[ProtCross issue tracker](https://github.com/GeraltZeroZhong/ProtCross/issues)
with `summary.json`, the exact command, platform, Python/PyTorch versions, and
sanitized diagnostics. Do not attach confidential structures or paths.

## Maintainer guide

### Maintained training workflow

This is the current maintained workflow, not the exact `0.1.2` release-training
invocation. The default configuration uses `train`/`train`/`val` splits and 70
epochs. Create the full development environment:

```bash
conda env create -f environment.yml
conda activate protcross
python -m pip install -e ".[dev,esm]"
```

The maintained pipeline is:

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

To match the recorded `0.1.2` split and epoch settings, use:

```bash
protcross train \
  data.train_split=all \
  data.target_split=all \
  data.val_split=null \
  trainer.max_epochs=150
```

Hydra overrides can change any configured value:

```bash
protcross train model.use_da=False
protcross train model.use_esm=False trainer.max_epochs=5
protcross train \
  data.data_dir_pdb=/abs/path/to/processed_pdb \
  data.data_dir_af2=/abs/path/to/processed_af2
```

Main configuration files are `configs/train.yaml`,
`configs/data/protein_seg.yaml`, `configs/model/da_module.yaml`, and
`configs/trainer/default.yaml`.

### Published-paper reproduction

The archived paper workflow uses PDBbind v2020 refined-set assets rather than
the default `0.1.2` release model:

```bash
protcross setup-assets \
  --asset-version 0.1.1-paper \
  --accept-esm-license
```

That bundle contains `best-epoch=59.ckpt` and `pca_esmc_128.pkl`. Historical
wrappers remain under `reproduction/legacy/`; maintained users should prefer
the unified `protcross` commands. An archived wrapper example is:

```bash
python reproduction/legacy/run_Predict_ProtCross.py \
  --pdb_file examples/6fhu.pdb \
  --asset-version 0.1.1-paper \
  --accept-esm-license
```

### Test the repository

```bash
python -m pytest -q
ruff check src tests desktop/backend desktop/installer

cd desktop/frontend
npm ci
npm run build
```

Desktop backend tests from the repository root:

```bash
python -m pytest -q tests/desktop
```

### Desktop architecture and development

The local Desktop path is:

```text
Tauri/React UI
  -> authenticated dynamic http://127.0.0.1:<port>
  -> protcross_desktop.server
  -> ProtCross Python API
```

The app never mutates a user-selected Conda environment. CPU and GPU/MPS
environments are separate so an acceleration failure cannot break CPU.
Before running Tauri, install Node.js 20, Rust 1.88, and the native packages in
the official Tauri v2 prerequisites guide; on Linux these include the GTK/WebKit
development stack.[^4]

```bash
python -m pip install -e ".[predict]"
python -m pip install -e desktop/backend
cd desktop/frontend
npm ci
export PROTCROSS_DESKTOP_BACKEND_PATH="../backend"
export PROTCROSS_DESKTOP_PYTHON="python"
npm run tauri:dev
```

PowerShell:

```powershell
python -m pip install -e ".[predict]"
python -m pip install -e desktop\backend
cd desktop\frontend
npm ci
$env:PROTCROSS_DESKTOP_BACKEND_PATH = "..\backend"
$env:PROTCROSS_DESKTOP_PYTHON = "python"
npm run tauri:dev
```

Release preparation validates bundled assets, locked wheelhouses, package
versions, and native artifacts:

```bash
desktop/installer/prepare_bundled_assets.sh
python -m build --wheel --outdir dist
python desktop/installer/prepare_runtime_wheelhouse.py \
  --runtime-dir desktop/runtime \
  --backend cpu \
  --local-protcross-wheel dist/protcross-0.2.1-py3-none-any.whl
python desktop/installer/validate_runtime_bundle.py \
  --runtime-dir desktop/runtime \
  --backend cpu
python desktop/installer/validate_version_consistency.py
```

Formal public installers require Windows code signing and macOS app/DMG signing
plus notarization. Unsigned artifacts must remain prerelease testing builds.
Clean-machine release acceptance should cover Windows 10/11 CPU, compatible and
incompatible CUDA systems, macOS Apple Silicon CPU/MPS, no-Conda machines, and
working or incomplete user-managed Conda environments.

## Version history

### 0.2.1

- Adds `protcross inspect` and matching Desktop structure checks before model
  loading.
- Makes asset setup resumable and verifiable, with progress, disk checks,
  locking, SHA256 validation, and persistent ESM-C license confirmation.
- Defaults to CPU, uses deterministic pure-PyTorch geometry, canonicalizes
  residue order, and chunks large distance searches without changing neighbor
  semantics.
- Defines v2 result/provenance schemas, preserves PDB records while annotating
  B-factors, retains mmCIF model identifiers, and prevents silent overwrite.
- Hardens Desktop setup and local API handling, restores interrupted work, and
  serializes Mol* structure and selection updates.
- Publishes unsigned Windows x64 and macOS Apple Silicon artifacts only as
  prerelease testing builds and consolidates scientific and release guidance in
  this README.

### 0.2.0

- Introduced the local Desktop workflow, Mol* visualization, backend setup,
  batch foundations, and diagnostics.
- Kept ESM-C weights outside installers and behind explicit license review.
- Promoted the complete annotated-structure, TSV, cluster JSON, and summary
  JSON result package.

### 0.1.3

- Added default result-package generation and clustered-residue metadata.
- Consolidated maintained commands under the `protcross` CLI.
- Archived paper-era wrappers under `reproduction/legacy/`.

### 0.1.2

- Added the default `0.1.2` checkpoint and paired PCA assets.
- Added the default HETATM/additive/ion exclusions documented above.
- Trained on all loaded source and target structures without a held-out release
  validation/test split.

### 0.1.1 and 0.1.0

- Added the original paper checkpoint/PCA, installable CLI, reusable
  preprocessing, domain-adaptation, evaluation, and model components.
- Established the initial PDB-to-AF2 point-cloud framework.

## License, citation, and references

ProtCross code is distributed under the
[MIT License](https://github.com/GeraltZeroZhong/ProtCross/blob/main/LICENSE).
ESM-C weights are not part of that code license and remain governed by
EvolutionaryScale's model terms.[^1] ProtCross checkpoint/PCA assets do not bundle ESM-C weights;
review applicable terms before redistributing custom ESM-derived assets.

If ProtCross supports your work, cite:

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

[^1]: EvolutionaryScale. [Cambrian Non-Commercial License Agreement](https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement).

[^2]: [Binding MOAD archive, Zenodo record 13375913](https://zenodo.org/records/13375913).

[^3]: EvolutionaryScale. [ESM-C 600M 2024-12 model repository](https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12).

[^4]: Tauri. [Prerequisites](https://v2.tauri.app/start/prerequisites/).
