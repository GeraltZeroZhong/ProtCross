# Original 0.1.0 Reproducibility Scripts

This directory archives the command-line scripts that were exposed at the
repository root or under `scripts/` in the 0.1.0-era workflow. They are kept for
paper reproduction and historical command references.

For normal prediction, training, and data preparation, prefer the installed
ProtCross CLI where a maintained equivalent exists:

```bash
protcross predict input.pdb --accept-esm-license
protcross train
protcross preprocess
protcross download-af2
protcross map-labels
```

Legacy entry points are still runnable from the repository root when the full
training/reproduction environment and required data assets are present. They may
write logs, outputs, checkpoints, or reports, so treat them as workflow scripts
rather than harmless help/smoke commands:

```bash
python reproduction/legacy/run_Predict_ProtCross.py --pdb_file examples/6fhu.pdb
python reproduction/legacy/train.py
python reproduction/legacy/preprocess_esm.py
python reproduction/legacy/get_af2.py
python reproduction/legacy/map_labels.py
```

These scripts are a mix of thin wrappers and task-specific analysis utilities.
Maintained equivalents live under `src/protcross/` and are exposed through the
`protcross` console commands where available.

## Original Paper Assets

The maintained 0.1.3 package defaults to the stable `0.1.2` release model for
practical prediction. For reproducing the original paper workflow, use the
archived scripts here together with the paper asset bundle:

```bash
protcross setup-assets --asset-version 0.1.1-paper --accept-esm-license
```

That bundle resolves the 0.1.1 paper checkpoint and PCA reducer:

```text
best-epoch=59.ckpt
pca_esmc_128.pkl
```

Those assets correspond to the original PDBbind v2020 refined-set experiments.
When using the current release model instead, stay on the default asset bundle
and the modern `protcross predict` workflow from the repository root README.

## Archived Analysis Helpers

Evaluation and multi-seed benchmark helpers are intentionally documented here
rather than in the main README. They are not maintained CLI entry points and
should be run only in a reproduction workspace:

| Script | Purpose |
| --- | --- |
| `test_adaptive.py` | Legacy checkpoint evaluation helper. |
| `run_multiseed_benchmark.py` | Legacy multi-seed training/evaluation workflow. |
| `run_Strategy.py` | Archived strategy-search entry point. |
| `analyze_geometric.py` | Archived geometric analysis utility. |
| `sensitivity-cutoff.py` | Archived cutoff sensitivity analysis. |
| `eval_run.py`, `eval_dataset.py`, `eval_utils.py` | Legacy evaluation support modules. |
| `setup_assets.py` | Historical wrapper; prefer `protcross setup-assets`. |
| `map_labels-o.py` | Older label-mapping variant retained for provenance. |

The archived `pdb_uniprot_mapping.json` file is kept here for the legacy
analysis scripts. Modern AF2 download and label-mapping workflows write their
mapping file to `artifacts/pdb_uniprot_mapping.json` by default.

## Historical Notes

### 0.1.1

Engineering-focused reproducibility release.

- Added installable package metadata and console commands such as
  `protcross-predict`, `protcross-preprocess`, and `protcross-train`.
- Added the unified `protcross` CLI plus setup/download/map-label helpers.
- Added one-command runtime asset setup for ESM-C weights, released checkpoint,
  and PCA reducer.
- Split prediction into a lightweight API that accepts one PDB/mmCIF and writes
  binding probabilities to the B-factor column.
- Split shared structure parsing, ESM-C feature extraction, PCA reduction, AF2
  downloads, label mapping, metrics, experiments, and domain weighting into
  reusable modules.
- Added pytest smoke/unit tests, including checkpoint CPU forward on a published
  processed sample.

### 0.1.0

Initial public release of ProtCross.

- Provided the core domain-adaptive point-cloud binding-site prediction
  framework across PDB and AlphaFold2 structures.
- Included PointNet++ segmentation, ESM-C residue embeddings, PCA feature
  reduction, pLDDT-aware target weighting, and DANN-style domain adaptation.
- Documented the original training/evaluation workflow, including preprocessing,
  label mapping, Hydra configuration, testing, and multi-seed benchmarking.
