# ProtCross Examples

`6fhu.pdb` is a small smoke-test structure for quick inference checks with the released checkpoint and PCA file.

Example:

```bash
protcross predict examples/6fhu.pdb --accept-esm-license --out-dir examples/protcross-results
```

If the default runtime assets are missing, prediction installs them after
explicit ESM-C license confirmation.
The command writes:

```text
examples/protcross-results/6fhu.protcross.pdb
examples/protcross-results/6fhu.protcross.scores.tsv
examples/protcross-results/6fhu.protcross.pockets.json
examples/protcross-results/6fhu.protcross.summary.json
```

`6fhu.protcross.pockets.json` contains the probability-weighted pocket center,
selected residues, residue probabilities, residue count, and clustered pockets
in the input coordinate frame.

With an explicit asset directory:

```bash
protcross predict examples/6fhu.pdb \
  --assets-dir /absolute/path/to/protcross-assets \
  --out-dir examples/protcross-results
```

For source checkout testing without the automatic asset cache, pass the current
0.1.2 release asset files explicitly:

```bash
protcross predict examples/6fhu.pdb \
  --checkpoint checkpoints/protcross-0.1.2-binding-moad-final.ckpt \
  --esm-weights /absolute/path/to/esmc_600m_2024_12_v0.pth \
  --pca data/pca_esmc_128_binding_moad_0.1.2.pkl \
  --out-dir examples/protcross-results
```

Use `--trust-unverified-assets` only for custom or experimental local assets
after verifying their provenance.
