# ProtCross Examples

`6fhu.pdb` is a small published smoke-test structure copied from the 0.1.0 release data. It is intended for quick inference checks with the released checkpoint and PCA file.

Example:

```bash
protcross setup-assets
protcross predict examples/6fhu.pdb --output examples/6fhu.pred.pdb
```

With an explicit asset directory:

```bash
protcross predict examples/6fhu.pdb \
  --assets-dir /absolute/path/to/protcross-assets \
  --output examples/6fhu.pred.pdb
```

For source checkout testing without the automatic asset cache, pass explicit files:

```bash
protcross-predict examples/6fhu.pdb \
  --checkpoint checkpoint/best-epoch=59.ckpt \
  --esm-weights /absolute/path/to/esmc_600m_2024_12_v0.pth \
  --pca data/pca_esmc_128.pkl \
  --output examples/6fhu.pred.pdb
```
