"""CLI for ESM-C + PCA preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path

from evopoint_da.data.preprocess import PreprocessConfig, preprocess_directory


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Preprocess PDB/mmCIF structures into ProtCross .pt files.")
    parser.add_argument("--data_dir", "--data-dir", dest="data_dir", required=True)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--model_name", "--model-name", dest="model_name", required=True)
    parser.add_argument("--pca_model_path", "--pca-model-path", dest="pca_model_path", default="pca_esmc_128.pkl")
    parser.add_argument("--fit_pca", "--fit-pca", dest="fit_pca", action="store_true")
    parser.add_argument("--pca_dim", "--pca-dim", dest="pca_dim", type=int, default=128)
    parser.add_argument("--is_af2", "--is-af2", dest="is_af2", action="store_true")
    parser.add_argument("--sample_ratio", "--sample-ratio", dest="sample_ratio", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-len", type=int, default=1022)
    return parser


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog=prog).parse_args(argv)
    config = PreprocessConfig(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        model_name=Path(args.model_name),
        pca_model_path=Path(args.pca_model_path),
        fit_pca=args.fit_pca,
        pca_dim=args.pca_dim,
        is_af2=args.is_af2,
        sample_ratio=args.sample_ratio,
        device=args.device,
        max_len=args.max_len,
    )
    preprocess_directory(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
