"""CLI for ESM-C + PCA preprocessing."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

MAX_ESM_RESIDUES = 1022


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Preprocess PDB/mmCIF structures into ProtCross .pt files.")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument(
        "--esm-weights",
        "--model-name",
        "--model_name",
        dest="model_name",
        required=True,
        help="Local ESM-C weights file, e.g. esmc_600m_2024_12_v0.pth.",
    )
    parser.add_argument("--pca", "--pca-model-path", "--pca_model_path", dest="pca_model_path", default="pca_esmc_128.pkl")
    parser.add_argument("--fit-pca", "--fit_pca", dest="fit_pca", action="store_true")
    parser.add_argument("--pca-dim", "--pca_dim", dest="pca_dim", type=_positive_int, default=128)
    parser.add_argument("--is-af2", "--is_af2", dest="is_af2", action="store_true")
    parser.add_argument("--sample-ratio", "--sample_ratio", dest="sample_ratio", type=_positive_float, default=0.1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-len", type=_max_len, default=MAX_ESM_RESIDUES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-going", action="store_true", help="Do not fail the command when individual files fail.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Keep existing .pt files in the output directory instead of quarantining files not produced by current inputs.",
    )
    parser.add_argument(
        "--accept-esm-license",
        action="store_true",
        help="Confirm that you reviewed and accept the ESM-C model license before using local ESM-C weights.",
    )
    return parser


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog=prog).parse_args(argv)
    from protcross.data.preprocess import PreprocessConfig, preprocess_directory

    try:
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
            seed=args.seed,
            fail_on_error=not args.keep_going,
            accept_esm_license=args.accept_esm_license,
            append=args.append,
        )
        preprocess_directory(config)
        return 0
    except Exception as exc:
        print(f"ProtCross preprocessing failed: {exc}", file=sys.stderr)
        return 1


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return number


def _max_len(value: str) -> int:
    number = _positive_int(value)
    if number > MAX_ESM_RESIDUES:
        raise argparse.ArgumentTypeError(f"must be <= {MAX_ESM_RESIDUES} for ESM-C")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
