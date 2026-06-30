"""CLI for mapping PDB binding labels onto AF2 structures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Map PDB-derived labels onto processed AF2 samples.")
    parser.add_argument(
        "--processed-pdb-dir",
        default="data/processed_pdb",
        help="Legacy compatibility option; current mapping reads labels from --raw-pdb-dir structures.",
    )
    parser.add_argument("--processed-af2-dir", default="data/processed_af2")
    parser.add_argument("--raw-pdb-dir", default="data/raw_pdb")
    parser.add_argument("--raw-af2-dir", default="data/raw_af2")
    parser.add_argument("--mapping-file", default="artifacts/pdb_uniprot_mapping.json")
    parser.add_argument("--output-csv", default="mapping_report_final.csv")
    parser.add_argument("--debug-limit", type=_non_negative_int, default=5)
    parser.add_argument("--min-chain-score", type=_non_negative_float, default=0.15)
    parser.add_argument("--max-rmsd", type=_positive_float, default=30.0)
    parser.add_argument("--dry-run", action="store_true", help="Compute mapping statistics without modifying AF2 .pt files.")
    return parser


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog=prog).parse_args(argv)
    from protcross.data.label_mapping import LabelMappingConfig, map_labels

    try:
        config = LabelMappingConfig(
            processed_pdb_dir=Path(args.processed_pdb_dir),
            processed_af2_dir=Path(args.processed_af2_dir),
            raw_pdb_dir=Path(args.raw_pdb_dir),
            raw_af2_dir=Path(args.raw_af2_dir),
            mapping_file=Path(args.mapping_file),
            output_csv=Path(args.output_csv),
            debug_limit=args.debug_limit,
            min_chain_score=args.min_chain_score,
            max_rmsd=args.max_rmsd,
            in_place=not args.dry_run,
        )
        map_labels(config)
        return 0
    except Exception as exc:
        print(f"ProtCross label mapping failed: {exc}", file=sys.stderr)
        return 1


def _non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return number


def _non_negative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
