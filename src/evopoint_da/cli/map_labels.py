"""CLI for mapping PDB binding labels onto AF2 structures."""

from __future__ import annotations

import argparse
from pathlib import Path

from evopoint_da.data.label_mapping import LabelMappingConfig, map_labels


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Map PDB-derived labels onto processed AF2 samples.")
    parser.add_argument("--processed-pdb-dir", default="data/processed_pdb")
    parser.add_argument("--processed-af2-dir", default="data/processed_af2")
    parser.add_argument("--raw-pdb-dir", default="data/raw_pdb")
    parser.add_argument("--raw-af2-dir", default="data/raw_af2")
    parser.add_argument("--mapping-file", default="pdb_uniprot_mapping.json")
    parser.add_argument("--output-csv", default="mapping_report_final.csv")
    parser.add_argument("--debug-limit", type=int, default=5)
    parser.add_argument("--min-chain-score", type=float, default=0.15)
    parser.add_argument("--max-rmsd", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog=prog).parse_args(argv)
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
    )
    map_labels(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
