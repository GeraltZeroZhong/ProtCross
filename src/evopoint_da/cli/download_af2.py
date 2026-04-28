"""CLI for downloading matching AlphaFold structures."""

from __future__ import annotations

import argparse
from pathlib import Path

from evopoint_da.data.af2 import AF2DownloadConfig, download_af2_structures


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Download AlphaFold structures for PDB IDs or PDB files.")
    parser.add_argument("--raw-pdb-dir", default="data/raw_pdb")
    parser.add_argument("--output-dir", default="data/raw_af2")
    parser.add_argument("--mapping-file", default="pdb_uniprot_mapping.json")
    parser.add_argument("--pdb-id-file", default=None, help="Text file with one 4-character PDB ID per line.")
    parser.add_argument(
        "--initial-mapping-file",
        default=None,
        help="Optional PDB-to-UniProt JSON used before falling back to UniProt search.",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--uniprot-candidates", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    return parser


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog=prog).parse_args(argv)
    config = AF2DownloadConfig(
        raw_pdb_dir=Path(args.raw_pdb_dir),
        output_dir=Path(args.output_dir),
        mapping_file=Path(args.mapping_file),
        pdb_id_file=Path(args.pdb_id_file) if args.pdb_id_file else None,
        initial_mapping_file=Path(args.initial_mapping_file) if args.initial_mapping_file else None,
        max_workers=args.max_workers,
        uniprot_candidates=args.uniprot_candidates,
        timeout_seconds=args.timeout_seconds,
    )
    download_af2_structures(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
