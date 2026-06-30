"""CLI for downloading matching AlphaFold structures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Download AlphaFold structures for PDB IDs or PDB files.")
    parser.add_argument("--raw-pdb-dir", default="data/raw_pdb")
    parser.add_argument("--output-dir", default="data/raw_af2")
    parser.add_argument("--mapping-file", default="artifacts/pdb_uniprot_mapping.json")
    parser.add_argument("--pdb-id-file", default=None, help="Text file with one 4-character PDB ID per line.")
    parser.add_argument(
        "--initial-mapping-file",
        default=None,
        help="Optional PDB-to-UniProt JSON used before falling back to UniProt search.",
    )
    parser.add_argument("--max-workers", type=_positive_int, default=8)
    parser.add_argument("--uniprot-candidates", type=_positive_int, default=3)
    parser.add_argument("--timeout-seconds", type=_positive_int, default=30)
    parser.add_argument("--allow-empty", action="store_true", help="Exit successfully when no PDB IDs are found.")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit successfully when some requested PDB IDs fail but at least one download succeeds.",
    )
    parser.add_argument(
        "--allow-empty-downloads",
        action="store_true",
        help="Exit successfully when requested IDs produce no AlphaFold downloads.",
    )
    parser.add_argument("--refresh", action="store_true", help="Re-download AlphaFold files even when a verified cache exists.")
    return parser


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog=prog).parse_args(argv)
    from protcross.data.af2 import AF2DownloadConfig, download_af2_structures

    config = AF2DownloadConfig(
        raw_pdb_dir=Path(args.raw_pdb_dir),
        output_dir=Path(args.output_dir),
        mapping_file=Path(args.mapping_file),
        pdb_id_file=Path(args.pdb_id_file) if args.pdb_id_file else None,
        initial_mapping_file=Path(args.initial_mapping_file) if args.initial_mapping_file else None,
        max_workers=args.max_workers,
        uniprot_candidates=args.uniprot_candidates,
        timeout_seconds=args.timeout_seconds,
        allow_empty=args.allow_empty or args.allow_empty_downloads,
        allow_partial=args.allow_partial,
        refresh=args.refresh,
    )
    try:
        download_af2_structures(config)
        return 0
    except Exception as exc:
        print(f"ProtCross AF2 download failed: {exc}", file=sys.stderr)
        return 1


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
