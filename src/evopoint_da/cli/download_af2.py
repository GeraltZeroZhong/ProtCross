"""CLI for downloading matching AlphaFold structures."""

from __future__ import annotations

import argparse
from pathlib import Path

from evopoint_da.data.af2 import AF2DownloadConfig, download_af2_structures


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Download AlphaFold structures for PDB files.")
    parser.add_argument("--raw-pdb-dir", default="data/raw_pdb")
    parser.add_argument("--output-dir", default="data/raw_af2")
    parser.add_argument("--mapping-file", default="pdb_uniprot_mapping.json")
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
        max_workers=args.max_workers,
        uniprot_candidates=args.uniprot_candidates,
        timeout_seconds=args.timeout_seconds,
    )
    download_af2_structures(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
