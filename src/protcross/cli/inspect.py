"""CLI for checking a structure before loading prediction assets."""

from __future__ import annotations

import argparse
import json
import sys

from protcross.data.inspection import format_structure_inspection, inspect_structure
from protcross.data.structure import MAX_ESM_RESIDUES


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Inspect structure compatibility and scientific input assumptions without loading model assets.",
        epilog="Example: protcross inspect protein.cif --chain A",
    )
    parser.add_argument("input_structure", help="Input .pdb, .cif, or .mmcif structure.")
    parser.add_argument("--chain", "--chain-id", dest="chain_id", help="Inspect one author chain ID.")
    parser.add_argument(
        "--max-len",
        type=int,
        default=MAX_ESM_RESIDUES,
        help=f"ESM-C per-chain context limit (default: {MAX_ESM_RESIDUES}).",
    )
    parser.add_argument("--json", action="store_true", help="Print the complete machine-readable inspection JSON.")
    return parser


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog=prog).parse_args(argv)
    try:
        inspection = inspect_structure(args.input_structure, chain_id=args.chain_id, max_len=args.max_len)
    except Exception as exc:
        print(f"ProtCross inspection failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(inspection, indent=2))
    else:
        print(format_structure_inspection(inspection))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
