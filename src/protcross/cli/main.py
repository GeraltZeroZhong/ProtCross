"""Unified ProtCross command line interface."""

from __future__ import annotations

import argparse
from importlib import import_module

from protcross import __version__


COMMANDS = {
    "predict": ("protcross.cli.predict:main", "Predict binding-site probabilities for one structure."),
    "setup-assets": ("protcross.cli.setup_assets:main", "Download checkpoint, PCA, and ESM-C weights."),
    "preprocess": ("protcross.cli.preprocess:main", "Convert PDB/mmCIF structures into training .pt files."),
    "download-af2": ("protcross.cli.download_af2:main", "Download matching AlphaFold structures."),
    "map-labels": ("protcross.cli.map_labels:main", "Map PDB-derived labels onto AF2 samples."),
    "train": ("protcross.cli.train:main", "Run the Hydra training workflow."),
}


def build_parser() -> argparse.ArgumentParser:
    command_help = "\n".join(f"  {name:12s} {description}" for name, (_, description) in sorted(COMMANDS.items()))
    parser = argparse.ArgumentParser(
        prog="protcross",
        description="ProtCross command line tools.",
        epilog=f"Commands:\n{command_help}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"ProtCross {__version__}")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command_args = args.args
    if command_args and command_args[0] == "--":
        command_args = command_args[1:]
    return _load_command(COMMANDS[args.command][0])(command_args, prog=f"protcross {args.command}")


def _load_command(target: str):
    module_name, function_name = target.split(":", 1)
    return getattr(import_module(module_name), function_name)


if __name__ == "__main__":
    raise SystemExit(main())
