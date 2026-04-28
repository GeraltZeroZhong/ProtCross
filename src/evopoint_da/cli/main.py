"""Unified ProtCross command line interface."""

from __future__ import annotations

import argparse
from importlib import import_module

from evopoint_da import __version__


COMMANDS = {
    "predict": "evopoint_da.cli.predict:main",
    "setup-assets": "evopoint_da.cli.setup_assets:main",
    "preprocess": "evopoint_da.cli.preprocess:main",
    "download-af2": "evopoint_da.cli.download_af2:main",
    "map-labels": "evopoint_da.cli.map_labels:main",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="protcross", description="ProtCross command line tools.")
    parser.add_argument("--version", action="version", version=f"ProtCross {__version__}")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command_args = args.args
    if command_args and command_args[0] == "--":
        command_args = command_args[1:]
    return _load_command(COMMANDS[args.command])(command_args, prog=f"protcross {args.command}")


def _load_command(target: str):
    module_name, function_name = target.split(":", 1)
    return getattr(import_module(module_name), function_name)


if __name__ == "__main__":
    raise SystemExit(main())
