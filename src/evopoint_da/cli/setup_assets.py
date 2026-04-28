"""CLI for downloading ProtCross runtime assets."""

from __future__ import annotations

import argparse

from evopoint_da.assets import (
    DEFAULT_CHECKPOINT_URL,
    DEFAULT_ESM_URL,
    DEFAULT_PCA_URL,
    setup_assets,
)


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Download ProtCross checkpoint, PCA reducer, and ESM-C weights.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory where assets will be installed. Defaults to PROTCROSS_ASSETS_DIR "
            "or ~/.cache/protcross/assets/v0.1.2."
        ),
    )
    parser.add_argument("--esm-url", default=DEFAULT_ESM_URL)
    parser.add_argument("--checkpoint-url", default=DEFAULT_CHECKPOINT_URL)
    parser.add_argument("--pca-url", default=DEFAULT_PCA_URL)
    parser.add_argument("--force", action="store_true", help="Re-download files even if they already exist.")
    parser.add_argument("--no-verify", action="store_true", help="Skip SHA256 verification.")
    parser.add_argument("--skip-esm", action="store_true", help="Only download ProtCross checkpoint and PCA reducer.")
    return parser


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog=prog).parse_args(argv)
    setup_assets(
        args.output_dir,
        esm_url=args.esm_url,
        checkpoint_url=args.checkpoint_url,
        pca_url=args.pca_url,
        force=args.force,
        verify=not args.no_verify,
        skip_esm=args.skip_esm,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
