"""CLI for downloading ProtCross runtime assets."""

from __future__ import annotations

import argparse
import os
import sys

from protcross.assets import (
    ESM_LICENSE_URL,
    setup_assets,
)


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Download ProtCross checkpoint, PCA reducer, and ESM-C weights.",
        epilog=(
            "ESM-C download: about 2.14 GiB; keep at least 2.4 GiB free on the asset target "
            "(about 5 GiB total if the Python environment shares that disk). Interrupted downloads "
            "retain a .part file and resume on the next identical command.\n"
            f"Review the ESM-C license before accepting: {ESM_LICENSE_URL}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory where assets will be installed. Defaults to PROTCROSS_ASSETS_DIR "
            "or ~/.cache/protcross/assets/v0.1.2."
        ),
    )
    parser.add_argument(
        "--asset-version",
        default=os.environ.get("PROTCROSS_ASSET_VERSION", "default"),
        choices=("default", "latest", "0.1.2", "0.1.1-paper"),
        help=(
            "Asset bundle version. Defaults to PROTCROSS_ASSET_VERSION when set, "
            "otherwise the current 0.1.2 bundle."
        ),
    )
    parser.add_argument("--esm-url")
    parser.add_argument("--checkpoint-url")
    parser.add_argument("--pca-url")
    parser.add_argument(
        "--force",
        "--refresh-assets",
        dest="force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    parser.add_argument("--no-verify", action="store_true", help="Skip SHA256 verification.")
    parser.add_argument("--skip-esm", action="store_true", help="Only download ProtCross checkpoint and PCA reducer.")
    parser.add_argument(
        "--accept-esm-license",
        action="store_true",
        help="Confirm that you reviewed the upstream ESM-C license; acceptance is recorded in the asset manifest.",
    )
    return parser


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog=prog).parse_args(argv)
    try:
        setup_assets(
            args.output_dir,
            asset_version=args.asset_version,
            esm_url=args.esm_url,
            checkpoint_url=args.checkpoint_url,
            pca_url=args.pca_url,
            force=args.force,
            verify=not args.no_verify,
            skip_esm=args.skip_esm,
            accept_esm_license=args.accept_esm_license,
        )
        return 0
    except Exception as exc:
        print(f"ProtCross asset setup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
