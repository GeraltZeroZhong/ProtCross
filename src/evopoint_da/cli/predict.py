"""Command line interface for lightweight ProtCross prediction."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


LOCAL_CHECKPOINTS = (
    Path("checkpoints/protcross-0.1.2-binding-moad-final.ckpt"),
    Path("checkpoint/best-epoch=59.ckpt"),
)
LOCAL_PCAS = (
    Path("data/pca_esmc_128_binding_moad_0.1.2.pkl"),
    Path("data/pca_esmc_128.pkl"),
)


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Predict binding-site probabilities for one PDB/mmCIF structure and write them to B-factors.",
    )
    parser.add_argument("input_pdb", nargs="?", help="Input PDB/mmCIF structure.")
    parser.add_argument("--pdb_file", "--pdb-file", dest="pdb_file", help="Legacy input structure argument.")
    parser.add_argument(
        "--assets-dir",
        help=(
            "Directory containing a ProtCross checkpoint, esmc_600m_2024_12_v0.pth, "
            "and a matching PCA reducer. Explicit file arguments override this."
        ),
    )
    parser.add_argument(
        "--ckpt_path",
        "--ckpt-path",
        "--checkpoint",
        dest="ckpt_path",
        default=os.environ.get("PROTCROSS_CHECKPOINT"),
        help=(
            "ProtCross Lightning checkpoint. Defaults to PROTCROSS_CHECKPOINT, "
            "installed assets, or a known local checkpoint when present."
        ),
    )
    parser.add_argument(
        "--esm_weights",
        "--esm-weights",
        dest="esm_weights",
        default=os.environ.get("PROTCROSS_ESM_WEIGHTS"),
        help="Local ESM-C 600M weights. Can also be set with PROTCROSS_ESM_WEIGHTS.",
    )
    parser.add_argument(
        "--pca_path",
        "--pca-path",
        "--pca",
        dest="pca_path",
        default=os.environ.get("PROTCROSS_PCA"),
        help=(
            "Fitted PCA pickle for ESM-C embeddings. Defaults to PROTCROSS_PCA, "
            "installed assets, or a known local PCA reducer when present."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        "--output_pdb",
        "--output-pdb",
        dest="output_pdb",
        help="Output PDB path. Predicted probabilities overwrite B-factors.",
    )
    parser.add_argument("--scores-tsv", help="Optional residue-level score table.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold used in the text summary.")
    parser.add_argument("--chain", "--chain-id", dest="chain_id", help="Restrict prediction to one chain.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N.")
    parser.add_argument("--pca_dim", "--pca-dim", dest="pca_dim", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=1022, help="Maximum residues passed to ESM-C.")
    parser.add_argument("--fail-on-truncation", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    input_pdb = args.input_pdb or args.pdb_file
    if not input_pdb:
        parser.error("an input PDB/mmCIF path is required")

    assets = _resolve_asset_directory(args.assets_dir)
    if assets:
        args.ckpt_path = args.ckpt_path or str(assets.checkpoint)
        args.esm_weights = args.esm_weights or str(assets.esm_weights)
        args.pca_path = args.pca_path or str(assets.pca)

    if not args.ckpt_path:
        args.ckpt_path = next((str(path) for path in LOCAL_CHECKPOINTS if path.exists()), None)
    if not args.pca_path:
        args.pca_path = next((str(path) for path in LOCAL_PCAS if path.exists()), None)

    if not args.esm_weights:
        parser.error(
            "--esm-weights is required unless --assets-dir, PROTCROSS_ESM_WEIGHTS, "
            "or installed assets from `protcross setup-assets` are available"
        )
    if not args.ckpt_path:
        parser.error("--checkpoint is required unless PROTCROSS_CHECKPOINT or installed assets are available")
    if not args.pca_path:
        parser.error("--pca is required unless PROTCROSS_PCA or installed assets are available")

    try:
        from evopoint_da.inference import ProtCrossPredictor

        predictor = ProtCrossPredictor.from_files(
            ckpt_path=args.ckpt_path,
            esm_weights=args.esm_weights,
            pca_path=args.pca_path,
            device=args.device,
            pca_dim=args.pca_dim,
            max_len=args.max_len,
        )
        result = predictor.predict(
            input_pdb,
            chain_id=args.chain_id,
            threshold=args.threshold,
            output_pdb=args.output_pdb,
            scores_tsv=args.scores_tsv,
        )
        if args.fail_on_truncation and result.truncated:
            raise RuntimeError(
                f"Input was truncated from {result.original_length} to {len(result.residue_ids)} residues."
            )
    except Exception as exc:
        print(f"ProtCross prediction failed: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(result.format_summary())
        if args.output_pdb:
            print(f"Wrote B-factor prediction PDB: {Path(args.output_pdb)}")
        if args.scores_tsv:
            print(f"Wrote score table: {Path(args.scores_tsv)}")

    return 0


def _resolve_asset_directory(assets_dir: str | None) -> PredictorAssets | None:
    from evopoint_da.inference import PredictorAssets

    if assets_dir:
        return PredictorAssets.from_dir(assets_dir)

    default_assets = PredictorAssets.from_default_dir()
    if default_assets.is_complete():
        return default_assets

    return None


if __name__ == "__main__":
    raise SystemExit(main())
