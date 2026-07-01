"""Command line interface for lightweight ProtCross prediction."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path


MAX_ESM_RESIDUES = 1022


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
        "--checkpoint",
        "--ckpt-path",
        "--ckpt_path",
        dest="ckpt_path",
        default=os.environ.get("PROTCROSS_CHECKPOINT"),
        help=(
            "ProtCross Lightning checkpoint. Defaults to PROTCROSS_CHECKPOINT, "
            "installed assets, or the selected asset bundle cache."
        ),
    )
    parser.add_argument(
        "--esm-weights",
        "--esm_weights",
        dest="esm_weights",
        default=os.environ.get("PROTCROSS_ESM_WEIGHTS"),
        help="Local ESM-C 600M weights. Can also be set with PROTCROSS_ESM_WEIGHTS.",
    )
    parser.add_argument(
        "--pca",
        "--pca-path",
        "--pca_path",
        dest="pca_path",
        default=os.environ.get("PROTCROSS_PCA"),
        help=(
            "Fitted PCA pickle for ESM-C embeddings. Defaults to PROTCROSS_PCA, "
            "installed assets, or the selected asset bundle cache."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        "--output_pdb",
        "--output-pdb",
        dest="output_pdb",
        help="Output annotated structure path. Predicted probabilities overwrite B-factors for scored residues.",
    )
    parser.add_argument("--out-dir", "--output-dir", dest="out_dir", help="Directory for default ProtCross outputs.")
    parser.add_argument("--scores-tsv", help="Optional residue-level score table.")
    parser.add_argument("--pocket-json", help="Optional pocket-level JSON summary for downstream workflows.")
    parser.add_argument("--summary-json", help="Optional machine-readable run summary JSON.")
    parser.add_argument(
        "--pocket-cluster-cutoff",
        type=_positive_float,
        default=8.0,
        help="CA-CA distance cutoff in Angstroms for clustered pocket components.",
    )
    parser.add_argument(
        "--threshold",
        type=_probability_threshold,
        default=0.5,
        help=(
            "Threshold for binary calls, TSV is_binding, pocket selection, clustering, "
            "and terminal summary. B-factors remain continuous probabilities."
        ),
    )
    parser.add_argument("--chain", "--chain-id", dest="chain_id", help="Restrict prediction to one chain.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N.")
    parser.add_argument("--pca-dim", "--pca_dim", dest="pca_dim", type=_positive_int, default=128)
    parser.add_argument("--max-len", type=_max_len, default=MAX_ESM_RESIDUES, help="Maximum residues passed to ESM-C.")
    parser.add_argument(
        "--embedding-cache-dir",
        help="Optional directory for cached reduced ESM/PCA residue features keyed by sequence and asset metadata.",
    )
    parser.add_argument(
        "--asset-version",
        default="default",
        help="default/latest (current packaged stable bundle, 0.1.2), 0.1.2, or 0.1.1-paper.",
    )
    parser.add_argument("--refresh-assets", action="store_true", help="Re-download and verify the selected asset bundle.")
    parser.add_argument(
        "--accept-esm-license",
        action="store_true",
        help="Confirm that you reviewed the upstream ESM-C license before downloading or using ESM-C weights.",
    )
    parser.add_argument(
        "--trust-unverified-assets",
        action="store_true",
        help="Allow explicit local checkpoint/PCA/ESM files whose SHA256 does not match the selected release bundle.",
    )
    parser.add_argument(
        "--no-auto-assets",
        dest="auto_assets",
        action="store_false",
        help="Do not automatically download missing default prediction assets.",
    )
    parser.add_argument("--offline", action="store_true", help="Disable automatic asset downloads.")
    parser.set_defaults(auto_assets=True)
    parser.add_argument(
        "--allow-truncation",
        action="store_true",
        help="Allow structures longer than --max-len to be truncated before ESM-C inference.",
    )
    parser.add_argument("--fail-on-truncation", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--unscored-bfactor-policy",
        choices=("keep", "zero"),
        default="zero",
        help=(
            "How to write B-factors for unscored atoms/residues in the annotated structure. "
            "Default zero keeps the B-factor column in ProtCross probability units; use keep "
            "to preserve original B-factors/pLDDT for unscored atoms."
        ),
    )
    parser.add_argument("--summary-only", action="store_true", help="Print the summary without creating default output files.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    input_pdb = args.input_pdb or args.pdb_file
    if not input_pdb:
        parser.error("an input PDB/mmCIF path is required")
    if args.offline and args.refresh_assets:
        parser.error("--offline cannot be combined with --refresh-assets")
    input_path = Path(input_pdb)
    if not input_path.exists():
        print(f"ProtCross prediction failed: Input structure not found: {input_path}", file=sys.stderr)
        return 1
    if input_path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
        print(
            f"ProtCross prediction failed: Unsupported input structure extension: {input_path.suffix or '<none>'}",
            file=sys.stderr,
        )
        return 1
    try:
        _preflight_structure(input_path, chain_id=args.chain_id, max_len=args.max_len, allow_truncation=args.allow_truncation)
    except Exception as exc:
        print(f"ProtCross prediction failed: {exc}", file=sys.stderr)
        return 1

    if not args.summary_only:
        output_paths = _default_output_paths(input_pdb, args.out_dir)
        args.output_pdb = args.output_pdb or str(output_paths["structure"])
        args.scores_tsv = args.scores_tsv or str(output_paths["scores_tsv"])
        args.pocket_json = args.pocket_json or str(output_paths["pockets_json"])
        args.summary_json = args.summary_json or str(output_paths["summary_json"])

    try:
        auto_setup = args.auto_assets and not args.offline
        assets = _resolve_prediction_asset_paths(
            args,
            auto_setup=auto_setup,
        )

        from protcross.inference import ProtCrossPredictor

        predictor = ProtCrossPredictor.from_files(
            ckpt_path=args.ckpt_path,
            esm_weights=args.esm_weights,
            pca_path=args.pca_path,
            device=args.device,
            pca_dim=args.pca_dim,
            max_len=args.max_len,
            asset_version=assets.asset_version,
            embedding_cache_dir=args.embedding_cache_dir,
            accept_esm_license=True,
        )
        result = predictor.predict(
            input_pdb,
            chain_id=args.chain_id,
            threshold=args.threshold,
            pocket_cluster_cutoff=args.pocket_cluster_cutoff,
            output_pdb=args.output_pdb,
            scores_tsv=args.scores_tsv,
            pocket_json=args.pocket_json,
            summary_json=args.summary_json,
            allow_truncation=args.allow_truncation,
            unscored_bfactor_policy=args.unscored_bfactor_policy,
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

    return 0


def _resolve_asset_directory(
    assets_dir: str | None,
    *,
    auto_setup: bool = False,
    asset_version: str = "default",
    refresh: bool = False,
    accept_esm_license: bool = False,
):
    from protcross.assets import resolve_prediction_assets

    resolved = resolve_prediction_assets(
        assets_dir=assets_dir,
        auto_setup_assets=auto_setup,
        asset_version=asset_version,
        refresh_assets=refresh,
        accept_esm_license=accept_esm_license,
    )
    return resolved.assets


def _resolve_prediction_asset_paths(args: argparse.Namespace, *, auto_setup: bool):
    from protcross.assets import resolve_prediction_assets

    resolved = resolve_prediction_assets(
        ckpt_path=args.ckpt_path,
        esm_weights=args.esm_weights,
        pca_path=args.pca_path,
        assets_dir=args.assets_dir,
        auto_setup_assets=auto_setup,
        asset_version=args.asset_version,
        refresh_assets=args.refresh_assets,
        offline=args.offline,
        accept_esm_license=args.accept_esm_license,
        require_esm_license_for_use=True,
        trust_unverified_assets=args.trust_unverified_assets,
    )
    args.ckpt_path = str(resolved.checkpoint)
    args.esm_weights = str(resolved.esm_weights)
    args.pca_path = str(resolved.pca)
    return resolved


def _default_output_paths(input_pdb: str | Path, out_dir: str | Path | None = None) -> dict[str, Path]:
    input_path = Path(input_pdb)
    output_dir = Path(out_dir) if out_dir else input_path.parent
    structure_suffix = ".cif" if input_path.suffix.lower() in {".cif", ".mmcif"} else ".pdb"
    stem = input_path.stem
    return {
        "structure": output_dir / f"{stem}.protcross{structure_suffix}",
        "scores_tsv": output_dir / f"{stem}.protcross.scores.tsv",
        "pockets_json": output_dir / f"{stem}.protcross.pockets.json",
        "summary_json": output_dir / f"{stem}.protcross.summary.json",
    }


def _probability_threshold(value: str) -> float:
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return threshold


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return number


def _max_len(value: str) -> int:
    number = _positive_int(value)
    if number > MAX_ESM_RESIDUES:
        raise argparse.ArgumentTypeError(f"must be <= {MAX_ESM_RESIDUES} for ESM-C")
    return number


def _preflight_structure(
    input_path: Path,
    *,
    chain_id: str | None,
    max_len: int,
    allow_truncation: bool,
) -> None:
    from protcross.data import StructureParser, parsed_structure_long_chunks

    parser = StructureParser()
    parsed = parser.parse_file_with_labels(input_path, chain_id=chain_id)
    if not parsed:
        if chain_id:
            any_chain = parser.parse_file_with_labels(input_path, chain_id=None)
            if any_chain:
                raise ValueError(f"No standard amino-acid residues with CA atoms found for chain {chain_id!r}.")
        raise ValueError(f"No standard amino-acid residues with CA atoms found in {input_path}.")
    long_chunks = parsed_structure_long_chunks(parsed, max_len)
    if long_chunks and not allow_truncation:
        longest = max(end - start for start, end in long_chunks)
        raise ValueError(
            f"Input has an ESM chain context of {longest} scored residues, which exceeds --max-len={max_len}. "
            "Pass --allow-truncation to score only the leading residues of each long chain."
        )


if __name__ == "__main__":
    raise SystemExit(main())
