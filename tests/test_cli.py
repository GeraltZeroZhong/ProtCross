from pathlib import Path

from evopoint_da.cli import setup_assets
from evopoint_da.cli.main import build_parser as build_main_parser
from evopoint_da.cli.predict import _resolve_asset_directory, build_parser


def test_predict_cli_accepts_new_and_legacy_arguments():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--pdb_file",
            "input.pdb",
            "--ckpt_path",
            "model.ckpt",
            "--esm_weights",
            "esm.pth",
            "--pca_path",
            "pca.pkl",
            "--output_pdb",
            "out.pdb",
        ]
    )

    assert args.pdb_file == "input.pdb"
    assert args.ckpt_path == "model.ckpt"
    assert args.output_pdb == "out.pdb"


def test_predict_cli_accepts_assets_dir():
    parser = build_parser()
    args = parser.parse_args(["input.pdb", "--assets-dir", "assets", "--output", "out.pdb"])

    assert args.input_pdb == "input.pdb"
    assert args.assets_dir == "assets"


def test_unified_cli_exposes_setup_assets():
    parser = build_main_parser()
    args = parser.parse_args(["setup-assets", "--skip-esm"])

    assert args.command == "setup-assets"
    assert args.args == ["--skip-esm"]


def test_setup_assets_parser_allows_env_default():
    parser = setup_assets.build_parser()
    args = parser.parse_args([])

    assert args.output_dir is None


def test_predict_discovers_default_asset_directory(tmp_path, monkeypatch):
    for filename in ("best-epoch=59.ckpt", "esmc_600m_2024_12_v0.pth", "pca_esmc_128.pkl"):
        (tmp_path / filename).write_bytes(b"asset")
    monkeypatch.setenv("PROTCROSS_ASSETS_DIR", str(tmp_path))

    assets = _resolve_asset_directory(None)

    assert assets is not None
    assert assets.checkpoint == tmp_path / "best-epoch=59.ckpt"


def test_pyproject_exposes_setup_assets_entry_point():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'protcross-setup-assets = "evopoint_da.cli.setup_assets:main"' in pyproject
    assert 'predict = ["esm>=3.1.0", "httpx"]' in pyproject
