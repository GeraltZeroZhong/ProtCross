from pathlib import Path

from evopoint_da.cli import setup_assets
from evopoint_da.cli.download_af2 import build_parser as build_download_af2_parser
from evopoint_da.cli.main import build_parser as build_main_parser
from evopoint_da.cli.predict import _resolve_asset_directory, build_parser
from evopoint_da.assets import DEFAULT_CHECKPOINT_FILENAME, DEFAULT_PCA_FILENAME
from evopoint_da.data.af2 import AF2DownloadConfig, AF2Downloader
from evopoint_da.data.dataset import EvoPointDataset


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


def test_download_af2_cli_accepts_pdb_id_file():
    parser = build_download_af2_parser()
    args = parser.parse_args(["--pdb-id-file", "pdb_ids.txt", "--initial-mapping-file", "mapping.json"])

    assert args.pdb_id_file == "pdb_ids.txt"
    assert args.initial_mapping_file == "mapping.json"


def test_af2_downloader_collects_text_pdb_ids_without_raw_pdb_dir(tmp_path):
    pdb_id_file = tmp_path / "pdb_ids.txt"
    pdb_id_file.write_text("1e12\n4APR\n# duplicate below\n1E12\n", encoding="utf-8")
    config = AF2DownloadConfig(raw_pdb_dir=tmp_path / "missing", pdb_id_file=pdb_id_file)

    pdb_ids = AF2Downloader(config).collect_pdb_ids()

    assert pdb_ids == ["1E12", "4APR"]


def test_af2_downloader_preloads_initial_mapping_file(tmp_path):
    mapping_file = tmp_path / "initial_mapping.json"
    mapping_file.write_text('{"1e12": "P12345", "not_a_pdb": "ignored"}', encoding="utf-8")
    config = AF2DownloadConfig(initial_mapping_file=mapping_file, mapping_file=tmp_path / "output_mapping.json")

    downloader = AF2Downloader(config)

    assert downloader.preloaded_mapping == {"1E12": "P12345"}
    assert downloader.mapping == {}


def test_dataset_all_split_uses_every_processed_file(tmp_path):
    for name in ("a.pt", "b.pt", "c.pt"):
        (tmp_path / name).write_bytes(b"")
    dataset = object.__new__(EvoPointDataset)
    dataset.root = str(tmp_path)
    dataset.split = "all"
    dataset.split_seed = 42

    files = dataset._split_files()

    assert [Path(file).name for file in files] == ["a.pt", "b.pt", "c.pt"]


def test_setup_assets_parser_allows_env_default():
    parser = setup_assets.build_parser()
    args = parser.parse_args([])

    assert args.output_dir is None


def test_predict_discovers_default_asset_directory(tmp_path, monkeypatch):
    for filename in (DEFAULT_CHECKPOINT_FILENAME, "esmc_600m_2024_12_v0.pth", DEFAULT_PCA_FILENAME):
        (tmp_path / filename).write_bytes(b"asset")
    monkeypatch.setenv("PROTCROSS_ASSETS_DIR", str(tmp_path))

    assets = _resolve_asset_directory(None)

    assert assets is not None
    assert assets.checkpoint == tmp_path / DEFAULT_CHECKPOINT_FILENAME


def test_pyproject_exposes_setup_assets_entry_point():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'protcross-setup-assets = "evopoint_da.cli.setup_assets:main"' in pyproject
    assert 'predict = ["esm>=3.1.0", "httpx"]' in pyproject
