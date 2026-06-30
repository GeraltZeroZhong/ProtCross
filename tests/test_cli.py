from pathlib import Path
import json
import subprocess
import sys

import numpy as np

from protcross.cli import setup_assets
from protcross.cli.download_af2 import build_parser as build_download_af2_parser
from protcross.cli.main import build_parser as build_main_parser
from protcross.cli.predict import (
    _default_output_paths,
    _resolve_asset_directory,
    _resolve_prediction_asset_paths,
    build_parser,
    main as predict_main,
)
from protcross.assets import (
    ASSET_MANIFEST_FILENAME,
    DEFAULT_ASSET_BUNDLE,
    DEFAULT_CHECKPOINT_FILENAME,
    DEFAULT_PCA_FILENAME,
)
from protcross.data.af2 import AF2DownloadConfig, AF2Downloader
from protcross.data.dataset import EvoPointDataset
from protcross.inference import PredictionResult


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
            "--pocket-json",
            "pocket.json",
            "--pocket-cluster-cutoff",
            "9.5",
            "--summary-json",
            "summary.json",
            "--out-dir",
            "outdir",
            "--embedding-cache-dir",
            "cache",
            "--asset-version",
            "0.1.2",
            "--refresh-assets",
        ]
    )

    assert args.pdb_file == "input.pdb"
    assert args.ckpt_path == "model.ckpt"
    assert args.output_pdb == "out.pdb"
    assert args.pocket_json == "pocket.json"
    assert args.summary_json == "summary.json"
    assert args.out_dir == "outdir"
    assert args.embedding_cache_dir == "cache"
    assert args.pocket_cluster_cutoff == 9.5
    assert args.asset_version == "0.1.2"
    assert args.refresh_assets is True
    assert args.auto_assets is True


def test_predict_cli_accepts_assets_dir():
    parser = build_parser()
    args = parser.parse_args(["input.pdb", "--assets-dir", "assets", "--output", "out.pdb"])

    assert args.input_pdb == "input.pdb"
    assert args.assets_dir == "assets"


def test_predict_cli_can_disable_auto_assets():
    parser = build_parser()
    args = parser.parse_args(["input.pdb", "--no-auto-assets"])

    assert args.auto_assets is False


def test_predict_cli_accepts_offline_and_summary_only():
    parser = build_parser()
    args = parser.parse_args(["input.pdb", "--offline", "--summary-only"])

    assert args.offline is True
    assert args.summary_only is True


def test_default_output_paths_follow_input_format(tmp_path):
    paths = _default_output_paths(tmp_path / "input.mmcif", tmp_path / "out")

    assert paths["structure"] == tmp_path / "out/input.protcross.cif"
    assert paths["scores_tsv"] == tmp_path / "out/input.protcross.scores.tsv"
    assert paths["pockets_json"] == tmp_path / "out/input.protcross.pockets.json"
    assert paths["summary_json"] == tmp_path / "out/input.protcross.summary.json"


def test_unified_cli_exposes_setup_assets():
    parser = build_main_parser()
    args = parser.parse_args(["setup-assets", "--skip-esm"])

    assert args.command == "setup-assets"
    assert args.args == ["--skip-esm"]


def test_unified_cli_exposes_train():
    parser = build_main_parser()
    args = parser.parse_args(["train", "trainer.max_epochs=0"])

    assert args.command == "train"
    assert args.args == ["trainer.max_epochs=0"]


def test_download_af2_cli_accepts_pdb_id_file():
    parser = build_download_af2_parser()
    args = parser.parse_args(["--pdb-id-file", "pdb_ids.txt", "--initial-mapping-file", "mapping.json"])

    assert args.pdb_id_file == "pdb_ids.txt"
    assert args.initial_mapping_file == "mapping.json"
    assert args.mapping_file == "artifacts/pdb_uniprot_mapping.json"


def test_map_labels_cli_default_mapping_file_is_under_artifacts():
    from protcross.cli.map_labels import build_parser as build_map_labels_parser

    args = build_map_labels_parser().parse_args([])

    assert args.mapping_file == "artifacts/pdb_uniprot_mapping.json"


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


def test_af2_downloader_creates_mapping_parent(tmp_path):
    mapping_file = tmp_path / "nested" / "pdb_uniprot_mapping.json"
    config = AF2DownloadConfig(mapping_file=mapping_file)
    downloader = AF2Downloader(config)
    downloader.mapping = {"1ABC": "P12345"}

    downloader.save_mapping()

    assert json.loads(mapping_file.read_text(encoding="utf-8")) == {"1ABC": "P12345"}


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
    args = parser.parse_args(["--asset-version", "0.1.1-paper"])

    assert args.output_dir is None
    assert args.asset_version == "0.1.1-paper"


def test_predict_discovers_default_asset_directory(tmp_path, monkeypatch):
    _trust_managed_asset_hashes(monkeypatch)
    for filename in (DEFAULT_CHECKPOINT_FILENAME, "esmc_600m_2024_12_v0.pth", DEFAULT_PCA_FILENAME):
        (tmp_path / filename).write_bytes(b"asset")
    monkeypatch.setenv("PROTCROSS_ASSETS_DIR", str(tmp_path))

    assets = _resolve_asset_directory(None)

    assert assets is not None
    assert assets.checkpoint == tmp_path / DEFAULT_CHECKPOINT_FILENAME


def test_predict_auto_installs_missing_default_assets(tmp_path, monkeypatch):
    _trust_managed_asset_hashes(monkeypatch)
    monkeypatch.setenv("PROTCROSS_ASSETS_DIR", str(tmp_path))

    def fake_setup_assets(output_dir=None, **kwargs):
        output_dir = Path(output_dir) if output_dir else tmp_path
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename in (DEFAULT_CHECKPOINT_FILENAME, "esmc_600m_2024_12_v0.pth", DEFAULT_PCA_FILENAME):
            (output_dir / filename).write_bytes(b"asset")
        (output_dir / ASSET_MANIFEST_FILENAME).write_text('{"asset_version": "0.1.2"}', encoding="utf-8")
        return output_dir

    monkeypatch.setattr("protcross.assets.setup_assets", fake_setup_assets)

    assets = _resolve_asset_directory(None, auto_setup=True)

    assert assets is not None
    assert assets.is_complete()
    assert assets.checkpoint == tmp_path / DEFAULT_CHECKPOINT_FILENAME


def test_predict_partial_assets_do_not_download_explicit_esm(tmp_path, monkeypatch):
    _trust_managed_asset_hashes(monkeypatch)
    (tmp_path / DEFAULT_CHECKPOINT_FILENAME).write_bytes(b"checkpoint")
    (tmp_path / DEFAULT_PCA_FILENAME).write_bytes(b"pca")
    explicit_esm = tmp_path / "external-esm.pth"
    explicit_esm.write_bytes(b"esm")
    calls = []

    def fake_setup_assets(*args, **kwargs):
        calls.append(kwargs)
        raise AssertionError("setup_assets should not be called when all paths can be resolved")

    monkeypatch.setattr("protcross.assets.setup_assets", fake_setup_assets)
    args = build_parser().parse_args(
        [
            "input.pdb",
            "--assets-dir",
            str(tmp_path),
            "--esm-weights",
            str(explicit_esm),
        ]
    )

    assets = _resolve_prediction_asset_paths(args, auto_setup=True)

    assert calls == []
    assert assets is not None
    assert args.ckpt_path == str(tmp_path / DEFAULT_CHECKPOINT_FILENAME)
    assert args.esm_weights == str(explicit_esm)
    assert args.pca_path == str(tmp_path / DEFAULT_PCA_FILENAME)


def test_predict_main_missing_input_does_not_setup_assets(monkeypatch, tmp_path, capsys):
    def fail_setup_assets(*args, **kwargs):
        raise AssertionError("asset setup must not run before input validation")

    monkeypatch.setattr("protcross.assets.setup_assets", fail_setup_assets)

    assert predict_main([str(tmp_path / "missing.pdb")]) == 1
    assert "Input structure not found" in capsys.readouterr().err


def test_predict_cli_rejects_invalid_numeric_values():
    parser = build_parser()

    for args in (
        ["input.pdb", "--threshold", "1.5"],
        ["input.pdb", "--pocket-cluster-cutoff", "0"],
        ["input.pdb", "--max-len", "0"],
        ["input.pdb", "--max-len", "1023"],
    ):
        try:
            parser.parse_args(args)
        except SystemExit as exc:
            assert exc.code != 0
        else:
            raise AssertionError(f"expected parse failure for {args}")


def test_predict_main_writes_default_result_package(tmp_path, monkeypatch, capsys):
    _trust_managed_asset_hashes(monkeypatch)
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text("MODEL\nEND\n", encoding="utf-8")
    for filename in (DEFAULT_CHECKPOINT_FILENAME, "esmc_600m_2024_12_v0.pth", DEFAULT_PCA_FILENAME):
        (tmp_path / filename).write_bytes(b"asset")
    monkeypatch.setenv("PROTCROSS_ASSETS_DIR", str(tmp_path))

    _patch_fake_predictor(monkeypatch)

    assert predict_main([str(input_pdb), "--out-dir", str(tmp_path / "results"), "--device", "cpu"]) == 0

    output_dir = tmp_path / "results"
    assert (output_dir / "input.protcross.pdb").exists()
    assert (output_dir / "input.protcross.scores.tsv").exists()
    assert (output_dir / "input.protcross.pockets.json").exists()
    summary = json.loads((output_dir / "input.protcross.summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "protcross-summary-v1"
    assert summary["top_pocket"]["residue_count"] == 1
    assert summary["unscored_bfactor_policy"] == "keep"
    stdout = capsys.readouterr().out
    assert "Wrote structure:" in stdout
    assert "Pocket center:" in stdout


def test_predict_main_summary_only_does_not_create_defaults(tmp_path, monkeypatch, capsys):
    _trust_managed_asset_hashes(monkeypatch)
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text("MODEL\nEND\n", encoding="utf-8")
    for filename in (DEFAULT_CHECKPOINT_FILENAME, "esmc_600m_2024_12_v0.pth", DEFAULT_PCA_FILENAME):
        (tmp_path / filename).write_bytes(b"asset")
    monkeypatch.setenv("PROTCROSS_ASSETS_DIR", str(tmp_path))

    _patch_fake_predictor(monkeypatch)

    assert predict_main([str(input_pdb), "--summary-only", "--device", "cpu"]) == 0

    assert not (tmp_path / "input.protcross.pdb").exists()
    assert "Predicted binding residues: 1" in capsys.readouterr().out


def test_pyproject_exposes_setup_assets_entry_point():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'protcross-setup-assets = "protcross.cli.setup_assets:main"' in pyproject
    assert 'predict = ["esm>=3.1.0", "httpx"]' in pyproject


def test_unified_cli_help_does_not_import_torch():
    script = r"""
import importlib.abc
import sys

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("blocked torch import during help")
        if fullname == "pytorch_lightning" or fullname.startswith("pytorch_lightning."):
            raise ImportError("blocked lightning import during help")
        return None

sys.meta_path.insert(0, BlockTorch())
from protcross.cli.main import main

for command in ("predict", "setup-assets", "preprocess", "download-af2", "map-labels", "train"):
    try:
        code = main([command, "--help"])
    except SystemExit as exc:
        code = exc.code
    if code not in (0, None):
        raise SystemExit(f"{command} --help returned {code}")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _patch_fake_predictor(monkeypatch):
    class FakePredictor:
        def __init__(self, *, asset_version):
            self.asset_version = asset_version

        def predict(
            self,
            input_pdb,
            *,
            threshold,
            pocket_cluster_cutoff,
            output_pdb=None,
            scores_tsv=None,
            pocket_json=None,
            summary_json=None,
            **kwargs,
        ):
            result = PredictionResult(
                input_pdb=Path(input_pdb),
                residue_ids=["A_1", "A_2"],
                probabilities=np.array([0.8, 0.2]),
                threshold=threshold,
                ca_coords=np.array([[1.0, 2.0, 3.0], [5.0, 2.0, 3.0]]),
                cluster_cutoff=pocket_cluster_cutoff,
                residue_metadata=[
                    {
                        "residue_id": "A_1",
                        "residue_key": "model:0|chain:A|het:ATOM|resseq:1|icode:|resname:ALA",
                        "model_id": "0",
                        "chain_id": "A",
                        "auth_asym_id": "A",
                        "label_asym_id": "A",
                        "residue_number": 1,
                        "auth_seq_id": 1,
                        "label_seq_id": 1,
                        "insertion_code": "",
                        "resname": "ALA",
                        "one_letter_code": "A",
                        "input_bfactor": 20.0,
                    },
                    {
                        "residue_id": "A_2",
                        "residue_key": "model:0|chain:A|het:ATOM|resseq:2|icode:|resname:GLY",
                        "model_id": "0",
                        "chain_id": "A",
                        "auth_asym_id": "A",
                        "label_asym_id": "A",
                        "residue_number": 2,
                        "auth_seq_id": 2,
                        "label_seq_id": 2,
                        "insertion_code": "",
                        "resname": "GLY",
                        "one_letter_code": "G",
                        "input_bfactor": 21.0,
                    },
                ],
                asset_version=self.asset_version,
                device="cpu",
                max_len=1022,
                unscored_bfactor_policy=kwargs.get("unscored_bfactor_policy", "keep"),
                output_files={
                    key: str(value)
                    for key, value in (
                        ("structure", output_pdb),
                        ("scores_tsv", scores_tsv),
                        ("pockets_json", pocket_json),
                        ("summary_json", summary_json),
                    )
                    if value
                },
            )
            if output_pdb:
                Path(output_pdb).parent.mkdir(parents=True, exist_ok=True)
                Path(output_pdb).write_text("PDB\n", encoding="utf-8")
            if scores_tsv:
                result.write_scores_tsv(scores_tsv)
            if pocket_json:
                result.write_pocket_json(pocket_json)
            if summary_json:
                result.write_summary_json(summary_json)
            return result

    def fake_from_files(cls, **kwargs):
        return FakePredictor(asset_version=kwargs.get("asset_version"))

    from protcross.inference import ProtCrossPredictor

    monkeypatch.setattr(ProtCrossPredictor, "from_files", classmethod(fake_from_files))


def _trust_managed_asset_hashes(monkeypatch):
    expected_by_name = {spec.filename: spec.sha256 for spec in DEFAULT_ASSET_BUNDLE.assets}
    monkeypatch.setattr("protcross.assets.sha256_file", lambda path: expected_by_name[Path(path).name])
