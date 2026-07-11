from pathlib import Path
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

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
from protcross.data.af2 import AF2_MODEL_VERSION, AF2DownloadConfig, AF2Downloader
from protcross.data.dataset import EvoPointDataset
from protcross.inference import PredictionResult


MINIMAL_PDB = """\
ATOM      1  N   ALA A   1      11.104  13.207   9.447  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.120   9.327  1.00 20.00           C
ATOM      3  C   ALA A   1      13.129  14.520   9.617  1.00 20.00           C
ATOM      4  N   GLY A   2      14.104  14.907   9.947  1.00 30.00           N
ATOM      5  CA  GLY A   2      15.560  14.920   9.827  1.00 30.00           C
ATOM      6  C   GLY A   2      16.129  16.320   9.517  1.00 30.00           C
END
"""


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
            "--accept-esm-license",
            "--trust-unverified-assets",
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
    assert args.accept_esm_license is True
    assert args.trust_unverified_assets is True
    assert args.auto_assets is True
    assert args.device == "cpu"


def test_predict_cli_accepts_assets_dir():
    parser = build_parser()
    args = parser.parse_args(["input.pdb", "--assets-dir", "assets", "--output", "out.pdb"])

    assert args.input_pdb == "input.pdb"
    assert args.assets_dir == "assets"


def test_predict_cli_can_disable_auto_assets():
    parser = build_parser()
    args = parser.parse_args(["input.pdb", "--no-auto-assets"])

    assert args.auto_assets is False


def test_predict_cli_rejects_refresh_when_auto_assets_are_disabled(tmp_path):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        predict_main([str(input_pdb), "--no-auto-assets", "--refresh-assets"])


def test_predict_cli_rejects_cross_format_output_before_assets(tmp_path, monkeypatch, capsys):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    monkeypatch.setattr(
        "protcross.cli.predict._resolve_prediction_asset_paths",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("assets must not be resolved")),
    )

    exit_code = predict_main(
        [str(input_pdb), "--output", str(tmp_path / "converted.cif"), "--summary-only"]
    )

    assert exit_code == 1
    assert "same PDB or mmCIF format" in capsys.readouterr().err


def test_predict_cli_forwards_resolved_asset_metadata(tmp_path, monkeypatch):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    metadata = {
        "asset_version": "0.1.2",
        "checkpoint": {"actual_sha256": "checkpoint-hash", "verified": True},
    }
    captured = {}

    def fake_resolve(args, *, auto_setup):
        args.ckpt_path = str(tmp_path / "model.ckpt")
        args.esm_weights = str(tmp_path / "esm.pth")
        args.pca_path = str(tmp_path / "pca.pkl")
        return SimpleNamespace(asset_version="0.1.2", asset_metadata=metadata)

    class FakePredictor:
        def predict(self, *args, **kwargs):
            return object()

    def fake_from_files(**kwargs):
        captured.update(kwargs)
        return FakePredictor()

    monkeypatch.setattr("protcross.cli.predict._resolve_prediction_asset_paths", fake_resolve)
    monkeypatch.setattr("protcross.inference.ProtCrossPredictor.from_files", staticmethod(fake_from_files))

    exit_code = predict_main([str(input_pdb), "--summary-only", "--quiet"])

    assert exit_code == 0
    assert captured["asset_version"] == "0.1.2"
    assert captured["asset_metadata"] is metadata


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
    args = parser.parse_args([
        "--pdb-id-file",
        "pdb_ids.txt",
        "--initial-mapping-file",
        "mapping.json",
        "--allow-partial",
        "--allow-empty-downloads",
        "--append",
    ])

    assert args.pdb_id_file == "pdb_ids.txt"
    assert args.initial_mapping_file == "mapping.json"
    assert args.mapping_file == "artifacts/pdb_uniprot_mapping.json"
    assert args.allow_partial is True
    assert args.allow_empty_downloads is True
    assert args.append is True


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


def test_af2_downloader_rejects_empty_inputs_by_default(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    config = AF2DownloadConfig(raw_pdb_dir=raw_dir, output_dir=tmp_path / "af2")

    try:
        AF2Downloader(config).run()
    except ValueError as exc:
        assert "No PDB IDs found" in str(exc)
    else:
        raise AssertionError("expected empty AF2 input failure")


def test_af2_downloader_preloads_initial_mapping_file(tmp_path):
    mapping_file = tmp_path / "initial_mapping.json"
    mapping_file.write_text('{"1e12": "P12345", "not_a_pdb": "ignored"}', encoding="utf-8")
    config = AF2DownloadConfig(initial_mapping_file=mapping_file, mapping_file=tmp_path / "output_mapping.json")

    downloader = AF2Downloader(config)

    assert downloader.preloaded_mapping == {"1E12": "P12345"}
    assert downloader.mapping == {}


def test_af2_downloader_append_mode_preserves_existing_mapping(tmp_path):
    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text('{"1e12": "P12345"}', encoding="utf-8")
    config = AF2DownloadConfig(mapping_file=mapping_file, append=True)

    downloader = AF2Downloader(config)

    assert downloader.preloaded_mapping == {"1E12": "P12345"}
    assert downloader.mapping == {"1E12": "P12345"}


def test_af2_downloader_creates_mapping_parent(tmp_path):
    mapping_file = tmp_path / "nested" / "pdb_uniprot_mapping.json"
    config = AF2DownloadConfig(mapping_file=mapping_file)
    downloader = AF2Downloader(config)
    downloader.mapping = {"1ABC": "P12345"}

    downloader.save_mapping()

    assert json.loads(mapping_file.read_text(encoding="utf-8")) == {"1ABC": "P12345"}


def test_af2_downloader_fails_when_requested_ids_do_not_download(tmp_path, monkeypatch):
    pdb_id_file = tmp_path / "ids.txt"
    pdb_id_file.write_text("1ABC\n", encoding="utf-8")
    config = AF2DownloadConfig(
        pdb_id_file=pdb_id_file,
        mapping_file=tmp_path / "mapping.json",
        output_dir=tmp_path / "af2",
    )
    downloader = AF2Downloader(config)
    monkeypatch.setattr(downloader, "fetch_uniprot_ids", lambda pdb_id: [])

    with pytest.raises(RuntimeError, match="AlphaFold download failed"):
        downloader.run()


def test_af2_downloader_allows_partial_success_when_requested(tmp_path, monkeypatch):
    pdb_id_file = tmp_path / "ids.txt"
    pdb_id_file.write_text("1ABC\n2DEF\n", encoding="utf-8")
    config = AF2DownloadConfig(
        pdb_id_file=pdb_id_file,
        mapping_file=tmp_path / "mapping.json",
        output_dir=tmp_path / "af2",
        allow_partial=True,
    )
    downloader = AF2Downloader(config)
    monkeypatch.setattr(downloader, "fetch_uniprot_ids", lambda pdb_id: ["P12345"] if pdb_id == "1ABC" else [])
    monkeypatch.setattr(downloader, "download_structure", lambda accession: True)

    mapping = downloader.run()

    assert mapping == {"1ABC": "P12345"}


def test_af2_downloader_quarantines_stale_structures_without_append(tmp_path, monkeypatch):
    pdb_id_file = tmp_path / "ids.txt"
    pdb_id_file.write_text("1ABC\n", encoding="utf-8")
    output_dir = tmp_path / "af2"
    output_dir.mkdir()
    stale = output_dir / "AF-OLD.pdb"
    stale.write_text("HEADER stale\nATOM      1  CA  ALA A   1       0.0     0.0     0.0  1.00 90.00           C\n", encoding="utf-8")
    stale_manifest = output_dir / "AF-OLD.pdb.protcross-af2.json"
    stale_manifest.write_text("{}", encoding="utf-8")
    config = AF2DownloadConfig(
        pdb_id_file=pdb_id_file,
        mapping_file=tmp_path / "mapping.json",
        output_dir=output_dir,
    )
    downloader = AF2Downloader(config)
    monkeypatch.setattr(downloader, "fetch_uniprot_ids", lambda pdb_id: ["P12345"])
    monkeypatch.setattr(downloader, "download_structure", lambda accession: True)

    mapping = downloader.run()

    assert mapping == {"1ABC": "P12345"}
    assert not stale.exists()
    assert not stale_manifest.exists()
    orphaned = list((output_dir / "_orphaned").glob("*/AF-OLD.pdb"))
    assert len(orphaned) == 1


def test_af2_downloader_refreshes_unverifiable_cached_file(tmp_path, monkeypatch):
    output_dir = tmp_path / "af2"
    output_dir.mkdir()
    cached = output_dir / "AF-P12345.pdb"
    cached.write_text("not a pdb", encoding="utf-8")
    content = (
        b"HEADER    ALPHAFOLD\n"
        b"ATOM      1  CA  ALA A   1      11.000  12.000  13.000  1.00 90.00           C\n"
    )

    class Response:
        status_code = 200

    response = Response()
    response.content = content

    monkeypatch.setattr("protcross.data.af2.requests.get", lambda *args, **kwargs: response)
    downloader = AF2Downloader(AF2DownloadConfig(output_dir=output_dir))

    assert downloader.download_structure("P12345") is True
    assert cached.read_bytes() == content
    manifest = json.loads((output_dir / "AF-P12345.pdb.protcross-af2.json").read_text(encoding="utf-8"))
    assert manifest["uniprot_id"] == "P12345"
    assert manifest["model_version"] == AF2_MODEL_VERSION


def test_af2_downloader_reuses_verified_cached_file_without_network(tmp_path, monkeypatch):
    output_dir = tmp_path / "af2"
    output_dir.mkdir()
    cached = output_dir / "AF-P12345.pdb"
    content = (
        b"HEADER    ALPHAFOLD\n"
        b"ATOM      1  CA  ALA A   1      11.000  12.000  13.000  1.00 90.00           C\n"
    )
    cached.write_bytes(content)
    downloader = AF2Downloader(AF2DownloadConfig(output_dir=output_dir))
    downloader._write_download_manifest(cached, "P12345", "https://example.org/af.pdb", content)
    monkeypatch.setattr("protcross.data.af2.requests.get", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))

    assert downloader.download_structure("P12345") is True


def test_dataset_all_split_uses_every_processed_file(tmp_path):
    for name in ("a.pt", "b.pt", "c.pt"):
        (tmp_path / name).write_bytes(b"")
    dataset = object.__new__(EvoPointDataset)
    dataset.root = str(tmp_path)
    dataset.split = "all"
    dataset.split_seed = 42

    files = dataset._split_files()

    assert [Path(file).name for file in files] == ["a.pt", "b.pt", "c.pt"]


def test_preprocess_quarantines_orphan_outputs(tmp_path):
    from protcross.data.preprocess import _quarantine_orphan_outputs

    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    stale = output_dir / "old.pt"
    stale.write_bytes(b"old")
    current = tmp_path / "current.pdb"
    current.write_text(MINIMAL_PDB, encoding="utf-8")

    _quarantine_orphan_outputs(output_dir, [current])

    assert not stale.exists()
    assert list((output_dir / "_orphaned").glob("*/*.pt"))


def test_dataset_rejects_preprocess_manifest_orphans(tmp_path):
    from protcross.data.dataset import PREPROCESS_MANIFEST

    root = tmp_path / "processed"
    root.mkdir()
    (root / "a.pt").write_bytes(b"a")
    (root / "stale.pt").write_bytes(b"stale")
    (root / PREPROCESS_MANIFEST).write_text(
        json.dumps({"schema_version": "protcross-preprocess-v1", "append_mode": False, "produced_outputs": ["a.pt"]}),
        encoding="utf-8",
    )
    dataset = object.__new__(EvoPointDataset)
    dataset.root = str(root)

    with pytest.raises(RuntimeError, match="not listed"):
        dataset._validate_preprocess_manifest()


def test_setup_assets_parser_allows_env_default():
    parser = setup_assets.build_parser()
    args = parser.parse_args(["--asset-version", "0.1.1-paper"])

    assert args.output_dir is None
    assert args.asset_version == "0.1.1-paper"


def test_setup_assets_parser_uses_environment_asset_version(monkeypatch):
    monkeypatch.setenv("PROTCROSS_ASSET_VERSION", "0.1.1-paper")

    args = setup_assets.build_parser().parse_args([])

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
            "--accept-esm-license",
            "--trust-unverified-assets",
        ]
    )

    assets = _resolve_prediction_asset_paths(args, auto_setup=True)

    assert calls == []
    assert assets is not None
    assert args.ckpt_path == str(tmp_path / DEFAULT_CHECKPOINT_FILENAME)
    assert args.esm_weights == str(explicit_esm)
    assert args.pca_path == str(tmp_path / DEFAULT_PCA_FILENAME)


def test_predict_asset_resolution_requires_esm_license_for_use(tmp_path, monkeypatch):
    _trust_managed_asset_hashes(monkeypatch)
    for filename in (DEFAULT_CHECKPOINT_FILENAME, "esmc_600m_2024_12_v0.pth", DEFAULT_PCA_FILENAME):
        (tmp_path / filename).write_bytes(b"asset")
    args = build_parser().parse_args(["input.pdb", "--assets-dir", str(tmp_path)])

    with pytest.raises(RuntimeError, match="accept-esm-license"):
        _resolve_prediction_asset_paths(args, auto_setup=False)


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
        ["input.pdb", "--threshold", "nan"],
        ["input.pdb", "--pocket-cluster-cutoff", "inf"],
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


def test_preprocess_preflight_does_not_create_output_before_input_validation(tmp_path):
    from protcross.data.preprocess import PreprocessConfig, preprocess_directory

    output_dir = tmp_path / "out"

    try:
        preprocess_directory(
            PreprocessConfig(
                data_dir=tmp_path / "missing",
                output_dir=output_dir,
                model_name=tmp_path / "missing-esm.pth",
                pca_model_path=tmp_path / "missing-pca.pkl",
            )
        )
    except FileNotFoundError as exc:
        assert "Input data directory not found" in str(exc)
    else:
        raise AssertionError("expected preprocess input failure")
    assert not output_dir.exists()


def test_map_labels_rejects_empty_processed_af2_dir(tmp_path):
    from protcross.data.label_mapping import LabelMappingConfig, map_labels

    processed_pdb = tmp_path / "processed_pdb"
    processed_af2 = tmp_path / "processed_af2"
    raw_pdb = tmp_path / "raw_pdb"
    raw_af2 = tmp_path / "raw_af2"
    for directory in (processed_pdb, processed_af2, raw_pdb, raw_af2):
        directory.mkdir()
    mapping = tmp_path / "mapping.json"
    mapping.write_text('{"1ABC": "P12345"}', encoding="utf-8")

    try:
        map_labels(
            LabelMappingConfig(
                processed_pdb_dir=processed_pdb,
                processed_af2_dir=processed_af2,
                raw_pdb_dir=raw_pdb,
                raw_af2_dir=raw_af2,
                mapping_file=mapping,
                output_csv=tmp_path / "report.csv",
            )
        )
    except FileNotFoundError as exc:
        assert "No processed AF2 .pt files" in str(exc)
    else:
        raise AssertionError("expected empty map-labels input failure")


def test_map_labels_rejects_zero_mapped_labels_by_default(tmp_path):
    from protcross.data.label_mapping import LabelMappingConfig, map_labels

    processed_pdb = tmp_path / "processed_pdb"
    processed_af2 = tmp_path / "processed_af2"
    raw_pdb = tmp_path / "raw_pdb"
    raw_af2 = tmp_path / "raw_af2"
    for directory in (processed_pdb, processed_af2, raw_pdb, raw_af2):
        directory.mkdir()
    (processed_af2 / "AF-Q99999.pt").write_bytes(b"not loaded because no mapping target")
    mapping = tmp_path / "mapping.json"
    mapping.write_text('{"1ABC": "P12345"}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="no mapped AF2 labels"):
        map_labels(
            LabelMappingConfig(
                processed_pdb_dir=processed_pdb,
                processed_af2_dir=processed_af2,
                raw_pdb_dir=raw_pdb,
                raw_af2_dir=raw_af2,
                mapping_file=mapping,
                output_csv=tmp_path / "report.csv",
            )
        )


def test_map_labels_can_allow_empty_mapping_for_diagnostics(tmp_path):
    from protcross.data.label_mapping import LabelMappingConfig, map_labels

    processed_pdb = tmp_path / "processed_pdb"
    processed_af2 = tmp_path / "processed_af2"
    raw_pdb = tmp_path / "raw_pdb"
    raw_af2 = tmp_path / "raw_af2"
    for directory in (processed_pdb, processed_af2, raw_pdb, raw_af2):
        directory.mkdir()
    (processed_af2 / "AF-Q99999.pt").write_bytes(b"not loaded because no mapping target")
    mapping = tmp_path / "mapping.json"
    mapping.write_text('{"1ABC": "P12345"}', encoding="utf-8")

    report = map_labels(
        LabelMappingConfig(
            processed_pdb_dir=processed_pdb,
            processed_af2_dir=processed_af2,
            raw_pdb_dir=raw_pdb,
            raw_af2_dir=raw_af2,
            mapping_file=mapping,
            output_csv=tmp_path / "report.csv",
            allow_empty_mapping=True,
        )
    )

    assert report["Mapped_Sites"] == 0
    assert (tmp_path / "report.csv.manifest.json").exists()


def test_predict_main_writes_default_result_package(tmp_path, monkeypatch, capsys):
    _trust_managed_asset_hashes(monkeypatch)
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    for filename in (DEFAULT_CHECKPOINT_FILENAME, "esmc_600m_2024_12_v0.pth", DEFAULT_PCA_FILENAME):
        (tmp_path / filename).write_bytes(b"asset")
    monkeypatch.setenv("PROTCROSS_ASSETS_DIR", str(tmp_path))

    _patch_fake_predictor(monkeypatch)

    assert (
        predict_main(
            [
                str(input_pdb),
                "--out-dir",
                str(tmp_path / "results"),
                "--device",
                "cpu",
                "--accept-esm-license",
            ]
        )
        == 0
    )

    output_dir = tmp_path / "results"
    assert (output_dir / "input.protcross.pdb").exists()
    assert (output_dir / "input.protcross.scores.tsv").exists()
    assert (output_dir / "input.protcross.pockets.json").exists()
    summary = json.loads((output_dir / "input.protcross.summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "protcross-summary-v2"
    assert summary["top_pocket"]["residue_count"] == 1
    assert summary["unscored_bfactor_policy"] == "zero"
    stdout = capsys.readouterr().out
    assert "Wrote structure:" in stdout
    assert "Top cluster score-weighted CA centroid:" in stdout


def test_predict_main_summary_only_does_not_create_defaults(tmp_path, monkeypatch, capsys):
    _trust_managed_asset_hashes(monkeypatch)
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    for filename in (DEFAULT_CHECKPOINT_FILENAME, "esmc_600m_2024_12_v0.pth", DEFAULT_PCA_FILENAME):
        (tmp_path / filename).write_bytes(b"asset")
    monkeypatch.setenv("PROTCROSS_ASSETS_DIR", str(tmp_path))

    _patch_fake_predictor(monkeypatch)

    assert predict_main([str(input_pdb), "--summary-only", "--device", "cpu", "--accept-esm-license"]) == 0

    assert not (tmp_path / "input.protcross.pdb").exists()
    assert "Residues above threshold: 1" in capsys.readouterr().out


def test_predict_rejects_unavailable_cuda_before_resolving_assets(tmp_path, monkeypatch, capsys):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "protcross.cli.predict._resolve_prediction_asset_paths",
        lambda *args, **kwargs: pytest.fail("assets must not be resolved for an unavailable explicit device"),
    )

    assert predict_main([str(input_pdb), "--summary-only", "--device", "cuda"]) == 1
    assert "CUDA was requested but" in capsys.readouterr().err


def test_summary_only_explicit_output_refuses_overwrite_before_assets(tmp_path, monkeypatch, capsys):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    output_pdb = tmp_path / "existing.pdb"
    output_pdb.write_text("do not replace\n", encoding="utf-8")
    monkeypatch.setattr(
        "protcross.cli.predict._resolve_prediction_asset_paths",
        lambda *args, **kwargs: pytest.fail("assets must not be resolved when output preflight fails"),
    )

    assert predict_main(
        [str(input_pdb), "--summary-only", "--output", str(output_pdb), "--device", "cpu"]
    ) == 1
    assert output_pdb.read_text(encoding="utf-8") == "do not replace\n"
    assert "already exist" in capsys.readouterr().err


def test_predict_rejects_colliding_outputs_before_assets(tmp_path, monkeypatch, capsys):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    shared_output = tmp_path / "shared.out"
    monkeypatch.setattr(
        "protcross.cli.predict._resolve_prediction_asset_paths",
        lambda *args, **kwargs: pytest.fail("assets must not be resolved for colliding outputs"),
    )

    assert (
        predict_main(
            [
                str(input_pdb),
                "--summary-only",
                "--output",
                str(shared_output),
                "--scores-tsv",
                str(shared_output),
            ]
        )
        == 1
    )
    assert "must be distinct" in capsys.readouterr().err
    assert input_pdb.read_text(encoding="utf-8") == MINIMAL_PDB


def test_predict_rejects_input_as_output_even_with_overwrite(tmp_path, monkeypatch, capsys):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    monkeypatch.setattr(
        "protcross.cli.predict._resolve_prediction_asset_paths",
        lambda *args, **kwargs: pytest.fail("assets must not be resolved when an output aliases input"),
    )

    assert (
        predict_main(
            [
                str(input_pdb),
                "--summary-only",
                "--summary-json",
                str(input_pdb),
                "--overwrite",
            ]
        )
        == 1
    )
    assert "must not overwrite the input structure" in capsys.readouterr().err
    assert input_pdb.read_text(encoding="utf-8") == MINIMAL_PDB


def test_predict_rejects_directory_output_before_assets(tmp_path, monkeypatch, capsys):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    monkeypatch.setattr(
        "protcross.cli.predict._resolve_prediction_asset_paths",
        lambda *args, **kwargs: pytest.fail("assets must not be resolved for a directory output"),
    )

    assert (
        predict_main(
            [
                str(input_pdb),
                "--summary-only",
                "--summary-json",
                str(tmp_path),
                "--overwrite",
            ]
        )
        == 1
    )
    assert "directory, not a file" in capsys.readouterr().err


def test_predict_expands_user_output_path_before_prediction(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    for filename in (
        DEFAULT_CHECKPOINT_FILENAME,
        "esmc_600m_2024_12_v0.pth",
        DEFAULT_PCA_FILENAME,
    ):
        (assets_dir / filename).write_bytes(b"asset")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PROTCROSS_ASSETS_DIR", str(assets_dir))
    _trust_managed_asset_hashes(monkeypatch)
    _patch_fake_predictor(monkeypatch)

    assert (
        predict_main(
            [
                str(input_pdb),
                "--summary-only",
                "--summary-json",
                "~/result.json",
                "--accept-esm-license",
                "--quiet",
            ]
        )
        == 0
    )

    assert (home / "result.json").exists()
    assert not (Path.cwd() / "~" / "result.json").exists()


def test_pyproject_exposes_setup_assets_entry_point():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'protcross-setup-assets = "protcross.cli.setup_assets:main"' in pyproject
    assert 'requires-python = ">=3.10,<3.11"' in pyproject
    assert '"esm>=3.1.0,<3.3"' in pyproject
    assert '"httpx>=0.27,<0.29"' in pyproject
    assert '"torchvision>=0.18,<0.19"' in pyproject
    assert '"torchtext>=0.18,<0.19"' in pyproject


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
                unscored_bfactor_policy=kwargs.get("unscored_bfactor_policy", "zero"),
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
    monkeypatch.setattr(
        "protcross.assets.sha256_file",
        lambda path: expected_by_name.get(Path(path).name)
        or hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    )
