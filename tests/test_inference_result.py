from pathlib import Path
import hashlib
import json
import pickle
import re
import sys
import types

import numpy as np
import pytest
import torch

from protcross.assets import (
    ASSET_MANIFEST_FILENAME,
    DEFAULT_ASSET_BUNDLE,
    DEFAULT_CHECKPOINT_FILENAME,
    DEFAULT_PCA_FILENAME,
)
from protcross.data.esm import ESMFeatureExtractor
from protcross.inference import PredictionResult, PredictorAssets, ProtCrossPredictor, predict_pdb
from protcross.inference.predictor import _resolve_predict_pdb_assets


MINIMAL_PDB = """\
ATOM      1  N   ALA A   1      11.104  13.207   9.447  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.120   9.327  1.00 20.00           C
ATOM      3  C   ALA A   1      13.129  14.520   9.617  1.00 20.00           C
ATOM      4  N   GLY A   2      14.104  14.907   9.947  1.00 30.00           N
ATOM      5  CA  GLY A   2      15.560  14.920   9.827  1.00 30.00           C
ATOM      6  C   GLY A   2      16.129  16.320   9.517  1.00 30.00           C
TER
END
"""


def test_prediction_result_summary_and_thresholding():
    result = PredictionResult(
        input_pdb=Path("input.pdb"),
        residue_ids=["A_1", "A_2", "A_3"],
        probabilities=np.array([0.1, 0.7, 0.4]),
        threshold=0.5,
    )

    hits = result.binding_residues

    assert [hit.residue_id for hit in hits] == ["A_2"]
    assert hits[0].score == hits[0].probability
    assert np.shares_memory(result.scores, result.probabilities)
    assert "Residues above threshold: 1" in result.format_summary()
    assert result.to_records()[0]["chain_id"] == "A"


def test_prediction_result_ranks_blank_and_named_chains_independently():
    result = PredictionResult(
        input_pdb=Path("input.pdb"),
        residue_ids=["_1", "_2", "A_1"],
        probabilities=np.array([0.1, 0.8, 0.5]),
        residue_metadata=[
            {"residue_id": "_1", "chain_id": ""},
            {"residue_id": "_2", "chain_id": ""},
            {"residue_id": "A_1", "chain_id": "A"},
        ],
    )

    assert result.chains_analyzed == ["", "A"]
    assert [record["rank_within_chain"] for record in result.to_records()] == [2, 1, 1]
    assert "Chains analyzed: <blank>, A" in result.format_summary()


def test_prediction_result_reports_weighted_pocket_center_and_clusters():
    result = PredictionResult(
        input_pdb=Path("input.pdb"),
        residue_ids=["A_1", "A_2", "A_3", "A_4"],
        probabilities=np.array([0.9, 0.6, 0.8, 0.2]),
        threshold=0.5,
        ca_coords=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [40.0, 0.0, 0.0],
            ]
        ),
        cluster_cutoff=8.0,
    )

    payload = result.to_pocket_dict()

    assert payload["schema_version"] == "protcross-pocket-v2"
    assert payload["coordinate_units"] == "angstrom"
    assert payload["center_type"] == "score_weighted_ca_centroid"
    assert payload["score_calibrated"] is False
    assert payload["selected_residue_count"] == 3
    np.testing.assert_allclose(payload["aggregate_pocket"]["center"], [17.2 / 2.3, 0.0, 0.0])
    np.testing.assert_allclose(payload["aggregate_pocket"]["center_unweighted"], [22.0 / 3.0, 0.0, 0.0])
    assert payload["aggregate_pocket"]["bbox_min"] == [0.0, 0.0, 0.0]
    assert payload["aggregate_pocket"]["bbox_max"] == [20.0, 0.0, 0.0]
    assert [pocket["residue_count"] for pocket in payload["clustered_pockets"]] == [2, 1]
    np.testing.assert_allclose(payload["clustered_pockets"][0]["center"], [0.8, 0.0, 0.0])
    assert payload["clustered_pockets"][0]["residues"][0]["residue_id"] == "A_1"
    summary = result.to_summary_dict()
    assert summary["aggregate_pocket"]["residue_count"] == 3
    assert summary["top_pocket"]["residue_count"] == 2
    np.testing.assert_allclose(summary["top_pocket"]["center"], [0.8, 0.0, 0.0])


def test_prediction_result_caches_internal_derivations_and_returns_defensive_copies():
    result = PredictionResult(
        input_pdb=Path("input.pdb"),
        residue_ids=["A_1", "A_2", "A_3"],
        probabilities=np.array([0.9, 0.6, 0.2]),
        threshold=0.5,
        ca_coords=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        cluster_cutoff=8.0,
    )

    records = result.to_records()
    pockets = result.to_pocket_dict()
    summary = result.to_summary_dict()

    records[0]["score"] = 9.9
    pockets["clustered_pockets"][0]["residue_count"] = 999
    summary["top_residues"][0]["score"] = 9.9

    assert result.to_records()[0]["score"] == pytest.approx(0.9)
    assert result.to_pocket_dict()["clustered_pockets"][0]["residue_count"] == 2
    assert result.to_summary_dict()["top_residues"][0]["score"] == pytest.approx(0.9)


def test_prediction_result_invalidates_derived_caches_when_threshold_changes():
    result = PredictionResult(
        input_pdb=Path("input.pdb"),
        residue_ids=["A_1", "A_2"],
        probabilities=np.array([0.6, 0.8]),
        threshold=0.5,
        ca_coords=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )
    original = result.to_summary_dict()

    result.threshold = 0.7
    updated = result.to_summary_dict()

    assert updated is not original
    assert updated["threshold"] == 0.7
    assert updated["selected_residue_count"] == 1
    assert [record["is_binding"] for record in result.to_records()] == [0, 1]


def test_prediction_result_rejects_nonfinite_geometry_and_protects_score_array():
    result = PredictionResult(
        input_pdb=Path("input.pdb"),
        residue_ids=["A_1"],
        probabilities=np.array([0.6]),
        ca_coords=np.array([[0.0, 0.0, 0.0]]),
    )

    with pytest.raises(ValueError, match="read-only"):
        result.probabilities[0] = 0.1
    with pytest.raises(ValueError, match="finite"):
        result.cluster_cutoff = float("nan")
    with pytest.raises(ValueError, match="finite"):
        PredictionResult(
            input_pdb=Path("input.pdb"),
            residue_ids=["A_1"],
            probabilities=np.array([0.6]),
            ca_coords=np.array([[float("nan"), 0.0, 0.0]]),
        )


def test_prediction_result_writers_never_replace_the_input_structure(tmp_path):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    result = PredictionResult(
        input_pdb=input_pdb,
        residue_ids=["A_1"],
        probabilities=np.array([0.6]),
        ca_coords=np.array([[0.0, 0.0, 0.0]]),
    )

    for writer in (result.write_pdb, result.write_scores_tsv, result.write_pocket_json, result.write_summary_json):
        with pytest.raises(ValueError, match="must not overwrite"):
            writer(input_pdb)

    assert input_pdb.read_text(encoding="utf-8") == MINIMAL_PDB


def test_prediction_result_rejects_cross_format_annotated_structure(tmp_path):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    result = PredictionResult(
        input_pdb=input_pdb,
        residue_ids=["A_1"],
        probabilities=np.array([0.6]),
        ca_coords=np.array([[0.0, 0.0, 0.0]]),
    )

    with pytest.raises(ValueError, match="same PDB or mmCIF format"):
        result.write_pdb(tmp_path / "converted.cif")


def test_prediction_result_empty_pocket_json_shape():
    result = PredictionResult(
        input_pdb=Path("input.pdb"),
        residue_ids=["A_1"],
        probabilities=np.array([0.1]),
        threshold=0.5,
        ca_coords=np.array([[1.0, 2.0, 3.0]]),
    )

    payload = result.to_pocket_dict()

    assert payload["selected_residue_count"] == 0
    assert payload["aggregate_pocket"] is None
    assert payload["clustered_pockets"] == []


def test_prediction_result_warnings_are_reported_in_outputs():
    result = PredictionResult(
        input_pdb=Path("models.pdb"),
        residue_ids=["A_1"],
        probabilities=np.array([0.8]),
        threshold=0.5,
        ca_coords=np.array([[1.0, 2.0, 3.0]]),
        warnings=["Input contains 2 models; only model 0 was parsed and scored."],
    )

    assert "only model 0" in result.to_summary_dict()["warnings"][0]
    assert "only model 0" in result.to_pocket_dict()["warnings"][0]
    assert "WARNING: Input contains 2 models" in result.format_summary()


def test_prediction_result_writes_enhanced_scores_tsv_and_pocket_json(tmp_path):
    result = PredictionResult(
        input_pdb=Path("input.pdb"),
        residue_ids=["A_1", "A_2"],
        probabilities=np.array([0.7, 0.1]),
        threshold=0.5,
        ca_coords=np.array([[1.0, 2.0, 3.0], [9.0, 8.0, 7.0]]),
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
                "input_bfactor": 30.0,
            },
        ],
        asset_version="0.1.2",
        device="cpu",
        max_len=1022,
    )
    scores_path = tmp_path / "scores.tsv"
    legacy_scores_path = tmp_path / "scores.legacy.tsv"
    pocket_path = tmp_path / "pocket.json"
    summary_path = tmp_path / "summary.json"

    result.write_scores_tsv(scores_path)
    result.write_scores_tsv(legacy_scores_path, schema="legacy")
    result.write_pocket_json(pocket_path)
    result.write_summary_json(summary_path)

    lines = scores_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t") == [
        "residue_id",
        "residue_key",
        "residue_id_namespace",
        "model_id",
        "chain_id",
        "auth_asym_id",
        "label_asym_id",
        "residue_number",
        "auth_seq_id",
        "label_seq_id",
        "insertion_code",
        "resname",
        "one_letter_code",
        "input_bfactor",
        "model_score",
        "probability",
        "is_binding",
        "x",
        "y",
        "z",
        "cluster_id",
        "is_scored",
        "rank_global",
        "rank_within_chain",
    ]
    assert lines[1].split("\t")[13:20] == [
        "20.000000", "0.700000", "0.700000", "1", "1.000000", "2.000000", "3.000000"
    ]
    assert lines[1].split("\t")[-3:] == ["1", "1", "1"]
    assert legacy_scores_path.read_text(encoding="utf-8").splitlines()[0].split("\t") == [
        "residue_id",
        "chain_id",
        "residue_number",
        "probability",
        "is_binding",
    ]
    assert json.loads(pocket_path.read_text(encoding="utf-8"))["aggregate_pocket"]["residue_count"] == 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "protcross-summary-v2"
    assert summary["score_calibrated"] is False
    assert summary["asset_version"] == "0.1.2"
    assert summary["top_pocket"]["residue_count"] == 1
    assert summary["aggregate_pocket"]["residue_count"] == 1


def test_prediction_result_validates_shapes():
    with pytest.raises(ValueError, match="same length"):
        PredictionResult(input_pdb=Path("input.pdb"), residue_ids=["A_1"], probabilities=np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="shape"):
        PredictionResult(
            input_pdb=Path("input.pdb"),
            residue_ids=["A_1"],
            probabilities=np.array([0.1]),
            ca_coords=np.array([[1.0, 2.0]]),
        )
    with pytest.raises(ValueError, match="finite"):
        PredictionResult(input_pdb=Path("input.pdb"), residue_ids=["A_1"], probabilities=np.array([np.nan]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        PredictionResult(input_pdb=Path("input.pdb"), residue_ids=["A_1"], probabilities=np.array([1.2]))
    with pytest.raises(ValueError, match="threshold"):
        PredictionResult(input_pdb=Path("input.pdb"), residue_ids=["A_1"], probabilities=np.array([0.2]), threshold=1.2)


def test_predictor_assets_from_dir(tmp_path):
    assets = PredictorAssets.from_dir(tmp_path)

    assert assets.checkpoint == tmp_path / DEFAULT_CHECKPOINT_FILENAME
    assert assets.esm_weights == tmp_path / "esmc_600m_2024_12_v0.pth"
    assert assets.pca == tmp_path / DEFAULT_PCA_FILENAME


def test_predict_pdb_asset_resolution_uses_assets_dir(tmp_path, monkeypatch):
    _trust_managed_asset_hashes(monkeypatch)
    for filename in (DEFAULT_CHECKPOINT_FILENAME, "esmc_600m_2024_12_v0.pth", DEFAULT_PCA_FILENAME):
        (tmp_path / filename).write_bytes(b"asset")

    ckpt, esm, pca = _resolve_predict_pdb_assets(None, None, None, assets_dir=tmp_path, auto_setup_assets=False)

    assert ckpt == tmp_path / DEFAULT_CHECKPOINT_FILENAME
    assert esm == tmp_path / "esmc_600m_2024_12_v0.pth"
    assert pca == tmp_path / DEFAULT_PCA_FILENAME


def test_predict_pdb_asset_resolution_reports_missing(tmp_path, monkeypatch):
    for name in ("PROTCROSS_CHECKPOINT", "PROTCROSS_ESM_WEIGHTS", "PROTCROSS_PCA"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PROTCROSS_ASSETS_DIR", str(tmp_path / "empty-assets"))

    with pytest.raises(ValueError, match="Run `protcross setup-assets`"):
        _resolve_predict_pdb_assets(None, None, None, auto_setup_assets=False)


def test_predict_pdb_asset_resolution_auto_installs_defaults(tmp_path, monkeypatch):
    _trust_managed_asset_hashes(monkeypatch)
    for name in ("PROTCROSS_CHECKPOINT", "PROTCROSS_ESM_WEIGHTS", "PROTCROSS_PCA"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PROTCROSS_ASSETS_DIR", str(tmp_path))

    def fake_setup_assets(output_dir=None, **kwargs):
        output_dir = Path(output_dir) if output_dir else tmp_path
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename in (DEFAULT_CHECKPOINT_FILENAME, "esmc_600m_2024_12_v0.pth", DEFAULT_PCA_FILENAME):
            (output_dir / filename).write_bytes(b"asset")
        (output_dir / ASSET_MANIFEST_FILENAME).write_text(
            '{"asset_version": "0.1.2"}',
            encoding="utf-8",
        )
        return output_dir

    monkeypatch.setattr("protcross.assets.setup_assets", fake_setup_assets)

    ckpt, esm, pca = _resolve_predict_pdb_assets(None, None, None)

    assert ckpt == tmp_path / DEFAULT_CHECKPOINT_FILENAME
    assert esm == tmp_path / "esmc_600m_2024_12_v0.pth"
    assert pca == tmp_path / DEFAULT_PCA_FILENAME


def test_predict_pdb_asset_resolution_uses_partial_assets_and_explicit_esm(tmp_path, monkeypatch):
    _trust_managed_asset_hashes(monkeypatch)
    for name in ("PROTCROSS_CHECKPOINT", "PROTCROSS_ESM_WEIGHTS", "PROTCROSS_PCA"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / DEFAULT_CHECKPOINT_FILENAME).write_bytes(b"checkpoint")
    (tmp_path / DEFAULT_PCA_FILENAME).write_bytes(b"pca")
    esm = tmp_path / "external-esm.pth"
    esm.write_bytes(b"esm")

    def fake_setup_assets(*args, **kwargs):
        raise AssertionError("setup_assets should not run when explicit ESM completes the bundle")

    monkeypatch.setattr("protcross.assets.setup_assets", fake_setup_assets)

    ckpt, resolved_esm, pca = _resolve_predict_pdb_assets(
        None,
        esm,
        None,
        assets_dir=tmp_path,
        auto_setup_assets=True,
        trust_unverified_assets=True,
    )

    assert ckpt == tmp_path / DEFAULT_CHECKPOINT_FILENAME
    assert resolved_esm == esm
    assert pca == tmp_path / DEFAULT_PCA_FILENAME


def test_protcross_predictor_fake_components_writes_result_package(tmp_path):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    output_pdb = tmp_path / "out.pdb"
    scores_tsv = tmp_path / "scores.tsv"
    pocket_json = tmp_path / "pockets.json"
    summary_json = tmp_path / "summary.json"

    predictor = ProtCrossPredictor(
        device="cpu",
        max_len=4,
        esm_extractor=_FakeESM(),
        pca_reducer=_FakePCA(),
        structure_parser=_FakeParser(),
        model=_FakeModel(),
        asset_version="test-assets",
    )

    result = predictor.predict(
        input_pdb,
        threshold=0.5,
        output_pdb=output_pdb,
        scores_tsv=scores_tsv,
        pocket_json=pocket_json,
        summary_json=summary_json,
        unscored_bfactor_policy="zero",
    )

    assert result.binding_residues[0].residue_id == "A_1"
    assert output_pdb.exists()
    assert scores_tsv.read_text(encoding="utf-8").splitlines()[0].startswith("residue_id\tresidue_key")
    assert json.loads(pocket_json.read_text(encoding="utf-8"))["clustered_pockets"][0]["residue_count"] == 1
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["asset_version"] == "test-assets"
    assert summary["unscored_bfactor_policy"] == "zero"
    assert summary["elapsed_seconds"] is not None


def test_predictor_result_package_staging_preserves_existing_outputs_on_writer_failure(
    tmp_path, monkeypatch
):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    outputs = {
        "output_pdb": tmp_path / "out.pdb",
        "scores_tsv": tmp_path / "scores.tsv",
        "pocket_json": tmp_path / "pockets.json",
        "summary_json": tmp_path / "summary.json",
    }
    for path in outputs.values():
        path.write_text(f"previous:{path.name}", encoding="utf-8")

    predictor = ProtCrossPredictor(
        device="cpu",
        max_len=4,
        esm_extractor=_FakeESM(),
        pca_reducer=_FakePCA(),
        structure_parser=_FakeParser(),
        model=_FakeModel(),
    )

    def fail_pocket_writer(self, output_json):
        raise OSError("simulated pocket writer failure")

    monkeypatch.setattr(PredictionResult, "write_pocket_json", fail_pocket_writer)

    with pytest.raises(OSError, match="simulated pocket writer failure"):
        predictor.predict(input_pdb, **outputs)

    for path in outputs.values():
        assert path.read_text(encoding="utf-8") == f"previous:{path.name}"
    assert not list(tmp_path.glob(".*.stage.*"))
    assert not list(tmp_path.glob(".*.backup"))


def test_predictor_result_package_rolls_back_existing_outputs_on_replace_failure(
    tmp_path, monkeypatch
):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    outputs = {
        "output_pdb": tmp_path / "out.pdb",
        "scores_tsv": tmp_path / "scores.tsv",
        "pocket_json": tmp_path / "pockets.json",
        "summary_json": tmp_path / "summary.json",
    }
    for path in outputs.values():
        path.write_text(f"previous:{path.name}", encoding="utf-8")
    predictor = ProtCrossPredictor(
        device="cpu",
        max_len=4,
        esm_extractor=_FakeESM(),
        pca_reducer=_FakePCA(),
        structure_parser=_FakeParser(),
        model=_FakeModel(),
    )
    original_replace = Path.replace

    def fail_scores_publication(path, target):
        if ".stage" in path.name and Path(target) == outputs["scores_tsv"]:
            raise OSError("simulated publication replace failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_scores_publication)

    with pytest.raises(OSError, match="simulated publication replace failure"):
        predictor.predict(input_pdb, **outputs)

    for path in outputs.values():
        assert path.read_text(encoding="utf-8") == f"previous:{path.name}"
    assert not list(tmp_path.glob(".*.stage.*"))
    assert not list(tmp_path.glob(".*.backup"))


def test_predict_pdb_missing_input_does_not_resolve_assets(tmp_path, monkeypatch):
    def fail_resolve_assets(*args, **kwargs):
        raise AssertionError("asset resolution must not run before input validation")

    monkeypatch.setattr("protcross.inference.predictor.resolve_prediction_assets", fail_resolve_assets)

    with pytest.raises(FileNotFoundError, match="Input structure not found"):
        predict_pdb(tmp_path / "missing.pdb")


def test_predict_pdb_invalid_options_do_not_resolve_assets(tmp_path, monkeypatch):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")

    def fail_resolve_assets(*args, **kwargs):
        raise AssertionError("asset resolution must not run before option validation")

    monkeypatch.setattr("protcross.inference.predictor.resolve_prediction_assets", fail_resolve_assets)

    invalid_calls = (
        {"threshold": 1.1},
        {"pocket_cluster_cutoff": 0},
        {"unscored_bfactor_policy": "drop"},
        {"max_len": 1023},
        {"device": "definitely-invalid"},
    )
    for kwargs in invalid_calls:
        with pytest.raises(ValueError):
            predict_pdb(input_pdb, **kwargs)


def test_predict_pdb_rejects_output_path_collisions_before_resolving_assets(tmp_path, monkeypatch):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    shared_output = tmp_path / "shared.out"

    def fail_resolve_assets(*args, **kwargs):
        raise AssertionError("asset resolution must not run for unsafe output paths")

    monkeypatch.setattr("protcross.inference.predictor.resolve_prediction_assets", fail_resolve_assets)

    with pytest.raises(ValueError, match="must be distinct"):
        predict_pdb(
            input_pdb,
            output_pdb=shared_output,
            scores_tsv=shared_output,
        )
    with pytest.raises(ValueError, match="must not overwrite the input structure"):
        predict_pdb(input_pdb, summary_json=input_pdb)
    with pytest.raises(IsADirectoryError, match="directory"):
        predict_pdb(input_pdb, summary_json=tmp_path)


def test_predictor_rejects_unsafe_output_paths_before_featurizing(tmp_path):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    esm = _CountingESM()
    pca = _CountingPCA()
    predictor = ProtCrossPredictor(
        device="cpu",
        esm_extractor=esm,
        pca_reducer=pca,
        structure_parser=_FakeParser(),
        model=_FakeModel(),
    )
    shared_output = tmp_path / "shared.out"

    with pytest.raises(ValueError, match="must be distinct"):
        predictor.predict(
            input_pdb,
            output_pdb=shared_output,
            scores_tsv=shared_output,
        )
    with pytest.raises(ValueError, match="must not overwrite the input structure"):
        predictor.predict(input_pdb, summary_json=input_pdb)
    with pytest.raises(IsADirectoryError, match="directory"):
        predictor.predict(input_pdb, summary_json=tmp_path)

    assert esm.calls == 0
    assert pca.calls == 0


def test_predictor_expands_user_output_paths(tmp_path, monkeypatch):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    predictor = ProtCrossPredictor(
        device="cpu",
        esm_extractor=_FakeESM(),
        pca_reducer=_FakePCA(),
        structure_parser=_FakeParser(),
        model=_FakeModel(),
    )

    result = predictor.predict(input_pdb, summary_json="~/result.json")

    assert (tmp_path / "result.json").exists()
    assert not (Path.cwd() / "~" / "result.json").exists()
    assert result.output_files["summary_json"] == str(tmp_path / "result.json")


def test_from_default_assets_uses_verified_resolver(monkeypatch, tmp_path):
    resolved = types.SimpleNamespace(
        checkpoint=tmp_path / "model.ckpt",
        esm_weights=tmp_path / "esm.pth",
        pca=tmp_path / "pca.pkl",
        asset_version="0.1.2",
        asset_metadata={"all_assets_verified": True},
    )
    captured_resolve = {}
    captured_files = {}
    expected = object()

    def fake_resolve(**kwargs):
        captured_resolve.update(kwargs)
        return resolved

    def fake_from_files(**kwargs):
        captured_files.update(kwargs)
        return expected

    monkeypatch.setattr("protcross.inference.predictor.resolve_prediction_assets", fake_resolve)
    monkeypatch.setattr(
        PredictorAssets,
        "from_default_dir",
        classmethod(
            lambda cls, **kwargs: PredictorAssets(
                checkpoint=tmp_path / "model.ckpt",
                esm_weights=tmp_path / "esm.pth",
                pca=tmp_path / "pca.pkl",
                asset_version="0.1.2",
            )
        ),
    )
    monkeypatch.setattr(ProtCrossPredictor, "from_files", staticmethod(fake_from_files))

    actual = ProtCrossPredictor.from_default_assets(
        device="cpu",
        accept_esm_license=False,
    )

    assert actual is expected
    assert captured_resolve["auto_setup_assets"] is False
    assert captured_resolve["offline"] is True
    assert captured_resolve["require_esm_license_for_use"] is True
    assert captured_resolve["ckpt_path"] == tmp_path / "model.ckpt"
    assert captured_files["asset_metadata"] == {"all_assets_verified": True}
    assert captured_files["accept_esm_license"] is True


def test_predict_pdb_forwards_resolved_asset_metadata(tmp_path, monkeypatch):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    paths = {
        "checkpoint": tmp_path / "model.ckpt",
        "esm_weights": tmp_path / "esm.pth",
        "pca": tmp_path / "pca.pkl",
    }
    metadata = {"asset_version": "custom", "checkpoint": {"actual_sha256": "abc"}}
    resolved = types.SimpleNamespace(
        **paths,
        asset_version="custom",
        asset_metadata=metadata,
    )
    captured = {}
    expected_result = object()

    class FakePredictor:
        def predict(self, *args, **kwargs):
            return expected_result

    def fake_from_files(**kwargs):
        captured.update(kwargs)
        return FakePredictor()

    monkeypatch.setattr("protcross.inference.predictor.resolve_prediction_assets", lambda *args, **kwargs: resolved)
    monkeypatch.setattr(ProtCrossPredictor, "from_files", staticmethod(fake_from_files))

    result = predict_pdb(input_pdb)

    assert result is expected_result
    assert captured["asset_version"] == "custom"
    assert captured["asset_metadata"] is metadata


def test_predictor_rejects_invalid_prediction_options_before_featurizing(tmp_path):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    esm = _CountingESM()
    pca = _CountingPCA()
    predictor = ProtCrossPredictor(
        device="cpu",
        max_len=4,
        esm_extractor=esm,
        pca_reducer=pca,
        structure_parser=_FakeParser(),
        model=_FakeModel(),
        asset_version="test-assets",
    )

    with pytest.raises(ValueError, match=re.escape("[0, 1]")):
        predictor.predict(input_pdb, threshold=1.1)
    with pytest.raises(ValueError, match="pocket_cluster_cutoff"):
        predictor.predict(input_pdb, pocket_cluster_cutoff=0)
    with pytest.raises(ValueError, match="unscored_bfactor_policy"):
        predictor.predict(input_pdb, unscored_bfactor_policy="drop")

    assert esm.calls == 0
    assert pca.calls == 0


def test_predictor_predict_many_matches_individual_predictions(tmp_path):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    predictor = ProtCrossPredictor(
        device="cpu",
        max_len=4,
        esm_extractor=_FakeESM(),
        pca_reducer=_FakePCA(),
        structure_parser=_FakeParser(),
        model=_VariableFakeModel(),
        asset_version="test-assets",
    )

    individual = [predictor.predict(input_pdb, threshold=0.5) for _ in range(2)]
    results = predictor.predict_many([input_pdb, input_pdb], threshold=0.5)

    assert len(results) == len(individual) == 2
    for batched, expected in zip(results, individual):
        assert batched.residue_ids == expected.residue_ids
        assert batched.residue_metadata == expected.residue_metadata
        np.testing.assert_allclose(batched.probabilities, expected.probabilities, rtol=1e-6, atol=1e-7)
        assert [record["is_binding"] for record in batched.to_records()] == [
            record["is_binding"] for record in expected.to_records()
        ]
        assert batched.runtime_metadata["execution_mode"] == "microbatch_fp32"
        assert batched.runtime_metadata["effective_microbatch_size"] == "2"


def test_predictor_predict_many_deduplicates_identical_chain_features(tmp_path):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    esm = _CountingESM()
    pca = _CountingPCA()
    predictor = ProtCrossPredictor(
        device="cpu",
        max_len=4,
        esm_extractor=esm,
        pca_reducer=pca,
        structure_parser=_FakeParser(),
        model=_VariableFakeModel(),
        asset_version="test-assets",
    )

    results = predictor.predict_many([input_pdb, input_pdb], batch_size=2)

    assert len(results) == 2
    assert esm.calls == 1
    assert pca.calls == 1


def test_predictor_predict_many_enforces_single_graph_batch_budgets(tmp_path):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    esm = _CountingESM()
    predictor = ProtCrossPredictor(
        device="cpu",
        max_len=4,
        esm_extractor=esm,
        pca_reducer=_FakePCA(),
        structure_parser=_FakeParser(),
        model=_FakeModel(),
    )

    results = predictor.predict_many(
        [input_pdb],
        max_batch_residues=1,
        max_batch_quadratic_cost=1,
        return_exceptions=True,
    )

    assert len(results) == 1
    assert isinstance(results[0], ValueError)
    assert "exceeding batch limits" in str(results[0])
    assert esm.calls == 0


def test_predictor_predict_many_rolls_back_failed_output_claims(tmp_path):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    first_output = tmp_path / "first.pdb"
    reusable_output = tmp_path / "reusable.pdb"
    predictor = ProtCrossPredictor(
        device="cpu",
        max_len=4,
        esm_extractor=_FakeESM(),
        pca_reducer=_FakePCA(),
        structure_parser=_FakeParser(),
        model=_VariableFakeModel(),
        asset_version="test-assets",
    )

    results = predictor.predict_many(
        [input_pdb, input_pdb, input_pdb],
        output_paths=[
            {"output_pdb": first_output},
            {"output_pdb": reusable_output, "scores_tsv": first_output},
            {"output_pdb": reusable_output},
        ],
        return_exceptions=True,
    )

    assert isinstance(results[0], PredictionResult)
    assert isinstance(results[1], ValueError)
    assert isinstance(results[2], PredictionResult)
    assert first_output.exists()
    assert reusable_output.exists()


def test_esm_padded_microbatches_match_single_sequence_extraction():
    extractor = object.__new__(ESMFeatureExtractor)
    extractor.device = "cpu"
    extractor.tokenizer = types.SimpleNamespace(pad_token_id=0)
    extractor._protein_cls = lambda sequence: types.SimpleNamespace(sequence=sequence)
    extractor.model = _TokenAwareESM()
    sequences = ["AG", "K", "MNPQR"]

    individual = [extractor.extract_residue_embeddings(sequence) for sequence in sequences]
    batched = extractor.extract_residue_embeddings_many(
        sequences,
        max_batch_size=3,
        max_padded_tokens=12,
    )

    assert len(batched) == len(individual)
    for actual, expected, sequence in zip(batched, individual, sequences):
        assert actual.shape == (len(sequence), 4)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_esm_padded_microbatches_enforce_budget_for_single_sequence():
    extractor = object.__new__(ESMFeatureExtractor)
    extractor.device = "cpu"
    extractor.tokenizer = types.SimpleNamespace(pad_token_id=0)
    extractor._protein_cls = lambda sequence: types.SimpleNamespace(sequence=sequence)
    extractor.model = _TokenAwareESM()

    with pytest.raises(ValueError, match="exceeding max_padded_tokens=4"):
        extractor.extract_residue_embeddings_many(
            ["AGK"],
            max_batch_size=1,
            max_padded_tokens=4,
        )


def test_esm_padded_microbatches_split_oom_and_preserve_order():
    extractor = object.__new__(ESMFeatureExtractor)
    extractor.device = "cpu"
    extractor.tokenizer = types.SimpleNamespace(pad_token_id=0)
    extractor._protein_cls = lambda sequence: types.SimpleNamespace(sequence=sequence)
    extractor.model = _OOMOnBatchESM()
    sequences = ["AG", "K", "MNP"]

    actual = extractor.extract_residue_embeddings_many(
        sequences,
        max_batch_size=3,
        max_padded_tokens=15,
    )

    expected_model = _TokenAwareESM()
    extractor.model = expected_model
    expected = [extractor.extract_residue_embeddings(sequence) for sequence in sequences]
    assert len(actual) == len(expected)
    for actual_item, expected_item in zip(actual, expected):
        torch.testing.assert_close(actual_item, expected_item, rtol=0, atol=0)


def test_predictor_rejects_invalid_max_len():
    with pytest.raises(ValueError, match="max_len"):
        ProtCrossPredictor(
            device="cpu",
            max_len=1023,
            esm_extractor=_FakeESM(),
            pca_reducer=_FakePCA(),
            structure_parser=_FakeParser(),
            model=_FakeModel(),
        )


def test_predictor_rejects_invalid_pca_dim():
    with pytest.raises(ValueError, match="pca_dim"):
        ProtCrossPredictor(
            device="cpu",
            pca_dim=0,
            esm_extractor=_FakeESM(),
            pca_reducer=_FakePCA(),
            structure_parser=_FakeParser(),
            model=_FakeModel(),
        )


def test_predictor_from_files_requires_esm_license_acceptance(tmp_path, monkeypatch):
    monkeypatch.delenv("PROTCROSS_ACCEPT_ESM_LICENSE", raising=False)
    ckpt = tmp_path / "model.ckpt"
    esm = tmp_path / "esm.pth"
    pca = tmp_path / "pca.pkl"
    for path in (ckpt, esm, pca):
        path.write_bytes(b"asset")

    with pytest.raises(RuntimeError, match="accept-esm-license"):
        ProtCrossPredictor.from_files(ckpt, esm, pca)


def test_predictor_from_assets_builds_traceable_metadata(tmp_path, monkeypatch):
    ckpt = tmp_path / "model.ckpt"
    esm = tmp_path / "esm.pth"
    pca = tmp_path / "pca.pkl"
    for path, payload in ((ckpt, b"checkpoint"), (esm, b"esm"), (pca, b"pca")):
        path.write_bytes(payload)
    assets = PredictorAssets(ckpt, esm, pca, asset_version="custom")
    captured = {}
    expected_predictor = object()

    def fake_from_files(**kwargs):
        captured.update(kwargs)
        return expected_predictor

    monkeypatch.setattr(ProtCrossPredictor, "from_files", staticmethod(fake_from_files))

    predictor = ProtCrossPredictor.from_assets(assets)

    assert predictor is expected_predictor
    metadata = captured["asset_metadata"]
    assert metadata
    assert metadata["asset_version"] == "custom"
    assert metadata["contains_unverified_assets"] is True
    for name, path in (("checkpoint", ckpt), ("esm_weights", esm), ("pca", pca)):
        assert metadata[name]["path"] == str(path.resolve())
        assert metadata[name]["actual_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert metadata[name]["verified"] is None
        assert metadata[name]["source"] == "predictor_assets"


def test_predictor_from_files_runs_offline_real_prediction_path(tmp_path, monkeypatch):
    _install_fake_esm_modules(monkeypatch)
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    ckpt = tmp_path / "model.ckpt"
    esm = tmp_path / "esmc_600m_2024_12_v0.pth"
    pca = tmp_path / "pca.pkl"
    output_pdb = tmp_path / "out.pdb"
    scores_tsv = tmp_path / "scores.tsv"
    pocket_json = tmp_path / "pockets.json"
    summary_json = tmp_path / "summary.json"
    ckpt.write_bytes(b"checkpoint")
    torch.save({"state_dict": {}}, esm)
    with pca.open("wb") as handle:
        pickle.dump(_OfflineSmokePCA(), handle)

    monkeypatch.setattr(
        "protcross.inference.predictor.EvoPointDALitModule.load_from_checkpoint",
        lambda *args, **kwargs: _VariableFakeModel(),
    )

    predictor = ProtCrossPredictor.from_files(
        ckpt,
        esm,
        pca,
        device="cpu",
        pca_dim=2,
        max_len=8,
        accept_esm_license=True,
    )
    result = predictor.predict(
        input_pdb,
        threshold=0.5,
        output_pdb=output_pdb,
        scores_tsv=scores_tsv,
        pocket_json=pocket_json,
        summary_json=summary_json,
    )

    assert result.residue_ids == ["A_1", "A_2"]
    assert output_pdb.exists()
    assert scores_tsv.exists()
    pocket_payload = json.loads(pocket_json.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert pocket_payload["schema_version"] == "protcross-pocket-v2"
    assert summary_payload["schema_version"] == "protcross-summary-v2"
    assert summary_payload["assets"] == pocket_payload["assets"]
    assert summary_payload["input_file"]["sha256"] == hashlib.sha256(input_pdb.read_bytes()).hexdigest()
    assert summary_payload["input_file"] == pocket_payload["input_file"]
    assert summary_payload["runtime"]["python"]
    assert summary_payload["runtime"]["torch"]
    assert summary_payload["assets"]
    for name, path in (("checkpoint", ckpt), ("esm_weights", esm), ("pca", pca)):
        entry = summary_payload["assets"][name]
        assert entry["path"] == str(path.resolve())
        assert entry["actual_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry["verified"] is None
        assert entry["source"] == "direct"


def test_predictor_embedding_cache_reuses_reduced_features(tmp_path):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    esm = _CountingESM()
    pca = _CountingPCA()
    predictor = ProtCrossPredictor(
        device="cpu",
        max_len=4,
        embedding_cache_dir=tmp_path / "feature-cache",
        feature_pipeline_fingerprint="counting-esm-pca-v1",
        esm_extractor=esm,
        pca_reducer=pca,
        structure_parser=_FakeParser(),
        model=_FakeModel(),
        asset_version="test-assets",
    )

    predictor.predict(input_pdb, threshold=0.5)
    predictor.predict(input_pdb, threshold=0.5)

    assert esm.calls == 1
    assert pca.calls == 1
    assert list((tmp_path / "feature-cache").glob("*.pt"))


def test_injected_feature_pipeline_requires_explicit_cache_fingerprint(tmp_path):
    with pytest.raises(ValueError, match="feature_pipeline_fingerprint"):
        ProtCrossPredictor(
            device="cpu",
            embedding_cache_dir=tmp_path / "feature-cache",
            esm_extractor=_FakeESM(),
            pca_reducer=_FakePCA(),
            structure_parser=_FakeParser(),
            model=_FakeModel(),
        )


def test_injected_feature_pipeline_fingerprint_isolates_cache_namespaces(tmp_path):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    cache_dir = tmp_path / "feature-cache"
    extractors = []
    for fingerprint in ("pipeline-a", "pipeline-b"):
        esm = _CountingESM()
        extractors.append(esm)
        predictor = ProtCrossPredictor(
            device="cpu",
            max_len=4,
            embedding_cache_dir=cache_dir,
            feature_pipeline_fingerprint=fingerprint,
            esm_extractor=esm,
            pca_reducer=_FakePCA(),
            structure_parser=_FakeParser(),
            model=_FakeModel(),
            asset_version="test-assets",
        )
        predictor.predict(input_pdb)

    assert [extractor.calls for extractor in extractors] == [1, 1]
    assert len(list(cache_dir.glob("*.pt"))) == 2


def test_predictor_rebuilds_corrupt_embedding_cache(tmp_path):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    cache_dir = tmp_path / "feature-cache"
    esm = _CountingESM()
    pca = _CountingPCA()
    predictor = ProtCrossPredictor(
        device="cpu",
        max_len=4,
        embedding_cache_dir=cache_dir,
        feature_pipeline_fingerprint="counting-esm-pca-v1",
        esm_extractor=esm,
        pca_reducer=pca,
        structure_parser=_FakeParser(),
        model=_FakeModel(),
        asset_version="test-assets",
    )
    predictor.predict(input_pdb)
    cache_path = next(cache_dir.glob("*.pt"))
    cache_path.write_bytes(b"truncated-cache")

    predictor.predict(input_pdb)
    predictor.predict(input_pdb)

    assert esm.calls == 2
    assert pca.calls == 2
    assert not list(cache_dir.glob("*.part.pt"))


@pytest.mark.parametrize(
    "cached",
    [
        torch.ones((1, 2), dtype=torch.float32),
        torch.tensor([[float("nan"), 1.0], [1.0, 1.0]], dtype=torch.float32),
    ],
)
def test_predictor_rebuilds_invalid_embedding_cache_tensor(tmp_path, cached):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    cache_dir = tmp_path / "feature-cache"
    esm = _CountingESM()
    predictor = ProtCrossPredictor(
        device="cpu",
        max_len=4,
        embedding_cache_dir=cache_dir,
        feature_pipeline_fingerprint="counting-esm-pca-v1",
        esm_extractor=esm,
        pca_reducer=_FakePCA(),
        structure_parser=_FakeParser(),
        model=_FakeModel(),
        asset_version="test-assets",
    )
    predictor.predict(input_pdb)
    cache_path = next(cache_dir.glob("*.pt"))
    torch.save(cached, cache_path)

    predictor.predict(input_pdb)

    assert esm.calls == 2
    rebuilt = torch.load(cache_path, map_location="cpu", weights_only=True)
    assert rebuilt.shape == (2, 2)
    assert torch.isfinite(rebuilt).all()


def test_predictor_rejects_loaded_pca_dimension_mismatch(tmp_path):
    pca_path = tmp_path / "pca.pkl"
    with pca_path.open("wb") as handle:
        pickle.dump(types.SimpleNamespace(n_components=2), handle)

    with pytest.raises(ValueError, match="has 2 components"):
        ProtCrossPredictor._load_pca(pca_path, 128)


def test_predictor_embedding_cache_key_includes_asset_identity(tmp_path):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    cache_dir = tmp_path / "feature-cache"
    esm_path_a = tmp_path / "esm-a.pth"
    esm_path_b = tmp_path / "esm-b.pth"
    pca_path_a = tmp_path / "pca-a.pkl"
    pca_path_b = tmp_path / "pca-b.pkl"
    for path, payload in (
        (esm_path_a, b"esm-a"),
        (esm_path_b, b"esm-b"),
        (pca_path_a, b"pca-a"),
        (pca_path_b, b"pca-b"),
    ):
        path.write_bytes(payload)

    esm_a = _CountingESM()
    pca_a = _CountingPCA()
    predictor_a = ProtCrossPredictor(
        ckpt_path=None,
        esm_weights=esm_path_a,
        pca_path=pca_path_a,
        device="cpu",
        max_len=4,
        embedding_cache_dir=cache_dir,
        feature_pipeline_fingerprint="counting-esm-pca-v1",
        esm_extractor=esm_a,
        pca_reducer=pca_a,
        structure_parser=_FakeParser(),
        model=_FakeModel(),
        asset_version="custom",
    )
    predictor_a.predict(input_pdb, threshold=0.5)

    esm_b = _CountingESM()
    pca_b = _CountingPCA()
    predictor_b = ProtCrossPredictor(
        ckpt_path=None,
        esm_weights=esm_path_b,
        pca_path=pca_path_b,
        device="cpu",
        max_len=4,
        embedding_cache_dir=cache_dir,
        feature_pipeline_fingerprint="counting-esm-pca-v1",
        esm_extractor=esm_b,
        pca_reducer=pca_b,
        structure_parser=_FakeParser(),
        model=_FakeModel(),
        asset_version="custom",
    )
    predictor_b.predict(input_pdb, threshold=0.5)

    assert esm_a.calls == 1
    assert pca_a.calls == 1
    assert esm_b.calls == 1
    assert pca_b.calls == 1
    assert len(list(cache_dir.glob("*.pt"))) == 2


def test_predictor_featurizes_multichain_inputs_per_chain(tmp_path):
    input_pdb = tmp_path / "multi.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    (tmp_path / "esm.pth").write_bytes(b"esm")
    (tmp_path / "pca.pkl").write_bytes(b"pca")
    esm = _RecordingESM()
    predictor = ProtCrossPredictor(
        ckpt_path=None,
        esm_weights=tmp_path / "esm.pth",
        pca_path=tmp_path / "pca.pkl",
        device="cpu",
        max_len=2,
        esm_extractor=esm,
        pca_reducer=_FakePCA(),
        structure_parser=_FakeMultiChainParser(),
        model=_VariableFakeModel(),
        asset_version="custom",
    )

    result = predictor.predict(input_pdb, threshold=0.5)

    assert esm.sequences == ["AG", "K"]
    assert result.residue_ids == ["A_1", "A_2", "B_1"]


def test_predictor_truncates_each_chain_independently(tmp_path):
    input_pdb = tmp_path / "multi.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    esm = _RecordingESM()
    predictor = ProtCrossPredictor(
        device="cpu",
        max_len=2,
        esm_extractor=esm,
        pca_reducer=_FakePCA(),
        structure_parser=_FakeLongMultiChainParser(),
        model=_VariableFakeModel(),
        asset_version="custom",
    )

    result = predictor.predict(input_pdb, threshold=0.5, allow_truncation=True)

    assert esm.sequences == ["AG", "KL"]
    assert result.truncated is True
    assert result.original_length == 6
    assert result.residue_ids == ["A_1", "A_2", "B_1", "B_2"]


class _FakeParser:
    def parse_file_with_labels(self, file_path, chain_id=None):
        return {
            "coords": np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype=np.float32),
            "raw_coords": np.array([[12.56, 13.12, 9.327], [15.56, 14.92, 9.827]], dtype=np.float32),
            "sequence": "AG",
            "plddts": np.array([20.0, 30.0], dtype=np.float32),
            "residue_ids": ["A_1", "A_2"],
            "residue_metadata": [
                {
                    "residue_id": "A_1",
                    "residue_key": "model:0|chain:A|het:ATOM|resseq:1|icode:|resname:ALA",
                    "residue_id_namespace": "pdb",
                    "model_id": "0",
                    "chain_id": "A",
                    "auth_asym_id": "A",
                    "label_asym_id": None,
                    "residue_number": 1,
                    "auth_seq_id": 1,
                    "label_seq_id": None,
                    "insertion_code": "",
                    "resname": "ALA",
                    "one_letter_code": "A",
                    "input_bfactor": 20.0,
                },
                {
                    "residue_id": "A_2",
                    "residue_key": "model:0|chain:A|het:ATOM|resseq:2|icode:|resname:GLY",
                    "residue_id_namespace": "pdb",
                    "model_id": "0",
                    "chain_id": "A",
                    "auth_asym_id": "A",
                    "label_asym_id": None,
                    "residue_number": 2,
                    "auth_seq_id": 2,
                    "label_seq_id": None,
                    "insertion_code": "",
                    "resname": "GLY",
                    "one_letter_code": "G",
                    "input_bfactor": 30.0,
                },
            ],
            "labels": np.array([1.0, 0.0], dtype=np.float32),
            "truncated": False,
            "original_length": 2,
        }


class _FakeESM:
    def extract_residue_embeddings(self, sequence):
        return torch.ones((len(sequence), 4), dtype=torch.float32)


class _TokenAwareESM:
    def encode(self, protein):
        residue_tokens = [3 + (ord(residue) % 20) for residue in protein.sequence]
        return types.SimpleNamespace(sequence=torch.tensor([1, *residue_tokens, 2], dtype=torch.long))

    def __call__(self, sequence_tokens):
        token_values = sequence_tokens.to(dtype=torch.float32)
        positions = torch.arange(sequence_tokens.shape[1], dtype=torch.float32).expand_as(token_values)
        embeddings = torch.stack(
            (
                token_values,
                token_values.square(),
                positions,
                (sequence_tokens != 0).to(dtype=torch.float32),
            ),
            dim=-1,
        )
        return types.SimpleNamespace(embeddings=embeddings)


class _OOMOnBatchESM(_TokenAwareESM):
    def __call__(self, sequence_tokens):
        if sequence_tokens.shape[0] > 1:
            raise RuntimeError("simulated accelerator out of memory")
        return super().__call__(sequence_tokens)


class _RecordingESM(_FakeESM):
    def __init__(self):
        self.sequences = []

    def extract_residue_embeddings(self, sequence):
        self.sequences.append(sequence)
        return super().extract_residue_embeddings(sequence)


class _FakePCA:
    def transform(self, embeddings):
        return embeddings[:, :2]


class _CountingESM(_FakeESM):
    def __init__(self):
        self.calls = 0

    def extract_residue_embeddings(self, sequence):
        self.calls += 1
        return super().extract_residue_embeddings(sequence)


class _CountingPCA(_FakePCA):
    def __init__(self):
        self.calls = 0

    def transform(self, embeddings):
        self.calls += 1
        return super().transform(embeddings)


class _FakeModel:
    hparams = type("HParams", (), {"use_esm": True})()

    def eval(self):
        return self

    def to(self, device):
        return self

    def backbone(self, x, pos, batch):
        return x, None

    def seg_head(self, feats):
        return torch.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=torch.float32)


class _VariableFakeModel(_FakeModel):
    def seg_head(self, feats):
        logits = torch.zeros((feats.shape[0], 2), dtype=torch.float32)
        logits[:, 1] = 2.0
        return logits


class _FakeMultiChainParser:
    def parse_file_with_labels(self, file_path, chain_id=None):
        return {
            "coords": np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [16.0, 0.0, 0.0]], dtype=np.float32),
            "raw_coords": np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [16.0, 0.0, 0.0]], dtype=np.float32),
            "sequence": "AGK",
            "plddts": np.array([20.0, 30.0, 40.0], dtype=np.float32),
            "residue_ids": ["A_1", "A_2", "B_1"],
            "residue_metadata": [
                _fake_residue_metadata("A_1", "A", 1, "ALA", "A"),
                _fake_residue_metadata("A_2", "A", 2, "GLY", "G"),
                _fake_residue_metadata("B_1", "B", 1, "LYS", "K"),
            ],
            "labels": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "truncated": False,
            "original_length": 3,
        }


class _FakeLongMultiChainParser:
    def parse_file_with_labels(self, file_path, chain_id=None):
        return {
            "coords": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [11.0, 0.0, 0.0],
                    [12.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            "raw_coords": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [11.0, 0.0, 0.0],
                    [12.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            "sequence": "AGHKLM",
            "plddts": np.array([20.0, 21.0, 22.0, 30.0, 31.0, 32.0], dtype=np.float32),
            "residue_ids": ["A_1", "A_2", "A_3", "B_1", "B_2", "B_3"],
            "residue_metadata": [
                _fake_residue_metadata("A_1", "A", 1, "ALA", "A"),
                _fake_residue_metadata("A_2", "A", 2, "GLY", "G"),
                _fake_residue_metadata("A_3", "A", 3, "HIS", "H"),
                _fake_residue_metadata("B_1", "B", 1, "LYS", "K"),
                _fake_residue_metadata("B_2", "B", 2, "LEU", "L"),
                _fake_residue_metadata("B_3", "B", 3, "MET", "M"),
            ],
            "labels": np.zeros(6, dtype=np.float32),
            "truncated": False,
            "original_length": 6,
        }


def _fake_residue_metadata(residue_id, chain_id, residue_number, resname, one_letter_code):
    return {
        "residue_id": residue_id,
        "residue_key": f"model:0|chain:{chain_id}|het:ATOM|resseq:{residue_number}|icode:|resname:{resname}",
        "residue_id_namespace": "pdb",
        "model_id": "0",
        "chain_id": chain_id,
        "auth_asym_id": chain_id,
        "label_asym_id": None,
        "residue_number": residue_number,
        "auth_seq_id": residue_number,
        "label_seq_id": None,
        "insertion_code": "",
        "resname": resname,
        "one_letter_code": one_letter_code,
        "input_bfactor": 20.0,
    }


class _OfflineSmokePCA:
    def transform(self, array):
        return np.asarray(array, dtype=np.float32)[:, :2]


def _install_fake_esm_modules(monkeypatch):
    esm_module = types.ModuleType("esm")
    models_module = types.ModuleType("esm.models")
    esmc_module = types.ModuleType("esm.models.esmc")
    sdk_module = types.ModuleType("esm.sdk")
    api_module = types.ModuleType("esm.sdk.api")
    tokenization_module = types.ModuleType("esm.tokenization")

    class FakeTokenizer:
        def __init__(self, *args, **kwargs):
            pass

    class FakeProtein:
        def __init__(self, sequence):
            self.sequence = sequence

    class FakeESMC:
        def __init__(self, tokenizer=None, **kwargs):
            self.tokenizer = tokenizer

        def load_state_dict(self, state_dict, strict=False):
            return types.SimpleNamespace(missing_keys=[], unexpected_keys=[])

        def to(self, device):
            return self

        def eval(self):
            return self

        def encode(self, protein):
            return types.SimpleNamespace(sequence=torch.arange(len(protein.sequence) + 2, dtype=torch.long))

        def __call__(self, input_ids, attention_mask=None):
            length = int(input_ids.shape[1])
            values = torch.arange(length * 4, dtype=torch.float32).reshape(1, length, 4)
            return types.SimpleNamespace(embeddings=values)

    esmc_module.ESMC = FakeESMC
    api_module.ESMProtein = FakeProtein
    tokenization_module.EsmSequenceTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "esm", esm_module)
    monkeypatch.setitem(sys.modules, "esm.models", models_module)
    monkeypatch.setitem(sys.modules, "esm.models.esmc", esmc_module)
    monkeypatch.setitem(sys.modules, "esm.sdk", sdk_module)
    monkeypatch.setitem(sys.modules, "esm.sdk.api", api_module)
    monkeypatch.setitem(sys.modules, "esm.tokenization", tokenization_module)


def _trust_managed_asset_hashes(monkeypatch):
    expected_by_name = {spec.filename: spec.sha256 for spec in DEFAULT_ASSET_BUNDLE.assets}
    monkeypatch.setattr(
        "protcross.assets.sha256_file",
        lambda path: expected_by_name.get(Path(path).name)
        or hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    )
