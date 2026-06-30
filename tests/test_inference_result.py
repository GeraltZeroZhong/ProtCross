from pathlib import Path
import json
import re

import numpy as np
import pytest
import torch

from protcross.assets import (
    ASSET_MANIFEST_FILENAME,
    DEFAULT_ASSET_BUNDLE,
    DEFAULT_CHECKPOINT_FILENAME,
    DEFAULT_PCA_FILENAME,
)
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
    assert "Predicted binding residues: 1" in result.format_summary()
    assert result.to_records()[0]["chain_id"] == "A"


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

    assert payload["schema_version"] == "protcross-pocket-v1"
    assert payload["coordinate_units"] == "angstrom"
    assert payload["center_type"] == "probability_weighted_ca_centroid"
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
    assert lines[1].split("\t")[13:19] == ["20.000000", "0.700000", "1", "1.000000", "2.000000", "3.000000"]
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
    assert summary["schema_version"] == "protcross-summary-v1"
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
    )
    for kwargs in invalid_calls:
        with pytest.raises(ValueError):
            predict_pdb(input_pdb, **kwargs)


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


def test_predictor_predict_many_reuses_fake_components(tmp_path):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    predictor = ProtCrossPredictor(
        device="cpu",
        max_len=4,
        esm_extractor=_FakeESM(),
        pca_reducer=_FakePCA(),
        structure_parser=_FakeParser(),
        model=_FakeModel(),
        asset_version="test-assets",
    )

    results = predictor.predict_many([input_pdb, input_pdb], threshold=0.5)

    assert [result.binding_residues[0].residue_id for result in results] == ["A_1", "A_1"]


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


def test_predictor_embedding_cache_reuses_reduced_features(tmp_path):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    esm = _CountingESM()
    pca = _CountingPCA()
    predictor = ProtCrossPredictor(
        device="cpu",
        max_len=4,
        embedding_cache_dir=tmp_path / "feature-cache",
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

    def backbone(self, x, pos, batch):
        return x, None

    def seg_head(self, feats):
        return torch.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=torch.float32)


def _trust_managed_asset_hashes(monkeypatch):
    expected_by_name = {spec.filename: spec.sha256 for spec in DEFAULT_ASSET_BUNDLE.assets}
    monkeypatch.setattr("protcross.assets.sha256_file", lambda path: expected_by_name[Path(path).name])
