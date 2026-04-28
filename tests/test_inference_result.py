from pathlib import Path

import numpy as np
import pytest

from evopoint_da.inference import PredictionResult, PredictorAssets
from evopoint_da.inference.predictor import _resolve_predict_pdb_assets


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


def test_predictor_assets_from_dir(tmp_path):
    assets = PredictorAssets.from_dir(tmp_path)

    assert assets.checkpoint == tmp_path / "best-epoch=59.ckpt"
    assert assets.esm_weights == tmp_path / "esmc_600m_2024_12_v0.pth"
    assert assets.pca == tmp_path / "pca_esmc_128.pkl"


def test_predict_pdb_asset_resolution_uses_assets_dir(tmp_path):
    ckpt, esm, pca = _resolve_predict_pdb_assets(None, None, None, assets_dir=tmp_path)

    assert ckpt == tmp_path / "best-epoch=59.ckpt"
    assert esm == tmp_path / "esmc_600m_2024_12_v0.pth"
    assert pca == tmp_path / "pca_esmc_128.pkl"


def test_predict_pdb_asset_resolution_reports_missing(tmp_path, monkeypatch):
    for name in ("PROTCROSS_CHECKPOINT", "PROTCROSS_ESM_WEIGHTS", "PROTCROSS_PCA"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PROTCROSS_ASSETS_DIR", str(tmp_path / "empty-assets"))

    with pytest.raises(ValueError, match="Run `protcross setup-assets`"):
        _resolve_predict_pdb_assets(None, None, None)
