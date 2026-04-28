from pathlib import Path

from evopoint_da.assets import sha256_file, write_env_file
from evopoint_da.inference import PredictorAssets


def test_asset_sha256_and_env_file(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"protcross")

    assert sha256_file(payload) == "db69f4e2069decf9bdc08ce84d965868d32b4968213f025aac5a0d3ab9d77421"

    write_env_file(tmp_path)
    env_text = (tmp_path / "protcross.env").read_text(encoding="utf-8")

    assert "PROTCROSS_ASSETS_DIR" in env_text
    assert "PROTCROSS_CHECKPOINT" in env_text
    assert "PROTCROSS_ESM_WEIGHTS" in env_text
    assert "PROTCROSS_PCA" in env_text


def test_predictor_assets_completion(tmp_path):
    assets = PredictorAssets.from_dir(tmp_path)

    assert not assets.is_complete()
    assert Path(tmp_path / "best-epoch=59.ckpt") in assets.missing_files()

    assets.checkpoint.write_bytes(b"checkpoint")
    assets.esm_weights.write_bytes(b"esm")
    assets.pca.write_bytes(b"pca")

    assert assets.is_complete()
    assert assets.missing_files() == []
