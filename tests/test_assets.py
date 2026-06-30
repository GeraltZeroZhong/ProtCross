from pathlib import Path
import json

import pytest

from protcross.assets import (
    ASSET_MANIFEST_FILENAME,
    AssetSpec,
    DEFAULT_CHECKPOINT_FILENAME,
    DEFAULT_CHECKPOINT_URL,
    DEFAULT_PCA_FILENAME,
    DEFAULT_ASSET_BUNDLE,
    LEGACY_CHECKPOINT_FILENAME,
    LEGACY_CHECKPOINT_URL,
    download_asset,
    resolve_prediction_assets,
    setup_assets,
    sha256_file,
    write_env_file,
)
from protcross.inference import PredictorAssets


def test_asset_sha256_and_env_file(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"protcross")

    assert sha256_file(payload) == "db69f4e2069decf9bdc08ce84d965868d32b4968213f025aac5a0d3ab9d77421"

    write_env_file(tmp_path)
    env_text = (tmp_path / "protcross.env").read_text(encoding="utf-8")

    assert "PROTCROSS_ASSETS_DIR" in env_text
    assert "PROTCROSS_ASSET_VERSION" in env_text
    assert "PROTCROSS_CHECKPOINT" in env_text
    assert "PROTCROSS_ESM_WEIGHTS" in env_text
    assert "PROTCROSS_PCA" in env_text


def test_default_checkpoint_url_matches_github_release_asset_name():
    assert DEFAULT_CHECKPOINT_URL.endswith(f"/{DEFAULT_CHECKPOINT_FILENAME}")


def test_legacy_checkpoint_url_uses_release_asset_name_but_keeps_local_filename():
    assert LEGACY_CHECKPOINT_FILENAME == "best-epoch=59.ckpt"
    assert LEGACY_CHECKPOINT_URL.endswith("/best-epoch.59.ckpt")


def test_predictor_assets_completion(tmp_path):
    assets = PredictorAssets.from_dir(tmp_path)

    assert not assets.is_complete()
    assert Path(tmp_path / DEFAULT_CHECKPOINT_FILENAME) in assets.missing_files()

    assets.checkpoint.write_bytes(b"checkpoint")
    assets.esm_weights.write_bytes(b"esm")
    assets.pca.write_bytes(b"pca")

    assert assets.is_complete()
    assert assets.missing_files() == []


def test_setup_assets_writes_manifest_without_network(tmp_path, monkeypatch):
    downloaded = []

    def fake_download(spec, output_path, *, force=False, verify=True):
        downloaded.append((spec.filename, force, verify))
        output_path.write_bytes(b"asset")

    monkeypatch.setattr("protcross.assets.download_asset", fake_download)

    output_dir = setup_assets(tmp_path, force=True, accept_esm_license=True)

    manifest = json.loads((output_dir / ASSET_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "protcross-assets-v1"
    assert manifest["package_version"] == "0.1.3"
    assert manifest["asset_version"] == "0.1.2"
    assert manifest["asset_bundle_version"] == "0.1.2"
    assert manifest["checkpoint_filename"] == DEFAULT_CHECKPOINT_FILENAME
    assert manifest["pca_filename"] == DEFAULT_PCA_FILENAME
    checkpoint_entry = manifest["files"]["ProtCross checkpoint"]
    assert checkpoint_entry["expected_sha256"]
    assert checkpoint_entry["actual_sha256"] == sha256_file(output_dir / DEFAULT_CHECKPOINT_FILENAME)
    assert checkpoint_entry["size_bytes"] == len(b"asset")
    assert checkpoint_entry["present"] is True
    assert {item[0] for item in downloaded} == {
        "esmc_600m_2024_12_v0.pth",
        DEFAULT_CHECKPOINT_FILENAME,
        DEFAULT_PCA_FILENAME,
    }


def test_manifestless_legacy_assets_are_detected(tmp_path):
    (tmp_path / LEGACY_CHECKPOINT_FILENAME).write_bytes(b"checkpoint")
    (tmp_path / "pca_esmc_128.pkl").write_bytes(b"pca")
    (tmp_path / "esmc_600m_2024_12_v0.pth").write_bytes(b"esm")

    assets = PredictorAssets.from_dir(tmp_path)

    assert assets.asset_version == "0.1.1-paper"
    assert assets.checkpoint == tmp_path / LEGACY_CHECKPOINT_FILENAME
    assert assets.pca == tmp_path / "pca_esmc_128.pkl"


def test_resolve_refresh_assets_refreshes_managed_esm(tmp_path, monkeypatch):
    for spec in DEFAULT_ASSET_BUNDLE.assets:
        (tmp_path / spec.filename).write_text(spec.sha256 or "asset", encoding="utf-8")
    calls = []

    def fake_setup_assets(output_dir=None, **kwargs):
        calls.append(kwargs)
        output_dir = Path(output_dir) if output_dir else tmp_path
        output_dir.mkdir(parents=True, exist_ok=True)
        for spec in DEFAULT_ASSET_BUNDLE.assets:
            (output_dir / spec.filename).write_text(spec.sha256 or "asset", encoding="utf-8")
        (output_dir / ASSET_MANIFEST_FILENAME).write_text('{"asset_version": "0.1.2"}', encoding="utf-8")
        return output_dir

    expected_by_name = {spec.filename: spec.sha256 for spec in DEFAULT_ASSET_BUNDLE.assets}
    monkeypatch.setattr("protcross.assets.setup_assets", fake_setup_assets)
    monkeypatch.setattr("protcross.assets.sha256_file", lambda path: expected_by_name[Path(path).name])

    resolve_prediction_assets(assets_dir=tmp_path, refresh_assets=True)

    assert calls == [{"asset_version": "default", "force": True, "skip_esm": False, "accept_esm_license": False}]


def test_resolve_refresh_assets_skips_explicit_external_esm(tmp_path, monkeypatch):
    for spec in (DEFAULT_ASSET_BUNDLE.assets[1], DEFAULT_ASSET_BUNDLE.assets[2]):
        (tmp_path / spec.filename).write_text(spec.sha256 or "asset", encoding="utf-8")
    explicit_esm = tmp_path / "external-esm.pth"
    explicit_esm.write_bytes(b"external")
    calls = []

    def fake_setup_assets(output_dir=None, **kwargs):
        calls.append(kwargs)
        output_dir = Path(output_dir) if output_dir else tmp_path
        output_dir.mkdir(parents=True, exist_ok=True)
        for spec in (DEFAULT_ASSET_BUNDLE.assets[1], DEFAULT_ASSET_BUNDLE.assets[2]):
            (output_dir / spec.filename).write_text(spec.sha256 or "asset", encoding="utf-8")
        (output_dir / ASSET_MANIFEST_FILENAME).write_text('{"asset_version": "0.1.2"}', encoding="utf-8")
        return output_dir

    expected_by_name = {spec.filename: spec.sha256 for spec in DEFAULT_ASSET_BUNDLE.assets}
    monkeypatch.setattr("protcross.assets.setup_assets", fake_setup_assets)
    monkeypatch.setattr("protcross.assets.sha256_file", lambda path: expected_by_name[Path(path).name])

    resolved = resolve_prediction_assets(
        esm_weights=explicit_esm,
        assets_dir=tmp_path,
        refresh_assets=True,
        trust_unverified_assets=True,
    )

    assert calls == [{"asset_version": "default", "force": True, "skip_esm": True, "accept_esm_license": False}]
    assert resolved.esm_weights == explicit_esm
    assert resolved.asset_version == "custom"


def test_resolve_rejects_bad_managed_assets_without_auto_download(tmp_path, monkeypatch):
    for spec in DEFAULT_ASSET_BUNDLE.assets:
        (tmp_path / spec.filename).write_bytes(b"bad")

    with pytest.raises(RuntimeError, match="SHA256"):
        resolve_prediction_assets(assets_dir=tmp_path, auto_setup_assets=False, offline=True)


def test_resolve_rejects_forged_verified_manifest_for_managed_assets(tmp_path):
    files = {}
    for spec in DEFAULT_ASSET_BUNDLE.assets:
        path = tmp_path / spec.filename
        path.write_text("manifest-verified-placeholder", encoding="utf-8")
        files[spec.name] = {
            "filename": spec.filename,
            "expected_sha256": spec.sha256,
            "actual_sha256": spec.sha256,
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "verified": True,
        }
    (tmp_path / ASSET_MANIFEST_FILENAME).write_text(
        json.dumps({"asset_version": "0.1.2", "files": files}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="SHA256"):
        resolve_prediction_assets(assets_dir=tmp_path, auto_setup_assets=False)


def test_resolve_labels_all_explicit_assets_as_custom(tmp_path):
    ckpt = tmp_path / "model.ckpt"
    esm = tmp_path / "esm.pth"
    pca = tmp_path / "pca.pkl"
    for path in (ckpt, esm, pca):
        path.write_bytes(b"asset")

    with pytest.raises(RuntimeError, match="trust-unverified-assets"):
        resolve_prediction_assets(ckpt, esm, pca, auto_setup_assets=False)

    resolved = resolve_prediction_assets(
        ckpt,
        esm,
        pca,
        auto_setup_assets=False,
        trust_unverified_assets=True,
    )

    assert resolved.asset_version == "custom"


def test_setup_assets_requires_esm_license_acceptance(tmp_path, monkeypatch):
    def fail_download(*args, **kwargs):
        raise AssertionError("download should not start before license acceptance")

    monkeypatch.setattr("protcross.assets.download_asset", fail_download)

    with pytest.raises(RuntimeError, match="accept-esm-license"):
        setup_assets(tmp_path, force=True)


def test_resolve_explicit_missing_asset_fails_before_auto_setup(tmp_path, monkeypatch):
    def fail_setup_assets(*args, **kwargs):
        raise AssertionError("setup_assets should not run for a missing explicit path")

    monkeypatch.setattr("protcross.assets.setup_assets", fail_setup_assets)

    with pytest.raises(FileNotFoundError, match="--checkpoint"):
        resolve_prediction_assets(ckpt_path=tmp_path / "missing.ckpt", auto_setup_assets=True)


def test_download_asset_sha_mismatch_removes_part_file(tmp_path, monkeypatch):
    class FakeResponse:
        headers = {"content-length": "3"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"bad"

    monkeypatch.setattr("protcross.assets._http_get", lambda *args, **kwargs: FakeResponse())
    output_path = tmp_path / "asset.bin"

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        download_asset(
            AssetSpec("bad asset", "asset.bin", "https://example.invalid/asset.bin", sha256="0" * 64),
            output_path,
        )

    assert not output_path.exists()
    assert not output_path.with_suffix(".bin.part").exists()
