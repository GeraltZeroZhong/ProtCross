import hashlib
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
import protcross.assets as assets_module

from protcross import __version__
from protcross.assets import (
    ASSET_MANIFEST_FILENAME,
    AssetSpec,
    DEFAULT_CHECKPOINT_FILENAME,
    DEFAULT_CHECKPOINT_URL,
    DEFAULT_PCA_FILENAME,
    DEFAULT_ASSET_BUNDLE,
    DownloadCancelled,
    LEGACY_ASSET_BUNDLE,
    LEGACY_CHECKPOINT_FILENAME,
    LEGACY_CHECKPOINT_URL,
    download_asset,
    resolve_prediction_assets,
    setup_assets,
    sha256_file,
    write_env_file,
)
from protcross.inference import PredictorAssets


class _FakeResponse:
    def __init__(self, chunks, *, status_code=200, content_length=None, content_range=None):
        self._chunks = chunks
        self.status_code = status_code
        self.headers = {}
        if content_length is not None:
            self.headers["content-length"] = str(content_length)
        if content_range is not None:
            self.headers["content-range"] = str(content_range)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        assert chunk_size == 1024 * 1024
        yield from self._chunks


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


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

    def fake_download(spec, output_path, *, force=False, verify=True, known_sha256=None):
        downloaded.append((spec.filename, force, verify))
        output_path.write_bytes(b"asset")

    monkeypatch.setattr("protcross.assets.download_asset", fake_download)

    output_dir = setup_assets(tmp_path, force=True, accept_esm_license=True)

    manifest = json.loads((output_dir / ASSET_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "protcross-assets-v1"
    assert manifest["package_version"] == __version__
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


def test_setup_assets_hashes_each_existing_asset_once(tmp_path, monkeypatch):
    expected_by_name = {spec.filename: spec.sha256 for spec in DEFAULT_ASSET_BUNDLE.assets}
    for spec in DEFAULT_ASSET_BUNDLE.assets:
        (tmp_path / spec.filename).write_bytes(spec.filename.encode("utf-8"))
    hash_calls = []

    def fake_sha256(path):
        hash_calls.append(Path(path).name)
        return expected_by_name[Path(path).name]

    monkeypatch.setattr("protcross.assets.sha256_file", fake_sha256)

    setup_assets(tmp_path, accept_esm_license=True)

    assert sorted(hash_calls) == sorted(expected_by_name)


def test_manifestless_legacy_assets_are_detected(tmp_path):
    (tmp_path / LEGACY_CHECKPOINT_FILENAME).write_bytes(b"checkpoint")
    (tmp_path / "pca_esmc_128.pkl").write_bytes(b"pca")
    (tmp_path / "esmc_600m_2024_12_v0.pth").write_bytes(b"esm")

    assets = PredictorAssets.from_dir(tmp_path)

    assert assets.asset_version == "0.1.1-paper"
    assert assets.checkpoint == tmp_path / LEGACY_CHECKPOINT_FILENAME
    assert assets.pca == tmp_path / "pca_esmc_128.pkl"


def test_explicit_asset_version_rejects_conflicting_directory_manifest(tmp_path):
    (tmp_path / ASSET_MANIFEST_FILENAME).write_text(
        '{"asset_version": "0.1.2"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="version conflict"):
        PredictorAssets.from_dir(tmp_path, asset_version="0.1.1-paper")

    assert PredictorAssets.from_dir(tmp_path).asset_version == "0.1.2"


def test_environment_asset_version_selects_legacy_bundle(tmp_path, monkeypatch):
    paths = {
        "PROTCROSS_CHECKPOINT": tmp_path / LEGACY_ASSET_BUNDLE.checkpoint_filename,
        "PROTCROSS_ESM_WEIGHTS": tmp_path / LEGACY_ASSET_BUNDLE.esm_filename,
        "PROTCROSS_PCA": tmp_path / LEGACY_ASSET_BUNDLE.pca_filename,
    }
    for name, path in paths.items():
        path.write_bytes(name.encode("utf-8"))
        monkeypatch.setenv(name, str(path))
    monkeypatch.setenv("PROTCROSS_ASSET_VERSION", LEGACY_ASSET_BUNDLE.version)
    expected_by_name = {
        spec.filename: spec.sha256
        for spec in LEGACY_ASSET_BUNDLE.assets
    }
    monkeypatch.setattr(
        "protcross.assets.sha256_file",
        lambda path: expected_by_name[Path(path).name],
    )

    resolved = resolve_prediction_assets(
        auto_setup_assets=False,
        offline=True,
    )

    assert resolved.asset_version == LEGACY_ASSET_BUNDLE.version
    assert resolved.checkpoint.name == LEGACY_ASSET_BUNDLE.checkpoint_filename
    assert resolved.pca.name == LEGACY_ASSET_BUNDLE.pca_filename
    assert resolved.asset_metadata["selected_bundle_version"] == LEGACY_ASSET_BUNDLE.version


def test_explicit_asset_version_wins_over_environment_default(tmp_path, monkeypatch):
    expected_by_name = {spec.filename: spec.sha256 for spec in DEFAULT_ASSET_BUNDLE.assets}
    for spec in DEFAULT_ASSET_BUNDLE.assets:
        (tmp_path / spec.filename).write_bytes(spec.filename.encode("utf-8"))
    monkeypatch.setenv("PROTCROSS_ASSET_VERSION", LEGACY_ASSET_BUNDLE.version)
    monkeypatch.setattr(
        "protcross.assets.sha256_file",
        lambda path: expected_by_name[Path(path).name],
    )

    resolved = resolve_prediction_assets(
        assets_dir=tmp_path,
        asset_version="default",
        auto_setup_assets=False,
    )

    assert resolved.asset_version == DEFAULT_ASSET_BUNDLE.version


@pytest.mark.parametrize("kwargs", [{"offline": True}, {"auto_setup_assets": False}])
def test_resolve_rejects_refresh_when_downloads_are_disabled(kwargs):
    with pytest.raises(ValueError, match="refresh_assets requires"):
        resolve_prediction_assets(refresh_assets=True, **kwargs)


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
    monkeypatch.setattr(
        "protcross.assets.sha256_file",
        lambda path: expected_by_name.get(Path(path).name) or sha256_file(path),
    )

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


def test_resolve_reuses_unchanged_verified_manifest_hashes(tmp_path, monkeypatch):
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
    monkeypatch.setattr(
        "protcross.assets.sha256_file",
        lambda path: (_ for _ in ()).throw(AssertionError(f"unexpected rehash: {path}")),
    )

    resolved = resolve_prediction_assets(assets_dir=tmp_path, auto_setup_assets=False)

    assert resolved.asset_metadata["all_assets_verified"] is True


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
    metadata = resolved.asset_metadata
    assert metadata["asset_version"] == "custom"
    assert metadata["selected_bundle_version"] == "0.1.2"
    assert metadata["all_assets_verified"] is False
    assert metadata["contains_unverified_assets"] is True
    for name, path, source_label in (
        ("checkpoint", ckpt, "--checkpoint"),
        ("esm_weights", esm, "--esm-weights"),
        ("pca", pca, "--pca"),
    ):
        entry = metadata[name]
        assert entry["path"] == str(path.resolve())
        assert entry["actual_sha256"] == hashlib.sha256(b"asset").hexdigest()
        assert entry["sha256"] == entry["actual_sha256"]
        assert entry["expected_sha256"]
        assert entry["verified"] is False
        assert entry["verification_status"] == "sha256_mismatch_trusted_by_user"
        assert entry["trust_unverified_override"] is True
        assert entry["source"] == "user"
        assert entry["source_label"] == source_label


def test_resolve_records_verified_managed_asset_metadata_without_rehashing(tmp_path, monkeypatch):
    expected_by_name = {spec.filename: spec.sha256 for spec in DEFAULT_ASSET_BUNDLE.assets}
    for spec in DEFAULT_ASSET_BUNDLE.assets:
        (tmp_path / spec.filename).write_bytes(spec.filename.encode("utf-8"))
    hash_calls = []

    def fake_sha256(path):
        hash_calls.append(Path(path).name)
        return expected_by_name[Path(path).name]

    monkeypatch.setattr("protcross.assets.sha256_file", fake_sha256)

    resolved = resolve_prediction_assets(assets_dir=tmp_path, auto_setup_assets=False)

    metadata = resolved.asset_metadata
    assert metadata["asset_version"] == "0.1.2"
    assert metadata["selected_bundle_version"] == "0.1.2"
    assert metadata["all_assets_verified"] is True
    assert metadata["contains_unverified_assets"] is False
    for name, path in (
        ("checkpoint", resolved.checkpoint),
        ("esm_weights", resolved.esm_weights),
        ("pca", resolved.pca),
    ):
        entry = metadata[name]
        assert entry["path"] == str(path.resolve())
        assert entry["actual_sha256"] == entry["expected_sha256"]
        assert entry["verified"] is True
        assert entry["verification_status"] == "verified_release_sha256"
        assert entry["trust_unverified_override"] is False
        assert entry["source"] == "managed"
        assert entry["source_label"] == "managed asset directory"
    assert sorted(hash_calls) == sorted(expected_by_name)


def test_resolve_rehashes_only_managed_asset_changed_since_manifest(tmp_path, monkeypatch):
    files = {}
    expected_by_name = {spec.filename: spec.sha256 for spec in DEFAULT_ASSET_BUNDLE.assets}
    for spec in DEFAULT_ASSET_BUNDLE.assets:
        path = tmp_path / spec.filename
        path.write_bytes(spec.filename.encode("utf-8"))
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
    changed = tmp_path / DEFAULT_CHECKPOINT_FILENAME
    changed.write_bytes(changed.read_bytes() + b"changed")
    hash_calls = []

    def fake_sha256(path):
        hash_calls.append(Path(path).name)
        return expected_by_name[Path(path).name]

    monkeypatch.setattr("protcross.assets.sha256_file", fake_sha256)

    resolve_prediction_assets(assets_dir=tmp_path, auto_setup_assets=False)

    assert hash_calls == [DEFAULT_CHECKPOINT_FILENAME]


def test_resolve_all_explicit_official_assets_ignores_bad_managed_manifest(tmp_path, monkeypatch):
    ckpt = tmp_path / DEFAULT_CHECKPOINT_FILENAME
    esm = tmp_path / DEFAULT_ASSET_BUNDLE.assets[0].filename
    pca = tmp_path / DEFAULT_PCA_FILENAME
    for path in (ckpt, esm, pca):
        path.write_bytes(path.name.encode("utf-8"))
    bad_cache = tmp_path / "bad-cache"
    bad_cache.mkdir()
    (bad_cache / ASSET_MANIFEST_FILENAME).write_text('{"asset_version": "unknown"}', encoding="utf-8")
    expected_by_name = {
        DEFAULT_CHECKPOINT_FILENAME: DEFAULT_ASSET_BUNDLE.assets[1].sha256,
        DEFAULT_ASSET_BUNDLE.assets[0].filename: DEFAULT_ASSET_BUNDLE.assets[0].sha256,
        DEFAULT_PCA_FILENAME: DEFAULT_ASSET_BUNDLE.assets[2].sha256,
    }
    monkeypatch.setattr("protcross.assets.sha256_file", lambda path: expected_by_name[Path(path).name])

    resolved = resolve_prediction_assets(
        ckpt,
        esm,
        pca,
        assets_dir=bad_cache,
        auto_setup_assets=False,
    )

    assert resolved.asset_version == "0.1.2"
    assert resolved.assets.asset_version == "0.1.2"


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


def test_download_asset_http_200_writes_new_file_and_reports_progress(tmp_path, monkeypatch):
    payload = b"protcross-asset"
    response = _FakeResponse(
        [payload[:5], b"", payload[5:]],
        status_code=200,
        content_length=len(payload),
    )
    requests = []

    def fake_http_get(url, **kwargs):
        requests.append((url, kwargs))
        return response

    monkeypatch.setattr("protcross.assets._http_get", fake_http_get)
    output_path = tmp_path / "asset.bin"
    progress = []

    download_asset(
        AssetSpec("test asset", output_path.name, "https://example.invalid/asset.bin", _sha256(payload)),
        output_path,
        progress_callback=lambda downloaded, total: progress.append((downloaded, total)),
    )

    assert output_path.read_bytes() == payload
    assert not output_path.with_suffix(".bin.part").exists()
    assert requests == [
        (
            "https://example.invalid/asset.bin",
            {"stream": True, "timeout": 30, "headers": None},
        )
    ]
    assert progress == [
        (0, len(payload)),
        (5, len(payload)),
        (len(payload), len(payload)),
        (len(payload), len(payload)),
    ]


def test_download_asset_http_206_resumes_existing_part(tmp_path, monkeypatch):
    payload = b"protcross-asset"
    prefix = payload[:6]
    remainder = payload[len(prefix) :]
    response = _FakeResponse(
        [remainder],
        status_code=206,
        content_length=len(remainder),
        content_range=f"bytes {len(prefix)}-{len(payload) - 1}/{len(payload)}",
    )
    requests = []

    def fake_http_get(url, **kwargs):
        requests.append((url, kwargs))
        return response

    monkeypatch.setattr("protcross.assets._http_get", fake_http_get)
    monkeypatch.setattr("protcross.assets._require_download_space", lambda *args, **kwargs: None)
    output_path = tmp_path / "asset.bin"
    output_path.with_suffix(".bin.part").write_bytes(prefix)
    progress = []

    download_asset(
        AssetSpec(
            "test asset",
            output_path.name,
            "https://example.invalid/asset.bin",
            _sha256(payload),
            len(payload),
        ),
        output_path,
        progress_callback=lambda downloaded, total: progress.append((downloaded, total)),
    )

    assert output_path.read_bytes() == payload
    assert requests[0][1]["headers"] == {"Range": f"bytes={len(prefix)}-"}
    assert progress[0] == (len(prefix), len(payload))
    assert progress[-1] == (len(payload), len(payload))


def test_download_asset_restarts_when_server_ignores_range(tmp_path, monkeypatch):
    payload = b"complete-response"
    stale_prefix = b"stale-"
    response = _FakeResponse(
        [payload],
        status_code=200,
        content_length=len(payload),
    )
    requests = []

    def fake_http_get(url, **kwargs):
        requests.append((url, kwargs))
        return response

    monkeypatch.setattr("protcross.assets._http_get", fake_http_get)
    output_path = tmp_path / "asset.bin"
    output_path.with_suffix(".bin.part").write_bytes(stale_prefix)
    progress = []

    download_asset(
        AssetSpec("test asset", output_path.name, "https://example.invalid/asset.bin", _sha256(payload)),
        output_path,
        progress_callback=lambda downloaded, total: progress.append((downloaded, total)),
    )

    assert requests[0][1]["headers"] == {"Range": f"bytes={len(stale_prefix)}-"}
    assert output_path.read_bytes() == payload
    assert progress[0] == (0, len(payload))


def test_download_asset_promotes_complete_verified_part_without_network(tmp_path, monkeypatch):
    payload = b"already-complete"
    output_path = tmp_path / "asset.bin"
    part_path = output_path.with_suffix(".bin.part")
    part_path.write_bytes(payload)
    progress = []

    def fail_http_get(*args, **kwargs):
        raise AssertionError("a complete verified .part file must not trigger a network request")

    monkeypatch.setattr("protcross.assets._http_get", fail_http_get)

    download_asset(
        AssetSpec(
            "test asset",
            output_path.name,
            "https://example.invalid/asset.bin",
            _sha256(payload),
            len(payload),
        ),
        output_path,
        progress_callback=lambda downloaded, total: progress.append((downloaded, total)),
    )

    assert output_path.read_bytes() == payload
    assert not part_path.exists()
    assert progress == [(len(payload), len(payload))]


def test_download_asset_promotes_sha_complete_part_without_known_size(tmp_path, monkeypatch):
    payload = b"complete-without-published-size"
    output_path = tmp_path / "asset.bin"
    output_path.with_suffix(".bin.part").write_bytes(payload)
    monkeypatch.setattr(
        "protcross.assets._http_get",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network must not be used")),
    )

    download_asset(
        AssetSpec("test asset", output_path.name, "https://example.invalid/asset.bin", _sha256(payload)),
        output_path,
    )

    assert output_path.read_bytes() == payload


def test_download_asset_retries_from_zero_for_wrong_content_range(tmp_path, monkeypatch):
    payload = b"complete-response"
    prefix = payload[:5]
    output_path = tmp_path / "asset.bin"
    output_path.with_suffix(".bin.part").write_bytes(prefix)
    responses = iter(
        [
            _FakeResponse(
                [payload[5:]],
                status_code=206,
                content_length=len(payload) - 5,
                content_range=f"bytes 4-{len(payload) - 1}/{len(payload)}",
            ),
            _FakeResponse([payload], status_code=200, content_length=len(payload)),
        ]
    )
    calls = []

    def fake_http_get(url, **kwargs):
        calls.append(kwargs.get("headers"))
        return next(responses)

    monkeypatch.setattr("protcross.assets._http_get", fake_http_get)
    download_asset(
        AssetSpec("test asset", output_path.name, "https://example.invalid/asset.bin", _sha256(payload)),
        output_path,
    )

    assert calls == [{"Range": "bytes=5-"}, None]
    assert output_path.read_bytes() == payload


def test_download_asset_retries_from_zero_after_range_not_satisfiable(tmp_path, monkeypatch):
    payload = b"fresh-complete-response"
    output_path = tmp_path / "asset.bin"
    output_path.with_suffix(".bin.part").write_bytes(b"corrupt-partial")
    responses = iter(
        [
            _FakeResponse([], status_code=416),
            _FakeResponse([payload], status_code=200, content_length=len(payload)),
        ]
    )
    calls = []

    def fake_http_get(url, **kwargs):
        calls.append(kwargs.get("headers"))
        return next(responses)

    monkeypatch.setattr("protcross.assets._http_get", fake_http_get)
    download_asset(
        AssetSpec("test asset", output_path.name, "https://example.invalid/asset.bin", _sha256(payload)),
        output_path,
    )

    assert calls == [{"Range": "bytes=15-"}, None]
    assert output_path.read_bytes() == payload


def test_download_asset_refuses_response_larger_than_expected(tmp_path, monkeypatch):
    output_path = tmp_path / "asset.bin"
    monkeypatch.setattr(
        "protcross.assets._http_get",
        lambda *args, **kwargs: _FakeResponse([b"too-long"], status_code=200, content_length=8),
    )

    with pytest.raises(RuntimeError, match="exceeds the expected size"):
        download_asset(
            AssetSpec("test asset", output_path.name, "https://example.invalid/asset.bin", size_bytes=3),
            output_path,
            verify=False,
        )

    assert not output_path.exists()
    assert not output_path.with_suffix(".bin.part").exists()


def test_download_asset_short_response_retains_resumable_part(tmp_path, monkeypatch):
    output_path = tmp_path / "asset.bin"
    monkeypatch.setattr(
        "protcross.assets._http_get",
        lambda *args, **kwargs: _FakeResponse([b"abc"], status_code=200, content_length=3),
    )

    with pytest.raises(RuntimeError, match="partial data retained"):
        download_asset(
            AssetSpec("test asset", output_path.name, "https://example.invalid/asset.bin", size_bytes=5),
            output_path,
            verify=False,
        )

    assert output_path.with_suffix(".bin.part").read_bytes() == b"abc"


def test_download_asset_discards_complete_corrupt_part_before_redownload(tmp_path, monkeypatch):
    payload = b"valid-payload"
    corrupt_payload = b"bad---payload"
    assert len(corrupt_payload) == len(payload)
    output_path = tmp_path / "asset.bin"
    part_path = output_path.with_suffix(".bin.part")
    part_path.write_bytes(corrupt_payload)
    response = _FakeResponse([payload], status_code=200, content_length=len(payload))
    requests = []

    def fake_http_get(url, **kwargs):
        requests.append((url, kwargs))
        return response

    monkeypatch.setattr("protcross.assets._http_get", fake_http_get)
    monkeypatch.setattr("protcross.assets._require_download_space", lambda *args, **kwargs: None)

    download_asset(
        AssetSpec(
            "test asset",
            output_path.name,
            "https://example.invalid/asset.bin",
            _sha256(payload),
            len(payload),
        ),
        output_path,
    )

    assert requests[0][1]["headers"] is None
    assert output_path.read_bytes() == payload
    assert not part_path.exists()


def test_download_asset_cancellation_retains_partial_file(tmp_path, monkeypatch):
    cancel_event = Event()

    def chunks():
        yield b"first-chunk"
        cancel_event.set()
        yield b"second-chunk"

    monkeypatch.setattr(
        "protcross.assets._http_get",
        lambda *args, **kwargs: _FakeResponse(chunks(), status_code=200),
    )
    output_path = tmp_path / "asset.bin"

    with pytest.raises(DownloadCancelled, match="partial data retained"):
        download_asset(
            AssetSpec("test asset", output_path.name, "https://example.invalid/asset.bin"),
            output_path,
            cancel_event=cancel_event,
        )

    assert not output_path.exists()
    assert output_path.with_suffix(".bin.part").read_bytes() == b"first-chunk"


def test_download_asset_skips_verified_target_with_single_complete_progress_event(tmp_path, monkeypatch):
    payload = b"already-installed"
    output_path = tmp_path / "asset.bin"
    output_path.write_bytes(payload)
    progress = []

    def fail_http_get(*args, **kwargs):
        raise AssertionError("an existing verified target must not trigger a network request")

    monkeypatch.setattr("protcross.assets._http_get", fail_http_get)

    download_asset(
        AssetSpec(
            "test asset",
            output_path.name,
            "https://example.invalid/asset.bin",
            _sha256(payload),
            len(payload),
        ),
        output_path,
        progress_callback=lambda downloaded, total: progress.append((downloaded, total)),
    )

    assert output_path.read_bytes() == payload
    assert progress == [(len(payload), len(payload))]


def test_download_space_preflight_fails_before_large_transfer(tmp_path, monkeypatch):
    monkeypatch.setattr(
        assets_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=128 * 1024 * 1024),
    )

    with pytest.raises(OSError, match="Not enough free disk space"):
        assets_module._require_download_space(
            tmp_path,
            2_300_275_866,
            already_downloaded=0,
        )


def test_setup_assets_reuses_persisted_esm_license_acceptance(tmp_path, monkeypatch):
    downloads = []

    def fake_download(spec, output_path, *, force=False, verify=True, known_sha256=None):
        downloads.append((spec.filename, force, verify))
        output_path.write_bytes(b"asset")

    monkeypatch.delenv("PROTCROSS_ACCEPT_ESM_LICENSE", raising=False)
    monkeypatch.setattr("protcross.assets.download_asset", fake_download)

    setup_assets(tmp_path, force=True, accept_esm_license=True)
    first_manifest = json.loads((tmp_path / ASSET_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    accepted_at = first_manifest["esm_license"]["accepted_at"]

    setup_assets(tmp_path, force=True, accept_esm_license=False)
    second_manifest = json.loads((tmp_path / ASSET_MANIFEST_FILENAME).read_text(encoding="utf-8"))

    assert second_manifest["esm_license"] == {
        "accepted": True,
        "url": first_manifest["esm_license"]["url"],
        "accepted_at": accepted_at,
    }
    assert len(downloads) == 6


def test_setup_assets_relative_directory_writes_absolute_environment_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_download(spec, output_path, **kwargs):
        output_path.write_bytes(spec.filename.encode("utf-8"))

    monkeypatch.setattr("protcross.assets.download_asset", fake_download)

    output_dir = setup_assets("relative-assets", force=True, accept_esm_license=True)

    assert output_dir == (tmp_path / "relative-assets").resolve()
    env_text = (output_dir / "protcross.env").read_text(encoding="utf-8")
    assert str(output_dir) in env_text
    assert "relative-assets/" not in env_text.replace(str(output_dir), "")


def test_setup_assets_refresh_quarantines_corrupt_manifest(tmp_path, monkeypatch):
    (tmp_path / ASSET_MANIFEST_FILENAME).write_text("{truncated", encoding="utf-8")

    def fake_download(spec, output_path, **kwargs):
        output_path.write_bytes(spec.filename.encode("utf-8"))

    monkeypatch.setattr("protcross.assets.download_asset", fake_download)

    setup_assets(tmp_path, force=True, accept_esm_license=True)

    manifest = json.loads((tmp_path / ASSET_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["asset_version"] == DEFAULT_ASSET_BUNDLE.version
    assert len(list(tmp_path.glob(f"{ASSET_MANIFEST_FILENAME}.corrupt-*"))) == 1


def test_prediction_acceptance_is_persisted_for_existing_managed_assets(tmp_path, monkeypatch):
    expected_by_name = {spec.filename: spec.sha256 for spec in DEFAULT_ASSET_BUNDLE.assets}
    for spec in DEFAULT_ASSET_BUNDLE.assets:
        (tmp_path / spec.filename).write_bytes(spec.filename.encode("utf-8"))
    (tmp_path / ASSET_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": "protcross-assets-v1",
                "asset_version": DEFAULT_ASSET_BUNDLE.version,
                "esm_license": {"accepted": False, "url": None, "accepted_at": None},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "protcross.assets.sha256_file",
        lambda path: expected_by_name[Path(path).name],
    )

    resolve_prediction_assets(
        assets_dir=tmp_path,
        auto_setup_assets=False,
        accept_esm_license=True,
        require_esm_license_for_use=True,
    )
    persisted = json.loads((tmp_path / ASSET_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert persisted["esm_license"]["accepted"] is True
    assert persisted["esm_license"]["accepted_at"]

    resolve_prediction_assets(
        assets_dir=tmp_path,
        auto_setup_assets=False,
        accept_esm_license=False,
        require_esm_license_for_use=True,
    )
