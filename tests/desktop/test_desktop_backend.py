from __future__ import annotations

import hashlib
import json
import sys
import http.client
import threading
import time
import zipfile
from pathlib import Path
from urllib.parse import quote

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "desktop" / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "desktop" / "installer"))

from protcross.assets import AssetSpec, DownloadCancelled, sha256_file
from protcross import __version__
import protcross_desktop.config as desktop_config
import protcross_desktop.manifest as desktop_manifest
import protcross_desktop.service as desktop_service
from protcross_desktop.config import DesktopPaths
from protcross_desktop.server import create_server
from protcross_desktop.service import DesktopBackend
from validate_bundled_assets import validate_bundled_assets


MINIMAL_PDB = (
    "HEADER    TEST STRUCTURE\n"
    "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C  \n"
    "END\n"
)


def test_desktop_manifest_recovers_corrupt_json(tmp_path):
    manifest_path = tmp_path / "desktop-assets.json"
    manifest_path.write_text("{truncated", encoding="utf-8")

    manifest = desktop_manifest.DesktopManifest.load(manifest_path)

    assert not manifest_path.exists()
    quarantined = list(tmp_path.glob("desktop-assets.json.corrupt-*"))
    assert len(quarantined) == 1
    assert manifest.extra["recovered_corrupt_manifest"] == str(quarantined[0])


def test_desktop_status_exposes_active_work_and_prunes_finished_history(tmp_path):
    backend = DesktopBackend(root=tmp_path)
    active = desktop_service.AssetDownloadJob("active", "esm.pth", 100, status="running")
    backend._asset_downloads[active.id] = active
    for index in range(desktop_service.MAX_RETAINED_FINISHED_JOBS + 5):
        job = desktop_service.AssetDownloadJob(
            f"done-{index}",
            "esm.pth",
            100,
            status="completed",
            completed_at=float(index),
        )
        backend._asset_downloads[job.id] = job

    with backend._lock:
        backend._prune_finished_jobs_locked(backend._asset_downloads)
    status = backend.status()

    assert len(backend._asset_downloads) == desktop_service.MAX_RETAINED_FINISHED_JOBS + 1
    assert any(job["id"] == "active" for job in status["activity"]["asset_downloads"])


class FakeResult:
    def __init__(self, input_path, output_pdb=None, scores_tsv=None, pocket_json=None, summary_json=None):
        self.input_path = str(input_path)
        self.output_files = {
            "structure": str(output_pdb),
            "scores_tsv": str(scores_tsv),
            "pockets_json": str(pocket_json),
            "summary_json": str(summary_json),
        }
        for path in (output_pdb, scores_tsv):
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text("fake", encoding="utf-8")
        if pocket_json:
            Path(pocket_json).parent.mkdir(parents=True, exist_ok=True)
            Path(pocket_json).write_text(json.dumps(self.to_pocket_dict()), encoding="utf-8")
        if summary_json:
            Path(summary_json).parent.mkdir(parents=True, exist_ok=True)
            Path(summary_json).write_text(json.dumps(self.to_summary_dict()), encoding="utf-8")

    def to_summary_dict(self):
        return {
            "schema_version": "protcross-summary-v2",
            "input_structure": self.input_path,
            "top_pocket": {
                "center": [1.0, 2.0, 3.0],
                "residue_count": 4,
                "score_mean": 0.75,
                "score_max": 0.9,
            },
            "output_files": self.output_files,
        }

    def to_pocket_dict(self):
        return {
            "schema_version": "protcross-pocket-v2",
            "clustered_pockets": [
                {
                    "cluster_id": 1,
                    "center": [1.0, 2.0, 3.0],
                    "residue_count": 2,
                    "score_mean": 0.75,
                    "score_max": 0.9,
                    "residues": [
                        {
                            "residue_id": "A_10",
                            "chain_id": "A",
                            "residue_number": 10,
                            "score": 0.9,
                            "probability": 0.9,
                        },
                        {
                            "residue_id": "A_11",
                            "chain_id": "A",
                            "residue_number": 11,
                            "score": 0.8,
                            "probability": 0.8,
                        },
                    ],
                }
            ],
            "aggregate_pocket": None,
        }


class FakePredictor:
    calls = []

    def predict(self, input_structure, **kwargs):
        self.calls.append((str(input_structure), kwargs))
        return FakeResult(
            input_structure,
            output_pdb=kwargs.get("output_pdb"),
            scores_tsv=kwargs.get("scores_tsv"),
            pocket_json=kwargs.get("pocket_json"),
            summary_json=kwargs.get("summary_json"),
        )


def _ready_backend(tmp_path, monkeypatch, predictor_factory=None):
    factory_calls = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return FakePredictor()

    backend = DesktopBackend(root=tmp_path, predictor_factory=predictor_factory or factory)
    backend.paths.default_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    backend.paths.default_checkpoint.write_bytes(b"checkpoint")
    backend.paths.default_pca.write_bytes(b"pca")
    esm_path = tmp_path / "esm.pth"
    esm_path.write_bytes(b"esm")
    monkeypatch.setattr(
        desktop_service,
        "DEFAULT_ASSETS",
        (
            AssetSpec("esm", "esm.pth", "", sha256_file(esm_path)),
            AssetSpec("checkpoint", "checkpoint.ckpt", "", sha256_file(backend.paths.default_checkpoint)),
            AssetSpec("pca", "pca.pkl", "", sha256_file(backend.paths.default_pca)),
        ),
    )
    backend.manifest.esm_expected_sha256 = sha256_file(esm_path)
    backend.confirm_esm_license()
    backend.import_esm_weights(esm_path)
    backend.configure_backend("conda", conda_python=sys.executable)
    backend.manifest.backend_test_ok = True
    backend.manifest.backend_test_mode = "conda"
    backend.manifest.backend_test_python = sys.executable
    backend.manifest.backend_test_package_version = __version__
    backend.manifest.save(backend.paths.manifest_path)
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    return backend, input_pdb, factory_calls


def _wait_for_asset_download(backend, job_id, terminal_statuses, timeout=3.0):
    deadline = time.time() + timeout
    status = backend.esm_download_status(job_id)
    while status["status"] not in terminal_statuses and time.time() < deadline:
        time.sleep(0.01)
        status = backend.esm_download_status(job_id)
    assert status["status"] in terminal_statuses, status
    return status


def test_esm_import_requires_license_confirmation(tmp_path):
    backend = DesktopBackend(root=tmp_path)
    esm_path = tmp_path / "esm.pth"
    esm_path.write_bytes(b"esm")

    with pytest.raises(RuntimeError, match="license"):
        backend.import_esm_weights(esm_path)


def test_license_and_import_write_manifest(tmp_path):
    backend = DesktopBackend(root=tmp_path)
    esm_path = tmp_path / "esm.pth"
    esm_path.write_bytes(b"esm")

    backend.confirm_esm_license()
    backend.manifest.esm_expected_sha256 = sha256_file(esm_path)
    status = backend.import_esm_weights(esm_path)

    assert status["present"] is True
    assert backend.paths.manifest_path.exists()
    assert backend.manifest.esm_license_confirmed is True
    assert backend.manifest.esm_source == "imported"


def test_async_esm_download_reports_progress_deduplicates_and_completes(tmp_path, monkeypatch):
    payload = b"verified-esm-weights"
    digest = hashlib.sha256(payload).hexdigest()
    entered = threading.Event()
    release = threading.Event()
    original_assets = desktop_service.DEFAULT_ASSETS
    monkeypatch.setattr(
        desktop_service,
        "DEFAULT_ASSETS",
        (
            AssetSpec("esm", desktop_service.ESM_FILENAME, "https://example.invalid/esm", digest, len(payload)),
            *original_assets[1:],
        ),
    )
    monkeypatch.setattr(desktop_service, "ESM_EXPECTED_SHA256", digest)

    def fake_download(
        spec,
        output_path,
        *,
        force=False,
        verify=True,
        progress_callback=None,
        cancel_event=None,
    ):
        assert spec.sha256 == digest
        assert force is False
        assert verify is True
        assert cancel_event is not None
        progress_callback(4, len(payload))
        entered.set()
        assert release.wait(timeout=2)
        output_path.write_bytes(payload)
        progress_callback(len(payload), len(payload))

    monkeypatch.setattr(desktop_service, "download_asset", fake_download)
    backend = DesktopBackend(root=tmp_path)
    backend.manifest.esm_expected_sha256 = digest
    backend.confirm_esm_license()

    started = backend.start_esm_download()
    assert entered.wait(timeout=2)
    running = backend.esm_download_status(started["id"])
    duplicate = backend.start_esm_download()

    assert running["status"] == "running"
    assert running["downloaded_bytes"] == 4
    assert running["total_bytes"] == len(payload)
    assert duplicate["id"] == started["id"]

    release.set()
    completed = _wait_for_asset_download(backend, started["id"], {"completed"})

    assert completed["downloaded_bytes"] == len(payload)
    assert completed["percent"] == 100.0
    assert backend.paths.managed_esm_weights.read_bytes() == payload
    assert backend.manifest.esm_source == "downloaded"
    assert backend.manifest.esm_verified is True


def test_async_esm_download_can_cancel_and_retains_partial_file(tmp_path, monkeypatch):
    entered = threading.Event()
    original_assets = desktop_service.DEFAULT_ASSETS
    monkeypatch.setattr(
        desktop_service,
        "DEFAULT_ASSETS",
        (
            AssetSpec(
                "esm",
                desktop_service.ESM_FILENAME,
                "https://example.invalid/esm",
                "0" * 64,
                12,
            ),
            *original_assets[1:],
        ),
    )

    def cancellable_download(
        spec,
        output_path,
        *,
        force=False,
        verify=True,
        progress_callback=None,
        cancel_event=None,
    ):
        partial = output_path.with_suffix(output_path.suffix + ".part")
        partial.write_bytes(b"partial")
        progress_callback(partial.stat().st_size, spec.size_bytes)
        entered.set()
        assert cancel_event.wait(timeout=2)
        raise DownloadCancelled("cancelled for test")

    monkeypatch.setattr(desktop_service, "download_asset", cancellable_download)
    backend = DesktopBackend(root=tmp_path)
    backend.confirm_esm_license()

    started = backend.start_esm_download()
    assert entered.wait(timeout=2)
    cancelling = backend.cancel_esm_download(started["id"])
    cancelled = _wait_for_asset_download(backend, started["id"], {"cancelled"})

    assert cancelling["status"] == "cancelling"
    assert cancelled["downloaded_bytes"] == len(b"partial")
    assert cancelled["error"] is None
    assert backend.paths.managed_esm_weights.exists() is False
    assert backend.paths.managed_esm_weights.with_suffix(
        backend.paths.managed_esm_weights.suffix + ".part"
    ).read_bytes() == b"partial"


def test_wrong_esm_hash_does_not_mark_assets_ready(tmp_path):
    backend = DesktopBackend(root=tmp_path)
    backend.paths.default_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    backend.paths.default_checkpoint.write_bytes(b"checkpoint")
    backend.paths.default_pca.write_bytes(b"pca")
    esm_path = tmp_path / "esm.pth"
    esm_path.write_bytes(b"not-the-release-weight")

    backend.confirm_esm_license()
    with pytest.raises(ValueError, match="SHA256"):
        backend.import_esm_weights(esm_path)
    backend.configure_backend("cpu")

    status = backend.asset_status()
    assert status["esm"]["present"] is False
    assert backend.manifest.esm_weights_path is None
    assert status["ready"] is False


def test_asset_status_uses_cached_real_hash_without_rehash(tmp_path, monkeypatch):
    backend, _, _ = _ready_backend(tmp_path, monkeypatch)
    status = backend.asset_status()
    assert status["checkpoint"]["verification_source"] == "sha256"
    assert status["pca"]["verification_source"] == "sha256"

    def fail_hash(path):
        raise AssertionError(f"unexpected hash read: {path}")

    monkeypatch.setattr(desktop_service, "sha256_file", fail_hash)
    monkeypatch.setattr(desktop_manifest, "sha256_file", fail_hash)

    status = backend.asset_status()

    assert status["esm"]["verified"] is True
    assert status["checkpoint"]["verification_source"] == "sha256"
    assert status["pca"]["verification_source"] == "sha256"


def test_import_checkpoint_and_pca_copy_into_managed_assets_and_clear_predictor(tmp_path, monkeypatch):
    backend, _, _ = _ready_backend(tmp_path, monkeypatch)
    source_checkpoint = tmp_path / "external" / "checkpoint.ckpt"
    source_pca = tmp_path / "external" / "pca.pkl"
    source_checkpoint.parent.mkdir()
    source_checkpoint.write_bytes(b"checkpoint")
    source_pca.write_bytes(b"pca")
    backend._predictor = object()
    backend._predictor_key = ("old", "old", "old", "cpu")

    checkpoint_status = backend.import_checkpoint(source_checkpoint)
    pca_status = backend.import_pca(source_pca)

    assert Path(backend.manifest.checkpoint_path) == backend.paths.assets_dir / desktop_service.DEFAULT_ASSETS[1].filename
    assert Path(backend.manifest.pca_path) == backend.paths.assets_dir / desktop_service.DEFAULT_ASSETS[2].filename
    assert Path(backend.manifest.checkpoint_path).read_bytes() == b"checkpoint"
    assert Path(backend.manifest.pca_path).read_bytes() == b"pca"
    assert checkpoint_status["verified"] is True
    assert pca_status["verified"] is True
    assert backend._predictor is None
    assert backend._predictor_key is None


def test_validate_bundled_assets_accepts_matching_manifest(tmp_path):
    checkpoint = tmp_path / "checkpoint.ckpt"
    pca = tmp_path / "pca.pkl"
    checkpoint.write_bytes(b"checkpoint")
    pca.write_bytes(b"pca")
    checkpoint_sha = sha256_file(checkpoint)
    pca_sha = sha256_file(pca)
    manifest = {
        "schema_version": "protcross-desktop-bundled-assets-v1",
        "checkpoint": {
            "filename": checkpoint.name,
            "sha256": checkpoint_sha,
            "size_bytes": checkpoint.stat().st_size,
        },
        "pca": {
            "filename": pca.name,
            "sha256": pca_sha,
            "size_bytes": pca.stat().st_size,
        },
    }
    (tmp_path / "protcross-desktop-bundled-assets.json").write_text(json.dumps(manifest), encoding="utf-8")

    validated = validate_bundled_assets(
        tmp_path,
        expected={
            "checkpoint": {"filename": checkpoint.name, "sha256": checkpoint_sha},
            "pca": {"filename": pca.name, "sha256": pca_sha},
        },
    )

    assert validated == [checkpoint, pca]


def test_validate_bundled_assets_rejects_extra_weight_files(tmp_path):
    checkpoint = tmp_path / "checkpoint.ckpt"
    pca = tmp_path / "pca.pkl"
    esm = tmp_path / "esmc_600m_2024_12_v0.pth"
    checkpoint.write_bytes(b"checkpoint")
    pca.write_bytes(b"pca")
    esm.write_bytes(b"esm")
    checkpoint_sha = sha256_file(checkpoint)
    pca_sha = sha256_file(pca)
    (tmp_path / "protcross-desktop-bundled-assets.json").write_text(
        json.dumps(
            {
                "checkpoint": {"filename": checkpoint.name, "sha256": checkpoint_sha},
                "pca": {"filename": pca.name, "sha256": pca_sha},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Unexpected files"):
        validate_bundled_assets(
            tmp_path,
            expected={
                "checkpoint": {"filename": checkpoint.name, "sha256": checkpoint_sha},
                "pca": {"filename": pca.name, "sha256": pca_sha},
            },
        )


def test_root_readme_uses_portable_desktop_test_command():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "conda run -n pyg5070" not in readme
    assert "python -m pytest -q tests/desktop" in readme


def test_single_prediction_uses_desktop_output_package(tmp_path, monkeypatch):
    backend, input_pdb, factory_calls = _ready_backend(tmp_path, monkeypatch)

    result = backend.predict_single(input_pdb, output_dir=tmp_path / "results")

    assert result["ok"] is True
    assert len(factory_calls) == 1
    assert result["output_files"]["structure"].endswith("input.protcross.pdb")
    assert Path(result["output_files"]["summary_json"]).exists()
    assert [residue["residue_id"] for residue in result["top_pocket_residues"]] == ["A_10", "A_11"]


def test_prediction_validates_input_before_asset_readiness(tmp_path):
    backend = DesktopBackend(root=tmp_path)

    with pytest.raises(FileNotFoundError, match="Input structure not found"):
        backend.predict_single(tmp_path / "missing.pdb")


@pytest.mark.parametrize(
    ("threshold", "cutoff", "message"),
    [
        (-0.01, 8.0, "threshold"),
        (1.01, 8.0, "threshold"),
        (0.5, 0.0, "pocket_cluster_cutoff"),
    ],
)
def test_prediction_rejects_invalid_numeric_options(tmp_path, monkeypatch, threshold, cutoff, message):
    backend, input_pdb, _ = _ready_backend(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match=message):
        backend.predict_single(input_pdb, threshold=threshold, pocket_cluster_cutoff=cutoff)


def test_macos_default_root_uses_application_support(monkeypatch, tmp_path):
    monkeypatch.setattr(desktop_config.sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.delenv("PROTCROSS_DESKTOP_HOME", raising=False)

    paths = DesktopPaths.discover()

    assert paths.root == tmp_path / "Library" / "Application Support" / "ProtCross"


def test_gpu_backend_uses_mps_device_on_macos(tmp_path, monkeypatch):
    backend, _, _ = _ready_backend(tmp_path, monkeypatch)
    backend.configure_backend("gpu")
    monkeypatch.setattr(desktop_service.sys, "platform", "darwin")

    assert backend._default_device() == "mps"


def test_batch_prediction_reuses_one_predictor(tmp_path, monkeypatch):
    backend, input_pdb, factory_calls = _ready_backend(tmp_path, monkeypatch)
    second = tmp_path / "second.pdb"
    second.write_text(MINIMAL_PDB, encoding="utf-8")

    job = backend.submit_batch([input_pdb, second], output_dir=tmp_path / "batch")
    deadline = time.time() + 5
    while time.time() < deadline:
        status = backend.batch_status(job["id"])
        if status["status"] in {"completed", "completed_with_errors", "failed"}:
            break
        time.sleep(0.05)

    status = backend.batch_status(job["id"])
    assert status["status"] == "completed"
    assert len(factory_calls) == 1
    assert [item["status"] for item in status["items"]] == ["completed", "completed"]
    assert "pockets" not in status["items"][0]
    assert "summary" not in status["items"][0]
    detail = backend.batch_item_result(job["id"], status["items"][0]["input_structure"])
    assert [residue["residue_id"] for residue in detail["top_pocket_residues"]] == ["A_10", "A_11"]
    assert detail["pockets"]["schema_version"] == "protcross-pocket-v2"


def test_batch_rejects_second_active_job(tmp_path, monkeypatch):
    release = threading.Event()

    class BlockingPredictor(FakePredictor):
        def predict(self, input_structure, **kwargs):
            release.wait(timeout=2)
            return super().predict(input_structure, **kwargs)

    backend, input_pdb, _ = _ready_backend(tmp_path, monkeypatch, predictor_factory=lambda **kwargs: BlockingPredictor())
    second = tmp_path / "second.pdb"
    second.write_text(MINIMAL_PDB, encoding="utf-8")

    first = backend.submit_batch([input_pdb], output_dir=tmp_path / "batch")
    deadline = time.time() + 2
    while time.time() < deadline:
        if backend.batch_status(first["id"])["status"] == "running":
            break
        time.sleep(0.05)
    with pytest.raises(RuntimeError, match="already running"):
        backend.submit_batch([second], output_dir=tmp_path / "batch2")
    release.set()


def test_batch_prediction_uses_unique_output_directories_for_same_stem(tmp_path, monkeypatch):
    backend, input_pdb, _ = _ready_backend(tmp_path, monkeypatch)
    other_dir = tmp_path / "nested"
    other_dir.mkdir()
    second = other_dir / "input.pdb"
    second.write_text(MINIMAL_PDB, encoding="utf-8")

    job = backend.submit_batch([input_pdb, second], output_dir=tmp_path / "batch")
    deadline = time.time() + 5
    while time.time() < deadline:
        status = backend.batch_status(job["id"])
        if status["status"] in {"completed", "completed_with_errors", "failed"}:
            break
        time.sleep(0.05)

    status = backend.batch_status(job["id"])
    output_dirs = [item["output_dir"] for item in status["items"]]
    assert output_dirs[0] != output_dirs[1]
    assert all(Path(path).name.startswith("input-") for path in output_dirs)


def test_batch_level_failure_marks_items_and_exposes_error(tmp_path, monkeypatch):
    def factory(**kwargs):
        raise RuntimeError("simulated model load failure")

    backend, input_pdb, _ = _ready_backend(tmp_path, monkeypatch, predictor_factory=factory)

    job = backend.submit_batch([input_pdb], output_dir=tmp_path / "batch")
    deadline = time.time() + 5
    while time.time() < deadline:
        status = backend.batch_status(job["id"])
        if status["status"] == "failed":
            break
        time.sleep(0.05)

    status = backend.batch_status(job["id"])
    assert status["status"] == "failed"
    assert status["failed"] == 1
    assert "simulated model load failure" in status["error"]
    assert status["items"][0]["status"] == "failed"
    assert "simulated model load failure" in status["items"][0]["error"]


def test_export_diagnostics_zip_is_local_and_reviewable(tmp_path, monkeypatch):
    backend, _, _ = _ready_backend(tmp_path, monkeypatch)

    zip_path = Path(backend.export_diagnostics(tmp_path / "diag.zip"))

    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as archive:
        assert "diagnostics.json" in archive.namelist()
        assert "README.txt" in archive.namelist()


def test_export_diagnostics_redacts_proxy_credentials_and_paths(tmp_path, monkeypatch):
    backend, _, _ = _ready_backend(tmp_path, monkeypatch)
    backend.configure_backend("conda", conda_python=sys.executable, proxy_url="http://user:pass@proxy.example:8080")

    zip_path = Path(backend.export_diagnostics(tmp_path / "diag.zip"))

    with zipfile.ZipFile(zip_path) as archive:
        diagnostics = archive.read("diagnostics.json").decode("utf-8")
    assert "user:pass" not in diagnostics
    assert "http://<redacted>@proxy.example:8080" in diagnostics
    assert str(tmp_path) not in diagnostics


@pytest.mark.network
def test_desktop_server_handles_cors_options(tmp_path):
    server = create_server("127.0.0.1", 0, root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        conn.request(
            "OPTIONS",
            "/license/confirm",
            headers={"Origin": "http://127.0.0.1:5173", "Access-Control-Request-Method": "POST"},
        )
        response = conn.getresponse()
        response.read()

        assert response.status == 204
        assert response.getheader("access-control-allow-origin") == "http://127.0.0.1:5173"
        assert "POST" in (response.getheader("access-control-allow-methods") or "")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.network
def test_desktop_server_inspect_route_returns_structure_qc_without_ready_assets(tmp_path):
    structure = tmp_path / "input.pdb"
    structure.write_text(MINIMAL_PDB, encoding="utf-8")
    server = create_server("127.0.0.1", 0, root=tmp_path, token="secret-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = json.dumps({"input_structure": str(structure), "chain_id": "A"})
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        conn.request(
            "POST",
            "/inspect",
            body=payload,
            headers={
                "Content-Type": "application/json",
                "X-ProtCross-Desktop-Token": "secret-token",
            },
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))

        assert response.status == 200
        assert body["schema_version"] == "protcross-structure-inspection-v1"
        assert body["selected_chains"] == ["A"]
        assert body["scorable_residue_count"] == 1
        assert body["requires_truncation"] is False
        assert body["input_interpretation"]["assembly"].startswith("coordinates_as_supplied")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.network
def test_desktop_server_serves_result_file_for_viewer(tmp_path):
    structure = tmp_path / "result.pdb"
    structure.write_text("HEADER    TEST\n", encoding="utf-8")
    backend = DesktopBackend(root=tmp_path)
    backend.register_readable_output(structure)
    server = create_server("127.0.0.1", 0, root=tmp_path, token="secret-token", backend=backend)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        conn.request("GET", f"/file?path={quote(str(structure))}&token=secret-token")
        response = conn.getresponse()
        body = response.read().decode("utf-8")

        assert response.status == 200
        assert body == "HEADER    TEST\n"
        assert response.getheader("content-type") == "chemical/x-pdb"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.network
def test_desktop_server_rejects_unregistered_file_reads(tmp_path):
    structure = tmp_path / "secret.pdb"
    structure.write_text("HEADER    SECRET\n", encoding="utf-8")
    server = create_server("127.0.0.1", 0, root=tmp_path, token="secret-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        conn.request("GET", f"/file?path={quote(str(structure))}&token=secret-token")
        response = conn.getresponse()
        response.read()

        assert response.status == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.network
def test_desktop_server_requires_token_for_stateful_api(tmp_path):
    server = create_server("127.0.0.1", 0, root=tmp_path, token="secret-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        conn.request("GET", "/status")
        response = conn.getresponse()
        response.read()

        assert response.status == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.network
def test_desktop_server_rejects_null_origin_file_access(tmp_path):
    structure = tmp_path / "result.pdb"
    structure.write_text("HEADER    TEST\n", encoding="utf-8")
    backend = DesktopBackend(root=tmp_path)
    backend.register_readable_output(structure)
    server = create_server("127.0.0.1", 0, root=tmp_path, token="secret-token", backend=backend)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        conn.request(
            "GET",
            f"/file?path={quote(str(structure))}&token=secret-token",
            headers={"Origin": "null"},
        )
        response = conn.getresponse()
        response.read()

        assert response.status == 403
        assert response.getheader("access-control-allow-origin") is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.network
def test_desktop_server_rejects_cross_origin_side_effects(tmp_path):
    output_zip = tmp_path / "evil.zip"
    server = create_server("127.0.0.1", 0, root=tmp_path, token="secret-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        conn.request(
            "POST",
            "/diagnostics/export",
            body=json.dumps({"output_zip": str(output_zip)}),
            headers={"Origin": "https://evil.example", "Content-Type": "text/plain"},
        )
        response = conn.getresponse()
        response.read()

        assert response.status == 403
        assert output_zip.exists() is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
