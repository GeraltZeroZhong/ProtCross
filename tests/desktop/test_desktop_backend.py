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
        self.records = [
            {
                "residue_id": "A_10",
                "residue_key": "model:0|chain:A|het:ATOM|resseq:10|icode:|resname:ALA",
                "residue_id_namespace": "pdb",
                "model_id": "0",
                "chain_id": "A",
                "auth_asym_id": "A",
                "label_asym_id": None,
                "residue_number": 10,
                "auth_seq_id": 10,
                "label_seq_id": None,
                "insertion_code": "",
                "resname": "ALA",
                "one_letter_code": "A",
                "input_bfactor": 20.0,
                "score": 0.9,
                "probability": 0.9,
                "is_binding": 1,
                "x": 1.0,
                "y": 2.0,
                "z": 3.0,
                "cluster_id": 1,
                "is_scored": 1,
                "rank_global": 1,
                "rank_within_chain": 1,
            }
        ]
        self.output_files = {
            "structure": str(output_pdb),
            "scores_tsv": str(scores_tsv),
            "pockets_json": str(pocket_json),
            "summary_json": str(summary_json),
        }
        if output_pdb:
            Path(output_pdb).parent.mkdir(parents=True, exist_ok=True)
            Path(output_pdb).write_text("fake", encoding="utf-8")
        if scores_tsv:
            Path(scores_tsv).parent.mkdir(parents=True, exist_ok=True)
            Path(scores_tsv).write_text(
                "residue_id\tresidue_key\tchain_id\tresidue_number\tmodel_score\tprobability\t"
                "is_scored\tx\ty\tz\trank_global\n"
                "A_10\tmodel:0|chain:A|het:ATOM|resseq:10|icode:|resname:ALA\tA\t10\t"
                "0.9\t0.9\t1\t1.0\t2.0\t3.0\t1\n",
                encoding="utf-8",
            )
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

    def to_records(self):
        return self.records

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


def _wait_for_batch(backend, job_id, timeout=5.0):
    terminal = {"completed", "completed_with_errors", "failed", "cancelled", "interrupted"}
    deadline = time.time() + timeout
    status = backend.batch_status(job_id)
    while status["status"] not in terminal and time.time() < deadline:
        time.sleep(0.01)
        status = backend.batch_status(job_id)
    assert status["status"] in terminal, status
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
    assert result["scores"][0]["residue_key"].startswith("model:0|chain:A")
    assert result["scores"][0]["is_scored"] == 1


def test_existing_result_package_can_be_reopened(tmp_path, monkeypatch):
    backend, input_pdb, _ = _ready_backend(tmp_path, monkeypatch)
    predicted = backend.predict_single(input_pdb, output_dir=tmp_path / "results")

    reopened = backend.open_result(predicted["output_files"]["summary_json"])

    assert reopened["ok"] is True
    assert reopened["summary"]["schema_version"] == "protcross-summary-v2"
    assert reopened["pockets"]["schema_version"] == "protcross-pocket-v2"
    assert reopened["output_files"] == predicted["output_files"]
    assert reopened["scores"][0]["rank_global"] == 1
    assert reopened["scores"][0]["x"] == pytest.approx(1.0)
    assert [residue["residue_id"] for residue in reopened["top_pocket_residues"]] == ["A_10", "A_11"]
    assert backend.readable_output_file(reopened["output_files"]["structure"]).exists()


def test_moved_result_package_relocates_historical_absolute_paths(tmp_path, monkeypatch):
    backend, input_path, _ = _ready_backend(tmp_path / "desktop", monkeypatch)
    predicted = backend.predict_single(input_path)
    original_dir = Path(predicted["output_files"]["summary_json"]).parent
    moved_dir = tmp_path / "moved-result"
    original_dir.rename(moved_dir)

    reopened = backend.open_result(moved_dir / Path(predicted["output_files"]["summary_json"]).name)

    assert Path(reopened["output_files"]["structure"]).parent == moved_dir
    assert Path(reopened["output_files"]["pockets_json"]).parent == moved_dir
    assert Path(reopened["output_files"]["summary_json"]).parent == moved_dir


def test_open_result_resolves_nested_relative_paths_from_summary(tmp_path):
    package = tmp_path / "package"
    members = package / "members"
    members.mkdir(parents=True)
    structure = members / "input.protcross.pdb"
    pockets = members / "input.protcross.pockets.json"
    scores = members / "input.protcross.scores.tsv"
    summary = package / "input.protcross.summary.json"
    structure.write_text("HEADER    TEST\n", encoding="utf-8")
    pockets.write_text(
        json.dumps({"schema_version": "protcross-pocket-v2", "clustered_pockets": []}),
        encoding="utf-8",
    )
    scores.write_text(
        "residue_id\tresidue_key\tchain_id\tresidue_number\tmodel_score\tis_scored\tx\ty\tz\trank_global\n"
        "A_1\tmodel:0|chain:A|het:ATOM|resseq:1|icode:|resname:ALA\tA\t1\t0.8\t1\t0\t0\t0\t1\n",
        encoding="utf-8",
    )
    summary.write_text(
        json.dumps(
            {
                "schema_version": "protcross-summary-v2",
                "output_files": {
                    "structure": "members/input.protcross.pdb",
                    "pockets_json": "members/input.protcross.pockets.json",
                    "scores_tsv": "members/input.protcross.scores.tsv",
                },
            }
        ),
        encoding="utf-8",
    )

    reopened = DesktopBackend(root=tmp_path / "desktop").open_result(summary)

    assert Path(reopened["output_files"]["structure"]) == structure.resolve()
    assert Path(reopened["output_files"]["pockets_json"]) == pockets.resolve()
    assert Path(reopened["output_files"]["scores_tsv"]) == scores.resolve()
    assert reopened["scores"][0]["score"] == pytest.approx(0.8)


def test_open_result_keeps_legacy_packages_viewable_without_extended_scores(tmp_path, monkeypatch):
    backend, input_pdb, _ = _ready_backend(tmp_path, monkeypatch)
    predicted = backend.predict_single(input_pdb)
    scores_path = Path(predicted["output_files"]["scores_tsv"])
    scores_path.unlink()

    reopened = backend.open_result(predicted["output_files"]["summary_json"])
    assert reopened["scores"] == []
    assert "Interactive regrouping is disabled" in reopened["summary"]["warnings"][-1]

    scores_path.write_text(
        "residue_id\tchain_id\tresidue_number\tprobability\tis_binding\nA_1\tA\t1\t0.8\t1\n",
        encoding="utf-8",
    )
    reopened = backend.open_result(predicted["output_files"]["summary_json"])
    assert reopened["scores"] == []
    assert "unsupported schema" in reopened["summary"]["warnings"][-1]


def test_open_result_rejects_unrelated_json(tmp_path):
    backend = DesktopBackend(root=tmp_path)
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text('{"schema_version": "other-v1"}', encoding="utf-8")

    with pytest.raises(ValueError, match="supported ProtCross summary"):
        backend.open_result(unrelated)


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


def test_batch_prediction_uses_bounded_predict_many_when_supported(tmp_path, monkeypatch):
    batch_calls = []

    class BatchPredictor(FakePredictor):
        def predict_many(self, structures, **kwargs):
            batch_calls.append((list(structures), kwargs))
            results = []
            for structure, paths in zip(structures, kwargs["output_paths"]):
                results.append(
                    self.predict(
                        structure,
                        output_pdb=paths["structure"],
                        scores_tsv=paths["scores_tsv"],
                        pocket_json=paths["pockets_json"],
                        summary_json=paths["summary_json"],
                    )
                )
            return results

    backend, input_pdb, _ = _ready_backend(
        tmp_path,
        monkeypatch,
        predictor_factory=lambda **kwargs: BatchPredictor(),
    )
    second = tmp_path / "second.pdb"
    second.write_text(MINIMAL_PDB, encoding="utf-8")

    job = backend.submit_batch(
        items=[
            {"input_structure": input_pdb, "chain_id": "A"},
            {"input_structure": second, "chain_id": None},
        ],
        output_dir=tmp_path / "batch",
    )
    deadline = time.time() + 5
    while time.time() < deadline:
        status = backend.batch_status(job["id"])
        if status["status"] in {"completed", "completed_with_errors", "failed"}:
            break
        time.sleep(0.05)

    status = backend.batch_status(job["id"])
    assert status["status"] == "completed"
    assert len(batch_calls) == 1
    assert batch_calls[0][1]["batch_size"] == 2
    assert batch_calls[0][1]["return_exceptions"] is True
    assert len(batch_calls[0][1]["structure_inspections"]) == 2
    assert batch_calls[0][1]["chain_ids"] == ["A", None]
    assert [item["chain_id"] for item in status["items"]] == ["A", None]


def test_batch_prediction_passes_per_item_chain_to_single_predictor(tmp_path, monkeypatch):
    calls = []

    class ChainPredictor(FakePredictor):
        def predict(self, input_structure, **kwargs):
            calls.append((Path(input_structure).name, kwargs.get("chain_id")))
            return super().predict(input_structure, **kwargs)

    backend, input_pdb, _ = _ready_backend(
        tmp_path,
        monkeypatch,
        predictor_factory=lambda **kwargs: ChainPredictor(),
    )
    second = tmp_path / "second.pdb"
    second.write_text(MINIMAL_PDB, encoding="utf-8")

    job = backend.submit_batch(
        items=[
            {"input_structure": input_pdb, "chain_id": "A"},
            {"input_structure": second},
        ]
    )
    status = _wait_for_batch(backend, job["id"])

    assert status["status"] == "completed"
    assert calls == [("input.pdb", "A"), ("second.pdb", None)]


def test_batch_history_restores_and_retry_only_failed_items(tmp_path, monkeypatch):
    attempts = {}

    class FlakyPredictor(FakePredictor):
        def predict(self, input_structure, **kwargs):
            name = Path(input_structure).name
            attempts[name] = attempts.get(name, 0) + 1
            if name == "second.pdb" and attempts[name] == 1:
                raise RuntimeError("simulated per-item failure")
            return super().predict(input_structure, **kwargs)

    def factory(**kwargs):
        return FlakyPredictor()

    backend, input_pdb, _ = _ready_backend(tmp_path, monkeypatch, predictor_factory=factory)
    second = tmp_path / "second.pdb"
    second.write_text(MINIMAL_PDB, encoding="utf-8")
    original = backend.submit_batch(
        items=[
            {"input_structure": input_pdb, "chain_id": "A"},
            {"input_structure": second, "chain_id": None},
        ],
        output_dir=tmp_path / "batch",
        threshold=0.65,
        pocket_cluster_cutoff=6.5,
        allow_truncation=True,
        device="cpu",
    )
    original_status = _wait_for_batch(backend, original["id"])
    assert original_status["status"] == "completed_with_errors"
    completed_output = Path(original_status["items"][0]["output_files"]["structure"])
    completed_bytes = completed_output.read_bytes()
    completed_mtime = completed_output.stat().st_mtime_ns

    restored = DesktopBackend(root=tmp_path, predictor_factory=factory)
    restored_status = restored.batch_status(original["id"])
    assert restored_status["status"] == "completed_with_errors"
    assert [item["status"] for item in restored_status["items"]] == ["completed", "failed"]
    assert any(job["id"] == original["id"] for job in restored.status()["activity"]["batch_jobs"])
    assert restored.readable_output_file(completed_output) == completed_output.resolve()
    restored_result = restored.batch_item_result(original["id"], str(input_pdb), chain_id="A")
    assert restored_result["chain_id"] == "A"
    assert restored_result["scores"][0]["is_scored"] == 1

    retry = restored.retry_failed_batch(original["id"])
    retry_status = _wait_for_batch(restored, retry["id"])

    assert retry_status["status"] == "completed"
    assert retry_status["retry_of"] == original["id"]
    assert retry_status["settings"] == original_status["settings"]
    assert [(item["input_structure"], item["chain_id"]) for item in retry_status["items"]] == [
        (str(second), None)
    ]
    assert attempts == {"input.pdb": 1, "second.pdb": 2}
    assert completed_output.read_bytes() == completed_bytes
    assert completed_output.stat().st_mtime_ns == completed_mtime
    with pytest.raises(ValueError, match="no failed or interrupted items"):
        restored.retry_failed_batch(retry["id"])


def test_batch_history_marks_inflight_work_interrupted_without_losing_completed_outputs(tmp_path):
    completed_structure = tmp_path / "completed.pdb"
    completed_structure.write_text("completed output", encoding="utf-8")
    backend = DesktopBackend(root=tmp_path)
    job = desktop_service.BatchJob(
        id="interrupted-job",
        items=[
            desktop_service.QueueItem(
                input_structure=str(tmp_path / "done.pdb"),
                chain_id="A",
                status="completed",
                output_files={"structure": str(completed_structure)},
            ),
            desktop_service.QueueItem(
                input_structure=str(tmp_path / "pending.pdb"),
                chain_id="B",
                status="running",
            ),
        ],
        status="running",
        created_at=time.time(),
        completed=1,
        output_dir=str(tmp_path / "batch"),
        threshold=0.7,
    )
    with backend._lock:
        backend._jobs[job.id] = job
        backend._persist_batch_jobs_locked()

    restored = DesktopBackend(root=tmp_path)
    status = restored.batch_status(job.id)

    assert status["status"] == "interrupted"
    assert [item["status"] for item in status["items"]] == ["completed", "interrupted"]
    assert [item["chain_id"] for item in status["items"]] == ["A", "B"]
    assert status["settings"]["threshold"] == pytest.approx(0.7)
    assert completed_structure.read_text(encoding="utf-8") == "completed output"
    assert restored.readable_output_file(completed_structure) == completed_structure.resolve()
    assert json.loads((tmp_path / desktop_service.BATCH_HISTORY_FILENAME).read_text(encoding="utf-8"))[
        "schema_version"
    ] == desktop_service.BATCH_HISTORY_SCHEMA
    assert not list(tmp_path.glob(".batch-jobs.json.*.tmp"))


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
    backend_log = backend.paths.logs_dir / "backend.log"
    backend_log.write_text(
        "beginning-only marker\n"
        + ("old output\n" * 10_000)
        + f"root={backend.paths.root}\n"
        + "proxy=http://user:pass@proxy.example:8080 token=diagnostic-secret\n"
        + "final model load failure\n",
        encoding="utf-8",
    )

    zip_path = Path(backend.export_diagnostics(tmp_path / "diag.zip"))

    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as archive:
        assert "diagnostics.json" in archive.namelist()
        assert "README.txt" in archive.namelist()
        assert "logs/backend.log" in archive.namelist()
        log_excerpt = archive.read("logs/backend.log").decode("utf-8")
    assert "final model load failure" in log_excerpt
    assert "earlier log output truncated" in log_excerpt
    assert "beginning-only marker" not in log_excerpt
    assert str(backend.paths.root) not in log_excerpt
    assert "user:pass" not in log_excerpt
    assert "diagnostic-secret" not in log_excerpt
    assert "http://<redacted>@proxy.example:8080" in log_excerpt


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
    server = create_server("127.0.0.1", 0, root=tmp_path, token="secret-token")
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
def test_desktop_server_retries_failed_batch_items(tmp_path):
    calls = []

    class RetryBackend:
        def retry_failed_batch(self, job_id):
            calls.append(job_id)
            return {"id": "retry-job", "status": "queued", "retry_of": job_id}

    server = create_server(
        "127.0.0.1",
        0,
        root=tmp_path,
        token="secret-token",
        backend=RetryBackend(),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        conn.request(
            "POST",
            "/batch/original-job/retry",
            body="{}",
            headers={
                "Content-Type": "application/json",
                "X-ProtCross-Desktop-Token": "secret-token",
            },
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))

        assert response.status == 200
        assert body == {"id": "retry-job", "status": "queued", "retry_of": "original-job"}
        assert calls == ["original-job"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.network
def test_desktop_server_preserves_blank_chain_in_batch_result_query(tmp_path):
    calls = []

    class BatchResultBackend:
        def batch_item_result(self, job_id, input_structure, *, chain_id=None):
            calls.append((job_id, input_structure, chain_id))
            return {"ok": True, "chain_id": chain_id}

    server = create_server(
        "127.0.0.1",
        0,
        root=tmp_path,
        token="secret-token",
        backend=BatchResultBackend(),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        conn.request(
            "GET",
            "/batch/job/result?input_structure=input.pdb&chain_id=",
            headers={"X-ProtCross-Desktop-Token": "secret-token"},
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))

        assert response.status == 200
        assert body == {"ok": True, "chain_id": ""}
        assert calls == [("job", "input.pdb", "")]
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
        conn.request(
            "GET",
            f"/file?path={quote(str(structure))}",
            headers={"X-ProtCross-Desktop-Token": "secret-token"},
        )
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
        conn.request(
            "GET",
            f"/file?path={quote(str(structure))}",
            headers={"X-ProtCross-Desktop-Token": "secret-token"},
        )
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


def test_desktop_server_refuses_to_start_without_token(tmp_path, monkeypatch):
    monkeypatch.delenv("PROTCROSS_DESKTOP_TOKEN", raising=False)

    with pytest.raises(ValueError, match="API token is required"):
        create_server("127.0.0.1", 0, root=tmp_path)


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
            f"/file?path={quote(str(structure))}",
            headers={"Origin": "null", "X-ProtCross-Desktop-Token": "secret-token"},
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
