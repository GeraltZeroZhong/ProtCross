"""Stateful backend service used by the ProtCross Desktop sidecar."""

from __future__ import annotations

import os
import shutil
import hashlib
import json
import sys
import threading
import traceback
import uuid
import time
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Callable

from protcross.assets import DownloadCancelled, AssetSpec, DEFAULT_ASSETS, download_asset, sha256_file

from .config import (
    DesktopPaths,
    ESM_EXPECTED_SHA256,
    ESM_FILENAME,
    ESM_LICENSE_URL,
    ESM_MODEL_URL,
)
from .diagnostics import export_diagnostics, test_python_env
from .manifest import DesktopManifest


PredictorFactory = Callable[..., Any]
DEFAULT_BATCH_STATUS_LIMIT = 500
MAX_RETAINED_FINISHED_JOBS = 20


@dataclass
class QueueItem:
    input_structure: str
    status: str = "queued"
    output_dir: str | None = None
    output_files: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    def to_status_dict(self) -> dict[str, Any]:
        return {
            "input_structure": self.input_structure,
            "status": self.status,
            "output_dir": self.output_dir,
            "output_files": dict(self.output_files),
            "error": self.error,
        }


@dataclass
class BatchJob:
    id: str
    items: list[QueueItem]
    status: str = "queued"
    created_at: float = 0.0
    completed: int = 0
    failed: int = 0
    cancel_requested: bool = False
    error: str | None = None

    def to_dict(self, *, limit: int | None = DEFAULT_BATCH_STATUS_LIMIT, offset: int = 0) -> dict[str, Any]:
        bounded_offset = max(0, int(offset))
        items = self.items[bounded_offset:]
        if limit is not None:
            items = items[: max(0, int(limit))]
        return {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "completed": self.completed,
            "failed": self.failed,
            "cancel_requested": self.cancel_requested,
            "error": self.error,
            "item_count": len(self.items),
            "items_offset": bounded_offset,
            "items_returned": len(items),
            "items": [item.to_status_dict() for item in items],
        }


@dataclass
class AssetDownloadJob:
    id: str
    filename: str
    expected_size_bytes: int | None
    status: str = "queued"
    downloaded_bytes: int = 0
    total_bytes: int | None = None
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> dict[str, Any]:
        total = self.total_bytes or self.expected_size_bytes
        end_time = self.completed_at or time.time()
        elapsed = max(0.0, end_time - self.started_at) if self.started_at else 0.0
        speed = self.downloaded_bytes / elapsed if elapsed > 0 else None
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": total,
            "percent": (100.0 * self.downloaded_bytes / total) if total else None,
            "bytes_per_second": speed,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "resumable": True,
        }


class DesktopBackend:
    """Desktop state manager for assets, diagnostics, prediction, and batch jobs."""

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        predictor_factory: PredictorFactory | None = None,
    ) -> None:
        self.paths = DesktopPaths.discover(root)
        self.paths.ensure()
        self.manifest = DesktopManifest.load(self.paths.manifest_path)
        self.manifest.checkpoint_path = _migrate_bundled_asset_path(
            self.manifest.checkpoint_path, self.paths.default_checkpoint
        )
        self.manifest.pca_path = _migrate_bundled_asset_path(self.manifest.pca_path, self.paths.default_pca)
        self.manifest.save(self.paths.manifest_path)
        self._predictor_factory = predictor_factory
        self._predictor: Any | None = None
        self._predictor_key: tuple[str, str, str, str] | None = None
        self._jobs: dict[str, BatchJob] = {}
        self._asset_downloads: dict[str, AssetDownloadJob] = {}
        self._readable_output_files: set[Path] = set()
        self._file_status_cache: dict[tuple[str, int, int, str | None], dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._predict_lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        with self._lock:
            assets = self.asset_status()
            issues = self.readiness_issues(assets)
            activity = {
                "batch_jobs": [job.to_dict(limit=0) for job in self._jobs.values()],
                "asset_downloads": [job.to_dict() for job in self._asset_downloads.values()],
            }
            return {
                "paths": {
                    "root": str(self.paths.root),
                    "assets_dir": str(self.paths.assets_dir),
                    "runtime_dir": str(self.paths.runtime_dir),
                    "outputs_dir": str(self.paths.outputs_dir),
                    "manifest": str(self.paths.manifest_path),
                },
                "manifest": self.manifest.to_dict(),
                "assets": assets,
                "backend": self.backend_status(),
                "readiness": {
                    "ready": len(issues) == 0,
                    "issues": issues,
                },
                "activity": activity,
            }

    def confirm_esm_license(
        self,
        *,
        license_url: str = ESM_LICENSE_URL,
        model_url: str = ESM_MODEL_URL,
    ) -> dict[str, Any]:
        with self._lock:
            self.manifest.confirm_esm_license(license_url=license_url, model_url=model_url)
            self.manifest.save(self.paths.manifest_path)
            return self.manifest.to_dict()

    def configure_backend(
        self,
        mode: str,
        *,
        conda_python: str | Path | None = None,
        proxy_url: str | None = None,
    ) -> dict[str, Any]:
        if mode not in {"cpu", "gpu", "conda"}:
            raise ValueError("backend mode must be one of: cpu, gpu, conda")
        if mode == "conda" and not conda_python:
            raise ValueError("conda_python is required for conda backend mode")
        with self._lock:
            self.manifest.backend_mode = mode
            self.manifest.conda_python = str(Path(conda_python).expanduser()) if conda_python else None
            self.manifest.backend_test_ok = None
            self.manifest.backend_tested_at = None
            self.manifest.backend_test_mode = None
            self.manifest.backend_test_python = None
            self.manifest.backend_test_package_version = None
            self.manifest.proxy_url = proxy_url
            self.manifest.save(self.paths.manifest_path)
            self._predictor = None
            self._predictor_key = None
        return self.backend_status()

    def import_esm_weights(self, path: str | Path, *, copy_to_cache: bool = False) -> dict[str, Any]:
        self._require_license()
        source_path = Path(path).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"ESM-C weights not found: {source_path}")
        _verify_import_source(source_path, self.manifest.esm_expected_sha256, "ESM-C weights")
        target = self.paths.managed_esm_weights if copy_to_cache else source_path
        if copy_to_cache and source_path.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy_atomic(source_path, target)
        with self._lock:
            self.manifest.set_esm_weights(target, source="imported")
            self.manifest.save(self.paths.manifest_path)
            self._invalidate_predictor()
            return self.manifest.esm_status()

    def import_checkpoint(self, path: str | Path) -> dict[str, Any]:
        source_path = Path(path).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"ProtCross checkpoint not found: {source_path}")
        _verify_import_source(source_path, DEFAULT_ASSETS[1].sha256, "ProtCross checkpoint")
        target = self.paths.assets_dir / DEFAULT_ASSETS[1].filename
        _copy_atomic(source_path, target)
        with self._lock:
            self.manifest.checkpoint_path = str(target)
            self.manifest.save(self.paths.manifest_path)
            self._invalidate_predictor()
        return self.asset_status()["checkpoint"]

    def import_pca(self, path: str | Path) -> dict[str, Any]:
        source_path = Path(path).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"ProtCross PCA asset not found: {source_path}")
        _verify_import_source(source_path, DEFAULT_ASSETS[2].sha256, "ProtCross PCA reducer")
        target = self.paths.assets_dir / DEFAULT_ASSETS[2].filename
        _copy_atomic(source_path, target)
        with self._lock:
            self.manifest.pca_path = str(target)
            self.manifest.save(self.paths.manifest_path)
            self._invalidate_predictor()
        return self.asset_status()["pca"]

    def download_esm_weights(
        self,
        *,
        url: str = ESM_MODEL_URL,
        force: bool = False,
        _progress_callback: Callable[[int, int | None], None] | None = None,
        _cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        self._require_license()
        spec = AssetSpec(
            name="ESM-C 600M weights",
            filename=ESM_FILENAME,
            url=url,
            sha256=ESM_EXPECTED_SHA256,
            size_bytes=DEFAULT_ASSETS[0].size_bytes,
        )
        with _temporary_proxy_env(self.manifest.proxy_url):
            download_asset(
                spec,
                self.paths.managed_esm_weights,
                force=force,
                verify=True,
                progress_callback=_progress_callback,
                cancel_event=_cancel_event,
            )
        with self._lock:
            self.manifest.set_esm_weights(self.paths.managed_esm_weights, source="downloaded")
            self.manifest.esm_model_url = url
            self.manifest.save(self.paths.manifest_path)
            self._invalidate_predictor()
            return self.manifest.esm_status()

    def start_esm_download(self, *, force: bool = False) -> dict[str, Any]:
        """Start the large ESM-C download without blocking the desktop API."""
        self._require_license()
        with self._lock:
            self._prune_finished_jobs_locked(self._asset_downloads)
            running = next(
                (job for job in self._asset_downloads.values() if job.status in {"queued", "running", "cancelling"}),
                None,
            )
            if running is not None:
                return running.to_dict()
            job = AssetDownloadJob(
                id=str(uuid.uuid4()),
                filename=ESM_FILENAME,
                expected_size_bytes=DEFAULT_ASSETS[0].size_bytes,
            )
            self._asset_downloads[job.id] = job
        threading.Thread(target=self._run_esm_download, args=(job.id, force), daemon=True).start()
        return job.to_dict()

    def esm_download_status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._asset_downloads:
                raise KeyError(f"Unknown asset download: {job_id}")
            return self._asset_downloads[job_id].to_dict()

    def cancel_esm_download(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._asset_downloads:
                raise KeyError(f"Unknown asset download: {job_id}")
            job = self._asset_downloads[job_id]
            if job.status in {"queued", "running"}:
                job.status = "cancelling"
                job.cancel_event.set()
            return job.to_dict()

    def _run_esm_download(self, job_id: str, force: bool) -> None:
        with self._lock:
            job = self._asset_downloads[job_id]
            job.status = "running"
            job.started_at = time.time()

        def progress(downloaded: int, total: int | None) -> None:
            with self._lock:
                job.downloaded_bytes = int(downloaded)
                if total:
                    job.total_bytes = int(total)

        try:
            self.download_esm_weights(
                force=force,
                _progress_callback=progress,
                _cancel_event=job.cancel_event,
            )
        except DownloadCancelled:
            with self._lock:
                job.status = "cancelled"
                job.error = None
                job.completed_at = time.time()
            return
        except Exception as exc:
            with self._lock:
                job.status = "failed"
                job.error = str(exc)
                job.completed_at = time.time()
            return
        with self._lock:
            job.status = "completed"
            job.completed_at = time.time()

    def asset_status(self) -> dict[str, Any]:
        checkpoint = Path(self.manifest.checkpoint_path).expanduser() if self.manifest.checkpoint_path else None
        pca = Path(self.manifest.pca_path).expanduser() if self.manifest.pca_path else None
        if self.manifest.refresh_esm_verification_if_stale():
            self.manifest.save(self.paths.manifest_path)
        esm = self.manifest.esm_status()
        checkpoint_status = self._cached_file_status(checkpoint, DEFAULT_ASSETS[1].sha256)
        pca_status = self._cached_file_status(pca, DEFAULT_ASSETS[2].sha256)
        return {
            "checkpoint": checkpoint_status,
            "pca": pca_status,
            "esm": esm,
            "ready": bool(
                _is_usable_file_status(checkpoint_status)
                and _is_usable_file_status(pca_status)
                and esm["license_confirmed"]
                and esm["present"]
                and esm["verified"] is True
            ),
        }

    def backend_status(self) -> dict[str, Any]:
        package_version = _required_package_version()
        mode = self.manifest.backend_mode
        python = self._configured_python()
        sidecar_python = Path(sys.executable).resolve()
        resolved_python = _resolve_existing_path(python)
        return {
            "mode": mode,
            "python": str(python) if python else None,
            "python_present": bool(python and python.exists()),
            "sidecar_python": str(sidecar_python),
            "runtime_matches_config": bool(resolved_python and resolved_python == sidecar_python),
            "backend_test_ok": self.manifest.backend_test_ok,
            "backend_tested_at": self.manifest.backend_tested_at,
            "backend_test_mode": self.manifest.backend_test_mode,
            "backend_test_python": self.manifest.backend_test_python,
            "backend_test_package_version": self.manifest.backend_test_package_version,
            "required_package_version": package_version,
            "proxy_url": self.manifest.proxy_url,
        }

    def readiness_issues(self, status: dict[str, Any] | None = None) -> list[str]:
        package_version = _required_package_version()
        status = status or self.asset_status()
        issues: list[str] = []
        if not status["esm"]["license_confirmed"]:
            issues.append("Confirm the ESM-C license before configuring weights.")
        if not self.manifest.backend_mode:
            issues.append("Select a backend mode: cpu, gpu, or conda.")
        else:
            backend = self.backend_status()
            if not backend["python_present"]:
                issues.append(f"Backend Python is missing: {backend['python']}")
            if self.manifest.backend_test_ok is not True:
                issues.append("Run and pass the backend environment test.")
            elif (
                self.manifest.backend_test_mode != self.manifest.backend_mode
                or self.manifest.backend_test_python != backend["python"]
            ):
                issues.append("Backend configuration changed; run the environment test again.")
            elif self.manifest.backend_test_package_version != package_version:
                installed = self.manifest.backend_test_package_version or "an older/unknown version"
                issues.append(
                    f"Backend was tested with ProtCross {installed}; install or test the {package_version} backend."
                )
            if backend["python_present"] and not backend["runtime_matches_config"]:
                issues.append("Restart the desktop backend so prediction runs inside the selected environment.")
        if not status["checkpoint"]["present"]:
            issues.append(f"Bundled checkpoint is missing: {status['checkpoint']['path']}")
        elif status["checkpoint"]["verified"] is False:
            issues.append("ProtCross checkpoint failed SHA256 verification; import the expected release checkpoint.")
        if not status["pca"]["present"]:
            issues.append(f"Bundled PCA asset is missing: {status['pca']['path']}")
        elif status["pca"]["verified"] is False:
            issues.append("ProtCross PCA asset failed SHA256 verification; import the expected release PCA.")
        if not status["esm"]["present"]:
            issues.append("Download or import ESM-C weights.")
        elif status["esm"]["verified"] is False:
            issues.append("ESM-C weights failed SHA256 verification; repair or import the expected file.")
        return issues

    def test_backend(self, mode: str | None = None) -> dict[str, Any]:
        selected = mode or self.manifest.backend_mode
        if selected not in {"cpu", "gpu", "conda"}:
            raise ValueError("Select a backend before running environment tests.")
        python = self.paths.env_python(selected) if selected in {"cpu", "gpu"} else self._configured_python()
        if python is None:
            raise ValueError("Choose a conda environment Python before testing the conda backend.")
        result = test_python_env(python, backend=selected).to_dict()
        package_version = _required_package_version()
        installed_version = result.get("checks", {}).get("protcross", {}).get("distribution_version")
        if result["ok"] and installed_version != package_version:
            result["ok"] = False
            result["error"] = (
                f"Backend has ProtCross {installed_version or 'unknown'}, but Desktop requires {package_version}. "
                "Reinstall this backend to upgrade it."
            )
        if selected == self.manifest.backend_mode:
            from .manifest import utc_now

            with self._lock:
                self.manifest.backend_test_ok = bool(result["ok"])
                self.manifest.backend_tested_at = utc_now()
                self.manifest.backend_test_mode = selected
                self.manifest.backend_test_python = str(python)
                self.manifest.backend_test_package_version = installed_version
                self.manifest.save(self.paths.manifest_path)
        return result

    def predict_single(
        self,
        input_structure: str | Path,
        *,
        output_dir: str | Path | None = None,
        threshold: float = 0.5,
        pocket_cluster_cutoff: float = 8.0,
        chain_id: str | None = None,
        allow_truncation: bool = False,
        device: str | None = None,
    ) -> dict[str, Any]:
        input_path = _validate_structure_path(input_structure)
        _validate_prediction_options(threshold, pocket_cluster_cutoff)
        self._require_ready()
        if output_dir is None:
            output_dir = self.paths.outputs_dir / input_path.stem
        with self._predict_lock:
            output_paths = _desktop_output_paths(input_path, output_dir)
            if any(path.exists() for path in output_paths.values()):
                output_dir = _next_available_run_dir(Path(output_dir).expanduser())
                output_paths = _desktop_output_paths(input_path, output_dir)
            predictor = self._get_predictor(device=device)
            result = predictor.predict(
                input_path,
                chain_id=chain_id,
                threshold=threshold,
                pocket_cluster_cutoff=pocket_cluster_cutoff,
                output_pdb=output_paths["structure"],
                scores_tsv=output_paths["scores_tsv"],
                pocket_json=output_paths["pockets_json"],
                summary_json=output_paths["summary_json"],
                allow_truncation=allow_truncation,
            )
        pockets = result.to_pocket_dict()
        top_pocket_residues = _top_pocket_residues(pockets)
        self.register_readable_output(output_paths["structure"])
        return {
            "ok": True,
            "summary": result.to_summary_dict(),
            "pockets": pockets,
            "top_pocket_residues": top_pocket_residues,
            "output_files": {key: str(path) for key, path in output_paths.items()},
        }

    def inspect_input_structure(
        self,
        input_structure: str | Path,
        *,
        chain_id: str | None = None,
    ) -> dict[str, Any]:
        """Inspect an input without loading model assets or requiring prediction readiness."""
        from protcross.data import inspect_structure

        input_path = _validate_structure_path(input_structure)
        return inspect_structure(input_path, chain_id=chain_id)

    def submit_batch(
        self,
        structures: list[str | Path],
        *,
        output_dir: str | Path | None = None,
        threshold: float = 0.5,
        pocket_cluster_cutoff: float = 8.0,
        allow_truncation: bool = False,
        device: str | None = None,
    ) -> dict[str, Any]:
        if not structures:
            raise ValueError("Batch requires at least one input structure.")
        input_paths = [_validate_structure_path(path) for path in structures]
        _validate_prediction_options(threshold, pocket_cluster_cutoff)
        from protcross.data import inspect_structure

        input_errors = []
        for path in input_paths:
            try:
                report = inspect_structure(path)
                if report["requires_truncation"] and not allow_truncation:
                    input_errors.append(
                        f"{path}: longest chain has {report['longest_chain_context']} scorable residues "
                        f"(limit {report['max_len']}); enable truncation or remove this input"
                    )
            except Exception as exc:
                input_errors.append(f"{path}: {exc}")
        if input_errors:
            shown = input_errors[:20]
            suffix = f"\n- … and {len(input_errors) - len(shown)} more" if len(input_errors) > len(shown) else ""
            raise ValueError(
                "Batch structure check failed before model loading:\n- " + "\n- ".join(shown) + suffix
            )
        self._require_ready()
        import time

        job = BatchJob(
            id=str(uuid.uuid4()),
            items=[QueueItem(input_structure=str(path)) for path in input_paths],
            created_at=time.time(),
        )
        with self._lock:
            self._prune_finished_jobs_locked(self._jobs)
            active = [existing.id for existing in self._jobs.values() if existing.status in {"queued", "running"}]
            if active:
                raise RuntimeError(f"Batch job already running: {active[0]}")
            self._jobs[job.id] = job
        thread = threading.Thread(
            target=self._run_batch,
            args=(job.id,),
            kwargs={
                "output_dir": output_dir,
                "threshold": threshold,
                "pocket_cluster_cutoff": pocket_cluster_cutoff,
                "allow_truncation": allow_truncation,
                "device": device,
            },
            daemon=True,
        )
        thread.start()
        return job.to_dict()

    def batch_status(self, job_id: str, *, limit: int | None = DEFAULT_BATCH_STATUS_LIMIT, offset: int = 0) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"Unknown batch job: {job_id}")
            return self._jobs[job_id].to_dict(limit=limit, offset=offset)

    def batch_item_result(self, job_id: str, input_structure: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"Unknown batch job: {job_id}")
            item = next(
                (candidate for candidate in self._jobs[job_id].items if candidate.input_structure == input_structure),
                None,
            )
            if item is None:
                raise KeyError(f"Unknown batch item for job {job_id}: {input_structure}")
            if item.status != "completed":
                raise RuntimeError(f"Batch item is not completed: {item.status}")
            output_files = dict(item.output_files)
        summary_path = output_files.get("summary_json")
        pockets_path = output_files.get("pockets_json")
        if not summary_path or not pockets_path:
            raise FileNotFoundError("Batch item result JSON files are missing.")
        summary = _read_json_file(summary_path)
        pockets = _read_json_file(pockets_path)
        structure_path = output_files.get("structure")
        if structure_path:
            self.register_readable_output(structure_path)
        return {
            "ok": True,
            "input_structure": input_structure,
            "summary": summary,
            "pockets": pockets,
            "top_pocket_residues": _top_pocket_residues(pockets),
            "output_files": output_files,
        }

    def cancel_batch(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"Unknown batch job: {job_id}")
            self._jobs[job_id].cancel_requested = True
            return self._jobs[job_id].to_dict()

    def export_diagnostics(self, output_zip: str | Path | None = None) -> str:
        if output_zip is None:
            output_zip = self.paths.diagnostics_dir / "protcross-desktop-diagnostics.zip"
        env_results = []
        if self.manifest.backend_mode:
            try:
                env_results.append(self.test_backend())
            except Exception as exc:
                env_results.append({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        path = export_diagnostics(
            output_zip,
            manifest=self.manifest.to_dict(),
            env_results=env_results,
            extra={"status": self.status()},
        )
        return str(path)

    def register_readable_output(self, path: str | Path) -> None:
        resolved = _resolve_output_file(path)
        with self._lock:
            self._readable_output_files.add(resolved)

    def readable_output_file(self, path: str | Path) -> Path:
        resolved = _resolve_output_file(path)
        if resolved.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
            raise PermissionError("Only annotated PDB/mmCIF result files can be read by the viewer endpoint.")
        with self._lock:
            if resolved not in self._readable_output_files:
                raise PermissionError("File is not a registered ProtCross Desktop prediction result.")
        return resolved

    def _run_batch(
        self,
        job_id: str,
        *,
        output_dir: str | Path | None,
        threshold: float,
        pocket_cluster_cutoff: float,
        allow_truncation: bool,
        device: str | None,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
        try:
            with self._predict_lock:
                predictor = self._get_predictor(device=device)
            for index, item in enumerate(job.items, start=1):
                with self._lock:
                    if job.cancel_requested:
                        job.status = "cancelled"
                        return
                    item.status = "running"
                input_path = Path(item.input_structure)
                batch_root = Path(output_dir).expanduser() if output_dir else self.paths.outputs_dir / "batch" / job.id
                item_output = _unique_batch_output_dir(batch_root, input_path)
                paths = _desktop_output_paths(input_path, item_output)
                try:
                    with self._predict_lock:
                        if any(path.exists() for path in paths.values()):
                            item_output = _next_available_run_dir(item_output)
                            paths = _desktop_output_paths(input_path, item_output)
                        predictor.predict(
                            input_path,
                            threshold=threshold,
                            pocket_cluster_cutoff=pocket_cluster_cutoff,
                            output_pdb=paths["structure"],
                            scores_tsv=paths["scores_tsv"],
                            pocket_json=paths["pockets_json"],
                            summary_json=paths["summary_json"],
                            allow_truncation=allow_truncation,
                        )
                    with self._lock:
                        item.status = "completed"
                        item.output_dir = str(item_output)
                        item.output_files = {key: str(path) for key, path in paths.items()}
                        self._readable_output_files.add(_resolve_output_file(paths["structure"]))
                        job.completed = index
                except Exception as exc:
                    with self._lock:
                        item.status = "failed"
                        item.error = f"{type(exc).__name__}: {exc}"
                        job.completed = index
                        job.failed += 1
            with self._lock:
                job.status = "completed" if job.failed == 0 else "completed_with_errors"
        except Exception as exc:
            with self._lock:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                for item in job.items:
                    if item.status in {"queued", "running"}:
                        item.status = "failed"
                        item.error = f"{type(exc).__name__}: {exc}"
                        job.failed += 1

    def _get_predictor(self, *, device: str | None) -> Any:
        checkpoint_path = Path(str(self.manifest.checkpoint_path)).expanduser()
        pca_path = Path(str(self.manifest.pca_path)).expanduser()
        esm_path = Path(str(self.manifest.esm_weights_path)).expanduser()
        checkpoint = str(checkpoint_path)
        pca = str(pca_path)
        esm = str(esm_path)
        selected_device = device or self._default_device()
        key = (
            _file_identity(checkpoint_path),
            _file_identity(pca_path),
            _file_identity(esm_path),
            selected_device,
        )
        with self._lock:
            if self._predictor is not None and self._predictor_key == key:
                return self._predictor
            factory = self._predictor_factory or _default_predictor_factory
            self._predictor = factory(
                ckpt_path=checkpoint,
                esm_weights=esm,
                pca_path=pca,
                device=selected_device,
                asset_version="0.1.2-desktop",
                asset_metadata=self._asset_metadata(),
                embedding_cache_dir=self.paths.root / "feature-cache",
                accept_esm_license=True,
            )
            self._predictor_key = key
            return self._predictor

    def _invalidate_predictor(self) -> None:
        with self._lock:
            self._predictor = None
            self._predictor_key = None
            self._file_status_cache.clear()

    @staticmethod
    def _prune_finished_jobs_locked(jobs: dict[str, Any]) -> None:
        active_statuses = {"queued", "running", "cancelling"}
        finished = [job for job in jobs.values() if job.status not in active_statuses]
        finished.sort(
            key=lambda job: float(
                getattr(job, "completed_at", None)
                or getattr(job, "created_at", None)
                or getattr(job, "started_at", None)
                or 0.0
            ),
            reverse=True,
        )
        for job in finished[MAX_RETAINED_FINISHED_JOBS:]:
            jobs.pop(job.id, None)

    def _cached_file_status(self, path: Path | None, expected_sha256: str | None) -> dict[str, Any]:
        if not path or not path.exists():
            return _file_status(path, expected_sha256)
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, expected_sha256)
        with self._lock:
            cached = self._file_status_cache.get(key)
            if cached is not None:
                return dict(cached)
        status = _file_status(path, expected_sha256)
        with self._lock:
            self._file_status_cache[key] = dict(status)
        return status

    def _default_device(self) -> str:
        if self.manifest.backend_mode == "cpu":
            return "cpu"
        if self.manifest.backend_mode == "gpu":
            if sys.platform == "darwin":
                return "mps"
            return "cuda"
        return "auto"

    def _configured_python(self) -> Path | None:
        mode = self.manifest.backend_mode
        if mode in {"cpu", "gpu"}:
            return self.paths.env_python(mode)
        if mode == "conda" and self.manifest.conda_python:
            return Path(self.manifest.conda_python).expanduser()
        return None

    def _asset_metadata(self) -> dict[str, Any]:
        status = self.asset_status()
        return {
            "asset_bundle_version": "0.1.2",
            "checkpoint": status["checkpoint"],
            "pca": status["pca"],
            "esm": status["esm"],
        }

    def _require_license(self) -> None:
        if not self.manifest.esm_license_confirmed:
            raise RuntimeError("ESM-C license must be reviewed and confirmed before configuring weights.")

    def _require_ready(self) -> None:
        issues = self.readiness_issues()
        if issues:
            raise RuntimeError("ProtCross Desktop is not ready for prediction: " + "; ".join(issues))
        if not self.manifest.backend_mode:
            raise RuntimeError("Select and test a backend before running prediction.")


def _default_predictor_factory(**kwargs: Any) -> Any:
    from protcross.inference import ProtCrossPredictor

    return ProtCrossPredictor.from_files(**kwargs)


def _migrate_bundled_asset_path(current: str | None, bundled_default: Path) -> str:
    if not current:
        return str(bundled_default)
    current_path = Path(current).expanduser()
    if current_path.exists():
        return str(current_path)
    if bundled_default.exists() and current_path.name == bundled_default.name:
        return str(bundled_default)
    return str(current_path)


def _required_package_version() -> str:
    from protcross import __version__ as package_version

    return os.environ.get("PROTCROSS_DESKTOP_VERSION", package_version)


def _file_status(path: Path | None, expected_sha256: str | None) -> dict[str, Any]:
    present = bool(path and path.exists())
    actual = sha256_file(path) if present and path else None
    return {
        "path": str(path) if path else None,
        "present": present,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "verified": actual == expected_sha256 if actual and expected_sha256 else None,
        "verification_source": "sha256" if actual else None,
    }


def _is_usable_file_status(status: dict[str, Any]) -> bool:
    return bool(status["present"] and status["verified"] is not False)


def _desktop_output_paths(input_structure: Path, output_dir: str | Path) -> dict[str, Path]:
    from protcross.cli.predict import _default_output_paths

    return _default_output_paths(input_structure, output_dir)


def _top_pocket_residues(pockets: dict[str, Any]) -> list[dict[str, Any]]:
    clustered = pockets.get("clustered_pockets") or []
    if not clustered:
        return []
    return list(clustered[0].get("residues") or [])


def _read_json_file(path: str | Path) -> dict[str, Any]:
    file_path = Path(path).expanduser()
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"JSON result file not found: {file_path}")
    return json.loads(file_path.read_text(encoding="utf-8"))


def _unique_batch_output_dir(root: str | Path, input_path: Path) -> Path:
    digest = hashlib.sha1(str(input_path.resolve()).encode("utf-8")).hexdigest()[:8]
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in input_path.stem)
    return Path(root).expanduser() / f"{safe_stem}-{digest}"


def _next_available_run_dir(root: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    for index in range(1000):
        suffix = "" if index == 0 else f"-{index + 1}"
        candidate = root / f"run-{stamp}{suffix}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"Could not allocate a unique result directory under {root}")


def _resolve_existing_path(path: Path | None) -> Path | None:
    if path is None or not path.exists():
        return None
    return path.resolve()


def _copy_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
    try:
        shutil.copyfile(source, tmp)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)


def _verify_import_source(source: Path, expected_sha256: str | None, label: str) -> None:
    if not expected_sha256:
        return
    actual = sha256_file(source)
    if actual != expected_sha256:
        raise ValueError(f"{label} failed SHA256 verification: expected {expected_sha256}, got {actual}")


def _file_identity(path: Path) -> str:
    stat = path.stat()
    return f"{path.resolve()}|size={stat.st_size}|mtime_ns={stat.st_mtime_ns}"


def _resolve_output_file(path: str | Path) -> Path:
    file_path = Path(path).expanduser()
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    return file_path.resolve()


def _validate_structure_path(input_structure: str | Path) -> Path:
    path = Path(input_structure).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Input structure not found: {path}")
    if path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
        raise ValueError(f"Unsupported input structure extension: {path.suffix or '<none>'}")
    return path


def _validate_prediction_options(threshold: float, pocket_cluster_cutoff: float) -> None:
    if threshold is None or not isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be a finite value in [0, 1]")
    if pocket_cluster_cutoff is None or not isfinite(float(pocket_cluster_cutoff)) or float(pocket_cluster_cutoff) <= 0:
        raise ValueError("pocket_cluster_cutoff must be a finite positive value")


class _temporary_proxy_env:
    def __init__(self, proxy_url: str | None) -> None:
        self.proxy_url = proxy_url
        self.old: dict[str, str | None] = {}

    def __enter__(self) -> None:
        if not self.proxy_url:
            return
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            self.old[name] = os.environ.get(name)
            os.environ[name] = self.proxy_url

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        for name, value in self.old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
