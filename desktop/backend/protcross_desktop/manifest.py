"""Desktop asset manifest helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from protcross.assets import sha256_file

from .config import ESM_EXPECTED_SHA256, ESM_FILENAME, ESM_LICENSE_URL, ESM_MODEL_URL


SCHEMA_VERSION = "protcross-desktop-assets-v1"


@dataclass
class DesktopManifest:
    schema_version: str = SCHEMA_VERSION
    esm_license_confirmed: bool = False
    esm_license_confirmed_at: str | None = None
    esm_license_url: str = ESM_LICENSE_URL
    esm_model_url: str = ESM_MODEL_URL
    esm_weights_path: str | None = None
    esm_source: str | None = None
    esm_expected_sha256: str | None = ESM_EXPECTED_SHA256
    esm_actual_sha256: str | None = None
    esm_size_bytes: int | None = None
    esm_mtime_ns: int | None = None
    esm_verified: bool | None = None
    checkpoint_path: str | None = None
    pca_path: str | None = None
    backend_mode: str | None = None
    conda_python: str | None = None
    backend_test_ok: bool | None = None
    backend_tested_at: str | None = None
    backend_test_mode: str | None = None
    backend_test_python: str | None = None
    proxy_url: str | None = None
    updated_at: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "DesktopManifest":
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {field.name for field in cls.__dataclass_fields__.values()}
        payload = {key: value for key, value in data.items() if key in known}
        extra = {key: value for key, value in data.items() if key not in known}
        manifest = cls(**payload)
        manifest.extra.update(extra)
        return manifest

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = utc_now()
        payload = self.to_dict()
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "esm_license_confirmed": self.esm_license_confirmed,
            "esm_license_confirmed_at": self.esm_license_confirmed_at,
            "esm_license_url": self.esm_license_url,
            "esm_model_url": self.esm_model_url,
            "esm_weights_path": self.esm_weights_path,
            "esm_source": self.esm_source,
            "esm_expected_sha256": self.esm_expected_sha256,
            "esm_actual_sha256": self.esm_actual_sha256,
            "esm_size_bytes": self.esm_size_bytes,
            "esm_mtime_ns": self.esm_mtime_ns,
            "esm_verified": self.esm_verified,
            "checkpoint_path": self.checkpoint_path,
            "pca_path": self.pca_path,
            "backend_mode": self.backend_mode,
            "conda_python": self.conda_python,
            "backend_test_ok": self.backend_test_ok,
            "backend_tested_at": self.backend_tested_at,
            "backend_test_mode": self.backend_test_mode,
            "backend_test_python": self.backend_test_python,
            "proxy_url": self.proxy_url,
            "updated_at": self.updated_at,
        }
        payload.update(self.extra)
        return payload

    def confirm_esm_license(self, *, license_url: str = ESM_LICENSE_URL, model_url: str = ESM_MODEL_URL) -> None:
        self.esm_license_confirmed = True
        self.esm_license_confirmed_at = utc_now()
        self.esm_license_url = license_url
        self.esm_model_url = model_url

    def set_esm_weights(self, path: str | Path, *, source: str) -> None:
        path = Path(path).expanduser()
        self.esm_weights_path = str(path)
        self.esm_source = source
        self._refresh_esm_metadata(path)

    def refresh_esm_verification_if_stale(self) -> bool:
        path = Path(self.esm_weights_path).expanduser() if self.esm_weights_path else None
        if not path or not path.exists():
            changed = any(
                value is not None
                for value in (self.esm_actual_sha256, self.esm_size_bytes, self.esm_mtime_ns, self.esm_verified)
            )
            self.esm_actual_sha256 = None
            self.esm_size_bytes = None
            self.esm_mtime_ns = None
            self.esm_verified = None
            return changed
        stat = path.stat()
        if (
            self.esm_actual_sha256
            and self.esm_size_bytes == stat.st_size
            and self.esm_mtime_ns == stat.st_mtime_ns
        ):
            return False
        self._refresh_esm_metadata(path, stat=stat)
        return True

    def esm_status(self) -> dict[str, Any]:
        path = Path(self.esm_weights_path).expanduser() if self.esm_weights_path else None
        present = bool(path and path.exists())
        return {
            "license_confirmed": self.esm_license_confirmed,
            "path": str(path) if path else None,
            "present": present,
            "source": self.esm_source,
            "expected_sha256": self.esm_expected_sha256,
            "actual_sha256": self.esm_actual_sha256,
            "size_bytes": self.esm_size_bytes,
            "mtime_ns": self.esm_mtime_ns,
            "verified": self.esm_verified,
            "filename": ESM_FILENAME,
        }

    def _refresh_esm_metadata(self, path: Path, *, stat: Any | None = None) -> None:
        if not path.exists():
            self.esm_actual_sha256 = None
            self.esm_size_bytes = None
            self.esm_mtime_ns = None
            self.esm_verified = None
            return
        stat = stat or path.stat()
        self.esm_actual_sha256 = sha256_file(path)
        self.esm_size_bytes = stat.st_size
        self.esm_mtime_ns = stat.st_mtime_ns
        self.esm_verified = (
            self.esm_actual_sha256 == self.esm_expected_sha256
            if self.esm_actual_sha256 and self.esm_expected_sha256
            else None
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
