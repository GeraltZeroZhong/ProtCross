"""Environment tests and local diagnostic export for ProtCross Desktop."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


DIAGNOSTIC_LOG_TEXT_LIMIT = 32_000
DIAGNOSTIC_LOG_READ_LIMIT = DIAGNOSTIC_LOG_TEXT_LIMIT * 2


IMPORT_CHECK = r"""
import json
from importlib import metadata
payload = {}
for name in ("torch", "torch_geometric", "esm", "protcross", "protcross_desktop"):
    try:
        mod = __import__(name)
        payload[name] = {"ok": True, "version": getattr(mod, "__version__", None)}
        distribution = "protcross-desktop-backend" if name == "protcross_desktop" else name
        try:
            payload[name]["distribution_version"] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            payload[name]["distribution_version"] = None
    except Exception as exc:
        payload[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
try:
    import torch
    payload["torch"]["cuda_available"] = bool(torch.cuda.is_available())
    payload["torch"]["cuda_version"] = getattr(torch.version, "cuda", None)
    payload["torch"]["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    payload["torch"]["mps_available"] = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    payload["torch"]["mps_built"] = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
    )
    x = torch.tensor([1.0, 2.0])
    payload["torch"]["tensor_ok"] = float(x.sum().item()) == 3.0
except Exception as exc:
    payload["torch_extra_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(payload))
"""


@dataclass(frozen=True)
class EnvTestResult:
    backend: str
    python: str
    ok: bool
    returncode: int | None
    checks: dict[str, Any]
    stdout: str
    stderr: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "python": self.python,
            "ok": self.ok,
            "returncode": self.returncode,
            "checks": self.checks,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
        }


def test_python_env(python: str | Path, *, backend: str, timeout: int = 60) -> EnvTestResult:
    python = Path(python).expanduser()
    if not python.exists():
        return EnvTestResult(
            backend=backend,
            python=str(python),
            ok=False,
            returncode=None,
            checks={},
            stdout="",
            stderr="",
            error="Python executable not found.",
        )
    try:
        completed = subprocess.run(
            [str(python), "-c", IMPORT_CHECK],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        return EnvTestResult(
            backend=backend,
            python=str(python),
            ok=False,
            returncode=None,
            checks={},
            stdout="",
            stderr="",
            error=f"{type(exc).__name__}: {exc}",
        )

    checks: dict[str, Any] = {}
    if completed.stdout.strip():
        try:
            checks = json.loads(completed.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError as exc:
            checks = {"parse_error": str(exc)}
    required = ("torch", "torch_geometric", "esm", "protcross", "protcross_desktop")
    imports_ok = all(bool(checks.get(name, {}).get("ok")) for name in required)
    tensor_ok = bool(checks.get("torch", {}).get("tensor_ok"))
    acceleration_ok = True
    if backend == "gpu":
        torch_checks = checks.get("torch", {})
        acceleration_ok = bool(
            torch_checks.get("cuda_available") or torch_checks.get("mps_available")
        )
    ok = completed.returncode == 0 and imports_ok and tensor_ok and acceleration_ok
    return EnvTestResult(
        backend=backend,
        python=str(python),
        ok=ok,
        returncode=completed.returncode,
        checks=checks,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def system_snapshot() -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version,
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def export_diagnostics(
    output_zip: str | Path,
    *,
    manifest: dict[str, Any],
    env_results: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
    logs_dir: str | Path | None = None,
) -> Path:
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    payload = sanitize_diagnostics_payload({
        "system": system_snapshot(),
        "manifest": manifest,
        "env_results": env_results,
        "extra": extra or {},
        "privacy": "Generated locally. Review before attaching to a GitHub issue.",
    })
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("diagnostics.json", json.dumps(payload, indent=2))
        included_logs = _sanitized_log_files(logs_dir)
        for name, content in included_logs:
            archive.writestr(f"logs/{name}", content)
        archive.writestr(
            "README.txt",
            "This diagnostic archive is generated locally by ProtCross Desktop. "
            "Review diagnostics.json and any sanitized log excerpts before attaching it to a GitHub issue.\n"
            f"Included log excerpts: {', '.join(name for name, _ in included_logs) or 'none'}\n",
        )
    return output_zip


def _sanitized_log_files(logs_dir: str | Path | None) -> list[tuple[str, str]]:
    if logs_dir is None:
        return []
    directory = Path(logs_dir)
    if not directory.is_dir():
        return []
    local_root = str(directory.parent)
    logs = []
    for path in sorted(directory.glob("*.log")):
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size > DIAGNOSTIC_LOG_READ_LIMIT:
                handle.seek(-DIAGNOSTIC_LOG_READ_LIMIT, os.SEEK_END)
            else:
                handle.seek(0)
            content = handle.read().decode("utf-8", errors="replace")
        logs.append(
            (
                path.name,
                _sanitize_log_text(
                    content,
                    local_root=local_root,
                    earlier_output_omitted=size > DIAGNOSTIC_LOG_READ_LIMIT,
                ),
            ),
        )
    return logs


def _sanitize_log_text(
    value: str,
    *,
    local_root: str,
    earlier_output_omitted: bool = False,
) -> str:
    sanitized = _redact_proxy_credentials(value)
    if local_root:
        sanitized = sanitized.replace(local_root, "<local-path>")
    sanitized = _redact_path(sanitized)
    sanitized = _redact_secret_words(sanitized)
    if len(sanitized) <= DIAGNOSTIC_LOG_TEXT_LIMIT and not earlier_output_omitted:
        return sanitized
    return (
        "[earlier log output truncated by ProtCross Desktop diagnostics]\n"
        + sanitized[-DIAGNOSTIC_LOG_TEXT_LIMIT:]
    )


def sanitize_diagnostics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _sanitize_value(payload)


def _sanitize_value(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {name: _sanitize_value(child, key=str(name)) for name, child in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(child, key=key) for child in value]
    if isinstance(value, str):
        sanitized = _redact_proxy_credentials(value)
        sanitized = _redact_path(sanitized)
        sanitized = _redact_secret_words(sanitized)
        if _is_local_path_key(key):
            sanitized = _redact_local_path_value(sanitized)
        if key in {"stdout", "stderr"}:
            sanitized = _truncate_text(sanitized)
        return sanitized
    return value


def _redact_proxy_credentials(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        url = match.group(0)
        try:
            parts = urlsplit(url)
        except ValueError:
            return url
        if not parts.hostname or "@" not in parts.netloc:
            return url
        host = parts.hostname
        if parts.port:
            host = f"{host}:{parts.port}"
        return urlunsplit((parts.scheme, f"<redacted>@{host}", parts.path, parts.query, parts.fragment))

    return re.sub(r"https?://[^\s\"'<>]+", replace, value)


def _redact_path(value: str) -> str:
    home = str(Path.home())
    if home and home in value:
        value = value.replace(home, "~")
    userprofile = os.environ.get("USERPROFILE")
    if userprofile and userprofile in value:
        value = value.replace(userprofile, "~")
    return value


def _is_local_path_key(key: str | None) -> bool:
    if not key:
        return False
    lowered = key.lower()
    return (
        lowered in {"python", "conda_python", "manifest", "root", "path"}
        or lowered.endswith("_path")
        or lowered.endswith("_dir")
        or lowered.endswith("directory")
    )


def _redact_local_path_value(value: str) -> str:
    if value.startswith(("http://", "https://")):
        return value
    if value.startswith("~"):
        return value
    if value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value):
        name = Path(value).name
        return f"<local-path>/{name}" if name else "<local-path>"
    return value


def _redact_secret_words(value: str) -> str:
    patterns = (
        r"(?i)(password|passwd|token|secret|api[_-]?key)=([^\s&]+)",
        r"(?i)(password|passwd|token|secret|api[_-]?key):\s*([^\s,;]+)",
    )
    for pattern in patterns:
        value = re.sub(pattern, r"\1=<redacted>", value)
    return value


def _truncate_text(value: str, *, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n[truncated by ProtCross Desktop diagnostics]"
