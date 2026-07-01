#!/usr/bin/env python3
"""Validate that packaged desktop resources can start the Python sidecar."""

from __future__ import annotations

import argparse
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

BUNDLED_ASSET_MANIFEST = "protcross-desktop-bundled-assets.json"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    resource_dir = args.resource_dir.resolve()
    errors = _validate_resource_layout(resource_dir)
    if errors:
        for error in errors:
            print(f"sidecar resource error: {error}", file=sys.stderr)
        return 1
    if args.no_start:
        print(f"Validated packaged sidecar resource layout at {resource_dir}")
        return 0
    _start_and_probe_sidecar(resource_dir, args.python, args.timeout)
    print(f"Validated packaged sidecar startup from {resource_dir}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--no-start", action="store_true", help="Only validate resource layout.")
    return parser


def _validate_resource_layout(resource_dir: Path) -> list[str]:
    required = {
        "backend package": resource_dir / "backend" / "protcross_desktop" / "server.py",
        "python source package": resource_dir / "python-src" / "protcross" / "__init__.py",
        "runtime directory": resource_dir / "runtime",
        "bundled assets directory": resource_dir / "bundled-assets",
        "bundled asset manifest": resource_dir / "bundled-assets" / BUNDLED_ASSET_MANIFEST,
    }
    return [f"missing {label}: {path}" for label, path in required.items() if not path.exists()]


def _start_and_probe_sidecar(resource_dir: Path, python: str, timeout: float) -> None:
    port = _free_port()
    token = secrets.token_urlsafe(24)
    with tempfile.TemporaryDirectory(prefix="protcross-sidecar-smoke-") as root:
        env = dict(os.environ)
        python_paths = [
            str(resource_dir / "backend"),
            str(resource_dir / "python-src"),
        ]
        if env.get("PYTHONPATH"):
            python_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        env["PROTCROSS_DESKTOP_BUNDLED_ASSETS"] = str(resource_dir / "bundled-assets")
        env["PROTCROSS_DESKTOP_TOKEN"] = token
        process = subprocess.Popen(
            [
                python,
                "-m",
                "protcross_desktop.server",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--root",
                root,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        try:
            _wait_for_json(f"http://127.0.0.1:{port}/health", timeout=timeout, process=process)
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/status",
                headers={"Authorization": f"Bearer {token}"},
            )
            _wait_for_json(request, timeout=timeout, process=process)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _wait_for_json(url_or_request, *, timeout: float, process: subprocess.Popen) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=1)
            raise RuntimeError(
                "sidecar exited before readiness probe succeeded "
                f"(code {process.returncode}). stdout={stdout!r} stderr={stderr!r}"
            )
        try:
            with urllib.request.urlopen(url_or_request, timeout=2) as response:
                if response.status == 200:
                    response.read()
                    return
        except Exception as exc:
            last_error = exc
            time.sleep(0.25)
    raise TimeoutError(f"sidecar readiness probe timed out: {last_error}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    raise SystemExit(main())
