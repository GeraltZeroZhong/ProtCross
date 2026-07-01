#!/usr/bin/env python3
"""Prepare generated inputs required by ProtCross Desktop release builds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    _prepare_bundled_assets(repo_root)
    _prepare_runtime_bundle(repo_root, backend=args.runtime_backend)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--runtime-backend",
        choices=("cpu", "gpu", "all"),
        default="cpu",
        help="Runtime wheelhouse to prepare for packaging. Public installers bundle the CPU backend only.",
    )
    return parser


def _prepare_bundled_assets(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root / "src"))
    from protcross.assets import DEFAULT_ASSET_BUNDLE, download_asset

    assets_dir = repo_root / "desktop" / "bundled-assets"
    validator = repo_root / "desktop" / "installer" / "validate_bundled_assets.py"
    if _command_ok([sys.executable, str(validator), "--assets-dir", str(assets_dir)]):
        return
    targets = {
        DEFAULT_ASSET_BUNDLE.assets[1].filename: repo_root / "checkpoints",
        DEFAULT_ASSET_BUNDLE.assets[2].filename: repo_root / "data",
    }
    for spec in (DEFAULT_ASSET_BUNDLE.assets[1], DEFAULT_ASSET_BUNDLE.assets[2]):
        output_dir = targets[spec.filename]
        output_dir.mkdir(parents=True, exist_ok=True)
        download_asset(spec, output_dir / spec.filename, force=False, verify=True)
    prepare_script = repo_root / "desktop" / "installer" / (
        "prepare_bundled_assets.ps1" if sys.platform == "win32" else "prepare_bundled_assets.sh"
    )
    if sys.platform == "win32":
        _run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(prepare_script)])
    else:
        _run(["bash", str(prepare_script)])


def _prepare_runtime_bundle(repo_root: Path, *, backend: str) -> None:
    runtime_dir = repo_root / "desktop" / "runtime"
    validator = repo_root / "desktop" / "installer" / "validate_runtime_bundle.py"
    if _command_ok([sys.executable, str(validator), "--runtime-dir", str(runtime_dir), "--backend", backend]):
        return
    wheel = _ensure_local_protcross_wheel(repo_root)
    _run(
        [
            sys.executable,
            str(repo_root / "desktop" / "installer" / "prepare_runtime_wheelhouse.py"),
            "--runtime-dir",
            str(runtime_dir),
            "--backend",
            backend,
            "--local-protcross-wheel",
            str(wheel),
        ]
    )
    _run([sys.executable, str(validator), "--runtime-dir", str(runtime_dir), "--backend", backend])


def _ensure_local_protcross_wheel(repo_root: Path) -> Path:
    wheels = sorted((repo_root / "dist").glob("protcross-*.whl"))
    if wheels:
        return wheels[-1]
    out_dir = repo_root / "dist"
    out_dir.mkdir(exist_ok=True)
    _run([sys.executable, "-m", "build", "--wheel", "--outdir", str(out_dir)], cwd=repo_root)
    wheels = sorted(out_dir.glob("protcross-*.whl"))
    if not wheels:
        raise RuntimeError("Local ProtCross wheel build did not produce a wheel.")
    return wheels[-1]


def _command_ok(command: list[str]) -> bool:
    return subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
