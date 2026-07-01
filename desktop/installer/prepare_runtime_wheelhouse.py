#!/usr/bin/env python3
"""Prepare hash-locked Python runtime wheels for ProtCross Desktop releases."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


CPU_TORCH = {
    "torch": "2.3.1",
    "torchvision": "0.18.1",
}
WINDOWS_CPU_TORCH = {
    "torch": "2.3.1+cpu",
    "torchvision": "0.18.1+cpu",
}
WINDOWS_GPU_TORCH = {
    "torch": "2.3.1+cu121",
    "torchvision": "0.18.1+cu121",
}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    runtime_dir = args.runtime_dir.resolve()
    wheelhouse = (args.wheelhouse or runtime_dir / "wheelhouse").resolve()
    wheelhouse.mkdir(parents=True, exist_ok=True)

    local_wheel = args.local_protcross_wheel.resolve() if args.local_protcross_wheel else None
    if local_wheel and not local_wheel.exists():
        raise FileNotFoundError(f"Local ProtCross wheel not found: {local_wheel}")

    copied_uv = _copy_uv(runtime_dir, args.uv_bin)
    if copied_uv:
        print(f"Copied uv bootstrapper: {copied_uv}")

    backends = ["cpu", "gpu"] if args.backend == "all" else [args.backend]
    desktop_backend_dir = args.desktop_backend_dir.resolve() if args.desktop_backend_dir else runtime_dir.parent / "backend"
    if not desktop_backend_dir.exists():
        raise FileNotFoundError(f"Desktop backend package not found: {desktop_backend_dir}")
    with tempfile.TemporaryDirectory(prefix="protcross-desktop-backend-wheel-") as shared_tmp:
        desktop_backend_wheel = _build_local_wheel(desktop_backend_dir, Path(shared_tmp))
        for backend in backends:
            _prepare_backend(
                runtime_dir=runtime_dir,
                wheelhouse=wheelhouse,
                backend=backend,
                local_desktop_backend_wheel=desktop_backend_wheel,
                local_protcross_wheel=local_wheel,
                extra_pip_args=args.extra_pip_arg,
            )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=Path("desktop/runtime"))
    parser.add_argument("--wheelhouse", type=Path, default=None)
    parser.add_argument("--backend", choices=("cpu", "gpu", "all"), default="all")
    parser.add_argument(
        "--desktop-backend-dir",
        type=Path,
        default=None,
        help="Path to desktop/backend; a wheel is built and included in the hash bundle.",
    )
    parser.add_argument(
        "--local-protcross-wheel",
        type=Path,
        default=None,
        help="Use a locally built protcross wheel instead of downloading protcross from PyPI.",
    )
    parser.add_argument(
        "--uv-bin",
        type=Path,
        default=None,
        help="uv executable to bundle. Defaults to the uv executable on PATH.",
    )
    parser.add_argument(
        "--extra-pip-arg",
        action="append",
        default=[],
        help="Additional argument forwarded to pip wheel; repeat as needed.",
    )
    return parser


def _prepare_backend(
    *,
    runtime_dir: Path,
    wheelhouse: Path,
    backend: str,
    local_desktop_backend_wheel: Path,
    local_protcross_wheel: Path | None,
    extra_pip_args: list[str],
) -> None:
    requirements_lock = runtime_dir / "requirements-common.lock"
    if not requirements_lock.exists():
        raise FileNotFoundError(f"Missing desktop runtime lock file: {requirements_lock}")

    with tempfile.TemporaryDirectory(prefix=f"protcross-desktop-{backend}-") as tmp:
        tmp_dir = Path(tmp)
        temp_requirements = tmp_dir / "requirements.txt"
        temp_requirements.write_text(
            _backend_requirements(
                requirements_lock=requirements_lock,
                backend=backend,
                use_local_protcross=local_protcross_wheel is not None,
            ),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--wheel-dir",
            str(tmp_dir / "wheels"),
            "-r",
            str(temp_requirements),
        ]
        command.extend(_torch_index_args(backend))
        command.extend(extra_pip_args)
        print("+ " + " ".join(command))
        subprocess.run(command, check=True)

        if local_protcross_wheel:
            shutil.copy2(local_protcross_wheel, tmp_dir / "wheels" / local_protcross_wheel.name)
        shutil.copy2(local_desktop_backend_wheel, tmp_dir / "wheels" / local_desktop_backend_wheel.name)

        backend_wheels = []
        for wheel in sorted((tmp_dir / "wheels").glob("*.whl")):
            copied = wheelhouse / wheel.name
            shutil.copy2(wheel, copied)
            backend_wheels.append(copied)

    _write_hash_requirements(runtime_dir / f"requirements-{backend}.hashes", backend_wheels)
    print(f"Wrote {runtime_dir / f'requirements-{backend}.hashes'}")


def _build_local_wheel(package_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(package_dir / "build", ignore_errors=True)
    shutil.rmtree(package_dir / "protcross_desktop_backend.egg-info", ignore_errors=True)
    before = {path.resolve() for path in output_dir.glob("*.whl")}
    command = [sys.executable, "-m", "build", "--wheel", "--outdir", str(output_dir)]
    print("+ " + " ".join(command) + f"  # cwd={package_dir}")
    subprocess.run(command, cwd=package_dir, check=True)
    wheels = [
        path
        for path in output_dir.glob("protcross_desktop_backend-*.whl")
        if path.resolve() not in before
    ]
    shutil.rmtree(package_dir / "build", ignore_errors=True)
    shutil.rmtree(package_dir / "protcross_desktop_backend.egg-info", ignore_errors=True)
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one desktop backend wheel from {package_dir}, found {len(wheels)}.")
    return wheels[0]


def _backend_requirements(*, requirements_lock: Path, backend: str, use_local_protcross: bool) -> str:
    lines = []
    for line in requirements_lock.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            lines.append(line)
            continue
        package_name = stripped.split("==", 1)[0].strip().lower().replace("_", "-")
        if use_local_protcross and package_name == "protcross":
            continue
        lines.append(line)

    torch_versions = _torch_versions(backend)
    lines.extend(f"{name}=={version}" for name, version in torch_versions.items())
    return "\n".join(lines) + "\n"


def _torch_versions(backend: str) -> dict[str, str]:
    system = platform.system()
    if system == "Windows":
        return WINDOWS_GPU_TORCH if backend == "gpu" else WINDOWS_CPU_TORCH
    return CPU_TORCH


def _torch_index_args(backend: str) -> list[str]:
    system = platform.system()
    if system != "Windows":
        return []
    index = "https://download.pytorch.org/whl/cu121" if backend == "gpu" else "https://download.pytorch.org/whl/cpu"
    return ["--index-url", index, "--extra-index-url", "https://pypi.org/simple"]


def _write_hash_requirements(output_path: Path, wheels: list[Path]) -> None:
    entries = []
    seen = set()
    for wheel in sorted(wheels):
        name, version = _wheel_name_version(wheel)
        key = (name, version, _sha256(wheel))
        if key in seen:
            continue
        seen.add(key)
        entries.append(f"{name}=={version} --hash=sha256:{key[2]}")
    if not entries:
        raise RuntimeError("No wheels were prepared for the backend runtime.")
    output_path.write_text("\n".join(entries) + "\n", encoding="utf-8")


def _wheel_name_version(wheel: Path) -> tuple[str, str]:
    stem = wheel.name[:-4] if wheel.name.endswith(".whl") else wheel.name
    parts = stem.split("-")
    if len(parts) < 5:
        raise ValueError(f"Invalid wheel filename: {wheel.name}")
    return parts[0].replace("_", "-").lower(), parts[1]


def _copy_uv(runtime_dir: Path, uv_bin: Path | None) -> Path | None:
    source = uv_bin or _which_uv()
    if source is None:
        print("uv executable not found on PATH; validate_runtime_bundle.py will fail until uv is bundled.", file=sys.stderr)
        return None
    source = source.resolve()
    suffix = ".exe" if os.name == "nt" else ""
    target = runtime_dir / "uv" / f"uv{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (target.parent / "uv.sha256").write_text(f"{_sha256(target)}  {target.name}\n", encoding="utf-8")
    return target


def _which_uv() -> Path | None:
    path = shutil.which("uv")
    return Path(path) if path else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
