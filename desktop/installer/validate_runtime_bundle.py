#!/usr/bin/env python3
"""Validate the hash-locked Python runtime bundle for ProtCross Desktop."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

from validate_version_consistency import core_version


HASH_LINE = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>\S+)\s+--hash=sha256:(?P<sha>[0-9a-f]{64})$")
REQUIRED_BACKEND_PACKAGES = {"protcross", "protcross-desktop-backend", "torch"}
VERSIONED_RUNTIME_PACKAGES = {"protcross", "protcross-desktop-backend"}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    runtime_dir = args.runtime_dir.resolve()
    backends = ["cpu", "gpu"] if args.backend == "all" else [args.backend]
    wheel_index = _index_wheels(runtime_dir / "wheelhouse")

    errors: list[str] = []
    try:
        expected_version = args.expected_protcross_version or core_version(args.repo_root)
    except Exception as exc:
        print(f"runtime bundle error: could not determine expected ProtCross version: {exc}", file=sys.stderr)
        return 1
    errors.extend(_validate_versioned_wheel_inventory(wheel_index, expected_version))
    if not (runtime_dir / "requirements-common.lock").exists():
        errors.append(f"missing runtime lock file: {runtime_dir / 'requirements-common.lock'}")
    else:
        errors.extend(_validate_common_lock(runtime_dir / "requirements-common.lock", expected_version))
    for backend in backends:
        errors.extend(
            _validate_hash_file(
                runtime_dir / f"requirements-{backend}.hashes",
                wheel_index,
                expected_version=expected_version,
            )
        )
    if not args.skip_uv:
        errors.extend(_validate_uv(runtime_dir))

    if errors:
        for error in errors:
            print(f"runtime bundle error: {error}", file=sys.stderr)
        return 1
    print(f"Validated ProtCross {expected_version} desktop runtime bundle at {runtime_dir}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=Path("desktop/runtime"))
    parser.add_argument("--backend", choices=("cpu", "gpu", "all"), default="all")
    parser.add_argument("--skip-uv", action="store_true", help="Do not require a bundled uv bootstrapper.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--expected-protcross-version",
        default=None,
        help="Required version for the ProtCross and desktop-backend wheels. Defaults to the root project version.",
    )
    return parser


def _index_wheels(wheelhouse: Path) -> dict[tuple[str, str, str], Path]:
    if not wheelhouse.is_dir():
        return {}
    index = {}
    for wheel in wheelhouse.glob("*.whl"):
        name, version = _wheel_name_version(wheel)
        index[(name, version, _sha256(wheel))] = wheel
    return index


def _validate_hash_file(
    hash_file: Path,
    wheel_index: dict[tuple[str, str, str], Path],
    *,
    expected_version: str,
) -> list[str]:
    errors = []
    if not hash_file.exists():
        return [f"missing backend hash file: {hash_file}"]
    lines = [
        line.strip()
        for line in hash_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        return [f"empty backend hash file: {hash_file}"]
    if not wheel_index:
        errors.append(f"missing or empty wheelhouse for {hash_file}")
    package_names = set()
    for line in lines:
        match = HASH_LINE.match(line)
        if not match:
            errors.append(f"invalid hash requirement in {hash_file}: {line}")
            continue
        package_name = match.group("name").replace("_", "-").lower()
        package_names.add(package_name)
        if package_name in VERSIONED_RUNTIME_PACKAGES and match.group("version") != expected_version:
            errors.append(
                f"{hash_file} pins {package_name}=={match.group('version')}; "
                f"expected {package_name}=={expected_version}"
            )
        key = (
            package_name,
            match.group("version"),
            match.group("sha"),
        )
        if key not in wheel_index:
            errors.append(f"hash requirement has no matching wheel in wheelhouse: {line}")
    missing = sorted(REQUIRED_BACKEND_PACKAGES - package_names)
    if missing:
        errors.append(f"{hash_file} is missing required package hash entries: {', '.join(missing)}")
    return errors


def _validate_common_lock(lock_file: Path, expected_version: str) -> list[str]:
    pins = []
    for line in lock_file.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^protcross==([^\s;]+)\s*$", line.strip(), re.IGNORECASE)
        if match:
            pins.append(match.group(1))
    if pins == [expected_version]:
        return []
    if not pins:
        return [f"{lock_file} is missing an exact protcross=={expected_version} pin"]
    return [
        f"{lock_file} pins protcross version(s) {', '.join(pins)}; expected only {expected_version}"
    ]


def _validate_versioned_wheel_inventory(
    wheel_index: dict[tuple[str, str, str], Path],
    expected_version: str,
) -> list[str]:
    errors = []
    for (name, version, _sha), path in wheel_index.items():
        if name in VERSIONED_RUNTIME_PACKAGES and version != expected_version:
            errors.append(
                f"stale {name} wheel in runtime wheelhouse: {path.name}; expected version {expected_version}"
            )
    return errors


def _validate_uv(runtime_dir: Path) -> list[str]:
    suffix = ".exe" if os.name == "nt" else ""
    uv_path = runtime_dir / "uv" / f"uv{suffix}"
    sha_path = runtime_dir / "uv" / "uv.sha256"
    if not uv_path.exists():
        return [f"missing bundled uv executable: {uv_path}"]
    if not sha_path.exists():
        return [f"missing bundled uv checksum: {sha_path}"]
    expected = sha_path.read_text(encoding="utf-8").split()[0].lower()
    actual = _sha256(uv_path)
    if expected != actual:
        return [f"uv checksum mismatch: expected {expected}, got {actual}"]
    return []


def _wheel_name_version(wheel: Path) -> tuple[str, str]:
    stem = wheel.name[:-4] if wheel.name.endswith(".whl") else wheel.name
    parts = stem.split("-")
    if len(parts) < 5:
        raise ValueError(f"Invalid wheel filename: {wheel.name}")
    return parts[0].replace("_", "-").lower(), parts[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
