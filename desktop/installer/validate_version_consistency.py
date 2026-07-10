#!/usr/bin/env python3
"""Validate that every ProtCross release surface uses one application version."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


VERSION_RE = r"[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9_.+-]*)?"


def core_version(repo_root: str | Path) -> str:
    """Return the canonical version from the root ``pyproject.toml``."""
    return _toml_string(Path(repo_root) / "pyproject.toml", "project", "version")


def validate_version_consistency(
    repo_root: str | Path,
    *,
    expected_version: str | None = None,
) -> list[str]:
    """Return human-readable release version consistency errors."""
    root = Path(repo_root).resolve()
    errors: list[str] = []
    try:
        canonical = core_version(root)
    except Exception as exc:
        return [f"core project version could not be read: {exc}"]
    expected = expected_version or canonical
    _expect(errors, "core pyproject version", canonical, expected)

    checks = (
        (
            "protcross.__version__",
            lambda: _python_version(root / "src" / "protcross" / "__init__.py"),
        ),
        (
            "desktop backend project version",
            lambda: _toml_string(root / "desktop" / "backend" / "pyproject.toml", "project", "version"),
        ),
        (
            "desktop backend protcross pin",
            lambda: _backend_core_pin(root / "desktop" / "backend" / "pyproject.toml"),
        ),
        (
            "frontend package version",
            lambda: _json(root / "desktop" / "frontend" / "package.json")["version"],
        ),
        (
            "frontend lockfile version",
            lambda: _json(root / "desktop" / "frontend" / "package-lock.json")["version"],
        ),
        (
            "frontend lockfile root package version",
            lambda: _json(root / "desktop" / "frontend" / "package-lock.json")["packages"][""]["version"],
        ),
        (
            "Tauri Cargo package version",
            lambda: _toml_string(root / "desktop" / "src-tauri" / "Cargo.toml", "package", "version"),
        ),
        (
            "Tauri Cargo lock package version",
            lambda: _cargo_lock_package_version(
                root / "desktop" / "src-tauri" / "Cargo.lock",
                "protcross-desktop",
            ),
        ),
        (
            "Tauri config version",
            lambda: _json(root / "desktop" / "src-tauri" / "tauri.conf.json")["version"],
        ),
        (
            "desktop runtime protcross pin",
            lambda: _requirements_exact_pin(
                root / "desktop" / "runtime" / "requirements-common.lock",
                "protcross",
            ),
        ),
    )
    for label, read_value in checks:
        try:
            _expect(errors, label, str(read_value()), expected)
        except Exception as exc:
            errors.append(f"{label} could not be read: {exc}")

    workflow = root / ".github" / "workflows" / "desktop-release.yml"
    try:
        workflow_text = workflow.read_text(encoding="utf-8")
        workflow_checks = (
            (
                "desktop release workflow DESKTOP_VERSION",
                _required_match(workflow_text, rf'^\s*DESKTOP_VERSION:\s*["\']?({VERSION_RE})["\']?\s*$', workflow),
                expected,
            ),
            (
                "desktop release workflow default tag",
                _required_match(workflow_text, rf'^\s*default:\s*["\']?v({VERSION_RE})["\']?\s*$', workflow),
                expected,
            ),
        )
        for label, actual, wanted in workflow_checks:
            _expect(errors, label, actual, wanted)
        release_note_version = _required_match(
            workflow_text,
            rf"^\s*# ProtCross Desktop ({VERSION_RE})\s*$",
            workflow,
        )
        _expect(errors, "desktop release notes heading", release_note_version, expected)
        artifact_versions = re.findall(rf"ProtCross_Desktop_({VERSION_RE})_", workflow_text)
        if not artifact_versions:
            errors.append("desktop release workflow contains no literal versioned artifact names")
        for artifact_version in sorted(set(artifact_versions)):
            _expect(errors, "desktop release artifact filename version", artifact_version, expected)
        workflow_copy_versions = re.findall(rf"for the ({VERSION_RE}) desktop workflow", workflow_text)
        for copy_version in sorted(set(workflow_copy_versions)):
            _expect(errors, "desktop release notes workflow version", copy_version, expected)
    except Exception as exc:
        errors.append(f"desktop release workflow version could not be read: {exc}")

    return errors


def _expect(errors: list[str], label: str, actual: str, expected: str) -> None:
    if actual != expected:
        errors.append(f"{label} is {actual!r}; expected {expected!r}")


def _toml_string(path: Path, section: str, key: str) -> str:
    current_section: str | None = None
    key_pattern = re.compile(rf"^{re.escape(key)}\s*=\s*[\"']([^\"']+)[\"']\s*$")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        section_match = re.fullmatch(r"\[([^]]+)\]", line)
        if section_match:
            current_section = section_match.group(1)
            continue
        if current_section == section:
            match = key_pattern.match(line)
            if match:
                return match.group(1)
    raise ValueError(f"missing [{section}] {key} in {path}")


def _python_version(path: Path) -> str:
    return _required_match(
        path.read_text(encoding="utf-8"),
        rf'^__version__\s*=\s*["\']({VERSION_RE})["\']\s*$',
        path,
    )


def _backend_core_pin(path: Path) -> str:
    return _required_match(
        path.read_text(encoding="utf-8"),
        rf'["\']protcross==({VERSION_RE})["\']',
        path,
    )


def _requirements_exact_pin(path: Path, package: str) -> str:
    pattern = re.compile(rf"^{re.escape(package)}==({VERSION_RE})\s*$", re.IGNORECASE)
    matches = [
        match.group(1)
        for raw_line in path.read_text(encoding="utf-8").splitlines()
        if (match := pattern.match(raw_line.strip()))
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one exact {package} pin in {path}, found {len(matches)}")
    return matches[0]


def _cargo_lock_package_version(path: Path, package: str) -> str:
    for block in path.read_text(encoding="utf-8").split("[[package]]")[1:]:
        name = re.search(r'^name\s*=\s*"([^"]+)"\s*$', block, re.MULTILINE)
        if name and name.group(1) == package:
            version = re.search(r'^version\s*=\s*"([^"]+)"\s*$', block, re.MULTILINE)
            if not version:
                break
            return version.group(1)
    raise ValueError(f"package {package!r} not found in {path}")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _required_match(text: str, pattern: str, source: Path) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise ValueError(f"version pattern {pattern!r} not found in {source}")
    return match.group(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--expected-version",
        default=None,
        help="Expected release version. Defaults to the root pyproject.toml project version.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    errors = validate_version_consistency(args.repo_root, expected_version=args.expected_version)
    if errors:
        for error in errors:
            print(f"version consistency error: {error}", file=sys.stderr)
        return 1
    version = args.expected_version or core_version(args.repo_root)
    print(f"Validated ProtCross {version} release version consistency.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
