"""Validate ProtCross Desktop bundled checkpoint and PCA assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_EXPECTED = {
    "checkpoint": {
        "filename": "protcross-0.1.2-binding-moad-final.ckpt",
        "sha256": "ccb56884b21402a027bfae9d4779f38c8f534513d980a96d7cd78c9931748b65",
    },
    "pca": {
        "filename": "pca_esmc_128_binding_moad_0.1.2.pkl",
        "sha256": "0f4e11806a622642c07dad539cec4216030220c1b5f3fc44c7926a2f6bca4d62",
    },
}


def validate_bundled_assets(
    assets_dir: str | Path,
    *,
    expected: dict[str, dict[str, str]] | None = None,
) -> list[Path]:
    assets_dir = Path(assets_dir)
    expected = expected or DEFAULT_EXPECTED
    manifest_path = assets_dir / "protcross-desktop-bundled-assets.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing bundled asset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed_names = {"protcross-desktop-bundled-assets.json", ".gitkeep"}
    validated = []
    for key, expected_entry in expected.items():
        entry = manifest.get(key)
        if not isinstance(entry, dict):
            raise RuntimeError(f"Bundled asset manifest is missing {key!r} entry.")
        filename = expected_entry["filename"]
        sha256 = expected_entry["sha256"]
        if entry.get("filename") != filename:
            raise RuntimeError(f"{key} filename mismatch: expected {filename}, got {entry.get('filename')}")
        recorded_sha = entry.get("sha256") or entry.get("expected_sha256")
        if recorded_sha != sha256:
            raise RuntimeError(f"{key} SHA256 manifest mismatch: expected {sha256}, got {recorded_sha}")
        path = assets_dir / filename
        if not path.exists():
            raise RuntimeError(f"Missing bundled {key} asset: {path}")
        size_bytes = entry.get("size_bytes")
        if size_bytes is not None and size_bytes != path.stat().st_size:
            raise RuntimeError(f"{key} size mismatch: expected {size_bytes}, got {path.stat().st_size}")
        actual = sha256_file(path)
        if actual != sha256:
            raise RuntimeError(f"{key} SHA256 mismatch: expected {sha256}, got {actual}")
        allowed_names.add(filename)
        validated.append(path)
    extras = sorted(path for path in assets_dir.iterdir() if path.name not in allowed_names)
    if extras:
        forbidden = ", ".join(str(path) for path in extras)
        raise RuntimeError(f"Unexpected files in bundled assets directory: {forbidden}")
    return validated


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate ProtCross Desktop bundled assets.")
    parser.add_argument(
        "--assets-dir",
        default=Path(__file__).resolve().parents[1] / "bundled-assets",
        help="Directory containing desktop bundled assets.",
    )
    args = parser.parse_args(argv)
    validated = validate_bundled_assets(args.assets_dir)
    for path in validated:
        print(f"[ok] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
