"""Configuration and filesystem layout for ProtCross Desktop."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from protcross.assets import (
    DEFAULT_ASSETS,
    DEFAULT_CHECKPOINT_FILENAME,
    DEFAULT_ESM_URL,
    DEFAULT_PCA_FILENAME,
)


APP_NAME = "ProtCross"
DESKTOP_MANIFEST = "protcross-desktop-assets.json"
ESM_LICENSE_URL = "https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement"
ESM_MODEL_URL = DEFAULT_ESM_URL
ESM_FILENAME = DEFAULT_ASSETS[0].filename
ESM_EXPECTED_SHA256 = DEFAULT_ASSETS[0].sha256


@dataclass(frozen=True)
class DesktopPaths:
    root: Path
    assets_dir: Path
    runtime_dir: Path
    logs_dir: Path
    outputs_dir: Path
    diagnostics_dir: Path
    manifest_path: Path
    bundled_assets_dir: Path

    @classmethod
    def discover(cls, root: str | Path | None = None) -> "DesktopPaths":
        root_path = Path(root).expanduser() if root else _default_root()
        bundled = Path(os.environ.get("PROTCROSS_DESKTOP_BUNDLED_ASSETS", root_path / "bundled-assets"))
        return cls(
            root=root_path,
            assets_dir=root_path / "assets",
            runtime_dir=root_path / "runtime",
            logs_dir=root_path / "logs",
            outputs_dir=root_path / "outputs",
            diagnostics_dir=root_path / "diagnostics",
            manifest_path=root_path / "assets" / DESKTOP_MANIFEST,
            bundled_assets_dir=bundled.expanduser(),
        )

    def ensure(self) -> None:
        for path in (
            self.root,
            self.assets_dir,
            self.runtime_dir,
            self.logs_dir,
            self.outputs_dir,
            self.diagnostics_dir,
            self.bundled_assets_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def default_checkpoint(self) -> Path:
        return self.bundled_assets_dir / DEFAULT_CHECKPOINT_FILENAME

    @property
    def default_pca(self) -> Path:
        return self.bundled_assets_dir / DEFAULT_PCA_FILENAME

    @property
    def managed_esm_weights(self) -> Path:
        return self.assets_dir / ESM_FILENAME

    def env_python(self, backend: str) -> Path:
        if backend not in {"cpu", "gpu"}:
            raise ValueError("backend must be 'cpu' or 'gpu'")
        env_dir = self.runtime_dir / f"{backend}-env"
        return env_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _default_root() -> Path:
    override = os.environ.get("PROTCROSS_DESKTOP_HOME")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path.home() / ".local" / "share" / "protcross-desktop"
