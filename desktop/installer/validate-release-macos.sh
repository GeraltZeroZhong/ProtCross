#!/usr/bin/env bash
set -euo pipefail

INSTALL_ROOT="${INSTALL_ROOT:-$HOME/Library/Application Support/ProtCross}"
APP_PATH="${APP_PATH:-}"
DMG_PATH="${DMG_PATH:-}"
CONDA_PYTHON="${CONDA_PYTHON:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-root)
      INSTALL_ROOT="$2"
      shift 2
      ;;
    --app)
      APP_PATH="$2"
      shift 2
      ;;
    --dmg)
      DMG_PATH="$2"
      shift 2
      ;;
    --conda-python)
      CONDA_PYTHON="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../runtime" && pwd)"
INSTALLER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$INSTALLER_DIR/validate_runtime_bundle.py" --runtime-dir "$RUNTIME_DIR" --backend cpu

validate_app_assets() {
  local app_path="$1"
  local assets_dir="$app_path/Contents/Resources/bundled-assets"
  python "$INSTALLER_DIR/validate_bundled_assets.py" --assets-dir "$assets_dir"
}

if [[ -n "$APP_PATH" ]]; then
  codesign --verify --deep --strict --verbose=2 "$APP_PATH"
  spctl --assess --type execute --verbose=2 "$APP_PATH"
  validate_app_assets "$APP_PATH"
fi

if [[ -n "$DMG_PATH" ]]; then
  codesign --verify --verbose=2 "$DMG_PATH"
  spctl --assess --type open --context context:primary-signature --verbose=2 "$DMG_PATH"
  MOUNT_DIR="$(mktemp -d)"
  cleanup_mount() {
    hdiutil detach "$MOUNT_DIR" >/dev/null 2>&1 || true
    rmdir "$MOUNT_DIR" >/dev/null 2>&1 || true
  }
  trap cleanup_mount EXIT
  hdiutil attach "$DMG_PATH" -nobrowse -readonly -mountpoint "$MOUNT_DIR" >/dev/null
  INNER_APP="$(find "$MOUNT_DIR" -maxdepth 1 -name '*.app' -print -quit)"
  if [[ -z "$INNER_APP" ]]; then
    echo "No .app bundle found inside DMG: $DMG_PATH" >&2
    exit 1
  fi
  codesign --verify --deep --strict --verbose=2 "$INNER_APP"
  spctl --assess --type execute --verbose=2 "$INNER_APP"
  validate_app_assets "$INNER_APP"
  cleanup_mount
  trap - EXIT
fi

"$RUNTIME_DIR/test_backend.sh" --backend cpu --install-root "$INSTALL_ROOT"

if [[ -x "$INSTALL_ROOT/runtime/gpu-env/bin/python" ]]; then
  "$RUNTIME_DIR/test_backend.sh" --backend gpu --install-root "$INSTALL_ROOT"
fi

if [[ -n "$CONDA_PYTHON" ]]; then
  "$RUNTIME_DIR/test_backend.sh" --backend conda --conda-python "$CONDA_PYTHON"
fi

echo "ProtCross Desktop macOS release validation completed."
