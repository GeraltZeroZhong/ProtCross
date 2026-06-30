#!/usr/bin/env bash
set -euo pipefail

APP_PATH="${APP_PATH:-}"
DMG_PATH="${DMG_PATH:-}"
DMG_VOLNAME="${DMG_VOLNAME:-ProtCross Desktop}"
SIGNING_IDENTITY="${APPLE_SIGNING_IDENTITY:-}"
APPLE_ID_VALUE="${APPLE_ID:-}"
APPLE_PASSWORD_VALUE="${APPLE_PASSWORD:-}"
APPLE_TEAM_ID_VALUE="${APPLE_TEAM_ID:-}"
SKIP_NOTARIZATION=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --app)
      APP_PATH="$2"
      shift 2
      ;;
    --dmg)
      DMG_PATH="$2"
      shift 2
      ;;
    --identity)
      SIGNING_IDENTITY="$2"
      shift 2
      ;;
    --apple-id)
      APPLE_ID_VALUE="$2"
      shift 2
      ;;
    --password)
      APPLE_PASSWORD_VALUE="$2"
      shift 2
      ;;
    --team-id)
      APPLE_TEAM_ID_VALUE="$2"
      shift 2
      ;;
    --skip-notarization)
      SKIP_NOTARIZATION=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SIGNING_IDENTITY" ]]; then
  echo "APPLE_SIGNING_IDENTITY or --identity is required." >&2
  exit 2
fi

if [[ -z "$APP_PATH" ]]; then
  APP_PATH="$(find desktop/src-tauri/target -path '*/bundle/macos/ProtCross Desktop.app' -print -quit 2>/dev/null || true)"
fi
if [[ -z "$DMG_PATH" ]]; then
  DMG_PATH="$(find desktop/src-tauri/target -path '*/bundle/dmg/*.dmg' -print -quit 2>/dev/null || true)"
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "$APP_PATH" ]]; then
  UV_PATH="$APP_PATH/Contents/Resources/runtime/uv/uv"
  if [[ -f "$UV_PATH" ]]; then
    codesign --force --options runtime --timestamp --sign "$SIGNING_IDENTITY" "$UV_PATH"
    UV_SHA="$(shasum -a 256 "$UV_PATH" | awk '{print $1}')"
    printf "%s  %s\n" "$UV_SHA" "$(basename "$UV_PATH")" > "$APP_PATH/Contents/Resources/runtime/uv/uv.sha256"
  fi
  python "$SCRIPT_DIR/validate_bundled_assets.py" --assets-dir "$APP_PATH/Contents/Resources/bundled-assets"
  python "$SCRIPT_DIR/validate_runtime_bundle.py" --runtime-dir "$APP_PATH/Contents/Resources/runtime" --backend all
  codesign --force --options runtime --timestamp --sign "$SIGNING_IDENTITY" "$APP_PATH"
  codesign --verify --deep --strict --verbose=2 "$APP_PATH"
fi

if [[ -n "$APP_PATH" && -n "$DMG_PATH" ]]; then
  mkdir -p "$(dirname "$DMG_PATH")"
  rm -f "$DMG_PATH"
  STAGING_DIR="$(mktemp -d)"
  cleanup_staging() {
    rm -rf "$STAGING_DIR"
  }
  trap cleanup_staging EXIT
  ditto "$APP_PATH" "$STAGING_DIR/$(basename "$APP_PATH")"
  hdiutil create -volname "$DMG_VOLNAME" -srcfolder "$STAGING_DIR" -ov -format UDZO "$DMG_PATH"
  cleanup_staging
  trap - EXIT
fi

if [[ -n "$DMG_PATH" ]]; then
  codesign --force --timestamp --sign "$SIGNING_IDENTITY" "$DMG_PATH"
  codesign --verify --verbose=2 "$DMG_PATH"
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
  python "$SCRIPT_DIR/validate_bundled_assets.py" --assets-dir "$INNER_APP/Contents/Resources/bundled-assets"
  python "$SCRIPT_DIR/validate_runtime_bundle.py" --runtime-dir "$INNER_APP/Contents/Resources/runtime" --backend all
  cleanup_mount
  trap - EXIT
fi

if [[ -n "$DMG_PATH" ]]; then
  if [[ "$SKIP_NOTARIZATION" -eq 1 ]]; then
    echo "Skipping notarization by explicit request."
  else
    if [[ -z "$APPLE_ID_VALUE" || -z "$APPLE_PASSWORD_VALUE" || -z "$APPLE_TEAM_ID_VALUE" ]]; then
      echo "Apple notarization credentials are required for DMG releases. Use --skip-notarization only for internal testing." >&2
      exit 2
    fi
    xcrun notarytool submit "$DMG_PATH" \
      --apple-id "$APPLE_ID_VALUE" \
      --password "$APPLE_PASSWORD_VALUE" \
      --team-id "$APPLE_TEAM_ID_VALUE" \
      --wait
    xcrun stapler staple "$DMG_PATH"
    xcrun stapler validate "$DMG_PATH"
  fi
fi

echo "macOS signing/notarization step completed."
