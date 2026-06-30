#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="$REPO_ROOT/desktop/bundled-assets"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      REPO_ROOT="$(cd "$2" && pwd)"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

CHECKPOINT="$REPO_ROOT/checkpoints/protcross-0.1.2-binding-moad-final.ckpt"
PCA="$REPO_ROOT/data/pca_esmc_128_binding_moad_0.1.2.pkl"
CHECKPOINT_SHA256="ccb56884b21402a027bfae9d4779f38c8f534513d980a96d7cd78c9931748b65"
PCA_SHA256="0f4e11806a622642c07dad539cec4216030220c1b5f3fc44c7926a2f6bca4d62"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint asset: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -f "$PCA" ]]; then
  echo "Missing PCA asset: $PCA" >&2
  exit 1
fi
actual_checkpoint_sha="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
actual_pca_sha="$(sha256sum "$PCA" | awk '{print $1}')"
checkpoint_size="$(wc -c < "$CHECKPOINT" | tr -d ' ')"
pca_size="$(wc -c < "$PCA" | tr -d ' ')"
if [[ "$actual_checkpoint_sha" != "$CHECKPOINT_SHA256" ]]; then
  echo "Checkpoint SHA256 mismatch: expected $CHECKPOINT_SHA256, got $actual_checkpoint_sha" >&2
  exit 1
fi
if [[ "$actual_pca_sha" != "$PCA_SHA256" ]]; then
  echo "PCA SHA256 mismatch: expected $PCA_SHA256, got $actual_pca_sha" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 ! -name ".gitkeep" -exec rm -rf {} +
cp "$CHECKPOINT" "$OUTPUT_DIR/protcross-0.1.2-binding-moad-final.ckpt"
cp "$PCA" "$OUTPUT_DIR/pca_esmc_128_binding_moad_0.1.2.pkl"
cat > "$OUTPUT_DIR/protcross-desktop-bundled-assets.json" <<JSON
{
  "schema_version": "protcross-desktop-bundled-assets-v1",
  "asset_bundle_version": "0.1.2",
  "checkpoint": {
    "filename": "protcross-0.1.2-binding-moad-final.ckpt",
    "sha256": "$CHECKPOINT_SHA256",
    "size_bytes": $checkpoint_size
  },
  "pca": {
    "filename": "pca_esmc_128_binding_moad_0.1.2.pkl",
    "sha256": "$PCA_SHA256",
    "size_bytes": $pca_size
  }
}
JSON

python "$REPO_ROOT/desktop/installer/validate_bundled_assets.py" --assets-dir "$OUTPUT_DIR"
echo "Bundled ProtCross assets written to $OUTPUT_DIR"
