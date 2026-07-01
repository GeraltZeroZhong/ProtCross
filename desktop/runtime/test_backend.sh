#!/usr/bin/env bash
set -euo pipefail

BACKEND="cpu"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/Library/Application Support/ProtCross}"
CONDA_PYTHON="${CONDA_PYTHON:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --install-root)
      INSTALL_ROOT="$2"
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

case "$BACKEND" in
  cpu|gpu)
    PYTHON="$INSTALL_ROOT/runtime/$BACKEND-env/bin/python"
    ;;
  conda)
    if [[ -z "$CONDA_PYTHON" ]]; then
      echo "--conda-python is required for conda backend tests" >&2
      exit 2
    fi
    PYTHON="$CONDA_PYTHON"
    ;;
  *)
    echo "backend must be cpu, gpu, or conda" >&2
    exit 2
    ;;
esac

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2
  exit 1
fi

"$PYTHON" - <<PY
import json
import sys

payload = {"backend": "$BACKEND", "python": sys.executable}
for name in ("torch", "torch_geometric", "esm", "protcross", "protcross_desktop"):
    try:
        module = __import__(name)
        payload[name] = {"ok": True, "version": getattr(module, "__version__", None)}
    except Exception as exc:
        payload[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

try:
    import torch
    payload["torch_extra"] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "mps_built": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built()),
        "tensor_ok": float(torch.tensor([1.0, 2.0]).sum().item()) == 3.0,
    }
except Exception as exc:
    payload["torch_extra_error"] = f"{type(exc).__name__}: {exc}"

print(json.dumps(payload, indent=2))

required = ("torch", "torch_geometric", "esm", "protcross", "protcross_desktop")
if not all(payload.get(name, {}).get("ok") for name in required):
    sys.exit(1)
if not payload.get("torch_extra", {}).get("tensor_ok"):
    sys.exit(1)
if "$BACKEND" == "gpu" and not (
    payload.get("torch_extra", {}).get("cuda_available")
    or payload.get("torch_extra", {}).get("mps_available")
):
    sys.exit(3)
PY
