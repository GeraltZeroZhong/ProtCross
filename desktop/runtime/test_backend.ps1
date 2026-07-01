param(
  [ValidateSet("cpu", "gpu", "conda")]
  [string]$Backend = "cpu",
  [string]$InstallRoot = "$env:LOCALAPPDATA\ProtCross",
  [string]$CondaPython = ""
)

$ErrorActionPreference = "Stop"

if ($Backend -eq "conda") {
  if ($CondaPython -eq "") {
    throw "CondaPython is required when Backend=conda"
  }
  $Python = $CondaPython
} else {
  $Python = Join-Path $InstallRoot "runtime\$Backend-env\Scripts\python.exe"
}

if (!(Test-Path $Python)) {
  throw "Python executable not found: $Python"
}

$Probe = & $Python -c @"
import json
import sys
payload = {}
for name in ("torch", "torch_geometric", "esm", "protcross", "protcross_desktop"):
    try:
        mod = __import__(name)
        payload[name] = {"ok": True, "version": getattr(mod, "__version__", None)}
    except Exception as exc:
        payload[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
try:
    import torch
    payload["torch_extra"] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
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
if "$Backend" == "gpu" and not payload.get("torch_extra", {}).get("cuda_available"):
    sys.exit(3)
"@
Write-Host $Probe
if ($LASTEXITCODE -ne 0) {
  throw "Backend test failed for $Backend with exit code $LASTEXITCODE"
}
