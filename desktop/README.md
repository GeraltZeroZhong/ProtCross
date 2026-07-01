# ProtCross Desktop

This directory contains the 0.2.0 desktop application scaffold for ProtCross
Desktop. The first desktop release targets Windows 10/11 x64 and macOS 12+ on
the architecture built by the release job. Separate Apple Silicon and Intel
macOS artifacts require separate runtime wheelhouses. The app is focused on
prediction workflows only.
The desktop backend package is versioned as 0.2.0 and currently pins the core
Python package to `protcross==0.1.3`.

## Product Boundaries

- No viewer-only mode: a completed setup must support prediction.
- No training, preprocessing, AF2 download, or label mapping UI in v1.
- Tauri + React is the desktop shell and UI stack.
- Mol* is the molecular viewer used for annotated ProtCross structures.
- Public Windows installers and macOS apps/DMGs must be code-signed before
  release. macOS public DMGs must also be notarized.
- The app does not upload structures, paths, logs, or predictions.
  Diagnostic packages are generated locally and users decide whether to attach
  them to a GitHub issue.

## Runtime Model

ProtCross Desktop uses a local Python sidecar:

```text
Tauri UI -> dynamic http://127.0.0.1:<port> -> protcross_desktop.server -> ProtCross Python API
```

On startup the app first honors `PROTCROSS_DESKTOP_PYTHON` for development.
Otherwise it reads the desktop manifest and starts the selected backend Python
when that environment already exists. The Tauri shell injects the bundled
backend and bundled asset resource paths into the sidecar environment. If no
configured environment is present, it falls back to `python` and the Setup page
guides the user to install or select a working backend, run the environment
test, then restart the backend. Sidecar stdout/stderr is written to the app data
`logs/backend.log` file.
If that fallback Python is missing on a clean machine, the Setup page can still
run the CPU/GPU installers through Tauri commands implemented in Rust.

- Windows CPU: `%LOCALAPPDATA%\ProtCross\runtime\cpu-env`
- Windows GPU: `%LOCALAPPDATA%\ProtCross\runtime\gpu-env`
- macOS CPU: `~/Library/Application Support/ProtCross/runtime/cpu-env`
- macOS GPU: `~/Library/Application Support/ProtCross/runtime/gpu-env`
- conda: user-selected environment, tested read-only

CPU and GPU environments are deliberately separate. GPU setup failures must not
break the CPU backend. On Windows, GPU means CUDA PyTorch. On macOS, GPU means
Apple Metal/MPS PyTorch; if MPS is unavailable, the GPU backend test fails and
users should use the CPU backend.

The CPU/GPU runtime installers try system Python 3.10 first. If it is missing,
release builds must provide a signed/hashed app-local Python bootstrapper such
as `runtime/uv/uv` or `runtime/uv/uv.exe`; the scripts do not download or
execute remote bootstrap installers.

Release dependency installation must use a bundled wheelhouse plus
`requirements-cpu.hashes` and `requirements-gpu.hashes`, generated for the
target platform. Online package-index installation is available only for
development by explicitly passing `--allow-online-package-index` or
`-AllowOnlinePackageIndex`.

## Asset Policy

The installer/bundle may include ProtCross-owned runtime assets:

- `protcross-0.1.2-binding-moad-final.ckpt`
- `pca_esmc_128_binding_moad_0.1.2.pkl`

These assets are trained or fitted from ProtCross release workflows using
ESM-C embeddings, but they do not contain or redistribute ESM-C model weights.

ESM-C weights are not bundled. Users must explicitly confirm the ESM-C model
license before downloading or importing:

- default file: `esmc_600m_2024_12_v0.pth`
- default source: the official ESM-C 600M model weight URL used by ProtCross
- expected SHA256 is inherited from `protcross.assets`

The desktop manifest records license confirmation, model URL, local weight path,
SHA256, backend selection, and proxy configuration. A hash mismatch keeps
prediction disabled.

## Development

From the repository root:

```bash
python -m pip install -e ".[predict]"
python -m pip install -e desktop/backend
cd desktop/frontend
npm install
export PROTCROSS_DESKTOP_BACKEND_PATH="../backend"
export PROTCROSS_DESKTOP_PYTHON="python"
npm run tauri:dev
```

PowerShell equivalent:

```powershell
python -m pip install -e ".[predict]"
python -m pip install -e desktop/backend
cd desktop\frontend
npm install
$env:PROTCROSS_DESKTOP_BACKEND_PATH = "..\backend"
$env:PROTCROSS_DESKTOP_PYTHON = "python"
npm run tauri:dev
```

For Python backend tests from the repository root:

```bash
python -m pytest -q tests/desktop
```

## Backend Installation

Windows:

```powershell
desktop\runtime\install_cpu_backend.ps1 -AllowOnlinePackageIndex
desktop\runtime\install_gpu_backend.ps1 -AllowOnlinePackageIndex
desktop\runtime\test_backend.ps1 -Backend cpu
```

macOS:

```bash
desktop/runtime/install_cpu_backend.sh --allow-online-package-index
desktop/runtime/install_gpu_backend.sh --allow-online-package-index
desktop/runtime/test_backend.sh --backend cpu
```

Use `-ProxyUrl` on Windows or `--proxy-url` on macOS when installing behind a
proxy.
For local development without a release wheelhouse, pass
`-AllowOnlinePackageIndex` or `--allow-online-package-index` intentionally.

Existing conda environments are read-only from ProtCross Desktop. The app tests
imports and acceleration availability, but it does not install, remove, or
upgrade packages in user-managed conda environments.

## Release Build Checklist

Prepare bundled assets:

```bash
desktop/installer/prepare_bundled_assets.sh
python -m build --wheel --outdir dist
python desktop/installer/prepare_runtime_wheelhouse.py \
  --runtime-dir desktop/runtime \
  --backend all \
  --local-protcross-wheel dist/protcross-0.1.3-py3-none-any.whl
python desktop/installer/validate_runtime_bundle.py --runtime-dir desktop/runtime --backend all
python desktop/installer/validate_packaged_sidecar.py \
  --resource-dir <packaged-or-staged-resource-dir>
```

Windows:

```powershell
desktop\installer\prepare_bundled_assets.ps1
cd desktop\frontend
npm run tauri:release-build -- --bundles nsis
cd ..\..
desktop\installer\sign-installer.ps1 `
  -AppExePath desktop\src-tauri\target\release\protcross-desktop.exe `
  -InstallerPath path\to\ProtCross-Setup-x64.exe `
  -CertificateThumbprint <thumbprint>
desktop\installer\validate-release.ps1 `
  -InstallerPath path\to\ProtCross-Setup-x64.exe `
  -RequireInstallerSignature `
  -ValidateInstalledPackage `
  -SkipLocalBackendTests
```

macOS:

```bash
desktop/installer/prepare_bundled_assets.sh
cd desktop/frontend
npm run tauri:release-build -- --bundles app
cd ../..
APPLE_SIGNING_IDENTITY="Developer ID Application: ..." \
APPLE_ID="developer@example.com" \
APPLE_PASSWORD="<app-specific-password>" \
APPLE_TEAM_ID="<team-id>" \
desktop/installer/sign-notarize-macos.sh \
  --app "desktop/src-tauri/target/release/bundle/macos/ProtCross Desktop.app" \
  --dmg "desktop/src-tauri/target/release/bundle/dmg/ProtCross_Desktop.dmg"
desktop/installer/validate-release-macos.sh \
  --dmg "desktop/src-tauri/target/release/bundle/dmg/ProtCross_Desktop.dmg"
```

The macOS signing script signs the `.app`, recreates the DMG from that signed
app bundle, signs/notarizes the DMG, and verifies the `.app` mounted from the
DMG. Use `--skip-notarization` only for internal unsigned test artifacts.

Run clean-machine scenarios:

- Windows 10 x64 without NVIDIA GPU
- Windows 11 x64 without NVIDIA GPU
- Windows with NVIDIA GPU present but incompatible/missing driver
- Windows with NVIDIA GPU and working CUDA PyTorch
- macOS release artifact architecture without MPS disabled
- macOS release artifact architecture CPU-only
- no conda
- conda installed but selected env missing dependencies
- conda installed with a working ProtCross env

Unsigned Windows installers and unsigned/unnotarized macOS DMGs are for
internal testing only and must not be published as formal releases.
