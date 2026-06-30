param(
  [string]$InstallRoot = "$env:LOCALAPPDATA\ProtCross",
  [string]$InstallerPath = "",
  [string]$CondaPython = "",
  [string]$BundledAssetsDir = (Join-Path (Split-Path -Parent $PSScriptRoot) "bundled-assets")
)

$ErrorActionPreference = "Stop"
$RuntimeDir = Join-Path (Split-Path -Parent $PSScriptRoot) "runtime"

function Invoke-Checked {
  param(
    [Parameter(Mandatory = $true)][string]$FilePath,
    [Parameter(Mandatory = $true)][string[]]$Arguments
  )
  & $FilePath @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code $LASTEXITCODE: $FilePath $($Arguments -join ' ')"
  }
}

Invoke-Checked "python" @((Join-Path $PSScriptRoot "validate_bundled_assets.py"), "--assets-dir", $BundledAssetsDir)
Invoke-Checked "python" @((Join-Path $PSScriptRoot "validate_runtime_bundle.py"), "--runtime-dir", $RuntimeDir, "--backend", "all")

if ($InstallerPath -ne "") {
  $signature = Get-AuthenticodeSignature -FilePath $InstallerPath
  if ($signature.Status -ne "Valid") {
    throw "Installer signature is not valid: $($signature.Status)"
  }
}

& (Join-Path $RuntimeDir "test_backend.ps1") -Backend cpu -InstallRoot $InstallRoot
if ($LASTEXITCODE -ne 0) { throw "CPU backend validation failed with exit code $LASTEXITCODE." }

$GpuPython = Join-Path $InstallRoot "runtime\gpu-env\Scripts\python.exe"
if (Test-Path $GpuPython) {
  & (Join-Path $RuntimeDir "test_backend.ps1") -Backend gpu -InstallRoot $InstallRoot
  if ($LASTEXITCODE -ne 0) { throw "GPU backend validation failed with exit code $LASTEXITCODE." }
}

if ($CondaPython -ne "") {
  & (Join-Path $RuntimeDir "test_backend.ps1") -Backend conda -CondaPython $CondaPython
  if ($LASTEXITCODE -ne 0) { throw "Conda backend validation failed with exit code $LASTEXITCODE." }
}

Write-Host "ProtCross Desktop release validation completed."
