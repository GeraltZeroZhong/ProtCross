param(
  [string]$InstallRoot = "$env:LOCALAPPDATA\ProtCross",
  [string]$InstallerPath = "",
  [string]$CondaPython = "",
  [string]$BundledAssetsDir = (Join-Path (Split-Path -Parent $PSScriptRoot) "bundled-assets"),
  [string]$PackagedResourceDir = "",
  [switch]$RequireInstallerSignature,
  [switch]$ValidateInstalledPackage,
  [switch]$SkipLocalBackendTests
)

$ErrorActionPreference = "Stop"
$RepoDesktopDir = Split-Path -Parent $PSScriptRoot
$RuntimeDir = Join-Path $RepoDesktopDir "runtime"

function Invoke-Checked {
  param(
    [Parameter(Mandatory = $true)][string]$FilePath,
    [Parameter(Mandatory = $true)][string[]]$Arguments
  )
  & $FilePath @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
  }
}

function Assert-ValidSignature {
  param([Parameter(Mandatory = $true)][string]$Path)
  if (!(Test-Path $Path)) {
    throw "Signed artifact not found: $Path"
  }
  $signature = Get-AuthenticodeSignature -FilePath $Path
  if ($signature.Status -ne "Valid") {
    throw "Signature is not valid for $Path`: $($signature.Status) $($signature.StatusMessage)"
  }
}

function Invoke-ResourceValidation {
  param([Parameter(Mandatory = $true)][string]$ResourceDir)
  Invoke-Checked "python" @((Join-Path $PSScriptRoot "validate_packaged_sidecar.py"), "--resource-dir", $ResourceDir, "--no-start")
  Invoke-Checked "python" @((Join-Path $PSScriptRoot "validate_runtime_bundle.py"), "--runtime-dir", (Join-Path $ResourceDir "runtime"), "--backend", "all")
}

function Find-PackagedResourceDir {
  param([Parameter(Mandatory = $true)][string]$Root)
  $server = Get-ChildItem -Path $Root -Filter "server.py" -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match "\\backend\\protcross_desktop\\server\.py$" } |
    Select-Object -First 1
  if ($null -eq $server) {
    throw "Could not locate packaged backend server.py under installed app root: $Root"
  }
  return $server.Directory.Parent.Parent.FullName
}

function Invoke-InstalledPackageValidation {
  param([Parameter(Mandatory = $true)][string]$Installer)
  if (!(Test-Path $Installer)) {
    throw "Installer not found: $Installer"
  }
  $tempRoot = Join-Path $env:TEMP ("protcross-desktop-install-" + [guid]::NewGuid().ToString("N"))
  $runtimeSmokeRoot = Join-Path $env:TEMP ("protcross-desktop-runtime-" + [guid]::NewGuid().ToString("N"))
  New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
  try {
    Write-Host "Installing NSIS package to $tempRoot"
    $process = Start-Process -FilePath $Installer -ArgumentList @("/S", "/D=$tempRoot") -Wait -PassThru
    if ($process.ExitCode -ne 0) {
      throw "NSIS silent install failed with exit code $($process.ExitCode)."
    }
    $resourceDir = Find-PackagedResourceDir -Root $tempRoot
    Invoke-ResourceValidation -ResourceDir $resourceDir
    & (Join-Path $resourceDir "runtime\install_cpu_backend.ps1") -InstallRoot $runtimeSmokeRoot
    if ($LASTEXITCODE -ne 0) { throw "Installed CPU backend setup failed with exit code $LASTEXITCODE." }
    & (Join-Path $resourceDir "runtime\test_backend.ps1") -Backend cpu -InstallRoot $runtimeSmokeRoot
    if ($LASTEXITCODE -ne 0) { throw "Installed CPU backend validation failed with exit code $LASTEXITCODE." }
  } finally {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $runtimeSmokeRoot -Recurse -Force -ErrorAction SilentlyContinue
  }
}

Invoke-Checked "python" @((Join-Path $PSScriptRoot "validate_bundled_assets.py"), "--assets-dir", $BundledAssetsDir)
Invoke-Checked "python" @((Join-Path $PSScriptRoot "validate_runtime_bundle.py"), "--runtime-dir", $RuntimeDir, "--backend", "all")

if ($InstallerPath -ne "" -and $RequireInstallerSignature) {
  Assert-ValidSignature -Path $InstallerPath
}

if ($PackagedResourceDir -ne "") {
  Invoke-ResourceValidation -ResourceDir $PackagedResourceDir
}

if ($ValidateInstalledPackage) {
  if ($InstallerPath -eq "") {
    throw "-ValidateInstalledPackage requires -InstallerPath."
  }
  Invoke-InstalledPackageValidation -Installer $InstallerPath
}

if (!$SkipLocalBackendTests) {
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
}

Write-Host "ProtCross Desktop release validation completed."
