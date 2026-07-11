param(
  [string]$InstallRoot = "$env:LOCALAPPDATA\ProtCross",
  [string]$PythonExe = "py",
  [string[]]$PythonArgs = @("-3.10"),
  [string]$ProxyUrl = "",
  [string]$UvExe = "",
  [string]$Wheelhouse = "",
  [switch]$AllowOnlinePackageIndex
)

$ErrorActionPreference = "Stop"
$RuntimeDir = Join-Path $InstallRoot "runtime"
$EnvDir = Join-Path $RuntimeDir "gpu-env"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$CommonRequirements = Join-Path $ScriptDir "requirements-common.lock"
$BackendHashRequirements = Join-Path $ScriptDir "requirements-gpu.hashes"
$BackendPackage = Join-Path (Split-Path -Parent $ScriptDir) "backend"
if ($Wheelhouse -eq "") {
  $Wheelhouse = Join-Path $ScriptDir "wheelhouse"
}

New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null

if ($ProxyUrl -ne "") {
  $env:HTTP_PROXY = $ProxyUrl
  $env:HTTPS_PROXY = $ProxyUrl
}

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

function Test-PythonCommand {
  try {
    & $PythonExe @PythonArgs --version *> $null
    return $LASTEXITCODE -eq 0
  } catch {
    return $false
  }
}

function Get-UvExe {
  $Candidate = $UvExe
  if ($Candidate -eq "") {
    $Candidate = Join-Path $ScriptDir "uv\uv.exe"
  }
  if (!(Test-Path $Candidate)) {
    throw "A usable Python 3.10 environment with pip was not found, and no bundled uv executable is available. Release builds must ship a signed/hashed Python bootstrapper; development installs may pass -PythonExe or -UvExe."
  }
  return $Candidate
}

function Test-EnvironmentPython {
  $Candidate = Join-Path $EnvDir "Scripts\python.exe"
  if (!(Test-Path $Candidate)) {
    return $false
  }
  try {
    & $Candidate --version *> $null
    return $LASTEXITCODE -eq 0
  } catch {
    return $false
  }
}

function Test-EnvironmentPip {
  if (!(Test-EnvironmentPython)) {
    return $false
  }
  $Candidate = Join-Path $EnvDir "Scripts\python.exe"
  try {
    & $Candidate -m pip --version *> $null
    return $LASTEXITCODE -eq 0
  } catch {
    return $false
  }
}

function Initialize-WithUv {
  $ResolvedUv = Get-UvExe
  Invoke-Checked $ResolvedUv @("python", "install", "3.10")
  Invoke-Checked $ResolvedUv @("venv", "--seed", "--allow-existing", "--python", "3.10", $EnvDir)
}

function Repair-EnvironmentPip {
  if (Test-EnvironmentPip) {
    return
  }
  Write-Warning "Backend environment is missing pip; attempting an in-place repair: $EnvDir"
  $Candidate = Join-Path $EnvDir "Scripts\python.exe"
  try {
    & $Candidate -m ensurepip --upgrade *> $null
  } catch {
    # Fall through to the bundled uv repair path.
  }
  if (Test-EnvironmentPip) {
    return
  }
  $ResolvedUv = Get-UvExe
  Invoke-Checked $ResolvedUv @("venv", "--seed", "--allow-existing", "--python", $Candidate, $EnvDir)
  if (!(Test-EnvironmentPip)) {
    throw "Backend environment repair did not produce a working pip: $EnvDir"
  }
}

function Install-Requirements {
  if ((Test-Path $Wheelhouse) -and (Test-Path $BackendHashRequirements)) {
    Invoke-Checked $Pip @("install", "--no-index", "--find-links", $Wheelhouse, "--require-hashes", "-r", $BackendHashRequirements)
    return
  }
  if ($AllowOnlinePackageIndex) {
    Write-Warning "Installing the optional GPU backend from fixed online package indexes."
    Invoke-Checked $Pip @("install", "torch==2.3.1+cu121", "torchvision==0.18.1+cu121", "--index-url", "https://download.pytorch.org/whl/cu121")
    Invoke-Checked $Pip @("install", "--find-links", $Wheelhouse, "-r", $CommonRequirements)
    Invoke-Checked $Pip @("install", $BackendPackage)
    return
  }
  throw "Missing GPU runtime wheelhouse: $Wheelhouse. Pass -AllowOnlinePackageIndex for the public optional GPU install path, or generate requirements-gpu.hashes for an internal offline GPU bundle."
}

if (!(Test-EnvironmentPython)) {
  if (Test-PythonCommand) {
    try {
      Invoke-Checked $PythonExe ($PythonArgs + @("-m", "venv", $EnvDir))
    } catch {
      Write-Warning "System Python could not create a complete environment; falling back to bundled uv."
      Initialize-WithUv
    }
  } else {
    Initialize-WithUv
  }
}

$Pip = Join-Path $EnvDir "Scripts\pip.exe"
$Python = Join-Path $EnvDir "Scripts\python.exe"

Repair-EnvironmentPip
Install-Requirements

$GpuCheck = & $Python -c "import sys, torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA'); sys.exit(0 if torch.cuda.is_available() else 3)"
Write-Host $GpuCheck
if ($LASTEXITCODE -ne 0) {
  throw "CUDA PyTorch is installed, but no compatible NVIDIA GPU/driver is available. Use the CPU backend or fix the NVIDIA driver, then rerun GPU setup."
}
Write-Host "ProtCross GPU backend installed at $EnvDir"
