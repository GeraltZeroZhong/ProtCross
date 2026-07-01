param(
  [string]$InstallerPath = "",
  [string]$AppExePath = "",
  [string]$RuntimeDir = (Join-Path (Split-Path -Parent $PSScriptRoot) "runtime"),
  [Parameter(Mandatory = $true)]
  [string]$CertificateThumbprint,
  [string]$TimestampServer = "http://timestamp.digicert.com"
)

$ErrorActionPreference = "Stop"

function Resolve-SigningCertificate {
  param([Parameter(Mandatory = $true)][string]$Thumbprint)
  $normalized = $Thumbprint.Replace(" ", "").ToUpperInvariant()
  $cert = Get-ChildItem Cert:\CurrentUser\My | Where-Object { $_.Thumbprint.Replace(" ", "").ToUpperInvariant() -eq $normalized } | Select-Object -First 1
  if ($null -eq $cert) {
    throw "Code signing certificate not found in CurrentUser\My: $Thumbprint"
  }
  return $cert
}

function Sign-And-Verify {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)]$Certificate
  )
  if (!(Test-Path $Path)) {
    throw "Sign target not found: $Path"
  }
  Write-Host "Signing $Path"
  $result = Set-AuthenticodeSignature -FilePath $Path -Certificate $Certificate -TimestampServer $TimestampServer
  if ($result.Status -ne "Valid") {
    throw "Signing failed for $Path`: $($result.Status) $($result.StatusMessage)"
  }
  $signature = Get-AuthenticodeSignature -FilePath $Path
  if ($signature.Status -ne "Valid") {
    throw "Signature verification failed for $Path`: $($signature.Status) $($signature.StatusMessage)"
  }
  return $signature
}

function Update-UvChecksum {
  param([Parameter(Mandatory = $true)][string]$UvPath)
  $shaPath = Join-Path (Split-Path -Parent $UvPath) "uv.sha256"
  $hash = (Get-FileHash -Algorithm SHA256 -Path $UvPath).Hash.ToLowerInvariant()
  Set-Content -Path $shaPath -Value "$hash  uv.exe" -Encoding ASCII
  Write-Host "Updated $shaPath"
}

$cert = Resolve-SigningCertificate -Thumbprint $CertificateThumbprint
$signedAny = $false

$UvPath = Join-Path $RuntimeDir "uv\uv.exe"
if (Test-Path $UvPath) {
  Sign-And-Verify -Path $UvPath -Certificate $cert | Out-Null
  Update-UvChecksum -UvPath $UvPath
  $signedAny = $true
}

if ($AppExePath -ne "") {
  Sign-And-Verify -Path $AppExePath -Certificate $cert | Out-Null
  $signedAny = $true
}

if ($InstallerPath -ne "") {
  Sign-And-Verify -Path $InstallerPath -Certificate $cert
  $signedAny = $true
}

if (!$signedAny) {
  throw "No signing targets were found. Pass -InstallerPath, -AppExePath, or a RuntimeDir containing uv\uv.exe."
}
