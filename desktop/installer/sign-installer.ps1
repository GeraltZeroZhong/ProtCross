param(
  [Parameter(Mandatory=$true)]
  [string]$InstallerPath,
  [Parameter(Mandatory=$true)]
  [string]$CertificateThumbprint,
  [string]$TimestampServer = "http://timestamp.digicert.com"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $InstallerPath)) {
  throw "Installer not found: $InstallerPath"
}

$cert = Get-ChildItem Cert:\CurrentUser\My | Where-Object { $_.Thumbprint -eq $CertificateThumbprint }
if ($null -eq $cert) {
  throw "Code signing certificate not found in CurrentUser\My: $CertificateThumbprint"
}

$result = Set-AuthenticodeSignature -FilePath $InstallerPath -Certificate $cert -TimestampServer $TimestampServer
if ($result.Status -ne "Valid") {
  throw "Signing failed: $($result.Status) $($result.StatusMessage)"
}

Get-AuthenticodeSignature -FilePath $InstallerPath
