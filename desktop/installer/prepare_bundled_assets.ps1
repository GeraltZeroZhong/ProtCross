param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")),
  [string]$OutputDir = (Join-Path $PSScriptRoot "..\bundled-assets")
)

$ErrorActionPreference = "Stop"

$Checkpoint = Join-Path $RepoRoot "checkpoints\protcross-0.1.2-binding-moad-final.ckpt"
$Pca = Join-Path $RepoRoot "data\pca_esmc_128_binding_moad_0.1.2.pkl"
$CheckpointSha256 = "ccb56884b21402a027bfae9d4779f38c8f534513d980a96d7cd78c9931748b65"
$PcaSha256 = "0f4e11806a622642c07dad539cec4216030220c1b5f3fc44c7926a2f6bca4d62"

if (!(Test-Path $Checkpoint)) {
  throw "Missing checkpoint asset: $Checkpoint"
}
if (!(Test-Path $Pca)) {
  throw "Missing PCA asset: $Pca"
}
$ActualCheckpointSha256 = (Get-FileHash -Algorithm SHA256 $Checkpoint).Hash.ToLowerInvariant()
$ActualPcaSha256 = (Get-FileHash -Algorithm SHA256 $Pca).Hash.ToLowerInvariant()
$CheckpointSize = (Get-Item $Checkpoint).Length
$PcaSize = (Get-Item $Pca).Length
if ($ActualCheckpointSha256 -ne $CheckpointSha256) {
  throw "Checkpoint SHA256 mismatch: expected $CheckpointSha256, got $ActualCheckpointSha256"
}
if ($ActualPcaSha256 -ne $PcaSha256) {
  throw "PCA SHA256 mismatch: expected $PcaSha256, got $ActualPcaSha256"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Get-ChildItem -Force $OutputDir | Where-Object { $_.Name -ne ".gitkeep" } | Remove-Item -Recurse -Force
Copy-Item $Checkpoint (Join-Path $OutputDir "protcross-0.1.2-binding-moad-final.ckpt") -Force
Copy-Item $Pca (Join-Path $OutputDir "pca_esmc_128_binding_moad_0.1.2.pkl") -Force
$Manifest = [ordered]@{
  schema_version = "protcross-desktop-bundled-assets-v1"
  asset_bundle_version = "0.1.2"
  checkpoint = [ordered]@{
    filename = "protcross-0.1.2-binding-moad-final.ckpt"
    sha256 = $CheckpointSha256
    size_bytes = $CheckpointSize
  }
  pca = [ordered]@{
    filename = "pca_esmc_128_binding_moad_0.1.2.pkl"
    sha256 = $PcaSha256
    size_bytes = $PcaSize
  }
}
$Manifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $OutputDir "protcross-desktop-bundled-assets.json")

& python (Join-Path $PSScriptRoot "validate_bundled_assets.py") --assets-dir $OutputDir
Write-Host "Bundled ProtCross assets written to $OutputDir"
