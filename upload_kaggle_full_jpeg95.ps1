param(
    [string]$Owner = "swannessie",
    [string]$Slug = "vcf-processed-jpeg95-6seq",
    [string]$Title = "VCF Processed Dataset - JPEG95 6 Sequences",
    [switch]$Public
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$archive = Join-Path $root "vcf-processed-jpeg95-6seq.zip"
$staging = Join-Path $root "kaggle-upload\jpeg95-6seq"
$python = Join-Path $env:USERPROFILE "miniconda3\envs\mediapipe_env\python.exe"

if (-not (Test-Path -LiteralPath $archive)) { throw "Archive not found: $archive" }
if (-not (Test-Path -LiteralPath $python)) { throw "Python not found: $python" }

New-Item -ItemType Directory -Force -Path $staging | Out-Null
$link = Join-Path $staging (Split-Path -Leaf $archive)
if (-not (Test-Path -LiteralPath $link)) {
    New-Item -ItemType HardLink -Path $link -Target $archive | Out-Null
}

$metadata = @{
    title = $Title
    id = "$Owner/$Slug"
    licenses = @(@{ name = "other" })
    isPrivate = -not $Public
    description = "Full VCF leakage-safe export: 9,465 accepted videos and all 55,174 quality-aware sequences, each containing 24 JPEG-95 frames at 300x300."
} | ConvertTo-Json -Depth 4
Set-Content -LiteralPath (Join-Path $staging "dataset-metadata.json") -Value $metadata -Encoding utf8

$kaggleArgs = @("datasets", "create", "-p", $staging, "-r", "skip")
if ($Public) { $kaggleArgs += "--public" }
& $python -m kaggle @kaggleArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Uploaded https://www.kaggle.com/datasets/$Owner/$Slug"
