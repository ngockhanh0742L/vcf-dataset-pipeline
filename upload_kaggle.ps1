param(
    [string]$Owner = "swannessie",
    [string]$Slug = "vcf-processed-dataset",
    [string]$Title = "VCF Processed Face Sequences",
    [switch]$Public
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$archive = Join-Path $root "vcf-processed-dataset.zip"
$staging = Join-Path $root "kaggle-upload"

if (-not (Test-Path -LiteralPath $archive)) {
    throw "Archive not found: $archive"
}
$python = Join-Path $env:USERPROFILE "miniconda3\envs\mediapipe_env\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Python environment not found: $python" }

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
    description = "Preprocessed VCF face sequences with leakage-safe train/val/test manifests."
} | ConvertTo-Json -Depth 4
Set-Content -LiteralPath (Join-Path $staging "dataset-metadata.json") -Value $metadata -Encoding utf8

$args = @("datasets", "create", "-p", $staging, "-r", "skip")
if ($Public) { $args += "--public" }
& $python -m kaggle @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
