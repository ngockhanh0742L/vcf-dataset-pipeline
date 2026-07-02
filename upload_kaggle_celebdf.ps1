param(
    [string]$Owner = "swannessie",
    [string]$Slug = "celebdf-v2-face-sequences-jpeg95-6seq",
    [string]$Title = "Celeb-DF v2 Face Sequences - JPEG95 6 Sequences",
    [switch]$Public,
    [switch]$Version,
    [string]$VersionNotes = "Refresh Celeb-DF v2 preprocessed JPEG95 sequence archive.",
    [switch]$BuildArchive,
    [switch]$SkipCompletenessCheck
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$description = @"
Preprocessed Celeb-DF v2 face sequences for deepfake detection.
Contains output_celebdf/manifests/sequence_manifest.csv with the official Celeb-DF test list preserved and train/val grouped by identity-connected components to avoid train/val identity and fake-pair leakage.
Each accepted sequence contains 24 JPEG-95 frames at 300x300. Use the split column as provided; do not randomly split sequence rows.
"@

& (Join-Path $root "upload_kaggle_processed.ps1") `
    -DatasetName "celebdf" `
    -OutputDir "output_celebdf" `
    -ArchiveName "output_celebdf.zip" `
    -Owner $Owner `
    -Slug $Slug `
    -Title $Title `
    -Description $description `
    -ExpectedVideos 6529 `
    -Public:$Public `
    -Version:$Version `
    -VersionNotes $VersionNotes `
    -BuildArchive:$BuildArchive `
    -SkipCompletenessCheck:$SkipCompletenessCheck
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
