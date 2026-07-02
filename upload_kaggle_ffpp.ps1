param(
    [string]$Owner = "swannessie",
    [string]$Slug = "ffpp-c23-face-sequences-jpeg95-6seq",
    [string]$Title = "FF++ C23 Face Sequences JPEG95",
    [switch]$Public,
    [switch]$Version,
    [string]$VersionNotes = "Refresh FaceForensics++ C23 preprocessed JPEG95 sequence archive.",
    [switch]$BuildArchive,
    [switch]$SkipCompletenessCheck
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$description = @"
Preprocessed FaceForensics++ C23 face sequences for deepfake detection.
Contains output_ffpp/manifests/sequence_manifest.csv with leakage-safe official splits, 33,265 accepted sequences from 5,992 videos, 24 JPEG-95 frames per sequence at 300x300.
Use the split column as provided; do not randomly split sequence rows.
"@

& (Join-Path $root "upload_kaggle_processed.ps1") `
    -DatasetName "ffpp" `
    -OutputDir "output_ffpp" `
    -ArchiveName "output_ffpp.zip" `
    -Owner $Owner `
    -Slug $Slug `
    -Title $Title `
    -Description $description `
    -ExpectedVideos 6000 `
    -Public:$Public `
    -Version:$Version `
    -VersionNotes $VersionNotes `
    -BuildArchive:$BuildArchive `
    -SkipCompletenessCheck:$SkipCompletenessCheck
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
