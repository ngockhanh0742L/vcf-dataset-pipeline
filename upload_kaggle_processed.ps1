param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetName,
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [Parameter(Mandatory = $true)]
    [string]$ArchiveName,
    [Parameter(Mandatory = $true)]
    [string]$Owner,
    [Parameter(Mandatory = $true)]
    [string]$Slug,
    [Parameter(Mandatory = $true)]
    [string]$Title,
    [Parameter(Mandatory = $true)]
    [string]$Description,
    [int]$ExpectedVideos = 0,
    [switch]$Public,
    [switch]$Version,
    [string]$VersionNotes = "",
    [switch]$BuildArchive,
    [switch]$SkipCompletenessCheck
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$outputPath = Join-Path $root $OutputDir
$archive = Join-Path $root $ArchiveName
$summaryPath = Join-Path $outputPath "manifests\dataset_summary.json"
$pidPath = Join-Path $outputPath "preprocess.pid"
$staging = Join-Path $root ("kaggle-upload\" + $DatasetName)
$python = Join-Path $env:USERPROFILE "miniconda3\envs\mediapipe_env\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}
if (-not (Test-Path -LiteralPath $outputPath)) {
    throw "Output directory not found: $outputPath"
}

if (-not $SkipCompletenessCheck) {
    if (Test-Path -LiteralPath $pidPath) {
        $pidText = (Get-Content -LiteralPath $pidPath -ErrorAction SilentlyContinue | Select-Object -First 1)
        if ($pidText) {
            $process = Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue
            if ($process) {
                throw "$DatasetName preprocess is still running as PID $pidText"
            }
        }
    }
    if (-not (Test-Path -LiteralPath $summaryPath)) {
        throw "Summary not found: $summaryPath"
    }
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    if ($ExpectedVideos -gt 0 -and [int]$summary.valid_videos -ne $ExpectedVideos) {
        throw "$DatasetName summary has valid_videos=$($summary.valid_videos), expected $ExpectedVideos"
    }
    if ($summary.failed_videos -and $summary.failed_videos.Count -gt 0) {
        throw "$DatasetName summary reports $($summary.failed_videos.Count) failed video(s)"
    }
}

if (-not (Test-Path -LiteralPath $archive)) {
    if (-not $BuildArchive) {
        throw "Archive not found: $archive. Re-run with -BuildArchive after preprocessing finishes."
    }
    Push-Location $root
    try {
        & tar.exe -a -cf $ArchiveName $OutputDir
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    finally {
        Pop-Location
    }
}

New-Item -ItemType Directory -Force -Path $staging | Out-Null
$link = Join-Path $staging (Split-Path -Leaf $archive)
if (Test-Path -LiteralPath $link) {
    Remove-Item -LiteralPath $link -Force
}
New-Item -ItemType HardLink -Path $link -Target $archive | Out-Null

$metadata = @{
    title = $Title
    id = "$Owner/$Slug"
    licenses = @(@{ name = "other" })
    isPrivate = -not $Public
    description = $Description
} | ConvertTo-Json -Depth 4
Set-Content -LiteralPath (Join-Path $staging "dataset-metadata.json") -Value $metadata -Encoding utf8

if ($Version) {
    $kaggleArgs = @("datasets", "version", "-p", $staging, "-m", $VersionNotes, "-r", "skip")
} else {
    $kaggleArgs = @("datasets", "create", "-p", $staging, "-r", "skip")
    if ($Public) { $kaggleArgs += "--public" }
}

& $python -m kaggle @kaggleArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Uploaded https://www.kaggle.com/datasets/$Owner/$Slug"
