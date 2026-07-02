param(
    [int]$StartPart = 2,
    [int]$EndPart = 8,
    [double]$MinimumFreeGiB = 20
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $env:USERPROFILE "miniconda3\envs\mediapipe_env\python.exe"

for ($part = $StartPart; $part -le $EndPart; $part++) {
    $disk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='D:'"
    $freeGiB = $disk.FreeSpace / 1GB
    if ($freeGiB -lt ($MinimumFreeGiB + 11)) {
        throw "Stopping before part ${part}: only $([math]::Round($freeGiB, 2)) GiB free."
    }
    Write-Host "Starting PNG shard $part (free: $([math]::Round($freeGiB, 2)) GiB)"
    & $python (Join-Path $root "export_png_drive_shard.py") `
        --destination (Join-Path $root "drive-shards") --part $part --target-gib 10
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
