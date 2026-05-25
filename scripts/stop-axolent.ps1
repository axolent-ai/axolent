#Requires -Version 5.1
<#
.SYNOPSIS
    Stop the AXOLENT bot via its PID file.
.DESCRIPTION
    Reads the PID from bridge/logs/axolent.pid, terminates the process,
    and removes the PID file.
#>
[CmdletBinding()]
param(
    [switch]$NoConfirm
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- Paths ---
$RepoRoot  = 'D:\Code\axolent'
$LogsDir   = Join-Path $RepoRoot 'bridge\logs'
$PidFile   = Join-Path $LogsDir  'axolent.pid'

# --- Check PID file ---
if (-not (Test-Path $PidFile)) {
    Write-Output 'NOT RUNNING'
    exit 0
}

$rawPid = (Get-Content $PidFile -Raw -Encoding UTF8).Trim()
if (-not ($rawPid -match '^\d+$')) {
    Write-Output "FAILED: corrupt PID file content: '$rawPid'"
    Remove-Item $PidFile -Force
    exit 1
}

$botPid = [int]$rawPid

# --- Kill process ---
try {
    $proc = Get-Process -Id $botPid -ErrorAction SilentlyContinue
    if ($null -eq $proc -or $proc.HasExited) {
        # Process already dead, clean up stale PID file
        Remove-Item $PidFile -Force
        Write-Output "STOPPED PID=$botPid (was already dead, cleaned stale PID file)"
        exit 0
    }

    Stop-Process -Id $botPid -Force
    # Wait briefly to confirm termination
    Start-Sleep -Milliseconds 500
    $recheck = Get-Process -Id $botPid -ErrorAction SilentlyContinue
    if ($null -ne $recheck -and -not $recheck.HasExited) {
        Write-Output "FAILED: process PID=$botPid did not terminate"
        exit 1
    }

    Remove-Item $PidFile -Force
    Write-Output "STOPPED PID=$botPid"
    exit 0

} catch {
    # If process not found, clean up
    if (Test-Path $PidFile) { Remove-Item $PidFile -Force }
    Write-Output "STOPPED PID=$botPid (process not found, cleaned PID file)"
    exit 0
}
