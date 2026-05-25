#Requires -Version 5.1
<#
.SYNOPSIS
    Report the status of the AXOLENT bot as JSON.
.DESCRIPTION
    Checks bridge/logs/axolent.pid and process state. Returns JSON suitable
    for parsing by orchestrators.
.PARAMETER Tail
    If specified, appends the last N lines of axolent.out.log to the JSON
    as a "log_tail" array field.
#>
[CmdletBinding()]
param(
    [int]$Tail = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- Paths ---
$RepoRoot  = 'D:\Code\axolent'
$LogsDir   = Join-Path $RepoRoot 'bridge\logs'
$PidFile   = Join-Path $LogsDir  'axolent.pid'
$OutLog    = Join-Path $LogsDir  'axolent.out.log'

# --- Helper: output JSON ---
function Write-StatusJson {
    param([hashtable]$Data)
    # Manual JSON construction to avoid ConvertTo-Json depth issues on PS 5.1
    $parts = @()
    foreach ($key in $Data.Keys) {
        $val = $Data[$key]
        if ($val -is [string]) {
            $parts += "`"$key`":`"$val`""
        } elseif ($val -is [bool]) {
            $parts += "`"$key`":$($val.ToString().ToLower())"
        } elseif ($val -is [int] -or $val -is [long] -or $val -is [double]) {
            $parts += "`"$key`":$val"
        } elseif ($val -is [array]) {
            $escaped = $val | ForEach-Object { "`"$($_ -replace '\\','\\\\' -replace '"','\"')`"" }
            $parts += "`"$key`":[$($escaped -join ',')]"
        } else {
            $parts += "`"$key`":`"$val`""
        }
    }
    Write-Output "{$($parts -join ',')}"
}

# --- No PID file = stopped ---
if (-not (Test-Path $PidFile)) {
    $result = @{ status = 'stopped' }
    if ($Tail -gt 0 -and (Test-Path $OutLog)) {
        $result['log_tail'] = @(Get-Content $OutLog -Tail $Tail -Encoding UTF8)
    }
    Write-StatusJson $result
    exit 0
}

# --- Read PID ---
$rawPid = (Get-Content $PidFile -Raw -Encoding UTF8).Trim()
if (-not ($rawPid -match '^\d+$')) {
    $result = @{ status = 'dead'; pid = 0; stale_pid_file = $true }
    Write-StatusJson $result
    exit 1
}

$botPid = [int]$rawPid

# --- Check process ---
$proc = Get-Process -Id $botPid -ErrorAction SilentlyContinue
if ($null -eq $proc -or $proc.HasExited) {
    $result = [ordered]@{ status = 'dead'; pid = $botPid; stale_pid_file = $true }
    Write-StatusJson $result
    exit 1
}

# --- Running ---
$startTime = ''
$uptimeSeconds = 0
try {
    $startTime = $proc.StartTime.ToString('yyyy-MM-ddTHH:mm:ss')
    $uptimeSeconds = [int]((Get-Date) - $proc.StartTime).TotalSeconds
} catch {
    # If we cannot read start time, still report running
}

$result = [ordered]@{
    status         = 'running'
    pid            = $botPid
    start_time     = $startTime
    uptime_seconds = $uptimeSeconds
}

if ($Tail -gt 0 -and (Test-Path $OutLog)) {
    $result['log_tail'] = @(Get-Content $OutLog -Tail $Tail -Encoding UTF8)
}

Write-StatusJson $result
exit 0
