#Requires -Version 5.1
<#
.SYNOPSIS
    Start the AXOLENT bot as a detached Windows process.
.DESCRIPTION
    Launches pythonw.exe main.py via Start-Process -WindowStyle Hidden so the
    process is adopted by Windows and survives the caller's session timeout.
    Writes PID to bridge/logs/axolent.pid for stop-axolent.ps1 / status-axolent.ps1.
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- Paths ---
$RepoRoot   = 'D:\Code\axolent'
$BridgeDir  = Join-Path $RepoRoot 'bridge'
$LogsDir    = Join-Path $BridgeDir 'logs'
$PidFile    = Join-Path $LogsDir   'axolent.pid'
$OutLog     = Join-Path $LogsDir   'axolent.out.log'
$ErrLog     = Join-Path $LogsDir   'axolent.err.log'
$PythonW    = Join-Path $BridgeDir '.venv\Scripts\pythonw.exe'

# --- Preflight checks ---
if (-not (Test-Path $PythonW)) {
    Write-Error "FAILED: pythonw.exe not found at $PythonW"
    exit 1
}
if (-not (Test-Path (Join-Path $BridgeDir 'main.py'))) {
    Write-Error "FAILED: main.py not found in $BridgeDir"
    exit 1
}

# Ensure logs directory exists
if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
}

# --- Double-start protection ---
if (Test-Path $PidFile) {
    $rawPid = (Get-Content $PidFile -Raw -Encoding UTF8).Trim()
    if ($rawPid -match '^\d+$') {
        $existingPid = [int]$rawPid
        $proc = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
        if ($null -ne $proc -and -not $proc.HasExited) {
            Write-Output "Already running PID=$existingPid"
            exit 1
        }
        # Stale PID file - remove and continue
        Remove-Item $PidFile -Force
    } else {
        # Corrupt PID file - remove
        Remove-Item $PidFile -Force
    }
}

# --- Start detached process ---
try {
    $process = Start-Process -FilePath $PythonW `
        -ArgumentList 'main.py' `
        -WorkingDirectory $BridgeDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $OutLog `
        -RedirectStandardError $ErrLog `
        -PassThru

    $newPid = $process.Id

    # Write PID file (UTF-8, no BOM, no trailing newline)
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($PidFile, "$newPid", $utf8NoBom)

} catch {
    Write-Output "FAILED: $($_.Exception.Message)"
    exit 1
}

# --- Health check: wait 3 seconds, verify process still alive ---
Start-Sleep -Seconds 3

$check = Get-Process -Id $newPid -ErrorAction SilentlyContinue
if ($null -eq $check -or $check.HasExited) {
    $errTail = ''
    if (Test-Path $ErrLog) {
        $errTail = Get-Content $ErrLog -Tail 20 -Encoding UTF8 | Out-String
    }
    # Clean up PID file since process died
    if (Test-Path $PidFile) { Remove-Item $PidFile -Force }
    Write-Output "FAILED: process exited within 3 seconds"
    if ($errTail) {
        Write-Output "--- last 20 lines of err.log ---"
        Write-Output $errTail
    }
    exit 1
}

# Check logs for positive startup signal (without leaking sensitive data)
# Note: python-telegram-bot logs to stderr via Python logging module
$startupOk = $false
foreach ($logPath in @($OutLog, $ErrLog)) {
    if ((Test-Path $logPath) -and -not $startupOk) {
        $logContent = Get-Content $logPath -Encoding UTF8 -ErrorAction SilentlyContinue
        if ($logContent -match 'Application started') {
            $startupOk = $true
        }
    }
}

Write-Output "STARTED PID=$newPid"
if ($startupOk) {
    Write-Output "Health: Application started signal detected"
} else {
    Write-Output "Health: process alive, startup signal not yet in log (may still be initializing)"
}
exit 0
