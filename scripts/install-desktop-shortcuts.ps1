#Requires -Version 5.1
<#
.SYNOPSIS
    Creates AXOLENT desktop shortcuts (Start, Stop, Status).
.DESCRIPTION
    Installs three .lnk files on the current user's Desktop that launch
    the corresponding PowerShell management scripts with -NoExit so the
    terminal window stays visible. Idempotent: re-running overwrites
    existing shortcuts.
.NOTES
    Author: Sigma (AI Engineer)
    Date:   2026-05-26
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- Configuration -----------------------------------------------------------

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
# Fallback: if invoked from outside the repo tree, use hard path
if (-not (Test-Path (Join-Path $RepoRoot 'scripts\start-axolent.ps1'))) {
    $RepoRoot = 'D:\Code\axolent'
}

$DesktopPath = Join-Path $env:USERPROFILE 'Desktop'
$ScriptsDir  = Join-Path $RepoRoot 'scripts'

$Shortcuts = @(
    @{
        Name        = 'AXOLENT starten'
        Target      = Join-Path $ScriptsDir 'start-axolent.ps1'
        Description = 'Start the AXOLENT bot as a background process'
    },
    @{
        Name        = 'AXOLENT stoppen'
        Target      = Join-Path $ScriptsDir 'stop-axolent.ps1'
        Description = 'Stop the running AXOLENT bot'
    },
    @{
        Name        = 'AXOLENT Status'
        Target      = Join-Path $ScriptsDir 'status-axolent.ps1'
        Description = 'Show current AXOLENT bot status'
    }
)

# --- Validation --------------------------------------------------------------

foreach ($sc in $Shortcuts) {
    if (-not (Test-Path $sc.Target)) {
        Write-Error "Required script missing: $($sc.Target)"
        exit 1
    }
}

# --- Icon detection (optional) -----------------------------------------------

$IconPath = $null
$CandidateIcon = Join-Path $RepoRoot 'assets\axolent.ico'
if (Test-Path $CandidateIcon) {
    $IconPath = $CandidateIcon
}

# --- Shortcut creation --------------------------------------------------------

$WshShell = New-Object -ComObject WScript.Shell
$CreatedCount = 0

foreach ($sc in $Shortcuts) {
    $LnkPath = Join-Path $DesktopPath "$($sc.Name).lnk"

    $Link = $WshShell.CreateShortcut($LnkPath)
    $Link.TargetPath       = 'powershell.exe'
    $Link.Arguments        = "-NoExit -ExecutionPolicy Bypass -File `"$($sc.Target)`""
    $Link.WorkingDirectory = $RepoRoot
    $Link.Description      = $sc.Description

    if ($IconPath) {
        $Link.IconLocation = "$IconPath,0"
    }

    $Link.Save()
    $CreatedCount++
}

Write-Host "$CreatedCount desktop shortcuts created." -ForegroundColor Green
