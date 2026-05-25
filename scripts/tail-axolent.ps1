#Requires -Version 5.1
<#
.SYNOPSIS
    Live log viewer for AXOLENT (tail -f equivalent).
.DESCRIPTION
    Follows bridge\logs\axolent.err.log in real time using Get-Content -Wait.
    Ctrl+C exits the tail without affecting the running bot process.
.NOTES
    Author: Sigma (AI Engineer)
    Date:   2026-05-26
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$LogFile = 'D:\Code\axolent\bridge\logs\axolent.err.log'

if (-not (Test-Path $LogFile)) {
    Write-Host ''
    Write-Host 'Log file not found. Is the bot running? Try starting via the AXOLENT starten shortcut.' -ForegroundColor Yellow
    Write-Host ''
    Write-Host "Expected path: $LogFile" -ForegroundColor DarkGray
    exit 0
}

Write-Host ''
Write-Host '===========================================' -ForegroundColor Cyan
Write-Host ' AXOLENT Live Logs (Ctrl+C to exit tail)' -ForegroundColor Cyan
Write-Host ' Following: bridge\logs\axolent.err.log' -ForegroundColor Cyan
Write-Host '===========================================' -ForegroundColor Cyan
Write-Host ''

Get-Content $LogFile -Wait -Tail 30
