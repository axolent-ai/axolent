@echo off
title Axolent Bot
cd /d %~dp0bridge
call .venv\Scripts\activate.bat
echo.
echo ===================================
echo  Axolent Bot wird gestartet...
echo ===================================
echo.
python main.py
echo.
echo ===================================
echo  Bot wurde beendet.
echo ===================================
pause
