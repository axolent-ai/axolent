@echo off
title Jarvis-LITE Bot
cd /d D:\Code\jarvis-lite\bridge
call .venv\Scripts\activate.bat
echo.
echo ===================================
echo  Jarvis-LITE Bot wird gestartet...
echo ===================================
echo.
python main.py
echo.
echo ===================================
echo  Bot wurde beendet.
echo ===================================
pause
