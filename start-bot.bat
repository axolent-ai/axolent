@echo off
title Axolent Bot
cd /d %~dp0bridge
call .venv\Scripts\activate.bat
echo.
echo ===================================
echo  Starting Axolent Bot...
echo ===================================
echo.
python main.py
echo.
echo ===================================
echo  Bot has been stopped.
echo ===================================
pause
