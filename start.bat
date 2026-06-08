@echo off
title Objectif.AI
echo.
echo  ================================================
echo   Objectif.AI v0.7.8
echo   BlueIris-compatible AI object detection
echo  ================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found. Please install Python 3.10 or newer.
    echo  https://www.python.org/downloads/
    pause
    exit /b 1
)

cd /d "%~dp0"

if not exist ".deps_installed" (
    echo  Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo  ERROR: Dependency installation failed.
        pause
        exit /b 1
    )
    echo installed > .deps_installed
    echo  Dependencies installed.
    echo.
)

echo  Dashboard:  http://localhost:32168
echo  API:        http://localhost:32168/v1/vision/detection
echo  Press Ctrl+C to stop
echo.

python main.py

pause
