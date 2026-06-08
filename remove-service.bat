@echo off
title Objectif.AI — Remove Service
setlocal EnableDelayedExpansion

echo.
echo  ================================================
echo   Objectif.AI — Remove Startup Service
echo  ================================================
echo.

REM ── Get app directory ────────────────────────────
cd /d "%~dp0"
set "APPDIR=%~dp0"
if "%APPDIR:~-1%"=="\" set "APPDIR=%APPDIR:~0,-1%"

REM ── Stop the running server ───────────────────────
python "%APPDIR%\_stop_server.py" "%APPDIR%"

REM ── Remove scheduled task ────────────────────────
schtasks /query /tn "ObjectifAI" >nul 2>&1
if errorlevel 1 (
    echo  No ObjectifAI scheduled task found.
) else (
    echo  Removing scheduled task...
    schtasks /delete /tn "ObjectifAI" /f >nul 2>&1
    if errorlevel 1 (
        echo  ERROR: Could not remove task. Try running as Administrator.
        pause & exit /b 1
    )
    echo  Scheduled task removed.
)

echo.
echo  Objectif.AI has been removed from startup.
echo  You can still run it manually with start.bat
echo.
pause
