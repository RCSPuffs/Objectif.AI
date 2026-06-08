@echo off
title Objectif.AI — Setup Service
setlocal EnableDelayedExpansion

echo.
echo  ================================================
echo   Objectif.AI — Install as Startup Service
echo  ================================================
echo.

REM ── Check Python ──────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found. Please install Python 3.10+
    echo  https://www.python.org/downloads/
    pause & exit /b 1
)

REM ── Get paths ─────────────────────────────────────
cd /d "%~dp0"
set "APPDIR=%~dp0"
if "%APPDIR:~-1%"=="\" set "APPDIR=%APPDIR:~0,-1%"

REM Get the full path to python.exe — this is what the task will use directly.
REM Using the full path ensures Task Scheduler runs the same Python as the user,
REM regardless of what PATH looks like in the scheduled task environment.
for /f "delims=" %%i in ('where python') do (
    set "PYEXE=%%i"
    goto :found_py
)
:found_py

echo  App directory : %APPDIR%
echo  Python        : %PYEXE%
echo.

REM ── Install dependencies ─────────────────────────
echo  Installing/checking dependencies...
python -m pip install -r "%APPDIR%\requirements.txt" --quiet
if errorlevel 1 (
    echo  WARNING: Some dependencies may not have installed correctly.
    echo  Check requirements.txt and try: pip install -r requirements.txt
)
echo  Dependencies OK.
echo.

REM ── Remove existing task if present ──────────────
schtasks /query /tn "ObjectifAI" >nul 2>&1
if not errorlevel 1 (
    echo  Removing existing ObjectifAI scheduled task...
    schtasks /delete /tn "ObjectifAI" /f >nul 2>&1
)

REM ── Get current username ──────────────────────────
for /f "tokens=*" %%u in ('whoami') do set "WHOAMI=%%u"

REM ── Write task XML ────────────────────────────────
set "TASKXML=%TEMP%\objectifai_task.xml"
python "%APPDIR%\_write_task_xml.py" "%PYEXE%" "%APPDIR%" "%WHOAMI%" "%TASKXML%"

if errorlevel 1 (
    echo  ERROR: Failed to write task XML.
    pause & exit /b 1
)

REM ── Register scheduled task ───────────────────────
schtasks /create /tn "ObjectifAI" /xml "%TASKXML%" /f
del "%TASKXML%" >nul 2>&1

if errorlevel 1 (
    echo.
    echo  ERROR: Failed to register scheduled task.
    echo  Try running this script as Administrator.
    pause & exit /b 1
)

echo.
echo  ================================================
echo   Service registered successfully!
echo.
echo   Objectif.AI will now start automatically
echo   when you log in to Windows.
echo.
echo   Starting now...
echo  ================================================
echo.

REM ── Start immediately ─────────────────────────────
schtasks /run /tn "ObjectifAI" >nul 2>&1

echo  Started. Dashboard will be available at:
echo  http://localhost:32168
echo.
echo  To remove the service, run: remove-service.bat
echo.

pause
