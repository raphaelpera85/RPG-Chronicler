@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [ERRO] Ambiente virtual nao encontrado. Execute setup_windows.bat.
    pause
    exit /b 1
)
where hermes >nul 2>nul
if errorlevel 1 (
    echo [ERRO] Hermes nao foi encontrado no PATH.
    pause
    exit /b 1
)
start "" /b powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0hermes_session.ps1"
exit /b 0
