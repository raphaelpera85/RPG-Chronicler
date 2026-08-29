@echo off
chcp 65001 >nul
title RPG Chronicler
cd /d "%~dp0"
if not exist "rpg_chronicler.py" (
    echo [ERRO] Execute este arquivo dentro da pasta do RPG Chronicler.
    pause
    exit /b 1
)
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [ERRO] Ambiente virtual n?o encontrado.
    echo Execute primeiro: setup_windows.bat
    pause
    exit /b 1
)
"%PYTHON_EXE%" "rpg_chronicler.py"
if %errorlevel% neq 0 (
    echo.
    echo [ERRO] Ocorreu um problema ao iniciar o RPG Chronicler.
    pause
)
