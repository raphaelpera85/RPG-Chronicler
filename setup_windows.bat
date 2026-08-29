@echo off
chcp 65001 >nul
setlocal
title RPG Chronicler - Instala??o
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERRO] Python 3.11 ou superior n?o foi encontrado no PATH.
    pause
    exit /b 1
)

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)"
if errorlevel 1 (
    echo [ERRO] Python 3.11 ou superior ? necess?rio.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto :error
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo [AVISO] FFmpeg n?o est? no PATH. Instale-o para usar formatos comprimidos.
)
where ffprobe >nul 2>nul
if errorlevel 1 (
    echo [AVISO] FFprobe n?o est? no PATH. Ele ? necess?rio para validar metadados de ?udio.
)

if not exist "rpg_chronicler_config.json" copy /Y "rpg_chronicler_config.example.json" "rpg_chronicler_config.json" >nul
if not exist "biblia_personagens_e_cenarios.md" copy /Y "biblia.example.md" "biblia_personagens_e_cenarios.md" >nul

echo.
echo [OK] Instala??o conclu?da. Execute iniciar_gravador_rpg.bat
pause
exit /b 0

:error
echo.
echo [ERRO] A instala??o falhou.
pause
exit /b 1
