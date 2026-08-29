@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo [ERRO] Ambiente .venv nao encontrado. Execute setup_windows.bat primeiro.
  pause
  exit /b 1
)
if not exist "D:\Users\rapha\Documents\Projetos\RPG\livros\pathfinder-rpg-poster-map-folio-inner-sea-biblioteca-elfica.pdf" (
  echo [ERRO] PDF de origem nao encontrado no caminho esperado.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" ".\mapas\traduzir_mapa_ptbr.py"
if errorlevel 1 pause
endlocal
