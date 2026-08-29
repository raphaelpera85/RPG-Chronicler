$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $project
$env:RPG_CHRONICLER_LM_URL = 'http://127.0.0.1:8645/v1'
$env:RPG_CHRONICLER_LM_MODEL = 'stepfun/step-3.7-flash:free'
$env:RPG_CHRONICLER_WHISPER_MODEL = 'large-v3-turbo'
$ownsProxy = $false
$proxy = $null
$listener = Get-NetTCPConnection -LocalPort 8645 -State Listen -ErrorAction SilentlyContinue
if (-not $listener) {
    $hermes = (Get-Command hermes -ErrorAction Stop).Source
    $proxy = Start-Process -FilePath $hermes -ArgumentList 'proxy start' -WindowStyle Hidden -PassThru
    $ownsProxy = $true
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        try {
            Invoke-WebRequest -Uri 'http://127.0.0.1:8645/v1/models' -UseBasicParsing -TimeoutSec 1 | Out-Null
            break
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
}
try {
    $pythonw = Join-Path $project '.venv\Scripts\pythonw.exe'
    if (-not (Test-Path -LiteralPath $pythonw)) { $pythonw = Join-Path $project '.venv\Scripts\python.exe' }
    if (-not (Test-Path -LiteralPath $pythonw)) { throw 'Ambiente virtual não encontrado. Execute setup_windows.bat.' }
    Start-Process -FilePath $pythonw -ArgumentList (Join-Path $project 'rpg_chronicler.py') -Wait
}
finally {
    if ($ownsProxy -and $proxy -and -not $proxy.HasExited) { Stop-Process -Id $proxy.Id -Force -ErrorAction SilentlyContinue }
}
