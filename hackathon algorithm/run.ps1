# Start the detector API + dashboard on Windows.
#   .\run.ps1                      -> http://localhost:8000
#   .\run.ps1 -Port 9000 -Workers 4
#   $env:SEMANTIC_MODE="uncertain"; $env:ANTHROPIC_API_KEY="sk-ant-..."; .\run.ps1
param([int]$Port = 8000, [int]$Workers = 2, [string]$Samples = "")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) {
  python -m venv .venv
  .\.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
  .\.venv\Scripts\python.exe -m pip install --quiet -r requirements.txt
}
if (Test-Path ".env") {
  Get-Content ".env" | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2].Trim('"'), "Process") }
  }
}
if ($Samples) { $env:SAMPLES_DIR = $Samples }
Write-Host "Altur Voice Shield -> http://localhost:$Port  (workers=$Workers, model=$($env:MODEL_PATH), semantic=$($env:SEMANTIC_MODE))"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port $Port --workers $Workers
