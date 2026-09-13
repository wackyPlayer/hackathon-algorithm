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

$ssl = @()
$scheme = "http"
if ($Https) {
  # Browsers only allow the microphone in a secure context, so a plain LAN link can never run the live
  # call on someone else's phone. A self-signed certificate fixes that without needing a tunnel.
  $cert = .\.venv\Scripts\python.exe -c "from backend.tls import ensure_cert; p = ensure_cert('certs'); print(p[0] + '|' + p[1] if p else '')"
  if (-not $cert) { throw "Could not create a certificate. Run: .\.venv\Scripts\python.exe -m pip install cryptography" }
  $parts = $cert.Trim().Split("|")
  $ssl = @("--ssl-certfile", $parts[0], "--ssl-keyfile", $parts[1])
  $scheme = "https"
  Write-Host ""
  Write-Host "Open on this machine:  https://localhost:$Port/" -ForegroundColor Green
  Write-Host "Open on other devices (same wifi):" -ForegroundColor Green
  .\.venv\Scripts\python.exe -c "from backend.tls import lan_addresses; [print('    https://%s:$Port/' % i) for i in lan_addresses()]"
  Write-Host "The certificate is self-signed: each device warns once - choose Advanced, then Proceed." -ForegroundColor Yellow
  Write-Host ""
}
Write-Host "Altur Voice Shield -> ${scheme}://localhost:$Port  (workers=$Workers, model=$($env:MODEL_PATH), semantic=$($env:SEMANTIC_MODE))"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port $Port --workers $Workers @ssl
