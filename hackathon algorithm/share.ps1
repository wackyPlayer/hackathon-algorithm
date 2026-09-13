# Share the running dashboard/API with anyone: opens a public HTTPS tunnel (Cloudflare quick tunnel, no account)
# and writes the URL to share_url.txt so the "Share link" button appears in the dashboard header.
#   .\share.ps1                 # tunnels http://localhost:8010
#   .\share.ps1 -Port 8000
# Keep this window open while people use the link. Ctrl+C closes the tunnel.
param([int]$Port = 8010)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Find-Cloudflared {
  $c = Get-Command cloudflared -ErrorAction SilentlyContinue
  if ($c) { return $c.Source }
  foreach ($p in @("$env:ProgramFiles\cloudflared\cloudflared.exe", "$env:LOCALAPPDATA\Microsoft\WinGet\Links\cloudflared.exe", "$PSScriptRoot\cloudflared.exe")) {
    if (Test-Path $p) { return $p }
  }
  return $null
}

$exe = Find-Cloudflared
if (-not $exe) {
  Write-Host "cloudflared not found - installing with winget (Cloudflare.cloudflared)..."
  winget install --id Cloudflare.cloudflared -e --accept-source-agreements --accept-package-agreements
  $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
  $exe = Find-Cloudflared
  if (-not $exe) { throw "cloudflared still not found. Install it from https://github.com/cloudflare/cloudflared/releases and re-run." }
}

try { Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3 | Out-Null }
catch { Write-Warning "Nothing is answering on port $Port yet. Start the server first (.\run.ps1 -Port $Port) - the tunnel will connect once it is up." }

$log = Join-Path $PSScriptRoot "cloudflared.log"
if (Test-Path $log) { Remove-Item $log -Force }
Remove-Item (Join-Path $PSScriptRoot "share_url.txt") -Force -ErrorAction SilentlyContinue
Write-Host "Starting tunnel to http://localhost:$Port ..."
$proc = Start-Process -FilePath $exe -ArgumentList "tunnel --url http://localhost:$Port --no-autoupdate" -NoNewWindow -PassThru -RedirectStandardError $log

$url = $null
for ($i = 0; $i -lt 60 -and -not $url; $i++) {
  Start-Sleep -Seconds 1
  if (Test-Path $log) {
    $m = Select-String -Path $log -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -AllMatches | Select-Object -First 1
    if ($m) { $url = $m.Matches[0].Value }
  }
  if ($proc.HasExited) { break }
}
if (-not $url) { Get-Content $log -ErrorAction SilentlyContinue | Select-Object -Last 20; throw "Tunnel did not come up (see cloudflared.log)." }

Set-Content -Path (Join-Path $PSScriptRoot "share_url.txt") -Value $url -Encoding ascii
Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  Public link:  $url" -ForegroundColor Green
Write-Host "  API:          $url/detect   $url/analyze   $url/health" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "Anyone with the link can use the dashboard, the live call (HTTPS, so their microphone works) and every API route."
Write-Host "Keep this window open. Press Ctrl+C to stop sharing."
try { Wait-Process -Id $proc.Id } finally { Remove-Item (Join-Path $PSScriptRoot "share_url.txt") -Force -ErrorAction SilentlyContinue }
