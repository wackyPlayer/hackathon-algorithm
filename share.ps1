# Share the running dashboard/API with anyone: opens a public HTTPS tunnel (Cloudflare quick tunnel, no account)
# and writes the URL to share_url.txt so the "Share link" button appears in the dashboard header.
#   .\share.ps1                 # tunnels http://localhost:8010
#   .\share.ps1 -Port 8000
# Keep this window open while people use the link. Ctrl+C closes the tunnel.
param([int]$Port = 8010, [string]$Hostname = "", [string]$TunnelName = "calliope",
      [ValidateSet("http2", "quic", "auto")][string]$Protocol = "http2")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Find-Cloudflared {
  $c = Get-Command cloudflared -ErrorAction SilentlyContinue
  if ($c) { return $c.Source }
  # winget installs the 64-bit package under "Program Files (x86)" on this machine, which the original
  # list did not include -- so the script reported "not found" on a box where cloudflared was installed.
  $candidates = @(
    "$env:ProgramFiles\cloudflared\cloudflared.exe",
    "${env:ProgramFiles(x86)}\cloudflared\cloudflared.exe",
    "$env:LOCALAPPDATA\Microsoft\WinGet\Links\cloudflared.exe",
    "$PSScriptRoot\cloudflared.exe"
  )
  foreach ($p in $candidates) { if ($p -and (Test-Path $p)) { return $p } }
  $found = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter cloudflared.exe -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($found) { return $found.FullName }
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

# A named tunnel keeps the SAME hostname across restarts, which a quick tunnel does not: every restart
# hands out a new *.trycloudflare.com name, and anything holding the old link (a judge's tab, a phone)
# breaks. Needs a one-time `cloudflared tunnel login` in a browser and a domain on your Cloudflare account.
if ($Hostname) {
  $cfgDir = Join-Path $env:USERPROFILE ".cloudflared"
  if (-not (Test-Path (Join-Path $cfgDir "cert.pem"))) {
    Write-Host "First run a browser login (once per machine):" -ForegroundColor Yellow
    Write-Host "    & '$exe' tunnel login" -ForegroundColor Yellow
    throw "no $cfgDir\cert.pem yet - log in, then re-run this with -Hostname $Hostname"
  }
  $existing = & $exe tunnel list 2>&1 | Select-String -Pattern "\s$TunnelName\s"
  if (-not $existing) { & $exe tunnel create $TunnelName }
  & $exe tunnel route dns --overwrite-dns $TunnelName $Hostname
  Write-Host ""
  Write-Host "==========================================================" -ForegroundColor Green
  Write-Host "  Stable link:  https://$Hostname" -ForegroundColor Green
  Write-Host "  API:          https://$Hostname/detect   https://$Hostname/health" -ForegroundColor Green
  Write-Host "==========================================================" -ForegroundColor Green
  Set-Content -Path (Join-Path $PSScriptRoot "share_url.txt") -Value "https://$Hostname" -Encoding ascii
  try { & $exe tunnel run --url "http://localhost:$Port" $TunnelName }
  finally { Remove-Item (Join-Path $PSScriptRoot "share_url.txt") -Force -ErrorAction SilentlyContinue }
  return
}

$log = Join-Path $PSScriptRoot "cloudflared.log"
if (Test-Path $log) { Remove-Item $log -Force }
Remove-Item (Join-Path $PSScriptRoot "share_url.txt") -Force -ErrorAction SilentlyContinue
Write-Host "Starting tunnel to http://localhost:$Port ..."
# --protocol http2 by default. A quick tunnel prefers QUIC (UDP 7844), which university and corporate
# wifi routinely block; cloudflared still prints a public URL, but nothing can reach it and every request
# comes back 530. HTTP/2 rides TCP 443 and gets through. Pass -Protocol auto to let cloudflared choose.
$cfArgs = "tunnel --url http://localhost:$Port --no-autoupdate"
if ($Protocol -ne "auto") { $cfArgs += " --protocol $Protocol" }
Write-Host "cloudflared $cfArgs" -ForegroundColor DarkGray
$proc = Start-Process -FilePath $exe -ArgumentList $cfArgs -NoNewWindow -PassThru -RedirectStandardError $log

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

# A URL is NOT proof the tunnel works: cloudflared prints one before it has a connection. Two different
# things can go wrong and they need different answers:
#   530 from the edge      -> the tunnel really is not connected. Fail; the link would be dead for everyone.
#   name does not resolve  -> usually just THIS machine's resolver being slow on a brand-new hostname
#                             (campus and corporate DNS cache negative answers). The link is fine for
#                             everyone else, so warn and hand it over rather than throwing it away.
Write-Host "Got $url - checking it actually serves..." -ForegroundColor DarkGray
$state = "unknown"
for ($i = 0; $i -lt 45 -and $state -ne "ok"; $i++) {
  Start-Sleep -Seconds 2
  if ($proc.HasExited) { break }
  if (-not (Select-String -Path $log -Pattern "Registered tunnel connection" -Quiet)) { continue }
  try {
    $r = Invoke-WebRequest -UseBasicParsing -Uri "$url/health" -TimeoutSec 10
    if ($r.StatusCode -eq 200) { $state = "ok" }
  } catch {
    $msg = "$($_.Exception.Message)"
    if ($msg -match "530") { $state = "edge530" }
    elseif ($msg -match "remote name|could not be resolved|No such host|NameResolutionFailure") { $state = "dns" }
    else { $state = "other" }
  }
}
if ($state -eq "edge530" -or ($state -ne "ok" -and $state -ne "dns")) {
  Get-Content $log -ErrorAction SilentlyContinue | Select-Object -Last 15
  Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
  throw "The tunnel came up but $url does not serve (530 = the edge has no connection to this machine). " +
        "Try -Protocol auto, or serve TLS locally instead: run.ps1 -Port $Port -Https"
}
if ($state -eq "dns") {
  Write-Warning "This machine cannot resolve $url yet - its DNS server is slow on new tunnel hostnames."
  Write-Warning "The tunnel IS connected, so the link should work for other people and will start working"
  Write-Warning "here shortly. To check from here now:  curl $url/health --doh-url https://1.1.1.1/dns-query"
}

Set-Content -Path (Join-Path $PSScriptRoot "share_url.txt") -Value $url -Encoding ascii
Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  Public link:  $url" -ForegroundColor Green
Write-Host "  API:          $url/detect   $url/analyze   $url/health" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "Anyone with the link can use the dashboard, the live call (HTTPS, so their microphone works) and every API route."
Write-Host "Keep this window open. Press Ctrl+C to stop sharing."
try { Wait-Process -Id $proc.Id } finally { Remove-Item (Join-Path $PSScriptRoot "share_url.txt") -Force -ErrorAction SilentlyContinue }
