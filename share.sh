#!/usr/bin/env bash
# Share the running dashboard/API with anyone: public HTTPS tunnel via Cloudflare quick tunnel (no account).
#   ./share.sh            # tunnels http://localhost:8010
#   PORT=8000 ./share.sh
# Keep it running while people use the link; Ctrl+C stops sharing.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8010}"
if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared not found. Install it: brew install cloudflared  |  apt: see https://pkg.cloudflare.com  |  or download from https://github.com/cloudflare/cloudflared/releases" >&2
  exit 1
fi
rm -f share_url.txt cloudflared.log
cloudflared tunnel --url "http://localhost:${PORT}" --no-autoupdate 2> cloudflared.log &
PID=$!
trap 'kill $PID 2>/dev/null; rm -f share_url.txt' EXIT
URL=""
for _ in $(seq 1 60); do
  sleep 1
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' cloudflared.log | head -1 || true)
  [ -n "$URL" ] && break
done
if [ -z "$URL" ]; then tail -20 cloudflared.log; echo "tunnel did not come up" >&2; exit 1; fi
printf '%s' "$URL" > share_url.txt
echo "=========================================================="
echo "  Public link:  $URL"
echo "  API:          $URL/detect   $URL/analyze   $URL/health"
echo "=========================================================="
echo "Anyone with the link can use the dashboard, the live call and every API route. Ctrl+C to stop."
wait $PID
