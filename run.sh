#!/usr/bin/env bash
# Start the detector API + dashboard (Linux / macOS / Git Bash).
#   ./run.sh              -> http://localhost:8000
#   HTTPS=1 PORT=8010 ./run.sh   -> https://<LAN ip>:8010, the only way other devices get a microphone
#   PORT=9000 WORKERS=4 SEMANTIC_MODE=uncertain ANTHROPIC_API_KEY=... ./run.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=".venv/bin/python"; [ -x "$PY" ] || PY=".venv/Scripts/python.exe"
if [ ! -x "$PY" ]; then
  python3 -m venv .venv || python -m venv .venv
  PY=".venv/bin/python"; [ -x "$PY" ] || PY=".venv/Scripts/python.exe"
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -r requirements.txt
fi
if [ -f .env ]; then set -a; . ./.env; set +a; fi
SSL=()
SCHEME=http
if [ "${HTTPS:-0}" = "1" ]; then
  # Browsers only hand out the microphone in a secure context, so a plain LAN link can never run the live
  # call on someone else's phone. A self-signed certificate fixes that without needing a tunnel.
  CERT=$("$PY" -c "from backend.tls import ensure_cert; p = ensure_cert('certs'); print((p[0] + '|' + p[1]) if p else '')")
  [ -n "$CERT" ] || { echo "Could not create a certificate: $PY -m pip install cryptography" >&2; exit 1; }
  SSL=(--ssl-certfile "${CERT%%|*}" --ssl-keyfile "${CERT##*|}")
  SCHEME=https
  echo "Open on other devices (same wifi):"
  "$PY" -c "from backend.tls import lan_addresses; [print('    https://%s:${PORT:-8000}/' % i) for i in lan_addresses()]"
  echo "The certificate is self-signed: each device warns once - choose Advanced, then Proceed."
fi
echo "Altur Voice Shield -> ${SCHEME}://localhost:${PORT:-8000} (workers=${WORKERS:-2}, semantic=${SEMANTIC_MODE:-off})"
exec "$PY" -m uvicorn backend.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers "${WORKERS:-2}" "${SSL[@]}"
