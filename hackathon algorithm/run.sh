#!/usr/bin/env bash
# Start the detector API + dashboard (Linux / macOS / Git Bash).
#   ./run.sh              -> http://localhost:8000
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
echo "Altur Voice Shield -> http://localhost:${PORT:-8000} (workers=${WORKERS:-2}, semantic=${SEMANTIC_MODE:-off})"
exec "$PY" -m uvicorn backend.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers "${WORKERS:-2}"
