"""Development launcher: sets convenient defaults (dataset samples dropdown) and starts uvicorn.

    python dev_server.py [--port 8000] [--reload]
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

CANDIDATE_SAMPLE_DIRS = [
    os.getenv("SAMPLES_DIR", ""),
    os.path.join(HERE, "audio"),
    os.path.join(os.path.expanduser("~"), "Downloads", "altur-challenge-audio", "audio"),
]
CANDIDATE_MANIFESTS = [
    os.getenv("SAMPLES_MANIFEST", ""),
    os.path.join(HERE, "manifest.csv"),
    os.path.join(os.path.expanduser("~"), "Downloads", "manifest.csv"),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--reload", action="store_true")
    a = ap.parse_args()
    for d in CANDIDATE_SAMPLE_DIRS:
        if d and os.path.isdir(d):
            os.environ.setdefault("SAMPLES_DIR", d)
            break
    for m in CANDIDATE_MANIFESTS:
        if m and os.path.exists(m):
            os.environ.setdefault("SAMPLES_MANIFEST", m)
            break
    import uvicorn
    print(f"samples: {os.environ.get('SAMPLES_DIR', '-')}  manifest: {os.environ.get('SAMPLES_MANIFEST', '-')}", file=sys.stderr)
    uvicorn.run("backend.main:app", host=a.host, port=a.port, reload=a.reload, log_level="info")
