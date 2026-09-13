"""Development launcher: sets convenient defaults (dataset samples dropdown) and starts uvicorn.

    python dev_server.py [--port 8000] [--reload]
    python dev_server.py --port 8010 --https      # https://<your LAN ip>:8010, so other devices get a
                                                  # microphone (browsers refuse it on plain http)
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
    # PORT is honoured so a launcher can hand us a free port (run.sh uses it too); --port still wins.
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT") or 8000))
    ap.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    ap.add_argument("--reload", action="store_true")
    ap.add_argument("--https", action="store_true",
                    help="serve TLS with a self-signed certificate so the live call works on other devices")
    ap.add_argument("--regen-cert", action="store_true", help="throw the certificate away and make a new one")
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
    ssl: dict = {}
    if a.https:
        from backend.tls import ensure_cert, lan_addresses
        pair = ensure_cert(os.path.join(HERE, "certs"), regenerate=a.regen_cert)
        if not pair:
            sys.exit("--https needs a certificate: pip install cryptography (or put openssl on PATH)")
        ssl = {"ssl_certfile": pair[0], "ssl_keyfile": pair[1]}
        print(f"HTTPS on. Open https://localhost:{a.port}/ here, and on other devices:", file=sys.stderr)
        for ip in lan_addresses():
            print(f"    https://{ip}:{a.port}/", file=sys.stderr)
        print("The certificate is self-signed, so each device shows a warning once: choose Advanced -> "
              "Proceed. The microphone works after that.", file=sys.stderr)
    uvicorn.run("backend.main:app", host=a.host, port=a.port, reload=a.reload, log_level="info", **ssl)
