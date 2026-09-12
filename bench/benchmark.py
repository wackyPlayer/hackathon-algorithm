"""Judge-style benchmark: POST base64 WAVs to /detect, measure latency and accuracy.

    python -m bench.benchmark --url http://127.0.0.1:8000/detect --audio <dir> --manifest manifest.csv --split val
    python -m bench.benchmark --url http://127.0.0.1:8000/detect --dir clips/   (no labels: just verdicts + latency)

Options: --limit N, --clip SECONDS (send only the first N seconds), --field audio (JSON key), --concurrency K
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def clip_wav(data: bytes, seconds: float) -> bytes:
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    n = int(seconds * sr)
    buf = io.BytesIO()
    sf.write(buf, x[:n], sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def auc_rank(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    if len(set(y.tolist())) < 2:
        return float("nan")
    order = s.argsort()
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/detect")
    ap.add_argument("--audio", default="")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--dir", default="")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clip", type=float, default=0)
    ap.add_argument("--field", default="audio")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=120)
    a = ap.parse_args()

    items = []
    if a.manifest:
        for r in csv.DictReader(open(a.manifest, encoding="utf-8")):
            if a.split and r["split"] != a.split:
                continue
            items.append((r["anon_id"], os.path.join(a.audio, r["anon_id"] + ".wav"), 1 if r["label"] == "synthetic" else 0))
    else:
        for f in sorted(os.listdir(a.dir)):
            if f.lower().endswith(".wav"):
                items.append((f, os.path.join(a.dir, f), None))
    if a.limit:
        items = items[: a.limit]

    def one(it):
        name, path, label = it
        data = open(path, "rb").read()
        if a.clip:
            data = clip_wav(data, a.clip)
        body = json.dumps({a.field: base64.b64encode(data).decode("ascii")})
        t0 = time.time()
        try:
            r = httpx.post(a.url, content=body, headers={"Content-Type": "application/json"}, timeout=a.timeout)
            dt = time.time() - t0
            j = r.json()
            return name, label, j.get("is_synthetic"), j.get("confidence"), dt, r.status_code, None
        except Exception as exc:
            return name, label, None, None, time.time() - t0, 0, str(exc)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        rows = list(ex.map(one, items))
    total = time.time() - t0
    lat = [r[4] for r in rows if r[5] == 200]
    print(f"{len(rows)} requests in {total:.1f}s; ok={len(lat)}; latency mean={statistics.mean(lat) if lat else float('nan'):.2f}s "
          f"p50={statistics.median(lat) if lat else float('nan'):.2f}s max={max(lat) if lat else float('nan'):.2f}s")
    errs = [r for r in rows if r[5] != 200]
    for r in errs[:5]:
        print("  error:", r[0], r[5], r[6])
    labelled = [r for r in rows if r[1] is not None and r[2] is not None]
    if labelled:
        y = np.array([r[1] for r in labelled])
        pred = np.array([1 if r[2] else 0 for r in labelled])
        conf = np.array([r[3] if r[3] is not None else 0.5 for r in labelled], dtype=float)
        p_syn = np.where(pred == 1, conf, 1 - conf)
        acc = float((pred == y).mean())
        tp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
        fp, tn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 0)).sum())
        brier = float(np.mean((p_syn - y) ** 2))
        print(f"accuracy={acc:.4f}  AUC={auc_rank(y, p_syn):.4f}  brier={brier:.4f}  TP={tp} FN={fn} FP={fp} TN={tn}")
        wrong = [r for r in labelled if (1 if r[2] else 0) != r[1]]
        for r in wrong[:10]:
            print(f"  wrong: {r[0]} label={'synthetic' if r[1] else 'human'} -> is_synthetic={r[2]} conf={r[3]:.3f}")
    else:
        for r in rows[:20]:
            print(f"  {r[0]}: is_synthetic={r[2]} confidence={r[3]} ({r[4]:.2f}s)")


if __name__ == "__main__":
    main()
