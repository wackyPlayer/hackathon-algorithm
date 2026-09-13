"""Evaluate the deployed pipeline (exactly what /detect computes) on a labelled corpus of WAV calls.

    python -m bench.eval_corpus --audio data/tts_corpus --manifest data/tts_corpus/manifest.csv
    python -m bench.eval_corpus --audio <dataset audio> --manifest <dataset manifest> --split val --clip 30
    python -m bench.eval_corpus ... --model models/detector_new.joblib --by profile engine

Prints detection rate / false-alarm rate, mean p(synthetic) and AUC overall and per group column, and lists
the worst misses with the feature groups that pushed them the wrong way. Writes a per-call CSV (--csv).
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_AN = None


def _init(model_path: str):
    global _AN
    from backend.scoring.pipeline import Analyzer
    _AN = Analyzer(model_path or None)


def _one(args):
    from backend.audio import load_call, call_from_arrays
    path, clip = args
    try:
        call = load_call(open(path, "rb").read())
        if clip:
            n = int(clip * call.sr)
            call = call_from_arrays(call.caller_raw[:n], None if call.agent is None else call.agent[:n], call.sr)
        r = _AN.analyze(call, want_ui=False, allow_semantic=False)
        g = (r.get("contributions") or {}).get("groups", {})
        return {"p": r["p_synthetic"], "is_syn": r["is_synthetic"], "conf": r["confidence"], "speech": r["speech_seconds"],
                "fast": r["signals"]["fast_model_p"], "heur": r["signals"]["heuristic_p"],
                "acoustic_p": r["signals"].get("acoustic_p"), "groups": g, "err": None}
    except Exception as exc:
        return {"p": None, "err": repr(exc)}


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    if len(set(y.tolist())) < 2:
        return float("nan")
    order = s.argsort()
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def summarize(rows, key=None):
    lab = np.array([r["y"] for r in rows])
    p = np.array([r["p"] for r in rows])
    pred = (p >= 0.5).astype(int)
    out = {"n": len(rows), "acc": float((pred == lab).mean())}
    if (lab == 1).any():
        out["recall_syn"] = float(pred[lab == 1].mean())
        out["p_syn_mean"] = float(p[lab == 1].mean())
        out["p_syn_min"] = float(p[lab == 1].min())
    if (lab == 0).any():
        out["fa_rate"] = float(pred[lab == 0].mean())
        out["p_hum_mean"] = float(p[lab == 0].mean())
        out["p_hum_max"] = float(p[lab == 0].max())
    out["auc"] = auc(lab, p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--split", default="")
    ap.add_argument("--clip", type=float, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--by", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--csv", default="")
    ap.add_argument("--worst", type=int, default=12)
    a = ap.parse_args()

    man = [r for r in csv.DictReader(open(a.manifest, encoding="utf-8")) if not a.split or r.get("split") == a.split]
    if a.limit:
        man = man[: a.limit]
    jobs = [(os.path.join(a.audio, r["anon_id"] + ".wav"), a.clip) for r in man]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.model,)) as ex:
        res = list(ex.map(_one, jobs, chunksize=2))
    dt = time.time() - t0
    rows = []
    for r, m in zip(res, man):
        if r["p"] is None:
            print("  error:", m["anon_id"], r["err"])
            continue
        rows.append({**m, **r, "y": 1 if m["label"] == "synthetic" else 0})
    print(f"{len(rows)} calls in {dt:.0f}s ({dt / max(len(rows), 1):.2f} s/call with {a.workers} workers)"
          + (f", clip={a.clip:.0f}s" if a.clip else ""))

    def show(name, sub):
        s = summarize(sub)
        parts = [f"n={s['n']:3d}", f"acc={s['acc']:.3f}"]
        if "recall_syn" in s:
            parts.append(f"detect={s['recall_syn']:.3f} (p mean {s['p_syn_mean']:.2f}, min {s['p_syn_min']:.2f})")
        if "fa_rate" in s:
            parts.append(f"false-alarm={s['fa_rate']:.3f} (p mean {s['p_hum_mean']:.2f}, max {s['p_hum_max']:.2f})")
        if not np.isnan(s["auc"]):
            parts.append(f"auc={s['auc']:.3f}")
        print(f"  {name:38s} " + "  ".join(parts))

    print("overall:")
    show("all", rows)
    for col in a.by:
        vals = sorted({r.get(col, "") for r in rows})
        print(f"by {col}:")
        for v in vals:
            show(str(v)[:38], [r for r in rows if r.get(col, "") == v])
    wrong = sorted([r for r in rows if (r["p"] >= 0.5) != (r["y"] == 1)], key=lambda r: -abs(r["p"] - 0.5))
    print(f"errors: {len(wrong)}/{len(rows)}")
    for r in wrong[: a.worst]:
        g = sorted(r["groups"].items(), key=lambda kv: -abs(kv[1]))[:5]
        gs = " ".join(f"{k}{v:+.1f}" for k, v in g)
        print(f"  {r['anon_id'][:60]:60s} {r['label']:9s} p={r['p']:.3f} fast={r['fast']:.2f} speech={r['speech']:.0f}s  {gs}")
    if a.csv:
        keys = [k for k in man[0].keys()] + ["p", "is_syn", "conf", "speech", "fast", "heur", "acoustic_p"]
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
