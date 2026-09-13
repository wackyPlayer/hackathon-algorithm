"""Detection rate per channel family on the out-of-distribution corpora, as JSON for the README table.

    python -m bench.ood_report --model models/detector.joblib --out data/ood_new.json

Each data/ood_<family>/ holds the same generated synthetic calls rendered through one channel, so the
numbers are paired: a difference between families is the channel and nothing else.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_AN = None


def _init(model_path: str):
    global _AN
    from backend.scoring.pipeline import Analyzer
    _AN = Analyzer(model_path or None)


def _one(path: str):
    from backend.audio import load_call
    try:
        r = _AN.analyze(load_call(open(path, "rb").read()), want_ui=False, allow_semantic=False)
        return os.path.basename(path)[:-4], float(r["p_synthetic"])
    except Exception as exc:  # pragma: no cover
        return os.path.basename(path)[:-4], float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--dirs", default="data/ood_*")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    from concurrent.futures import ProcessPoolExecutor

    out: dict = {}
    for d in sorted(glob.glob(a.dirs)):
        man = os.path.join(d, "manifest.csv")
        if not os.path.isdir(d) or not os.path.exists(man):
            continue
        fam = os.path.basename(d).replace("ood_", "")
        meta = {r["anon_id"]: r for r in csv.DictReader(open(man, encoding="utf-8"))}
        files = sorted(glob.glob(os.path.join(d, "*.wav")))
        with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.model,)) as ex:
            res = list(ex.map(_one, files))
        ps, prof = [], []
        for name, p in res:
            if not np.isfinite(p):
                continue
            ps.append(p)
            prof.append(meta.get(name, {}).get("profile", ""))
        ps, prof = np.array(ps), np.array(prof)
        rec = {"n": int(len(ps)), "detect": float((ps >= 0.5).mean()), "mean_p": round(float(ps.mean()), 4),
               "by_profile": {}}
        for pname in sorted(set(prof.tolist())):
            m = prof == pname
            if m.sum() >= 3:
                rec["by_profile"][pname or "-"] = {"n": int(m.sum()), "detect": float((ps[m] >= 0.5).mean())}
        out[fam] = rec
        prof_s = "  ".join(f"{k} {v['detect']:.0%}" for k, v in rec["by_profile"].items())
        print(f"{fam:<12} n={rec['n']:3d}  detect={rec['detect']:.1%}  mean p={rec['mean_p']:.3f}   {prof_s}",
              flush=True)

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
