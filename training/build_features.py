"""Extract the fast feature table for every call in the manifest (parallel).

    python -m training.build_features --audio <dir with anon_id.wav> --manifest manifest.csv --out data/features.csv

Optional:
    --workers N        processes (default: cpu_count - 1)
    --limit N          only the first N manifest rows (smoke test)
    --clip SECONDS     also add clipped variants (first N seconds of the call) as extra training rows
                       so the model sees short excerpts too (ids get a "#clipN" suffix).
    --embeddings       also compute SSL embeddings (needs torch + transformers) -> data/embeddings.npz
    --extra DIR CSV    additional labelled corpus (e.g. data/tts_corpus built by training/tts_corpus.py); its
                       manifest may carry `group` (cross-validation group) and `source` columns
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.audio import call_from_arrays, load_call  # noqa: E402
from backend.scoring.pipeline import extract  # noqa: E402


def _one(args):
    anon_id, path, clips = args
    rows = []
    try:
        data = open(path, "rb").read()
        call = load_call(data)
        ex = extract(call, want_ui=False)
        rows.append((anon_id, ex.features))
        for c in clips:
            n = int(c * call.sr)
            if n >= len(call.caller):
                continue
            sub = call_from_arrays(call.caller_raw[:n], None if call.agent is None else call.agent[:n], call.sr)
            rows.append((f"{anon_id}#clip{int(c)}", extract(sub, want_ui=False).features))
        return rows, None
    except Exception as exc:  # pragma: no cover
        return rows, f"{anon_id}: {exc!r}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="data/features.csv")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clip", type=float, action="append", default=[])
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--extra", nargs=2, action="append", default=[], metavar=("AUDIO_DIR", "MANIFEST"))
    a = ap.parse_args()

    rows = [dict(r, _dir=a.audio, _source="dataset") for r in csv.DictReader(open(a.manifest, encoding="utf-8"))]
    if a.limit:
        rows = rows[: a.limit]
    for d, m in a.extra:
        extra = [dict(r, _dir=d, _source=(r.get("source") or "extra")) for r in csv.DictReader(open(m, encoding="utf-8"))]
        print(f"extra corpus {m}: {len(extra)} calls")
        rows.extend(extra)
    meta = {r["anon_id"]: r for r in rows}
    jobs = [(r["anon_id"], os.path.join(r["_dir"], r["anon_id"] + ".wav"), a.clip) for r in rows]
    t0 = time.time()
    results: dict = {}
    errors = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rs, err = f.result()
            for rid, feats in rs:
                results[rid] = feats
            if err:
                errors.append(err)
            if i % 25 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} done ({time.time() - t0:.0f}s)", flush=True)
    keys = sorted({k for f in results.values() for k in f})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "anon_id", "label", "split", "duration_s", "group", "source"] + keys)
        for rid in sorted(results):
            base = rid.split("#")[0]
            m = meta[base]
            f = results[rid]
            w.writerow([rid, base, m["label"], m["split"], m.get("duration_s", ""), m.get("group") or base, m["_source"]] +
                       [("" if (f.get(k) is None or not np.isfinite(f.get(k, np.nan))) else f[k]) for k in keys])
    print(f"wrote {a.out}: {len(results)} rows x {len(keys)} features in {time.time() - t0:.0f}s; errors: {len(errors)}")
    for e in errors[:10]:
        print("  ", e)

    if a.embeddings:
        from backend.features.embeddings import Embedder
        from backend.vad import stft_power, vad_from_power
        emb = Embedder()
        ids, vecs = [], []
        for r in rows:
            call = load_call(open(os.path.join(a.audio, r["anon_id"] + ".wav"), "rb").read())
            P, _ = stft_power(call.caller)
            v = emb.embed(call.caller, vad_from_power(P))
            ids.append(r["anon_id"])
            vecs.append(v)
        out = os.path.join(os.path.dirname(os.path.abspath(a.out)), "embeddings.npz")
        np.savez(out, ids=np.array(ids), X=np.stack(vecs), model=emb.model_name, layer=emb.layer)
        print("wrote", out)


if __name__ == "__main__":
    main()
