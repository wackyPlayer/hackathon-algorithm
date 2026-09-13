"""Re-derive the turn_taking heuristic's centres from data, using only the calls the model is fitted on.

    python -m training.recalibrate_heuristics --features data/features_v4.csv

The interpretable layer in backend/scoring/heuristics.py scores each term with a logistic on
(value - centre) / width, so the centres and widths are model constants and have to be fitted like any
others: derived from the held-out calls they would leak, and the aspect scores printed next to a verdict
would flatter themselves. This reads the same held-out carve training/train.py uses (same flag defaults,
same seed, same grouping) and computes every constant on the fitted side only.

Prints a ready-to-paste term list plus the evidence behind each centre.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.train import carve_holdout, load_table  # noqa: E402

# (feature, higher_is_synthetic, weight, label) -- the centre and width come from the data below.
TERMS = [
    ("conv_dead_caller_fill_frac", True, 1.0, "caller, not the agent, breaks the dead air"),
    ("conv_resp_mad_norm", False, 1.0, "scatter of reply delays around its own median"),
    ("conv_pause_cv", False, 0.9, "variability of pauses inside a caller turn"),
    ("conv_resp_entropy_norm", False, 0.8, "spread of the reply-delay histogram"),
    ("conv_int_yield_cv", True, 0.6, "how repeatably the caller yields when talked over"),
    ("conv_turn_short_frac", False, 0.6, "short acknowledgement turns"),
    ("conv_backchannel_rate_per_agent_min", False, 0.5, "back-channels per agent minute"),
    ("conv_resp_predictability", True, 0.3, "reply delay explained by the agent's turn length"),
    ("conv_resp_slow_frac", True, 0.3, "replies slower than 3 s"),
    ("conv_resp_median", True, 0.3, "median response latency (s)"),
    ("conv_resp_frac_over_2s", True, 0.3, "responses slower than 2 s"),
    ("conv_turn_dur_mean", True, 0.2, "mean caller turn length (s)"),
]


def auc(y, x):
    """Rank AUC, ignoring missing values. > 0.5 means a higher value goes with synthetic."""
    ok = np.isfinite(x)
    y, x = y[ok], x[ok]
    if len(set(y.tolist())) < 2:
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ranks over ties
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features_v4.csv")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--holdout-seed", type=int, default=20260912)
    a = ap.parse_args()

    X, y, groups, split, ids, names, source, chan, prof = load_table(a.features)
    split, held = carve_holdout(split, source, groups, y, a.holdout_frac, a.holdout_seed)
    full = np.array(["#" not in i for i in ids])
    # Constants come from full calls AND their 30 s / 60 s prefixes. Fitting on full calls alone puts every
    # centre where a three-minute conversation lands, and the same caller clipped to a minute then sits on
    # the wrong side of it -- measured at a 33 % false-flag rate on human 60 s clips. The server is sent
    # clips of any length, so the constants have to hold across lengths.
    fit = (split == "train")                        # fitted side only: never the held-out calls
    idx = {n: j for j, n in enumerate(names)}
    ds = fit & (source == "dataset")
    print(f"{a.features}: constants from {int(ds.sum())} fitted dataset rows "
          f"({int((ds & full).sum())} full calls + their clips; "
          f"{int((ds & (y == 0)).sum())} human / {int((ds & (y == 1)).sum())} synthetic); "
          f"{int(held.sum())} held-out rows excluded\n")

    print('    "turn_taking": {')
    print('        "label": "Turn-taking behaviour",')
    print('        "weight": 1.3,')
    print('        "terms": [')
    ev = []
    proposed = []
    for feat, hi_syn, w, label in TERMS:
        j = idx.get(feat)
        if j is None:
            print(f"        # {feat}: not in the feature table, skipped")
            continue
        col = X[:, j]
        h = col[ds & (y == 0)]
        s = col[ds & (y == 1)]
        h, s = h[np.isfinite(h)], s[np.isfinite(s)]
        if len(h) < 20 or len(s) < 20:
            print(f"        # {feat}: too few values ({len(h)} human / {len(s)} synthetic), skipped")
            continue
        mh, ms = float(np.median(h)), float(np.median(s))
        centre = (mh + ms) / 2
        iqr = (np.subtract(*np.percentile(h, [75, 25])) + np.subtract(*np.percentile(s, [75, 25]))) / 2
        width = max(abs(float(iqr)) / 2, 1e-3)
        # keep the declared direction honest: if the fitted data disagrees with it, say so
        a_syn = auc(y[ds & np.isfinite(col)], col[ds & np.isfinite(col)])
        measured_hi = a_syn > 0.5
        flag = "" if measured_hi == hi_syn else "   # NOTE: measured direction disagrees with this flag"
        print(f'            ("{feat}", {centre:.3g}, {width:.3g}, {hi_syn}, {w}, "{label}"),{flag}')
        ev.append((feat, mh, ms, centre, width, a_syn, len(h), len(s)))
        proposed.append((feat, centre, width, hi_syn, w, label))
    print("        ],")
    print("    },")

    print(f"\n{'feature':<38} {'human med':>10} {'synth med':>10} {'centre':>8} {'width':>8} {'AUC':>7}  n")
    for feat, mh, ms, c, w, a_, nh, ns in ev:
        print(f"{feat:<38} {mh:>10.3f} {ms:>10.3f} {c:>8.3f} {w:>8.3f} {a_:>7.3f}  {nh}/{ns}")

    # ---- does the new list actually beat the shipped one? Score both on the HELD-OUT side.
    from backend.scoring.heuristics import ASPECTS, _sig

    def score(terms, i):
        """The same weighted mean of _sig terms that heuristics.aspect_scores computes."""
        num = den = 0.0
        for feat, centre, width, hi, w, _lab in terms:
            j = idx.get(feat)
            if j is None or not np.isfinite(X[i, j]):
                continue
            num += w * _sig(float(X[i, j]), centre, width, hi)
            den += w
        return num / den if den else float("nan")

    current = ASPECTS["turn_taking"]["terms"]
    fitted = {f: (c, w) for f, c, w, _hi, _wt, _lb in proposed}

    # A middle path. The shipped list is highly specific on real calls and nearly blind to a caller that
    # answers fast; the proposed one catches those and accuses far too many real customers. The hybrid keeps
    # the shipped magnitude terms, at reduced weight, and adds the three consistency terms that hold their
    # direction on every corpus - so a fast bot has something to trip, without spending the false-alarm
    # budget that matters most to a bank.
    keep = {"conv_resp_median": 0.8, "conv_resp_min": 0.6, "conv_resp_frac_over_2s": 0.6,
            "conv_dead_caller_fill_frac": 0.8, "conv_backchannel_rate_per_agent_min": 0.4,
            "conv_int_through_frac": 0.3, "conv_turn_dur_mean": 0.3}
    hybrid = [(f, c, w, hi, keep[f], lb) for f, c, w, hi, _wt, lb in current if f in keep]
    for f, wt in (("conv_resp_mad_norm", 0.7), ("conv_pause_cv", 0.6), ("conv_resp_entropy_norm", 0.5)):
        if f in fitted:
            c, w = fitted[f]
            hi = next(t[1] for t in TERMS if t[0] == f)
            lb = next(t[3] for t in TERMS if t[0] == f)
            hybrid.append((f, c, w, hi, wt, lb))

    lengths = (("full calls", full),
               ("60 s clips", np.array([i.endswith("#clip60") for i in ids])),
               ("30 s clips", np.array([i.endswith("#clip30") for i in ids])))
    test = held if held.any() else (split == "val")
    print(f"\nHeld-out dataset calls -- the false-alarm budget (accusing a real customer is the costly error)")
    print(f"{'term list':<10} {'length':<12} {'human false-flag':>17} {'synthetic detected':>19}   n(h)/n(s)")
    for label, terms in (("current", current), ("proposed", proposed), ("hybrid", hybrid)):
        for length, lmask in lengths:
            m = test & lmask & (source == "dataset")
            if m.sum() < 10:
                continue
            sc = np.array([score(terms, i) for i in np.flatnonzero(m)])
            yy = y[m]
            ok = np.isfinite(sc)
            fa = float((sc[ok & (yy == 0)] > 0.5).mean()) if (ok & (yy == 0)).any() else float("nan")
            det = float((sc[ok & (yy == 1)] > 0.5).mean()) if (ok & (yy == 1)).any() else float("nan")
            print(f"{label:<10} {length:<12} {fa:>16.1%} {det:>18.1%}   "
                  f"{int((ok & (yy == 0)).sum())}/{int((ok & (yy == 1)).sum())}")

    print(f"\nGenerated synthetic callers, full calls -- can the aspect catch a caller that answers fast?")
    print(f"{'term list':<10} {'bot':>8} {'realtime':>10} {'humanlike':>11}")
    for label, terms in (("current", current), ("proposed", proposed), ("hybrid", hybrid)):
        cells = []
        for pname in ("bot", "realtime", "humanlike"):
            m = full & (prof == pname)
            if m.sum() < 5:
                cells.append(float("nan"))
                continue
            sc = np.array([score(terms, i) for i in np.flatnonzero(m)])
            sc = sc[np.isfinite(sc)]
            cells.append(float((sc > 0.5).mean()))
        print(f"{label:<10} {cells[0]:>7.0%} {cells[1]:>9.0%} {cells[2]:>10.0%}")
    print("\nPick the list that catches the fast caller WITHOUT raising the false-flag rate on real calls.")

    print("\n" + "=" * 96)
    print("HYBRID, ready to paste into backend/scoring/heuristics.py as ASPECTS['turn_taking']['terms']:")
    print("=" * 96)
    for f, c, w, hi, wt, lb in hybrid:
        print(f'            ("{f}", {c:.3g}, {w:.3g}, {hi}, {wt}, "{lb}"),')


if __name__ == "__main__":
    main()
