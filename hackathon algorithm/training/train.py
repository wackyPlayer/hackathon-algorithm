"""Train, select, calibrate and save the detector.

    python -m training.train --features data/features.csv --out models/detector.joblib [--embeddings data/embeddings.npz]

Protocol (speaker-disjoint everywhere):
  1. model selection with GroupKFold(5) on the `train` split, groups = anon_id (clipped variants of a
     call stay in the same fold);
  2. honest evaluation on the untouched `val` split (full calls; clipped variants reported separately);
  3. Platt calibration fitted on out-of-fold train predictions;
  4. final refit on train+val for deployment (val metrics are kept in the model's meta as the estimate).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

META_COLS = {"id", "anon_id", "label", "split", "duration_s"}
# features whose scale depends on clip length rather than on the caller -> excluded by default
COUNT_LIKE = ("_count", "_n", "cut_hard_count", "rhythm_mod_windows", "text_words", "speech_seconds", "duration_s",
              "breath_count", "conv_caller_turn_count", "conv_dead_fill_latency_n")


def load_table(path: str):
    with open(path, encoding="utf-8") as fh:
        r = csv.reader(fh)
        header = next(r)
        rows = list(r)
    cols = {c: i for i, c in enumerate(header)}
    feat_names = [c for c in header if c not in META_COLS]
    X = np.full((len(rows), len(feat_names)), np.nan)
    for i, row in enumerate(rows):
        for j, c in enumerate(feat_names):
            v = row[cols[c]]
            if v != "":
                X[i, j] = float(v)
    ids = np.array([row[cols["id"]] for row in rows])
    groups = np.array([row[cols["anon_id"]] for row in rows])
    y = np.array([1 if row[cols["label"]] == "synthetic" else 0 for row in rows])
    split = np.array([row[cols["split"]] for row in rows])
    return X, y, groups, split, ids, feat_names


def eer(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def metrics(y, p, thr=0.5):
    y, p = np.asarray(y), np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    pred = (p >= thr).astype(int)
    out = {"n": int(len(y)), "acc": float((pred == y).mean()), "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
           "eer": eer(y, p) if len(set(y)) > 1 else float("nan"), "brier": float(brier_score_loss(y, p)),
           "logloss": float(log_loss(y, p, labels=[0, 1])),
           "tp": int(((pred == 1) & (y == 1)).sum()), "fn": int(((pred == 0) & (y == 1)).sum()),
           "fp": int(((pred == 1) & (y == 0)).sum()), "tn": int(((pred == 0) & (y == 0)).sum())}
    return out


def make_candidates():
    c = {}
    for C in (0.03, 0.1, 0.3, 1.0):
        c[f"lr_C{C}"] = Pipeline([("scaler", StandardScaler()),
                                  ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=5000))])
    for depth, lr in ((3, 0.05), (4, 0.05), (3, 0.1)):
        c[f"hgb_d{depth}_lr{lr}"] = HistGradientBoostingClassifier(max_depth=depth, learning_rate=lr, max_iter=300,
                                                                    l2_regularization=1.0, min_samples_leaf=10,
                                                                    class_weight="balanced", random_state=0)
    return c


def oof_predict(model_factory, X, y, groups, n_splits=5):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = model_factory()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


from backend.scoring.calibration import CalibratedModel, Identity, Platt  # noqa: E402  (shared with serving)


def choose_calibrator(p_oof, y, seed=0):
    """Cross-fitted (2-fold) log-loss decides whether Platt scaling helps the OOF probabilities."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    halves = [idx[: len(y) // 2], idx[len(y) // 2:]]
    raw = float(log_loss(y, np.clip(p_oof, 1e-6, 1 - 1e-6), labels=[0, 1]))
    cal = 0.0
    for a, b in (halves, halves[::-1]):
        pl = Platt().fit(p_oof[a], y[a])
        cal += float(log_loss(y[b], np.clip(pl.transform(p_oof[b]), 1e-6, 1 - 1e-6), labels=[0, 1])) / 2
    print(f"calibration check: raw OOF logloss={raw:.4f}  platt(cross-fit)={cal:.4f} -> {'platt' if cal < raw else 'raw'}")
    return (Platt() if cal < raw else Identity()), {"raw_logloss": raw, "platt_logloss": cal}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features.csv")
    ap.add_argument("--out", default="models/detector.joblib")
    ap.add_argument("--embeddings", default="")
    ap.add_argument("--keep-counts", action="store_true")
    ap.add_argument("--max-missing", type=float, default=0.4)
    a = ap.parse_args()
    import joblib

    X, y, groups, split, ids, names = load_table(a.features)
    full = np.array(["#" not in i for i in ids])
    keep = []
    tr_mask = split == "train"
    for j, n in enumerate(names):
        if not a.keep_counts and any(t in n for t in COUNT_LIKE):
            continue
        col = X[tr_mask, j]
        if np.isnan(col).mean() > a.max_missing or np.nanstd(col) < 1e-9:
            continue
        keep.append(j)
    names = [names[j] for j in keep]
    X = X[:, keep]
    medians = np.nanmedian(X[tr_mask], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    Xi = np.where(np.isnan(X), medians, X)
    print(f"{len(y)} rows ({full.sum()} full calls), {len(names)} features; train={tr_mask.sum()} val={(~tr_mask).sum()}")

    Xtr, ytr, gtr = Xi[tr_mask], y[tr_mask], groups[tr_mask]
    Xva, yva, fva = Xi[~tr_mask], y[~tr_mask], full[~tr_mask]
    t0 = time.time()
    results = {}
    oofs = {}
    for name, proto in make_candidates().items():
        from sklearn.base import clone
        oof = oof_predict(lambda: clone(proto), Xtr, ytr, gtr)
        oofs[name] = oof
        m = metrics(ytr, oof)
        results[name] = m
        print(f"  {name:14s} oof-auc={m['auc']:.4f} acc={m['acc']:.3f} eer={m['eer']:.3f} ({time.time() - t0:.0f}s)")
    best = max(results, key=lambda k: results[k]["auc"])
    print("best by OOF AUC:", best)

    from sklearn.base import clone
    proto = make_candidates()[best]
    calib, calinfo = choose_calibrator(oofs[best], ytr)
    platt = type(calib)().fit(oofs[best], ytr)
    base = clone(proto).fit(Xtr, ytr)
    cal = CalibratedModel(base, platt)
    p_val = cal.predict_proba(Xva)[:, 1]
    p_raw = base.predict_proba(Xva)[:, 1]
    # The train OOF predictions are near-separable, which makes Platt too steep for unseen speakers.
    # The speaker-disjoint val split is the honest reference: keep Platt only if it lowers val log-loss.
    fv = full[~tr_mask]
    ll_cal = log_loss(yva[fv], np.clip(p_val[fv], 1e-6, 1 - 1e-6), labels=[0, 1])
    ll_raw = log_loss(yva[fv], np.clip(p_raw[fv], 1e-6, 1 - 1e-6), labels=[0, 1])
    if ll_raw <= ll_cal:
        print(f"val logloss raw={ll_raw:.4f} < platt={ll_cal:.4f}: deploying uncalibrated probabilities")
        calib = Identity()
        calinfo["val_logloss_raw"], calinfo["val_logloss_platt"] = float(ll_raw), float(ll_cal)
        platt = Identity()
        p_val = p_raw
    else:
        print(f"val logloss platt={ll_cal:.4f} < raw={ll_raw:.4f}: keeping Platt calibration")
        calinfo["val_logloss_raw"], calinfo["val_logloss_platt"] = float(ll_raw), float(ll_cal)
    rep = {
        "best": best,
        "oof_train": results[best],
        "val_full_calls": metrics(yva[fva], p_val[fva]),
        "val_full_calls_uncalibrated": metrics(yva[fva], p_raw[fva]),
        "val_clips": metrics(yva[~fva], p_val[~fva]) if (~fva).any() else None,
        "all_candidates_oof": results,
        "n_features": len(names),
    }
    for clip in ("#clip30", "#clip60"):
        m = np.array([i.endswith(clip) for i in ids[~tr_mask]])
        if m.any():
            rep[f"val_{clip[1:]}"] = metrics(yva[m], p_val[m])
    print(json.dumps({k: v for k, v in rep.items() if k != "all_candidates_oof"}, indent=1))

    # deployment fit on train + val
    base_all = clone(proto).fit(Xi, y)
    platt_all = type(calib)().fit(np.concatenate([oofs[best], p_raw]), np.concatenate([ytr, yva]))
    final = CalibratedModel(base_all, platt_all)
    rep["calibration"] = {**calinfo, "used": type(calib).__name__}
    payload = {
        "kind": "lr" if best.startswith("lr") else "hgb",
        "feature_names": names,
        "medians": np.nanmedian(X, axis=0),
        "clf": final,
        "base_lr": None,
        "scaler": None,
        "meta": {"best": best, "val_auc": rep["val_full_calls"]["auc"], "val_acc": rep["val_full_calls"]["acc"],
                 "val_eer": rep["val_full_calls"]["eer"], "val_brier": rep["val_full_calls"]["brier"],
                 "oof_auc": results[best]["auc"], "n_train": int(len(y)), "n_features": len(names),
                 "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")},
        "embedding_head": None,
        "fusion": None,
    }
    payload["medians"] = np.where(np.isfinite(payload["medians"]), payload["medians"], 0.0)
    if best.startswith("lr"):
        payload["base_lr"] = base_all.named_steps["clf"]
        payload["scaler"] = base_all.named_steps["scaler"]
        coef = payload["base_lr"].coef_[0]
        order = np.argsort(-np.abs(coef))[:25]
        rep["top_coefficients"] = [{"feature": names[i], "coef": float(coef[i])} for i in order]
        print("top |coef| features:")
        for r in rep["top_coefficients"][:15]:
            print(f"   {r['coef']:+.3f}  {r['feature']}")

    # optional embedding head + stacker
    if a.embeddings and os.path.exists(a.embeddings):
        z = np.load(a.embeddings, allow_pickle=True)
        eid = {i: k for k, i in enumerate(z["ids"])}
        E = z["X"]
        rows_full = [k for k, i in enumerate(ids) if "#" not in i and i in eid]
        Xe = np.stack([E[eid[ids[k]]] for k in rows_full])
        ye, ge, se = y[rows_full], groups[rows_full], split[rows_full]
        head_proto = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000))])
        trm = se == "train"
        oof_e = oof_predict(lambda: clone(head_proto), Xe[trm], ye[trm], ge[trm])
        head = clone(head_proto).fit(Xe[trm], ye[trm])
        pe_val = head.predict_proba(Xe[~trm])[:, 1]
        rep["embedding_val"] = metrics(ye[~trm], pe_val)
        print("embedding head val:", rep["embedding_val"])
        # stacker on OOF logits (train) : fast + embed
        fast_oof = {ids[k]: v for k, v in zip(np.flatnonzero(tr_mask), oofs[best])}
        zf = np.array([fast_oof[ids[k]] for k in np.array(rows_full)[trm]])
        L = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        S = np.stack([L(zf), L(oof_e)], axis=1)
        st = LogisticRegression(C=1.0, max_iter=1000).fit(S, ye[trm])
        payload["embedding_head"] = {"clf": clone(head_proto).fit(Xe, ye), "dim": int(Xe.shape[1]),
                                     "model_name": str(z["model"]), "layer": int(z["layer"])}
        payload["fusion"] = {"weights": {"fast": float(st.coef_[0][0]), "embed": float(st.coef_[0][1]), "semantic": 0.8},
                             "bias": float(st.intercept_[0])}
        print("fusion:", payload["fusion"])

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    joblib.dump(payload, a.out)
    with open(os.path.join(os.path.dirname(os.path.abspath(a.out)), "train_report.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
