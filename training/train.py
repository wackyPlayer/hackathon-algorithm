"""Train, select, calibrate and save the detector.

    python -m training.train --features data/features.csv --out models/detector.joblib [--embeddings data/embeddings.npz]
    python -m training.train ... --acoustic-head --fusion max      # add the acoustic-only head (TTS texture)

Protocol (speaker-disjoint everywhere):
  1. model selection with GroupKFold(5) on the `train` split, groups = the `group` column (anon_id for dataset
     calls, engine:voice for TTS-corpus calls; clipped / channel-augmented variants stay in the same fold);
  2. honest evaluation on the untouched `val` split: dataset calls (full and clipped) and, per extra source
     (tts = synthetic calls from other TTS engines with unseen voices, aug = channel-augmented dataset calls);
  3. Platt calibration fitted on out-of-fold train predictions, kept only if it lowers val log-loss;
  4. optional acoustic-only head (no conversational / cross-channel features) whose job is to catch a
     synthetic *voice* even when the caller's timing looks human; fused with the main model in logit space
     (`--fusion none|max|mean|stack`, chosen from the val numbers printed by this script);
  5. final refit on train+val for deployment (val metrics are kept in the model's meta as the estimate).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.scoring.calibration import CalibratedModel, Identity, Platt  # noqa: E402  (shared with serving)
from backend.scoring.model import fuse_heads  # noqa: E402

META_COLS = {"id", "anon_id", "label", "split", "duration_s", "group", "source"}
# features whose scale depends on clip length rather than on the caller -> excluded by default
COUNT_LIKE = ("_count", "_n", "cut_hard_count", "rhythm_mod_windows", "text_words", "speech_seconds", "duration_s",
              "breath_count", "conv_caller_turn_count", "conv_dead_fill_latency_n")
ACOUSTIC_EXCLUDE = ("conv_", "chan_", "text_", "sem_")


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
    anon = np.array([row[cols["anon_id"]] for row in rows])
    groups = np.array([row[cols["group"]] or row[cols["anon_id"]] for row in rows]) if "group" in cols else anon
    source = np.array([row[cols["source"]] or "dataset" for row in rows]) if "source" in cols else np.array(["dataset"] * len(rows))
    y = np.array([1 if row[cols["label"]] == "synthetic" else 0 for row in rows])
    split = np.array([row[cols["split"]] for row in rows])
    return X, y, groups, split, ids, feat_names, source


def eer(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def metrics(y, p, thr=0.5):
    y, p = np.asarray(y), np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    if len(y) == 0:
        return None
    pred = (p >= thr).astype(int)
    out = {"n": int(len(y)), "acc": float((pred == y).mean()), "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
           "eer": eer(y, p) if len(set(y)) > 1 else float("nan"), "brier": float(brier_score_loss(y, p)),
           "logloss": float(log_loss(y, p, labels=[0, 1])),
           "tp": int(((pred == 1) & (y == 1)).sum()), "fn": int(((pred == 0) & (y == 1)).sum()),
           "fp": int(((pred == 1) & (y == 0)).sum()), "tn": int(((pred == 0) & (y == 0)).sum())}
    return out


def fmt(m):
    if not m:
        return "n/a"
    return (f"n={m['n']:4d} acc={m['acc']:.3f} auc={m['auc']:.4f} eer={m['eer']:.3f} logloss={m['logloss']:.3f} "
            f"tp={m['tp']} fn={m['fn']} fp={m['fp']} tn={m['tn']}")


def make_candidates(only_lr: bool = False):
    c = {}
    for C in (0.03, 0.1, 0.3, 1.0):
        c[f"lr_C{C}"] = Pipeline([("scaler", StandardScaler()),
                                  ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=5000))])
    if only_lr:
        return c
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


def per_source(y, p, source, full, ids, prefix, rep):
    """Val metrics: dataset full calls / clips (legacy keys) + every extra source."""
    ds = source == "dataset"
    rep[f"{prefix}full_calls"] = metrics(y[ds & full], p[ds & full])
    rep[f"{prefix}clips"] = metrics(y[ds & ~full], p[ds & ~full]) if (ds & ~full).any() else None
    for clip in ("#clip30", "#clip60"):
        m = ds & np.array([i.endswith(clip) for i in ids])
        if m.any():
            rep[f"{prefix}{clip[1:]}"] = metrics(y[m], p[m])
    for s in sorted(set(source.tolist()) - {"dataset"}):
        m = source == s
        rep[f"{prefix}{s}_full"] = metrics(y[m & full], p[m & full]) if (m & full).any() else None
        rep[f"{prefix}{s}_clips"] = metrics(y[m & ~full], p[m & ~full]) if (m & ~full).any() else None
    both = np.ones(len(y), bool)
    rep[f"{prefix}all_full"] = metrics(y[both & full], p[both & full])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features.csv")
    ap.add_argument("--out", default="models/detector.joblib")
    ap.add_argument("--embeddings", default="")
    ap.add_argument("--keep-counts", action="store_true")
    ap.add_argument("--max-missing", type=float, default=0.4)
    ap.add_argument("--exclude-source", action="append", default=[], help="drop these sources from training (evaluation keeps them)")
    ap.add_argument("--acoustic-head", action="store_true", help="also train an acoustic-only head (no conv_/chan_ features)")
    ap.add_argument("--fusion", default="none", choices=["none", "max", "mean", "stack"], help="how the acoustic head joins the main model")
    ap.add_argument("--head-C", type=float, default=0.03)
    ap.add_argument("--force", default="", help="force a candidate name (e.g. lr_C0.03) instead of the best OOF AUC")
    ap.add_argument("--only-lr", action="store_true", help="skip the (slow) gradient-boosting candidates")
    ap.add_argument("--exclude-group-prefix", action="append", default=[],
                    help="hold out every call whose group starts with this prefix (e.g. elevenlabs:) from training and report it as source 'heldout'")
    a = ap.parse_args()
    import joblib

    X, y, groups, split, ids, names, source = load_table(a.features)
    full = np.array(["#" not in i for i in ids])
    if a.exclude_group_prefix:
        held = np.array([g.startswith(tuple(a.exclude_group_prefix)) for g in groups])
        source = np.where(held, "heldout", source)
        split = np.where(held, "val", split)
        print(f"holding out {int(held.sum())} rows ({', '.join(a.exclude_group_prefix)}) from training -> reported as 'heldout'")
    keep = []
    tr_mask = (split == "train") & ~np.isin(source, a.exclude_source)
    ds_tr = tr_mask & (source == "dataset")
    for j, n in enumerate(names):
        if not a.keep_counts and any(t in n for t in COUNT_LIKE):
            continue
        col = X[ds_tr, j]
        if np.isnan(col).mean() > a.max_missing or np.nanstd(col) < 1e-9:
            continue
        keep.append(j)
    names = [names[j] for j in keep]
    X = X[:, keep]
    medians = np.nanmedian(X[tr_mask], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    Xi = np.where(np.isnan(X), medians, X)
    va_mask = split == "val"
    print(f"{len(y)} rows ({full.sum()} full calls), {len(names)} features; train={tr_mask.sum()} val={va_mask.sum()}; "
          f"sources: " + ", ".join(f"{s}={int((source == s).sum())}" for s in sorted(set(source.tolist()))))

    Xtr, ytr, gtr = Xi[tr_mask], y[tr_mask], groups[tr_mask]
    Xva, yva, fva, sva, iva = Xi[va_mask], y[va_mask], full[va_mask], source[va_mask], ids[va_mask]
    t0 = time.time()
    results, oofs = {}, {}
    for name, proto in make_candidates(a.only_lr).items():
        oof = oof_predict(lambda: clone(proto), Xtr, ytr, gtr)
        oofs[name] = oof
        m = metrics(ytr, oof)
        results[name] = m
        print(f"  {name:14s} oof-auc={m['auc']:.4f} acc={m['acc']:.3f} eer={m['eer']:.3f} ({time.time() - t0:.0f}s)")
    best = a.force or max(results, key=lambda k: results[k]["auc"])
    print("selected:", best)

    proto = make_candidates(a.only_lr)[best]
    calib, calinfo = choose_calibrator(oofs[best], ytr)
    platt = type(calib)().fit(oofs[best], ytr)
    base = clone(proto).fit(Xtr, ytr)
    cal = CalibratedModel(base, platt)
    p_val = cal.predict_proba(Xva)[:, 1]
    p_raw = base.predict_proba(Xva)[:, 1]
    # The train OOF predictions are near-separable, which makes Platt too steep for unseen speakers.
    # The speaker-disjoint val split is the honest reference: keep Platt only if it lowers val log-loss.
    dsf = fva & (sva == "dataset")
    ll_cal = log_loss(yva[dsf], np.clip(p_val[dsf], 1e-6, 1 - 1e-6), labels=[0, 1])
    ll_raw = log_loss(yva[dsf], np.clip(p_raw[dsf], 1e-6, 1 - 1e-6), labels=[0, 1])
    if ll_raw <= ll_cal:
        print(f"val logloss raw={ll_raw:.4f} <= platt={ll_cal:.4f}: deploying uncalibrated probabilities")
        calib, platt, p_val = Identity(), Identity(), p_raw
    else:
        print(f"val logloss platt={ll_cal:.4f} < raw={ll_raw:.4f}: keeping Platt calibration")
    calinfo["val_logloss_raw"], calinfo["val_logloss_platt"] = float(ll_raw), float(ll_cal)

    rep: dict = {"best": best, "oof_train": results[best], "all_candidates_oof": results, "n_features": len(names),
                 "train_sources": {s: int((source[tr_mask] == s).sum()) for s in sorted(set(source[tr_mask].tolist()))}}
    per_source(yva, p_val, sva, fva, iva, "val_", rep)
    print("main model, val:")
    for k in sorted(rep):
        if k.startswith("val_"):
            print(f"  {k:22s} {fmt(rep[k])}")

    # ------------------------------------------------------------------ acoustic-only head
    head_payload = None
    fusion_rule = None
    if a.acoustic_head:
        ac_idx = [j for j, n in enumerate(names) if not n.startswith(ACOUSTIC_EXCLUDE)]
        ac_names = [names[j] for j in ac_idx]
        head_proto = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=a.head_C, class_weight="balanced", max_iter=5000))])
        oof_ac = oof_predict(lambda: clone(head_proto), Xtr[:, ac_idx], ytr, gtr)
        head = clone(head_proto).fit(Xtr[:, ac_idx], ytr)
        p_ac = head.predict_proba(Xva[:, ac_idx])[:, 1]
        rep["acoustic_head"] = {"n_features": len(ac_names), "oof_train": metrics(ytr, oof_ac)}
        per_source(yva, p_ac, sva, fva, iva, "val_", rep["acoustic_head"])
        print(f"acoustic head ({len(ac_names)} features), val:")
        for k in sorted(rep["acoustic_head"]):
            if k.startswith("val_"):
                print(f"  {k:22s} {fmt(rep['acoustic_head'][k])}")
        # stacker on OOF logits (train only)
        L = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        S = np.stack([L(oofs[best]), L(oof_ac)], axis=1)
        st = LogisticRegression(C=0.3, max_iter=1000).fit(S, ytr)
        stack = {"w_main": float(st.coef_[0][0]), "w_ac": float(st.coef_[0][1]), "bias": float(st.intercept_[0])}
        rep["fusion_candidates"] = {}
        print("fusion rules, val (main model + acoustic head):")
        for rule in ("max", "mean", "stack"):
            r = {"kind": rule, **(stack if rule == "stack" else {})}
            pf = np.array([fuse_heads(pm, pa, r) for pm, pa in zip(p_val, p_ac)])
            sub: dict = {}
            per_source(yva, pf, sva, fva, iva, "val_", sub)
            rep["fusion_candidates"][rule] = sub
            for k in sorted(sub):
                print(f"  {rule:6s} {k:22s} {fmt(sub[k])}")
        if a.fusion != "none":
            fusion_rule = {"kind": a.fusion, **(stack if a.fusion == "stack" else {})}
            pf = np.array([fuse_heads(pm, pa, fusion_rule) for pm, pa in zip(p_val, p_ac)])
            rep["deployed_fusion"] = {}
            per_source(yva, pf, sva, fva, iva, "val_", rep["deployed_fusion"])
        head_payload = {"clf": clone(head_proto).fit(Xi[tr_mask | va_mask][:, ac_idx], y[tr_mask | va_mask]),
                        "feature_names": ac_names, "medians": medians[ac_idx], "C": a.head_C,
                        "val": {k: v for k, v in rep["acoustic_head"].items() if k.startswith("val_")}}

    # ------------------------------------------------------------------ deployment fit on train + val
    dep = tr_mask | va_mask
    base_all = clone(proto).fit(Xi[dep], y[dep])
    platt_all = type(calib)().fit(np.concatenate([oofs[best], p_raw]), np.concatenate([ytr, yva]))
    final = CalibratedModel(base_all, platt_all)
    rep["calibration"] = {**calinfo, "used": type(calib).__name__}
    vf = rep["val_full_calls"]
    payload = {
        "kind": "lr" if best.startswith("lr") else "hgb",
        "feature_names": names,
        "medians": np.where(np.isfinite(np.nanmedian(X[dep], axis=0)), np.nanmedian(X[dep], axis=0), 0.0),
        "clf": final,
        "base_lr": None,
        "scaler": None,
        "meta": {"best": best, "val_auc": vf["auc"], "val_acc": vf["acc"], "val_eer": vf["eer"], "val_brier": vf["brier"],
                 "val_tts_detect": (rep.get("val_tts_full") or {}).get("acc"),
                 "oof_auc": results[best]["auc"], "n_train": int(dep.sum()), "n_features": len(names),
                 "acoustic_head": bool(head_payload), "fusion": (fusion_rule or {}).get("kind", "none"),
                 "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")},
        "embedding_head": None,
        "fusion": None,
        "acoustic_head": head_payload,
        "fusion_rule": fusion_rule,
    }
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
    with open(os.path.splitext(os.path.abspath(a.out))[0].replace("detector", "train_report") + ".json"
              if os.path.basename(a.out).startswith("detector") else os.path.join(os.path.dirname(os.path.abspath(a.out)), "train_report.json"),
              "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
