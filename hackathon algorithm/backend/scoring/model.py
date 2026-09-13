"""Trained detector wrapper (scikit-learn), saved with joblib by training/train.py.

Saved payload (dict):
  kind            "lr" | "hgb"
  feature_names   list[str]   (order of the feature vector)
  medians         np.ndarray  (imputation values for missing features)
  clf             fitted sklearn Pipeline (scaler + classifier), possibly calibrated
  base_lr         fitted LogisticRegression on standardized features (for attributions) or None
  scaler          fitted StandardScaler matching base_lr, or None
  meta            dict of training metrics / notes
  embedding_head  optional dict {clf, dim, model_name, layer} for SSL embeddings
  fusion          optional dict {"weights": {...}, "bias": float} learned on out-of-fold logits
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("detector.model")

GROUP_PREFIXES = [
    ("conv_", "conversational"), ("f0_", "prosody"), ("hnr_", "prosody"), ("shimmer", "prosody"),
    ("energy_", "prosody"), ("level_", "level"), ("rhythm_", "rhythm"), ("ltas_", "spectrum"),
    ("spec_", "spectrum"), ("lfcc_", "cepstral"), ("mfcc_", "cepstral"), ("floor_", "noise_floor"),
    ("snr_", "noise_floor"), ("cut_", "cuts"), ("breath_", "breathing"), ("formant_", "formants"),
    ("hum_", "hum"), ("chan_", "cross_channel"), ("text_", "text"), ("sem_", "semantic"),
]


def feature_group(name: str) -> str:
    for p, g in GROUP_PREFIXES:
        if name.startswith(p):
            return g
    return "other"


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


@dataclass
class Detector:
    kind: str
    feature_names: list
    medians: np.ndarray
    clf: object
    base_lr: object = None
    scaler: object = None
    meta: dict = field(default_factory=dict)
    embedding_head: dict | None = None
    fusion: dict | None = None
    path: str = ""

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str) -> "Detector | None":
        if not path or not os.path.exists(path):
            return None
        try:
            import joblib
            d = joblib.load(path)
            det = cls(kind=d["kind"], feature_names=list(d["feature_names"]), medians=np.asarray(d["medians"]),
                      clf=d["clf"], base_lr=d.get("base_lr"), scaler=d.get("scaler"), meta=d.get("meta", {}),
                      embedding_head=d.get("embedding_head"), fusion=d.get("fusion"), path=path)
            log.info("loaded detector %s (%s, %d features) meta=%s", path, det.kind, len(det.feature_names),
                     {k: v for k, v in det.meta.items() if not isinstance(v, (list, dict))})
            return det
        except Exception as exc:  # pragma: no cover
            log.exception("failed to load detector from %s: %s", path, exc)
            return None

    # ------------------------------------------------------------------ inference
    def vector(self, features: dict) -> np.ndarray:
        v = np.array([features.get(n, np.nan) for n in self.feature_names], dtype=np.float64)
        bad = ~np.isfinite(v)
        v[bad] = self.medians[bad]
        return v

    def predict_proba(self, features: dict) -> float:
        v = self.vector(features)[None, :]
        return float(self.clf.predict_proba(v)[0, 1])

    def contributions(self, features: dict, top: int = 12) -> dict:
        """Per-feature-group logit contributions (LR only) and the top individual features."""
        if self.base_lr is None or self.scaler is None:
            return {}
        v = self.vector(features)[None, :]
        z = self.scaler.transform(v)[0]
        coef = np.asarray(self.base_lr.coef_)[0]
        c = coef * z
        groups: dict = {}
        for name, val in zip(self.feature_names, c):
            groups[feature_group(name)] = groups.get(feature_group(name), 0.0) + float(val)
        order = np.argsort(-np.abs(c))[:top]
        feats = [{"feature": self.feature_names[i], "value": float(v[0, i]), "contribution": float(c[i])} for i in order]
        return {"groups": groups, "top_features": feats, "intercept": float(self.base_lr.intercept_[0])}

    def predict_embedding(self, emb: np.ndarray) -> float | None:
        if self.embedding_head is None or emb is None:
            return None
        try:
            return float(self.embedding_head["clf"].predict_proba(np.asarray(emb, dtype=np.float64)[None, :])[0, 1])
        except Exception as exc:  # pragma: no cover
            log.warning("embedding head failed: %s", exc)
            return None

    def fuse(self, p_fast: float, p_embed: float | None, p_sem: float | None, defaults: dict) -> float:
        """Combine available probabilities in logit space with learned (or default) weights."""
        return fuse_probabilities(p_fast, p_embed, p_sem, defaults, self.fusion)


def fuse_probabilities(p_fast: float, p_embed: float | None, p_sem: float | None,
                       defaults: dict, fusion: dict | None = None) -> float:
    """Logit-space fusion. With a learned stacker: z = b + sum(w_i * logit(p_i)) over available
    signals. Without one: weighted mean of the available logits (a single signal keeps its scale)."""
    avail = [(k, p) for k, p in (("fast", p_fast), ("embed", p_embed), ("semantic", p_sem))
             if p is not None and np.isfinite(p)]
    if not avail:
        return 0.5
    if fusion:
        w = fusion.get("weights", {})
        z = float(fusion.get("bias", 0.0)) + sum(w.get(k, 1.0) * _logit(p) for k, p in avail)
        return _sigmoid(z)
    num = sum(defaults.get(k, 1.0) * _logit(p) for k, p in avail)
    den = sum(defaults.get(k, 1.0) for k, _ in avail) or 1.0
    return _sigmoid(num / den)
