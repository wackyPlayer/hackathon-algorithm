"""Probability calibration wrappers shared by training and serving (must be importable from both,
otherwise the pickled detector cannot be loaded)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class Platt:
    """Logistic calibration on the logit; regularised so it cannot saturate to 0/1."""

    def __init__(self, C: float = 1.0):
        self.lr = LogisticRegression(C=C, max_iter=1000)

    @staticmethod
    def _z(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))[:, None]

    def fit(self, p, y):
        self.lr.fit(self._z(p), y)
        return self

    def transform(self, p):
        return self.lr.predict_proba(self._z(p))[:, 1]


class Identity:
    def fit(self, p, y):
        return self

    def transform(self, p):
        return np.asarray(p, dtype=float)


class CalibratedModel:
    """sklearn-like wrapper: base estimator + calibrator on its positive-class probability."""

    def __init__(self, base, calibrator):
        self.base, self.platt = base, calibrator

    def predict_proba(self, X):
        p = self.platt.transform(self.base.predict_proba(X)[:, 1])
        return np.stack([1 - p, p], axis=1)
