"""End-to-end analysis: audio -> features -> probability -> verdict + explanation.

`extract()` is shared by training and serving so there is no train/serve skew.
`analyze()` is what the HTTP endpoints and the live WebSocket call.
"""
from __future__ import annotations

import logging
import math
import threading
import time

import numpy as np

from ..audio import Call
from ..config import settings
from ..features.acoustic import acoustic_features
from ..features.conversational import conversational_features
from ..vad import Vad, stft_power, vad_from_power, vad_from_segments
from . import heuristics
from .model import Detector, fuse_probabilities

log = logging.getLogger("detector.pipeline")


class Extraction:
    def __init__(self):
        self.features: dict = {}
        self.ui: dict = {}
        self.events: list = []
        self.timeline: dict = {}
        self.vad_c: Vad | None = None
        self.vad_a: Vad | None = None
        self.timing: dict = {}


def extract(call: Call, agent_segments: list | None = None, want_ui: bool = False) -> Extraction:
    """Compute every fast feature (acoustic + conversational). ~0.5-1 s for a 3 minute call."""
    ex = Extraction()
    t0 = time.time()
    P, freqs = stft_power(call.caller)
    vad_c = vad_from_power(P)
    vad_a = None
    if agent_segments is not None:
        vad_a = vad_from_segments(agent_segments, vad_c.n_frames)
    elif call.has_agent:
        Pa, _ = stft_power(call.agent)
        vad_a = vad_from_power(Pa)
    t1 = time.time()
    ac = acoustic_features(call, P, freqs, vad_c, vad_a, want_ui=want_ui)
    t2 = time.time()
    cv = conversational_features(vad_c, vad_a, call.duration)
    t3 = time.time()
    ex.features = {**ac.features, **cv.features}
    ex.ui = ac.ui
    ex.events = cv.events
    ex.timeline = cv.timeline
    ex.vad_c, ex.vad_a = vad_c, vad_a
    ex.timing = {"stft_vad_s": round(t1 - t0, 3), "acoustic_s": round(t2 - t1, 3), "conversational_s": round(t3 - t2, 3)}
    return ex


class Analyzer:
    """Holds the loaded model(s); thread-safe for concurrent requests."""

    def __init__(self, model_path: str | None = None):
        self.detector: Detector | None = Detector.load(model_path or settings.model_path)
        self.embedder = None
        self._emb_lock = threading.Lock()
        if settings.enable_embeddings:
            try:
                from ..features.embeddings import Embedder
                self.embedder = Embedder(settings.embedding_model, settings.embedding_layer)
            except Exception as exc:  # pragma: no cover
                log.warning("embeddings disabled: %s", exc)

    # ------------------------------------------------------------------ helpers
    @property
    def mode(self) -> str:
        return "trained_model" if self.detector else "heuristic"

    def warmup(self) -> None:
        rng = np.random.default_rng(0)
        x = (rng.standard_normal(8000 * 6) * 0.01).astype(np.float32)
        from ..audio import call_from_arrays
        self.analyze(call_from_arrays(x, x * 0.5), want_ui=False, allow_semantic=False)

    def _embedding_prob(self, call: Call, vad_c: Vad) -> float | None:
        if self.embedder is None or self.detector is None or self.detector.embedding_head is None:
            return None
        try:
            with self._emb_lock:
                emb = self.embedder.embed(call.caller, vad_c, settings.embedding_max_seconds)
            return self.detector.predict_embedding(emb)
        except Exception as exc:  # pragma: no cover
            log.warning("embedding inference failed: %s", exc)
            return None

    # ------------------------------------------------------------------ main entry
    def analyze(self, call: Call, agent_segments: list | None = None, agent_text_turns: list | None = None,
                want_ui: bool = True, allow_semantic: bool = True, force_semantic: bool = False) -> dict:
        t0 = time.time()
        ex = extract(call, agent_segments=agent_segments, want_ui=want_ui)
        feats = ex.features
        speech_s = float(feats.get("speech_seconds", 0.0) or 0.0)

        aspects = heuristics.aspect_scores(feats)
        p_heur = heuristics.heuristic_probability(aspects)
        contributions: dict = {}
        if self.detector is not None:
            p_fast = self.detector.predict_proba(feats)
            contributions = self.detector.contributions(feats)
        else:
            p_fast = p_heur
        p_embed = self._embedding_prob(call, ex.vad_c)
        weights = {"fast": settings.w_fast, "embed": settings.w_embed, "semantic": settings.w_semantic}
        fusion = self.detector.fusion if self.detector else None
        p = fuse_probabilities(p_fast, p_embed, None, weights, fusion)

        semantic: dict | None = None
        p_sem = None
        mode = settings.semantic_mode
        want_sem = force_semantic or (allow_semantic and (
            mode == "always" or (mode == "uncertain" and settings.semantic_low < p < settings.semantic_high)))
        if want_sem:
            from ..features.semantic import semantic_analysis
            semantic = semantic_analysis(call, ex.vad_c, ex.vad_a, agent_text_turns=agent_text_turns)
            if semantic.get("available"):
                feats.update(semantic.get("features", {}))
                aspects = heuristics.aspect_scores(feats)
                p_sem = semantic["features"].get("sem_synthetic_p")
                if p_sem is not None:
                    p = fuse_probabilities(p_fast, p_embed, p_sem, weights, fusion)

        # evidence shrink: not enough caller speech -> pull toward 0.5
        lo, hi = settings.min_speech_seconds, settings.full_confidence_speech_seconds
        evid = 0.0 if speech_s <= lo else min(1.0, (speech_s - lo) / max(hi - lo, 1e-6))
        p_final = 0.5 + (p - 0.5) * evid
        is_syn = bool(p_final >= settings.decision_threshold)
        confidence = float(max(p_final, 1.0 - p_final))
        attacks = heuristics.attack_profile(aspects, feats, p_final)

        out = {
            "is_synthetic": is_syn,
            "confidence": round(confidence, 4),
            "p_synthetic": round(float(p_final), 4),
            "verdict": "SYNTHETIC" if is_syn else "HUMAN",
            "evidence_level": round(evid, 3),
            "speech_seconds": round(speech_s, 2),
            "duration_seconds": round(call.duration, 2),
            "mode": self.mode,
            "signals": {
                "fast_model_p": round(float(p_fast), 4),
                "heuristic_p": round(float(p_heur), 4),
                "embedding_p": None if p_embed is None else round(float(p_embed), 4),
                "semantic_p": None if p_sem is None else round(float(p_sem), 4),
                "fused_p": round(float(p), 4),
            },
            "aspects": aspects,
            "attack_profile": attacks,
            "contributions": contributions,
            "events": ex.events,
            "timeline": ex.timeline,
            "features": {k: (None if (isinstance(v, float) and not math.isfinite(v)) else v) for k, v in feats.items()},
            "timing": {**ex.timing, "total_s": round(time.time() - t0, 3)},
            "input": {"sample_rate_in": call.sr_in, "channels_in": call.n_channels_in, "format": call.fmt,
                      "has_agent_channel": bool(ex.vad_a is not None)},
        }
        if semantic is not None:
            out["semantic"] = {k: v for k, v in semantic.items() if k != "features"}
        if want_ui:
            out["ui"] = ex.ui
        return out
