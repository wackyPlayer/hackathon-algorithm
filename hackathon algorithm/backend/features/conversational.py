"""Conversational-behaviour features from the two-channel turn timeline.

The agent (channel 1) interrupts, falls silent and talks over the caller. Humans recover from
those moments instantly and messily; a speech-recognition -> LLM -> TTS pipeline recovers with
machine-like consistency and a long, stable response latency. Everything here is derived from
the VAD of both channels only (no transcription), so it costs microseconds.

  conv_resp_*       response latency after each agent turn (mean/std/cv/median/min/p90)
  conv_int_*        agent interruptions: how fast the caller yields, restarts, or talks through
  conv_cint_*       caller interruptions and back-channels ("aja", "si") during agent speech
  conv_dead_*       dead-air windows: who fills the silence and after how long
  conv_turn_*       caller turn duration statistics, short-turn fraction, speech fractions
  conv_pause_*      pauses inside caller turns
  conv_false_start  short false starts followed by a proper turn
  conv_regularity   aggregate regularity index (1 = perfectly regular)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..vad import HOP_S, Vad


@dataclass
class ConvResult:
    features: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    timeline: dict = field(default_factory=dict)


def _safe(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _stats(prefix: str, vals, out: dict, min_n: int = 1) -> None:
    a = np.asarray(vals, dtype=np.float64)
    out[f"{prefix}_n"] = float(a.size)
    if a.size < min_n or a.size == 0:
        return
    out[f"{prefix}_mean"] = _safe(a.mean())
    out[f"{prefix}_median"] = _safe(np.median(a))
    out[f"{prefix}_min"] = _safe(a.min())
    out[f"{prefix}_max"] = _safe(a.max())
    if a.size >= 3:
        out[f"{prefix}_std"] = _safe(a.std())
        out[f"{prefix}_cv"] = _safe(a.std() / abs(a.mean())) if abs(a.mean()) > 1e-6 else float("nan")
        out[f"{prefix}_p90"] = _safe(np.percentile(a, 90))
        out[f"{prefix}_iqr"] = _safe(np.percentile(a, 75) - np.percentile(a, 25))


def _active_at(mask: np.ndarray, t: float) -> bool:
    i = int(t / HOP_S)
    return bool(0 <= i < mask.size and mask[i])


def conversational_features(vad_c: Vad, vad_a: Vad | None, duration: float) -> ConvResult:
    out: dict = {}
    events: list = []
    c_phr = [p for p in vad_c.phrases if p[1] - p[0] >= 0.12]
    c_turns = vad_c.turns
    out["conv_caller_speech_frac"] = _safe(vad_c.speech_seconds / max(duration, 1e-3))
    out["conv_caller_turn_count"] = float(len(c_turns))
    if c_turns:
        d = np.array([e - s for s, e in c_turns])
        _stats("conv_turn_dur", d, out)
        out["conv_turn_short_frac"] = _safe((d < 0.6).mean())
        out["conv_turn_long_frac"] = _safe((d > 6.0).mean())
    # pauses inside caller turns
    pauses = []
    for s, e in c_turns:
        inside = [p for p in c_phr if p[0] >= s - 1e-6 and p[1] <= e + 1e-6]
        for a, b in zip(inside[:-1], inside[1:]):
            g = b[0] - a[1]
            if 0.05 <= g <= 0.7:
                pauses.append(g)
    _stats("conv_pause", pauses, out)
    out["conv_pause_rate"] = _safe(len(pauses) / max(vad_c.speech_seconds, 1e-3))
    # false starts: short phrase, gap, then a proper phrase
    fs = 0
    for i in range(len(c_phr) - 1):
        a, b = c_phr[i], c_phr[i + 1]
        if (a[1] - a[0]) < 0.4 and 0.2 <= (b[0] - a[1]) <= 1.0 and (b[1] - b[0]) >= 0.8:
            if vad_a is None or not _active_at(vad_a.speech, a[0]):
                fs += 1
                events.append({"type": "false_start", "t": round(a[0], 2)})
    out["conv_false_start_count"] = float(fs)
    out["conv_false_start_rate"] = _safe(fs / max(len(c_turns), 1))

    if vad_a is None or not vad_a.turns:
        out["conv_has_agent"] = 0.0
        return ConvResult(features=out, events=events, timeline=_timeline(vad_c, vad_a))
    out["conv_has_agent"] = 1.0
    a_turns = vad_a.turns
    a_mask, c_mask = vad_a.speech, vad_c.speech
    n = min(a_mask.size, c_mask.size)
    out["conv_agent_speech_frac"] = _safe(vad_a.speech_seconds / max(duration, 1e-3))
    both = a_mask[:n] & c_mask[:n]
    out["conv_overlap_frac_of_caller"] = _safe(both.sum() / max(c_mask[:n].sum(), 1))
    out["conv_overlap_frac_of_agent"] = _safe(both.sum() / max(a_mask[:n].sum(), 1))

    # --- response latency after each agent turn -------------------------------------------
    lat = []
    for i, (s, e) in enumerate(a_turns):
        nxt = a_turns[i + 1][0] if i + 1 < len(a_turns) else duration + 10
        if _active_at(c_mask, e - HOP_S):     # caller already talking when agent stops: overlap, not a response
            continue
        cand = [p for p in c_phr if p[0] >= e - 0.3 and p[0] < min(nxt, e + 8.0)]
        if cand:
            L = cand[0][0] - e
            lat.append(L)
            events.append({"type": "response", "t": round(cand[0][0], 2), "latency": round(L, 2)})
    _stats("conv_resp", lat, out)
    if lat:
        out["conv_resp_first"] = _safe(lat[0])
        out["conv_resp_frac_over_1s"] = _safe(np.mean(np.array(lat) > 1.0))
        out["conv_resp_frac_over_2s"] = _safe(np.mean(np.array(lat) > 2.0))
        out["conv_resp_frac_under_300ms"] = _safe(np.mean(np.array(lat) < 0.3))
    out["conv_resp_missing_frac"] = _safe(1 - len(lat) / max(len(a_turns), 1))

    # --- agent interruptions: agent starts while the caller is speaking ------------------------
    yields, restarts, through = [], 0, 0
    for s, e in a_turns:
        if not _active_at(c_mask, s):
            continue
        # caller phrase containing s
        cur = [p for p in c_phr if p[0] <= s <= p[1]]
        if not cur:
            continue
        p_end = cur[0][1]
        y = min(p_end - s, 5.0)
        yields.append(y)
        if y >= min(e - s, 3.0) - 0.1:
            through += 1
        after = [p for p in c_phr if p_end < p[0] <= p_end + 2.0]
        if after:
            restarts += 1
        events.append({"type": "agent_interrupt", "t": round(s, 2), "yield": round(y, 2),
                       "restart": bool(after)})
    _stats("conv_int_yield", yields, out)
    out["conv_int_count"] = float(len(yields))
    if yields:
        out["conv_int_restart_frac"] = _safe(restarts / len(yields))
        out["conv_int_through_frac"] = _safe(through / len(yields))
        out["conv_int_yield_fast_frac"] = _safe(np.mean(np.array(yields) < 0.6))

    # --- caller interruptions and back-channels during agent speech -----------------------------
    cint, back = 0, 0
    for p in c_phr:
        if _active_at(a_mask, p[0]):
            dur = p[1] - p[0]
            if dur < 0.6:
                back += 1
                events.append({"type": "backchannel", "t": round(p[0], 2)})
            else:
                cint += 1
                events.append({"type": "caller_interrupt", "t": round(p[0], 2), "dur": round(dur, 2)})
    out["conv_cint_count"] = float(cint)
    out["conv_cint_rate_per_min"] = _safe(cint / max(duration / 60.0, 1e-3))
    out["conv_backchannel_count"] = float(back)
    out["conv_backchannel_rate_per_agent_min"] = _safe(back / max(vad_a.speech_seconds / 60.0, 1e-3))

    # --- dead air: both silent for >= 2.5 s; who fills it, and how fast -------------------------
    silent = ~(a_mask[:n] | c_mask[:n])
    d = np.diff(np.concatenate([[0], silent.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    dead, fills = 0, []
    for s, e in zip(starts, ends):
        if (e - s) * HOP_S < 2.5 or e >= n:
            continue
        dead += 1
        t_end = e * HOP_S
        if c_mask[e] and not a_mask[e]:
            fills.append(t_end - s * HOP_S)
            events.append({"type": "silence_fill", "t": round(t_end, 2), "after": round(t_end - s * HOP_S, 2)})
        else:
            events.append({"type": "agent_fills_silence", "t": round(t_end, 2), "after": round(t_end - s * HOP_S, 2)})
    out["conv_dead_count"] = float(dead)
    if dead:
        out["conv_dead_caller_fill_frac"] = _safe(len(fills) / dead)
    _stats("conv_dead_fill_latency", fills, out)

    # --- regularity index -----------------------------------------------------------------
    cvs = [out.get(k) for k in ("conv_resp_cv", "conv_turn_dur_cv", "conv_pause_cv")]
    cvs = [c for c in cvs if c is not None and np.isfinite(c)]
    if cvs:
        out["conv_regularity"] = _safe(1.0 - np.mean(np.clip(cvs, 0, 1.5)) / 1.5)
    return ConvResult(features=out, events=events, timeline=_timeline(vad_c, vad_a))


def _timeline(vad_c: Vad, vad_a: Vad | None) -> dict:
    return {
        "caller": [[round(s, 2), round(e, 2)] for s, e in vad_c.turns],
        "caller_phrases": [[round(s, 2), round(e, 2)] for s, e in vad_c.phrases],
        "agent": [[round(s, 2), round(e, 2)] for s, e in (vad_a.turns if vad_a else [])],
    }
