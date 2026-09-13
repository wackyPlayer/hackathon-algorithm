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
  conv_consistency  the same idea built only from scale-free measures, so a caller that answers *faster*
                    than a human (a speech-to-speech model) is still judged on how repeatable it is

On speed versus consistency: the dataset's synthetic callers all ran recognition -> LLM -> speech and took
two to three seconds, so a model trained on raw latency learns "slow = machine" and then waves through a
realtime voice agent that answers in 300 ms. The brief says the signal is that machines recover
*consistently*; the conv_resp_log_std / _mad_norm / _predictability / _entropy_norm features measure exactly
that, and say nothing about how fast the caller is.
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
    lat, lat_agent_dur = [], []
    for i, (s, e) in enumerate(a_turns):
        nxt = a_turns[i + 1][0] if i + 1 < len(a_turns) else duration + 10
        if _active_at(c_mask, e - HOP_S):     # caller already talking when agent stops: overlap, not a response
            continue
        cand = [p for p in c_phr if p[0] >= e - 0.3 and p[0] < min(nxt, e + 8.0)]
        if cand:
            L = cand[0][0] - e
            lat.append(L)
            lat_agent_dur.append(e - s)       # paired with this latency, for the predictability fit below
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

    # --- consistency, independent of speed ------------------------------------------------
    # The brief is explicit: "Humans recover from these instantly and messily. Machines recover
    # consistently, and consistency is a signal." The first version of this detector measured how *slow*
    # the caller was instead, because every synthetic caller in the dataset ran a speech-recognition ->
    # LLM -> speech pipeline and took 2-3 seconds. A speech-to-speech model answers in 300 ms, faster than
    # any human, and a "slow answers = machine" rule then votes confidently for HUMAN. Everything below is
    # scale-free on purpose: it asks how *repeatable* the caller's timing is, not how long it takes.
    lat_a = np.asarray(lat, dtype=np.float64)
    if lat_a.size >= 3:
        med = float(np.median(lat_a)) or 1e-6
        out["conv_resp_range_norm"] = _safe((np.percentile(lat_a, 90) - np.percentile(lat_a, 10)) / abs(med))
        out["conv_resp_iqr_norm"] = _safe((np.percentile(lat_a, 75) - np.percentile(lat_a, 25)) / abs(med))
        out["conv_resp_mad_norm"] = _safe(np.median(np.abs(lat_a - med)) / abs(med))
        # spread of the *log* latency: a person's replies scatter over an order of magnitude (an instant
        # "sí" and a ten-second think), a scheduler's do not, whatever its mean
        out["conv_resp_log_std"] = _safe(np.std(np.log(np.clip(lat_a, 0.05, None))))
        # how much of the variation a straight line through the agent's turn length explains: a pipeline's
        # latency is a constant plus processing time, so it is nearly predictable; a person's is not
        dur = np.asarray(lat_agent_dur, dtype=np.float64)
        if dur.size == lat_a.size and np.std(dur) > 1e-6 and np.std(lat_a) > 1e-9:
            r = float(np.corrcoef(dur, lat_a)[0, 1])
            out["conv_resp_predictability"] = _safe(r * r if np.isfinite(r) else np.nan)
        # entropy of the latency histogram, normalised to [0, 1]: low = the same answer delay every time
        h, _ = np.histogram(lat_a, bins=min(8, max(3, lat_a.size // 2)))
        pr = h / max(h.sum(), 1)
        pr = pr[pr > 0]
        out["conv_resp_entropy_norm"] = _safe(-(pr * np.log(pr)).sum() / np.log(len(pr)) if len(pr) > 1 else 0.0)
        # a human sometimes answers before the agent has finished; a turn-taking state machine waits
        out["conv_resp_fast_frac"] = _safe(np.mean(lat_a < 0.5))
        out["conv_resp_slow_frac"] = _safe(np.mean(lat_a > 3.0))
    if len(yields) >= 3:
        ya = np.asarray(yields, dtype=np.float64)
        m = float(np.median(ya)) or 1e-6
        # the brief's own probe: the agent talks over the caller. People stop at wildly different points
        # (mid-word, after finishing the thought, not at all); a machine yields the same way every time.
        out["conv_int_yield_range_norm"] = _safe((ya.max() - ya.min()) / abs(m))
        out["conv_int_yield_log_std"] = _safe(np.std(np.log(np.clip(ya, 0.05, None))))

    # --- regularity index -----------------------------------------------------------------
    cvs = [out.get(k) for k in ("conv_resp_cv", "conv_turn_dur_cv", "conv_pause_cv")]
    cvs = [c for c in cvs if c is not None and np.isfinite(c)]
    if cvs:
        out["conv_regularity"] = _safe(1.0 - np.mean(np.clip(cvs, 0, 1.5)) / 1.5)
    # the same idea over the scale-free measures only, so it survives a fast caller
    reg = [out.get(k) for k in ("conv_resp_log_std", "conv_resp_mad_norm", "conv_int_yield_log_std")]
    reg = [c for c in reg if c is not None and np.isfinite(c)]
    if reg:
        out["conv_consistency"] = _safe(1.0 - np.mean(np.clip(reg, 0, 1.2)) / 1.2)
    return ConvResult(features=out, events=events, timeline=_timeline(vad_c, vad_a))


def _timeline(vad_c: Vad, vad_a: Vad | None) -> dict:
    return {
        "caller": [[round(s, 2), round(e, 2)] for s, e in vad_c.turns],
        "caller_phrases": [[round(s, 2), round(e, 2)] for s, e in vad_c.phrases],
        "agent": [[round(s, 2), round(e, 2)] for s, e in (vad_a.turns if vad_a else [])],
    }
