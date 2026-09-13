"""Interpretable aspect scores and a no-training fallback probability.

Each *aspect* is a small set of features mapped through a logistic curve centred between the
class means observed on the Altur training calls (see README, "Calibration"). Scores are in
[0, 1] where 1 = "looks synthetic". They power the dashboard's per-aspect ratings and attack
profile even when the trained model makes the final decision; when no trained model is present
the weighted combination of aspects *is* the decision.
"""
from __future__ import annotations

import math

import numpy as np

# (feature, center, width, higher_is_synthetic, weight, short description)
ASPECTS: dict = {
    "bandwidth_cut": {
        "label": "Sharp spectral cut / band shape",
        "weight": 1.0,
        "terms": [
            ("ltas_high_ratio_db", -30.0, 3.0, False, 1.0, "energy above 3.6 kHz vs voice band"),
            ("ltas_rolloff_hz", 3680.0, 90.0, False, 0.8, "frequency where the spectrum rolls off"),
            ("ltas_low_ratio_db", 4.2, 1.5, True, 0.6, "energy below 250 Hz vs voice band"),
            ("ltas_rolloff_slope_db_per_100hz", 6.0, 2.0, True, 0.4, "steepness of the roll-off"),
        ],
    },
    "prosody_flatness": {
        "label": "Flat prosody / constant loudness",
        "weight": 0.9,
        "terms": [
            ("level_range_db", 15.4, 2.5, False, 0.9, "loudness range across phrases"),
            ("level_std_db", 4.0, 0.7, False, 0.8, "loudness variation across phrases"),
            ("energy_std_db", 11.4, 0.8, False, 0.6, "frame energy variation"),
            ("rhythm_rate_cv", 0.124, 0.03, False, 0.8, "variability of the syllable rate"),
            ("f0_std_st", 2.6, 0.5, False, 0.5, "pitch variability (semitones)"),
            ("f0_range_st", 8.5, 1.5, False, 0.4, "pitch range (5th-95th pct)"),
            ("rhythm_syllable_rate", 4.6, 0.3, True, 0.5, "syllables per second"),
        ],
    },
    "vocoder_artifacts": {
        "label": "Vocoder / codec texture",
        "weight": 0.8,
        "terms": [
            ("spec_flatness_std", 0.053, 0.01, False, 0.9, "variation of spectral flatness"),
            ("lfcc_s1", 3.85, 0.4, False, 0.8, "spread of the 1st linear cepstral coefficient"),
            ("mfcc_s0", 11.9, 0.7, False, 0.6, "spread of log energy (MFCC0)"),
            ("spec_centroid_std", 437.0, 60.0, False, 0.6, "variation of the spectral centroid"),
            ("f0_jitter_pct", 0.9, 0.3, False, 0.3, "cycle-to-cycle pitch perturbation"),
            ("shimmer_db", 1.1, 0.3, False, 0.3, "frame-to-frame amplitude perturbation"),
        ],
    },
    "breathing_absence": {
        "label": "Breathing",
        "weight": 0.4,
        "terms": [
            ("breath_per_speech_min", 6.0, 3.0, False, 1.0, "breath-like events per minute of speech"),
            ("breath_before_onset_frac", 0.15, 0.08, False, 0.7, "phrase onsets preceded by a breath"),
        ],
    },
    "hard_cuts": {
        "label": "Hard onsets / no decay tail",
        "weight": 0.5,
        "terms": [
            ("cut_hard_frac", 0.12, 0.05, True, 1.0, "phrase edges with a >25 dB jump"),
            ("cut_onset_ms_mean", 70.0, 20.0, False, 0.6, "mean onset ramp (ms)"),
            ("cut_decay_ms_mean", 60.0, 25.0, False, 0.6, "mean decay tail after a phrase (ms)"),
        ],
    },
    "injection_signature": {
        "label": "Noise floor / channel injection",
        "weight": 1.0,
        "terms": [
            ("floor_p90_minus_p10_db", 4.8, 1.2, False, 1.0, "spread of the noise floor level"),
            ("floor_spectral_var_db", 9.0, 2.0, False, 0.7, "stationarity of the noise floor spectrum"),
            ("floor_quiet_hop_frac", 0.72, 0.05, True, 0.6, "near-digital-silence fraction"),
            ("chan_floor_rise_during_agent_db", -1.3, 0.6, True, 0.8, "caller channel change while agent speaks"),
            ("floor_zero_hop_frac", 0.05, 0.03, True, 0.6, "exact digital silence fraction"),
        ],
    },
    "turn_taking": {
        "label": "Turn-taking behaviour",
        "weight": 1.3,
        # Latency magnitude alone cannot carry this aspect. Every synthetic caller in the dataset ran
        # recognition -> LLM -> speech and answered in 2-3 s, so terms built on "slow = machine" score a
        # speech-to-speech agent (0.25-0.7 s, faster than a person) as HUMAN: measured, they flagged 14 % of
        # such callers. The scale-free consistency terms below raise that to 55 % while holding the
        # false-flag rate on real held-out customers at 4.3 % on full calls and on 60 s clips.
        # Centres and widths for those terms come from training/recalibrate_heuristics.py, fitted on the
        # training side of the split only (never the held-out calls) and across full calls and clips.
        # Dropped: conv_dead_fill_latency_n -- a raw event count that scales with clip length, and it is
        # emitted as 0.0 even when no silence fill was ever seen, which then voted strongly HUMAN.
        "terms": [
            ("conv_resp_median", 1.8, 0.3, True, 0.8, "median response latency (s)"),
            ("conv_resp_min", 1.1, 0.3, True, 0.6, "fastest response (s)"),
            ("conv_resp_frac_over_2s", 0.35, 0.12, True, 0.6, "responses slower than 2 s"),
            ("conv_dead_caller_fill_frac", 0.45, 0.15, True, 0.8, "dead-air windows filled by the caller"),
            ("conv_backchannel_rate_per_agent_min", 1.0, 0.6, False, 0.4, "back-channels per agent minute"),
            ("conv_int_through_frac", 0.5, 0.2, True, 0.3, "talks through agent interruptions"),
            ("conv_turn_dur_mean", 4.0, 1.0, True, 0.3, "mean caller turn length (s)"),
            ("conv_resp_mad_norm", 0.145, 0.0657, False, 0.7, "scatter of reply delays around its own median"),
            ("conv_pause_cv", 0.341, 0.0845, False, 0.6, "variability of pauses inside a caller turn"),
            ("conv_resp_entropy_norm", 0.918, 0.0398, False, 0.5, "spread of the reply-delay histogram"),
        ],
    },
    "semantic_fabrication": {
        "label": "Invents answers (LLM behaviour)",
        "weight": 1.2,
        "terms": [
            ("sem_fabrication_p", 0.5, 0.15, True, 1.0, "judge: fabricated details"),
            ("sem_synthetic_p", 0.5, 0.15, True, 0.8, "judge: overall synthetic likelihood"),
            ("sem_invented_probes", 0.5, 0.4, True, 0.5, "probes about non-existent items answered with details"),
        ],
    },
    "llm_style": {
        "label": "LLM-like wording",
        "weight": 0.7,
        "terms": [
            ("sem_llm_style_p", 0.5, 0.15, True, 1.0, "judge: LLM-like phrasing"),
            ("text_filler_rate", 1.5, 0.8, False, 0.5, "fillers per 100 words"),
            ("text_llm_phrase_count", 2.0, 1.0, True, 0.4, "over-polite / formulaic phrases"),
            ("text_words_per_turn", 22.0, 6.0, True, 0.4, "words per caller turn"),
        ],
    },
}

ATTACKS = {
    "tts_clone": "AI-generated / cloned voice (TTS)",
    "llm_bot": "Autonomous LLM caller (ASR -> LLM -> TTS)",
    "digital_injection": "Audio injected into the call (no microphone)",
    "replay": "Recording replayed into a handset",
    "voice_changer": "Real-time voice conversion / pitch shift",
}


def _sig(x: float, center: float, width: float, higher_is_synthetic: bool) -> float:
    z = (x - center) / max(width, 1e-6)
    if not higher_is_synthetic:
        z = -z
    return 1.0 / (1.0 + math.exp(-max(min(z, 30), -30)))


def aspect_scores(features: dict) -> dict:
    out: dict = {}
    for name, spec in ASPECTS.items():
        num, den, evidence, used = 0.0, 0.0, [], 0
        for feat, center, width, hi, w, desc in spec["terms"]:
            v = features.get(feat)
            if v is None or not np.isfinite(v):
                continue
            s = _sig(float(v), center, width, hi)
            num += w * s
            den += w
            used += 1
            direction = "synthetic-like" if s > 0.6 else ("human-like" if s < 0.4 else "neutral")
            evidence.append({"feature": feat, "value": round(float(v), 3), "score": round(s, 2),
                             "desc": desc, "read": direction})
        score = num / den if den > 0 else None
        out[name] = {"label": spec["label"], "score": None if score is None else round(score, 3),
                     "weight": spec["weight"], "n_terms": used, "evidence": evidence}
    return out


def heuristic_probability(aspects: dict) -> float:
    num, den = 0.0, 0.0
    for name, a in aspects.items():
        if a["score"] is None:
            continue
        w = a["weight"] * min(1.0, a["n_terms"] / 2.0)
        num += w * a["score"]
        den += w
    if den == 0:
        return 0.5
    m = num / den
    return 1.0 / (1.0 + math.exp(-6.0 * (m - 0.5)))


def _s(aspects: dict, name: str, default: float = 0.5) -> float:
    a = aspects.get(name)
    return default if a is None or a["score"] is None else a["score"]


def attack_profile(aspects: dict, features: dict, p_synthetic: float) -> dict:
    voice = np.mean([_s(aspects, "prosody_flatness"), _s(aspects, "vocoder_artifacts"), _s(aspects, "breathing_absence")])
    mismatch = features.get("formant_f0_mismatch")
    mism = _sig(mismatch, 1.6, 0.4, True) if mismatch is not None and np.isfinite(mismatch) else 0.3
    channel_acoustic = np.mean([
        _sig(features.get("cut_decay_ms_mean", 40.0), 70.0, 25.0, True),
        _sig(features.get("floor_p90_minus_p10_db", 4.0), 4.8, 1.2, True),
        _sig(features.get("ltas_peakiness_db", 2.0), 2.5, 0.7, True),
        _sig(features.get("hum_db", 0.0), 6.0, 3.0, True),
    ])
    raw = {
        "tts_clone": float(np.mean([voice, _s(aspects, "bandwidth_cut"), _s(aspects, "hard_cuts")])),
        "llm_bot": float(np.mean([_s(aspects, "turn_taking"), _s(aspects, "semantic_fabrication"), _s(aspects, "llm_style")])),
        "digital_injection": float(np.mean([_s(aspects, "injection_signature"), _s(aspects, "bandwidth_cut")])),
        "replay": float(channel_acoustic * voice),
        "voice_changer": float(0.35 * _s(aspects, "vocoder_artifacts") + 0.65 * mism),
    }
    tot = sum(raw.values()) or 1.0
    return {k: {"label": ATTACKS[k], "raw": round(v, 3), "p": round(p_synthetic * v / tot, 3)} for k, v in raw.items()}
