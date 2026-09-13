"""Microphone quality check before a live call.

The live call classifies the *person at the microphone*, so anything the microphone path does that resembles a
synthetic pipeline (digital silence from a noise gate, loudness normalisation from AGC, narrow-band Bluetooth
audio, clipping) can produce a false positive, and anything that masks the cues (a very noisy line, a very low
level) makes the verdict unreliable and can let a synthetic voice pass as human (false negative).

`mic_check_report` turns the detector's own acoustic features of a short sample (a couple of seconds of
silence followed by a spoken sentence) into a quality grade and plain-language warnings. Because it reads the
same feature vector the detector uses, the warnings describe what the model will actually see.
"""
from __future__ import annotations

import numpy as np


def _ok(v):
    return v is not None and not (isinstance(v, float) and not np.isfinite(v))


def mic_check_report(feats: dict, client: dict | None = None) -> dict:
    client = client or {}
    w: list = []

    def warn(level: str, risk: str, text: str) -> None:
        w.append({"level": level, "risk": risk, "text": text})

    def g(k, default=None):
        v = feats.get(k)
        return v if _ok(v) else default

    speech = float(g("speech_seconds", 0.0) or 0.0)
    snr, floor, level = g("snr_db"), g("floor_db"), g("level_rms_speech_dbfs")
    peak, clip = g("level_peak_dbfs"), g("level_clip_frac", 0.0)
    zero, quiet, spread = g("floor_zero_hop_frac", 0.0), g("floor_quiet_hop_frac", 0.0), g("floor_p90_minus_p10_db")
    hum, rolloff, high = g("hum_db"), g("ltas_rolloff_hz"), g("ltas_high_ratio_db")

    if speech < 1.0:
        warn("bad", "unreliable", "No speech was detected. Say a full sentence during the test and check that the "
                                  "right input device is selected.")
    else:
        if snr is not None:
            if snr < 20:
                warn("bad", "false_negative", f"Very noisy input (SNR {snr:.0f} dB). Noise masks the breathing, noise-floor "
                                              "and texture cues: a synthetic voice could pass as human and any verdict is less reliable.")
            elif snr < 30:
                warn("warn", "false_negative", f"Noisy input (SNR {snr:.0f} dB): some acoustic cues are masked, expect lower confidence.")
        if zero > 0.5 or (quiet > 0.97 and spread is not None and spread < 0.8):
            warn("warn", "false_positive", "When you are quiet the microphone path produces digital silence or a perfectly constant "
                                           "floor (a noise gate or software noise suppression). Injected synthetic audio looks exactly "
                                           "like this, so a real voice can be flagged as synthetic.")
        if level is not None and level < -36:
            warn("warn", "unreliable", f"Input level is very low ({level:.0f} dBFS). Speak closer to the microphone or raise its gain; "
                                       "quiet speech loses the fine acoustic detail the detector needs.")
        if (clip or 0) > 0.005 or (peak is not None and peak > -0.3):
            warn("warn", "false_positive", "The signal clips. Distortion adds a codec/vocoder-like texture that reads as synthetic.")
        if rolloff is not None and high is not None and (rolloff < 3300 or high < -34):
            warn("info", "false_positive", f"Narrow-band input (roll-off at {rolloff:.0f} Hz, typical of a Bluetooth headset in call "
                                           "mode). The sharp band edge is one of the synthetic-pipeline cues.")
        if hum is not None and hum > 8:
            warn("info", "false_negative", f"Mains hum is present ({hum:.0f} dB above the floor). It is an analogue/human cue: a "
                                           "synthetic voice played through this microphone would look more human.")
    if client.get("noise_suppression"):
        warn("warn", "false_positive", "Browser noise suppression is active on this input. It gates silence and flattens dynamics, "
                                       "which the detector reads as a synthetic pipeline; disable it in the browser or OS if possible.")
    if client.get("auto_gain"):
        warn("warn", "false_positive", "Automatic gain control is active. It normalises loudness the way a TTS engine does "
                                       "(constant level, no dynamics), so a real voice can look synthetic.")
    sr = client.get("sample_rate")
    if sr and sr < 16000:
        warn("info", "false_positive", f"The audio device runs at {sr} Hz (narrow-band headset profile); the band shape will "
                                       "resemble a synthetic pipeline.")

    bad = sum(1 for x in w if x["level"] == "bad")
    warns = sum(1 for x in w if x["level"] == "warn")
    quality = "poor" if bad else ("fair" if warns else "good")
    if quality == "good":
        summary = "Microphone looks good: the verdict on this call should be reliable."
    else:
        risks = {x["risk"] for x in w}
        parts = []
        if "false_positive" in risks:
            parts.append("a real voice could be flagged as synthetic (false positive)")
        if "false_negative" in risks:
            parts.append("a synthetic voice could pass as human (false negative)")
        if "unreliable" in risks and not parts:
            parts.append("the verdict will be unreliable")
        summary = "Microphone issues: " + "; ".join(parts) + "."
    metrics = {
        "speech_seconds": round(speech, 1),
        "snr_db": None if snr is None else round(float(snr), 1),
        "floor_dbfs": None if floor is None else round(float(floor), 1),
        "level_dbfs": None if level is None else round(float(level), 1),
        "peak_dbfs": None if peak is None else round(float(peak), 1),
        "clip_frac": round(float(clip or 0), 4),
        "digital_silence_frac": round(float(zero or 0), 3),
        "floor_spread_db": None if spread is None else round(float(spread), 1),
        "rolloff_hz": None if rolloff is None else int(round(float(rolloff))),
        "hum_db": None if hum is None else round(float(hum), 1),
    }
    return {"quality": quality, "summary": summary, "warnings": w, "metrics": metrics, "client": client}
