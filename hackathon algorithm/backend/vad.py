"""Energy-based voice activity detection on a fixed 10 ms frame grid.

All feature modules share one STFT frame grid (50 ms window, 10 ms hop at 8 kHz), so the
VAD is derived from the same power spectrogram instead of a separate framing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

SR = 8000
WIN = 400       # 50 ms
HOP = 80        # 10 ms
NFFT = 512
HOP_S = HOP / SR
EPS = 1e-12


def stft_power(x: np.ndarray, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    """Power spectrogram (n_bins x n_frames) and bin frequencies."""
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    f, _, Z = signal.stft(x, fs=sr, window="hann", nperseg=WIN, noverlap=WIN - HOP, nfft=NFFT,
                          boundary=None, padded=True)
    return (np.abs(Z) ** 2).astype(np.float32), f.astype(np.float32)


def frame_db(P: np.ndarray) -> np.ndarray:
    return (10.0 * np.log10(P.sum(axis=0) + EPS)).astype(np.float32)


def segments_from_mask(mask: np.ndarray, hop_s: float = HOP_S, min_gap_s: float = 0.2,
                       min_dur_s: float = 0.12) -> list[tuple[float, float]]:
    """Boolean frame mask -> merged (start, end) segments in seconds."""
    if mask.size == 0 or not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    segs: list[list[float]] = []
    for s, e in zip(starts, ends):
        st, en = s * hop_s, e * hop_s
        if segs and st - segs[-1][1] < min_gap_s:
            segs[-1][1] = en
        else:
            segs.append([st, en])
    return [(s, e) for s, e in segs if e - s >= min_dur_s]


def mask_from_segments(segs, n_frames: int, hop_s: float = HOP_S) -> np.ndarray:
    m = np.zeros(n_frames, dtype=bool)
    for s, e in segs:
        a, b = int(round(s / hop_s)), int(round(e / hop_s))
        m[max(a, 0):min(b, n_frames)] = True
    return m


@dataclass
class Vad:
    db: np.ndarray                 # per-frame dB
    speech: np.ndarray             # per-frame bool
    floor_db: float
    threshold_db: float
    turns: list = field(default_factory=list)      # gaps < 0.7 s merged
    phrases: list = field(default_factory=list)    # gaps < 0.15 s merged

    @property
    def speech_seconds(self) -> float:
        return float(self.speech.sum()) * HOP_S

    @property
    def n_frames(self) -> int:
        return int(self.db.size)


def vad_from_power(P: np.ndarray, rise_db: float = 8.0, peak_drop_db: float = 32.0,
                   abs_min_db: float = -75.0) -> Vad:
    db = frame_db(P)
    if db.size == 0:
        return Vad(db=db, speech=np.zeros(0, bool), floor_db=-100.0, threshold_db=0.0)
    floor = float(np.percentile(db, 10))
    peak = float(np.percentile(db, 97))
    thr = max(floor + rise_db, peak - peak_drop_db, abs_min_db)
    raw = db > thr
    sm = median_filter(raw.astype(np.uint8), size=5).astype(bool) if raw.size >= 5 else raw
    v = Vad(db=db, speech=sm, floor_db=floor, threshold_db=thr)
    v.phrases = segments_from_mask(sm, min_gap_s=0.15, min_dur_s=0.08)
    v.turns = segments_from_mask(sm, min_gap_s=0.7, min_dur_s=0.15)
    v.speech = mask_from_segments(v.phrases, db.size)
    return v


def vad_from_segments(segs, n_frames: int) -> Vad:
    """Vad built from known segments (agent timeline in live mode, or dataset turns)."""
    m = mask_from_segments(segs, n_frames)
    db = np.where(m, -20.0, -90.0).astype(np.float32)
    v = Vad(db=db, speech=m, floor_db=-90.0, threshold_db=-50.0)
    v.phrases = segments_from_mask(m, min_gap_s=0.15, min_dur_s=0.0)
    v.turns = segments_from_mask(m, min_gap_s=0.7, min_dur_s=0.0)
    return v
