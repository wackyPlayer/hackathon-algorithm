"""Acoustic features of the caller channel (8 kHz telephony).

Everything is numpy/scipy on one shared STFT grid (50 ms window / 10 ms hop), so a 3 minute
call is analysed in well under a second on a laptop CPU. Feature groups:

  ltas_*        long-term spectrum: out-of-band energy, roll-off steepness ("sharp cut"), tilt, ripples
  f0_* / hnr_*  prosody and micro-perturbation: pitch variability, smoothness, jitter, shimmer, HNR
  rhythm_*      syllable rate and its variability, modulation spectrum
  lfcc_*/mfcc_* cepstral statistics (classic anti-spoofing front-end)
  floor_*       noise floor level, stationarity, digital silence (injection cue)
  cut_*         onset/offset ramps, hard cuts, reverberant decay tails (injection / replay cue)
  breath_*      breath events between phrases
  formant_*     LPC formants and their consistency with F0 (voice changer / pitch shift cue)
  hum_*         50/60 Hz mains hum (analogue acquisition cue)
  chan_*        cross-channel cues: acoustic echo of the agent, caller floor while the agent speaks
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field

import numpy as np
from scipy import signal
from scipy.fft import dct
from scipy.linalg import solve_toeplitz
from scipy.ndimage import binary_dilation, gaussian_filter1d, uniform_filter1d

from ..audio import Call
from ..vad import EPS, HOP, HOP_S, SR, WIN, Vad

_HANN = np.hanning(WIN).astype(np.float32)
_HANN_AC = None  # autocorrelation of the analysis window (lazy)


@dataclass
class AcousticResult:
    features: dict = field(default_factory=dict)
    ui: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- helpers

def _band(freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (freqs >= lo) & (freqs < hi)


def _safe(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _cv(a) -> float:
    a = np.asarray(a, dtype=np.float64)
    if a.size < 3 or abs(a.mean()) < 1e-9:
        return float("nan")
    return _safe(a.std() / abs(a.mean()))


def _frames_view(x: np.ndarray, n_frames: int) -> np.ndarray:
    need = (n_frames - 1) * HOP + WIN
    if len(x) < need:
        x = np.pad(x, (0, need - len(x)))
    x = np.ascontiguousarray(x, dtype=np.float32)
    return np.lib.stride_tricks.as_strided(x, shape=(n_frames, WIN), strides=(x.strides[0] * HOP, x.strides[0]))


def _filterbank(freqs: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Triangular filterbank (n_filters x n_bins) for the given edge frequencies."""
    n = len(edges) - 2
    fb = np.zeros((n, len(freqs)), dtype=np.float32)
    for i in range(n):
        lo, c, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (freqs - lo) / max(c - lo, 1e-6)
        down = (hi - freqs) / max(hi - c, 1e-6)
        fb[i] = np.clip(np.minimum(up, down), 0, 1)
    return fb


def _mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _imel(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


# ----------------------------------------------------------------------------- feature groups

def frame_descriptors(P: np.ndarray, freqs: np.ndarray) -> dict:
    tot = P.sum(axis=0) + EPS
    centroid = (freqs[:, None] * P).sum(axis=0) / tot
    logP = np.log(P + EPS)
    flatness = np.exp(logP[1:].mean(axis=0)) / (P[1:].mean(axis=0) + EPS)
    p = P / tot
    entropy = -(p * np.log(p + EPS)).sum(axis=0) / np.log(P.shape[0])
    flux = np.zeros(P.shape[1], dtype=np.float32)
    if P.shape[1] > 1:
        flux[1:] = np.abs(np.diff(logP, axis=1)).mean(axis=0)
    cum = np.cumsum(P, axis=0) / tot
    rolloff = freqs[np.argmax(cum >= 0.85, axis=0)]
    return {"centroid": centroid, "flatness": flatness, "entropy": entropy, "flux": flux, "rolloff85": rolloff}


def ltas_features(P: np.ndarray, freqs: np.ndarray, speech: np.ndarray) -> dict:
    out: dict = {}
    if speech.sum() < 20:
        return out
    Ps = P[:, speech]
    ltas = 10 * np.log10(np.median(Ps, axis=1) + EPS)
    sm = uniform_filter1d(ltas, 5)
    mean_p = Ps.mean(axis=1)

    def band_db(lo, hi):
        m = _band(freqs, lo, hi)
        return 10 * np.log10(mean_p[m].mean() + EPS) if m.any() else float("nan")

    voice = band_db(300, 3400)
    out["ltas_low_ratio_db"] = _safe(band_db(30, 250) - voice)
    out["ltas_high_ratio_db"] = _safe(band_db(3600, 3950) - voice)
    out["ltas_mid_high_ratio_db"] = _safe(band_db(2500, 3400) - band_db(300, 1000))
    # roll-off: highest frequency still within 25 dB of the voice-band plateau
    vb = _band(freqs, 300, 3000)
    plateau = np.percentile(sm[vb], 90)
    above = np.flatnonzero((sm > plateau - 25) & (freqs >= 300))
    if above.size:
        k = int(above.max())
        out["ltas_rolloff_hz"] = float(freqs[k])
        step = int(round(150 / (freqs[1] - freqs[0])))
        a, b = max(k - step, 0), min(k + step, len(freqs) - 1)
        span = (freqs[b] - freqs[a]) / 100.0
        out["ltas_rolloff_slope_db_per_100hz"] = _safe((sm[a] - sm[b]) / span) if span > 0 else float("nan")
    m = _band(freqs, 300, 3400)
    x = np.log2(freqs[m])
    if x.size > 5:
        out["ltas_tilt_db_per_oct"] = _safe(np.polyfit(x, sm[m], 1)[0])
    resid = sm[m] - uniform_filter1d(sm, 25)[m]
    out["ltas_peakiness_db"] = _safe(resid.std())
    hp = mean_p[_band(freqs, 3400, 4001)]
    out["ltas_high_flatness"] = _safe(np.exp(np.log(hp + EPS).mean()) / (hp.mean() + EPS))
    return out


def f0_features(x: np.ndarray, vad: Vad, want_track: bool = True) -> tuple[dict, dict]:
    global _HANN_AC
    n = vad.n_frames
    out: dict = {}
    ui: dict = {}
    if n < 5:
        return out, ui
    F = _frames_view(x, n) * _HANN
    spec = np.fft.rfft(F, n=1024, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :WIN]
    if _HANN_AC is None:
        w = np.fft.irfft(np.abs(np.fft.rfft(_HANN, n=1024)) ** 2)[:WIN]
        _HANN_AC = (w / (w[0] + EPS)).astype(np.float32)
    r = ac / (ac[:, :1] + EPS)
    r = r / (_HANN_AC[None, :] + 1e-3)
    lo, hi = 20, 134                        # 400 Hz .. 60 Hz
    sub = r[:, lo:hi]
    peak = sub.max(axis=1)
    cand = sub >= (0.90 * peak[:, None])    # shortest near-maximal lag avoids octave-down errors
    lag = lo + np.argmax(cand, axis=1)
    rows = np.arange(n)
    lm1 = r[rows, np.clip(lag - 1, 0, WIN - 1)]
    l0 = r[rows, lag]
    lp1 = r[rows, np.clip(lag + 1, 0, WIN - 1)]
    denom = lm1 - 2 * l0 + lp1
    ok = np.abs(denom) > 1e-6
    delta = np.where(ok, 0.5 * (lm1 - lp1) / np.where(ok, denom, 1.0), 0.0)
    f0 = SR / (lag + np.clip(delta, -1, 1))
    voiced = (peak > 0.55) & vad.speech & (f0 > 60) & (f0 < 420)
    pk = np.clip(peak, 1e-3, 0.999)
    hnr = 10 * np.log10(pk / (1 - pk))
    nv = int(voiced.sum())
    out["f0_voiced_frac"] = _safe(nv / max(int(vad.speech.sum()), 1))
    if nv >= 10:
        st = 12 * np.log2(f0[voiced] / 100.0)
        out["f0_median_hz"] = _safe(np.median(f0[voiced]))
        out["f0_std_st"] = _safe(st.std())
        out["f0_range_st"] = _safe(np.percentile(st, 95) - np.percentile(st, 5))
        out["f0_iqr_st"] = _safe(np.percentile(st, 75) - np.percentile(st, 25))
        out["hnr_mean_db"] = _safe(hnr[voiced].mean())
        out["hnr_std_db"] = _safe(hnr[voiced].std())
        d = np.diff(np.concatenate([[0], voiced.astype(np.int8), [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        st_all = 12 * np.log2(f0 / 100.0)
        smooth, jit, shim, slopes, lens = [], [], [], [], []
        for s, e in zip(starts, ends):
            L = e - s
            lens.append(L)
            if L >= 3:
                seg = st_all[s:e]
                smooth.append(np.abs(np.diff(seg, 2)).mean())
                T = 1.0 / f0[s:e]
                jit.append(np.abs(np.diff(T)).mean() / T.mean() * 100.0)
                shim.append(np.abs(np.diff(vad.db[s:e])).mean())
            if L >= 8:
                t = np.arange(L) * HOP_S
                slopes.append(np.polyfit(t, st_all[s:e], 1)[0])
        out["f0_smoothness_st"] = _safe(np.mean(smooth)) if smooth else float("nan")
        out["f0_jitter_pct"] = _safe(np.mean(jit)) if jit else float("nan")
        out["shimmer_db"] = _safe(np.mean(shim)) if shim else float("nan")
        out["f0_slope_std_st_per_s"] = _safe(np.std(slopes)) if len(slopes) >= 3 else float("nan")
        out["f0_slope_abs_mean"] = _safe(np.mean(np.abs(slopes))) if slopes else float("nan")
        out["f0_run_mean_frames"] = _safe(np.mean(lens)) if lens else float("nan")
        out["f0_runs_per_speech_s"] = _safe(len(lens) / max(vad.speech_seconds, 1e-3))
    if want_track:
        step = max(1, n // 1200)
        idx = np.arange(0, n, step)
        ui["f0"] = [[round(i * HOP_S, 2), (round(float(f0[i]), 1) if voiced[i] else None)] for i in idx]
    return out, ui


def energy_features(vad: Vad) -> dict:
    out: dict = {}
    sp = vad.speech
    if sp.sum() < 10:
        return out
    db = vad.db
    out["energy_std_db"] = _safe(db[sp].std())
    out["energy_smoothness_db"] = _safe(np.abs(np.diff(db[sp], 2)).mean()) if sp.sum() > 3 else float("nan")
    lv = []
    for s, e in vad.phrases:
        if e - s >= 0.3:
            a, b = int(s / HOP_S), int(e / HOP_S)
            lv.append(db[a:b].mean())
    if len(lv) >= 3:
        lv = np.array(lv)
        out["level_std_db"] = _safe(lv.std())
        out["level_range_db"] = _safe(lv.max() - lv.min())
    return out


def rhythm_features(P: np.ndarray, freqs: np.ndarray, vad: Vad) -> dict:
    out: dict = {}
    sp = vad.speech
    if sp.sum() < 30:
        return out
    env = 10 * np.log10(P[_band(freqs, 300, 2500)].sum(axis=0) + EPS)
    env_s = gaussian_filter1d(env, 2.0)
    peaks, _ = signal.find_peaks(env_s, distance=10, prominence=2.0)
    peaks = peaks[sp[peaks]]
    out["rhythm_syllable_rate"] = _safe(len(peaks) / max(vad.speech_seconds, 1e-3))
    rates = []
    for s, e in vad.turns:
        if e - s >= 1.5:
            a, b = int(s / HOP_S), int(e / HOP_S)
            k = ((peaks >= a) & (peaks < b)).sum()
            rates.append(k / (sp[a:b].sum() * HOP_S + 1e-3))
    out["rhythm_rate_cv"] = _cv(np.array(rates)) if len(rates) >= 3 else float("nan")
    L = 256
    spec_acc = np.zeros(L // 2 + 1)
    k = 0
    win = np.hanning(L)
    for a in range(0, len(env) - L, L // 2):
        if sp[a:a + L].mean() >= 0.7:
            seg = env[a:a + L] - env[a:a + L].mean()
            spec_acc += np.abs(np.fft.rfft(seg * win)) ** 2
            k += 1
    if k:
        mf = np.fft.rfftfreq(L, HOP_S)
        ms = spec_acc / k
        tot = ms[(mf >= 0.5) & (mf < 50)].sum() + EPS
        for name, lo, hi in (("1_3", 1, 3), ("3_7", 3, 7), ("7_15", 7, 15), ("15_30", 15, 30)):
            out[f"rhythm_mod_{name}_frac"] = _safe(ms[(mf >= lo) & (mf < hi)].sum() / tot)
        band = (mf >= 1) & (mf < 15)
        out["rhythm_mod_peak_hz"] = _safe(mf[band][np.argmax(ms[band])])
        out["rhythm_mod_windows"] = float(k)
    return out


def cepstral_features(P: np.ndarray, freqs: np.ndarray, speech: np.ndarray) -> dict:
    out: dict = {}
    if speech.sum() < 20:
        return out
    Ps = P[:, speech]
    lin_edges = np.linspace(0, 4000, 22)
    mel_edges = _imel(np.linspace(_mel(0), _mel(4000), 26))
    for name, edges, nc in (("lfcc", lin_edges, 20), ("mfcc", mel_edges, 13)):
        fb = _filterbank(freqs, edges)
        E = np.log(fb @ Ps + EPS)
        C = dct(E, type=2, norm="ortho", axis=0)[:nc]
        mu, sd = C.mean(axis=1), C.std(axis=1)
        for i in range(nc):
            out[f"{name}_m{i}"] = _safe(mu[i])
            out[f"{name}_s{i}"] = _safe(sd[i])
        if C.shape[1] > 2:
            out[f"{name}_delta_std"] = _safe(np.abs(np.diff(C, axis=1)).std(axis=1).mean())
    return out


def spectral_shape_features(desc: dict, speech: np.ndarray) -> dict:
    out: dict = {}
    if speech.sum() < 10:
        return out
    for k in ("centroid", "flatness", "entropy", "flux", "rolloff85"):
        v = desc[k][speech]
        out[f"spec_{k}_mean"] = _safe(v.mean())
        out[f"spec_{k}_std"] = _safe(v.std())
    return out


def floor_features(call: Call, P: np.ndarray, freqs: np.ndarray, vad: Vad, desc: dict) -> dict:
    out: dict = {}
    n = vad.n_frames
    if n < 10:
        return out
    ns = ~binary_dilation(vad.speech, iterations=3)
    out["floor_nonspeech_frac"] = _safe(ns.mean())
    if ns.sum() >= 5:
        db = vad.db[ns]
        out["floor_db"] = _safe(np.median(db))
        out["floor_std_db"] = _safe(db.std())
        out["floor_p90_minus_p10_db"] = _safe(np.percentile(db, 90) - np.percentile(db, 10))
        out["floor_flatness_mean"] = _safe(desc["flatness"][ns].mean())
        out["floor_centroid_mean"] = _safe(desc["centroid"][ns].mean())
        edges = np.geomspace(100, 4000, 9)
        bands = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = _band(freqs, lo, hi)
            bands.append(10 * np.log10(P[m][:, ns].sum(axis=0) + EPS))
        out["floor_spectral_var_db"] = _safe(np.mean([b.std() for b in bands]))
        if vad.speech.sum() >= 10:
            out["snr_db"] = _safe(np.median(vad.db[vad.speech]) - np.median(db))
    raw = call.caller_raw
    m = n * HOP
    r = raw[:m] if len(raw) >= m else np.pad(raw, (0, m - len(raw)))
    hop_max = np.abs(r.reshape(n, HOP)).max(axis=1)
    if ns.sum() >= 5:
        out["floor_zero_hop_frac"] = _safe((hop_max[ns] <= 2.0 / 32768).mean())
        out["floor_quiet_hop_frac"] = _safe((hop_max[ns] <= 16.0 / 32768).mean())
    if len(raw):
        out["level_peak_dbfs"] = _safe(20 * np.log10(np.abs(raw).max() + EPS))
        out["level_clip_frac"] = _safe((np.abs(raw) >= 0.985).mean())
    if vad.speech.sum() >= 10:
        sp_hops = hop_max[vad.speech]
        rms = np.sqrt(np.mean(r.reshape(n, HOP)[vad.speech] ** 2) + EPS)
        out["level_rms_speech_dbfs"] = _safe(20 * np.log10(rms))
        out["level_crest_db"] = _safe(20 * np.log10(sp_hops.max() + EPS) - 20 * np.log10(rms))
    return out


def cut_features(vad: Vad) -> dict:
    out: dict = {}
    db, n = vad.db, vad.n_frames
    if not vad.phrases or n < 20:
        return out
    floor = vad.floor_db
    onsets, offsets, decays, hard, total = [], [], [], 0, 0
    for s, e in vad.phrases:
        a, b = int(round(s / HOP_S)), int(round(e / HOP_S))
        if b - a < 3:
            continue
        pk = db[a:b].max()
        k0 = a
        for k in range(a, max(a - 30, 0) - 1, -1):
            if db[k] < floor + 6:
                k0 = k
                break
        k1 = a + int(np.argmax(db[a:b] >= pk - 6))
        onsets.append((k1 - k0) * 10.0)
        total += 1
        if a >= 2 and db[a] - db[a - 2] > 25:
            hard += 1
        k2 = b - 1 - int(np.argmax(db[a:b][::-1] >= pk - 6))
        k3 = b
        for k in range(b, min(b + 50, n)):
            if db[k] < floor + 6:
                k3 = k
                break
        offsets.append((k3 - k2) * 10.0)
        decays.append((k3 - b) * 10.0)
        total += 1
        if b + 2 < n and db[b - 1] - db[b + 1] > 25:
            hard += 1
    if onsets:
        out["cut_onset_ms_mean"] = _safe(np.mean(onsets))
        out["cut_onset_ms_std"] = _safe(np.std(onsets))
        out["cut_offset_ms_mean"] = _safe(np.mean(offsets))
        out["cut_decay_ms_mean"] = _safe(np.mean(decays))
        out["cut_decay_ms_p90"] = _safe(np.percentile(decays, 90))
        out["cut_hard_frac"] = _safe(hard / max(total, 1))
        out["cut_hard_count"] = float(hard)
    return out


def breath_features(vad: Vad, desc: dict, duration: float) -> tuple[dict, list]:
    out: dict = {}
    n = vad.n_frames
    if n < 50 or not vad.phrases:
        return out, []
    floor, thr = vad.floor_db, vad.threshold_db
    cand = (~vad.speech) & (vad.db > floor + 4) & (vad.db < thr) & (desc["flatness"] > 0.25) \
        & (desc["centroid"] > 400) & (desc["centroid"] < 3000)
    d = np.diff(np.concatenate([[0], cand.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    breaths = [(s * HOP_S, e * HOP_S) for s, e in zip(starts, ends) if 8 <= e - s <= 70]
    out["breath_count"] = float(len(breaths))
    out["breath_per_min"] = _safe(len(breaths) / max(duration / 60.0, 1e-3))
    out["breath_per_speech_min"] = _safe(len(breaths) / max(vad.speech_seconds / 60.0, 1e-3))
    ends_arr = np.array([b[1] for b in breaths]) if breaths else np.zeros(0)
    onsets = [s for i, (s, e) in enumerate(vad.phrases) if i == 0 or s - vad.phrases[i - 1][1] >= 0.2]
    if onsets:
        hit = sum(1 for s in onsets if ends_arr.size and np.any((ends_arr <= s + 0.05) & (ends_arr >= s - 0.4)))
        out["breath_before_onset_frac"] = _safe(hit / len(onsets))
    return out, breaths


def formant_features(x: np.ndarray, vad: Vad, f0_median) -> dict:
    out: dict = {}
    n = vad.n_frames
    sp = np.flatnonzero(vad.speech)
    if sp.size < 30 or n < 30:
        return out
    idx = sp[np.linspace(0, sp.size - 1, min(240, sp.size)).astype(int)]
    F = _frames_view(x, n)
    order = 10
    hann = np.hanning(WIN)
    f1s, f2s = [], []
    for i in idx:
        fr = F[i].astype(np.float64)
        fr = np.append(fr[0], fr[1:] - 0.97 * fr[:-1]) * hann
        if np.abs(fr).max() < 1e-4:
            continue
        r = np.correlate(fr, fr, "full")[WIN - 1:WIN + order]
        if r[0] <= 0:
            continue
        try:
            a = solve_toeplitz((r[:order], r[:order]), r[1:order + 1])
        except Exception:
            continue
        roots = np.roots(np.concatenate([[1.0], -a]))
        roots = roots[np.imag(roots) > 0]
        fr_hz = np.angle(roots) * SR / (2 * np.pi)
        bw = -SR / np.pi * np.log(np.abs(roots) + EPS)
        keep = (fr_hz > 90) & (fr_hz < 3800) & (bw < 400)
        f = np.sort(fr_hz[keep])
        if f.size >= 2:
            f1s.append(f[0])
            f2s.append(f[1])
    if len(f1s) >= 10:
        f1, f2 = np.array(f1s), np.array(f2s)
        out["formant_f1_mean"] = _safe(np.median(f1))
        out["formant_f1_std"] = _safe(f1.std())
        out["formant_f2_mean"] = _safe(np.median(f2))
        out["formant_f2_std"] = _safe(f2.std())
        if f0_median and np.isfinite(f0_median):
            out["formant_f0_f1_ratio"] = _safe(f0_median / max(np.median(f1), 1))
            z_f0 = (np.log(f0_median) - np.log(150.0)) / 0.28
            z_f2 = (np.log(np.median(f2)) - np.log(1500.0)) / 0.17
            out["formant_f0_mismatch"] = _safe(abs(z_f0 - z_f2))
    return out


def hum_features(raw: np.ndarray) -> dict:
    out: dict = {}
    if len(raw) < 16384:
        return out
    f, p = signal.welch(raw.astype(np.float64), fs=SR, nperseg=8192)
    pdb = 10 * np.log10(p + EPS)
    ref_m = (f >= 30) & (f <= 250)
    for base in (50, 60):
        prom = []
        for k in (1, 2, 3):
            m = np.abs(f - base * k) <= 1.5
            excl = ref_m & ~(np.abs(f - base * k) <= 4)
            if m.any() and excl.any():
                prom.append(pdb[m].max() - np.median(pdb[excl]))
        out[f"hum_{base}_db"] = _safe(max(prom)) if prom else float("nan")
    vals = [v for v in (out.get("hum_50_db"), out.get("hum_60_db")) if v is not None and np.isfinite(v)]
    out["hum_db"] = max(vals) if vals else float("nan")
    return out


def channel_features(call: Call, vad_c: Vad, vad_a: Vad | None) -> dict:
    out: dict = {}
    if vad_a is None or call.agent is None:
        return out
    n = min(vad_c.n_frames, vad_a.n_frames)
    if n < 50:
        return out
    a_only = vad_a.speech[:n] & ~binary_dilation(vad_c.speech[:n], iterations=5)
    if a_only.sum() >= 20:
        out["chan_caller_db_during_agent"] = _safe(np.median(vad_c.db[:n][a_only]))
        ns = ~binary_dilation(vad_c.speech[:n], iterations=3) & ~vad_a.speech[:n]
        if ns.sum() >= 5:
            out["chan_floor_rise_during_agent_db"] = _safe(
                np.median(vad_c.db[:n][a_only]) - np.median(vad_c.db[:n][ns]))
    d = np.diff(np.concatenate([[0], a_only.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    runs = sorted([(e - s, s, e) for s, e in zip(starts, ends) if e - s >= 100], reverse=True)[:6]
    if runs:
        sos = signal.butter(4, [300, 3400], btype="band", fs=SR, output="sos")
        L = 3200  # 400 ms max lag
        acc = np.zeros(L + 1)
        k = 0
        for _, s, e in runs:
            sa, se = s * HOP, min(e * HOP, len(call.agent))
            if se - sa < 4000:
                continue
            se = min(se, sa + 5 * SR)
            a = signal.sosfiltfilt(sos, call.agent[sa:se])
            c = signal.sosfiltfilt(sos, call.caller[sa:min(se + L, len(call.caller))])
            if len(c) < len(a) + 10:
                continue
            nfft = 1 << int(np.ceil(np.log2(len(c) + len(a))))
            xc = np.fft.irfft(np.fft.rfft(c, nfft) * np.conj(np.fft.rfft(a, nfft)), nfft)[:L + 1]
            norm = np.linalg.norm(a) * np.linalg.norm(c[:len(a)]) + EPS
            acc += xc / norm
            k += 1
        if k:
            acc /= k
            out["chan_echo_xcorr_max"] = _safe(np.abs(acc).max())
            out["chan_echo_lag_ms"] = _safe(np.argmax(np.abs(acc)) / SR * 1000)
    return out


def spectrogram_ui(P: np.ndarray, freqs: np.ndarray, max_cols: int = 1200, rows: int = 128) -> dict:
    n_bins, n = P.shape
    if n == 0:
        return {}
    usable = P[: (n_bins // rows) * rows]
    R = usable.reshape(rows, -1, n).mean(axis=1)
    step = max(1, int(np.ceil(n / max_cols)))
    if step > 1:
        pad = (-n) % step
        R = np.pad(R, ((0, 0), (0, pad)), constant_values=0).reshape(rows, -1, step).max(axis=2)
    D = 10 * np.log10(R + EPS)
    lo, hi = np.percentile(D, 3), np.percentile(D, 99.5)
    U = np.clip((D - lo) / max(hi - lo, 1e-3), 0, 1)
    U = (U[::-1] * 255).astype(np.uint8)
    return {"rows": rows, "cols": int(U.shape[1]), "hop_s": step * HOP_S,
            "fmax": float(freqs[(n_bins // rows) * rows - 1]),
            "data": base64.b64encode(np.ascontiguousarray(U).tobytes()).decode("ascii")}


# ----------------------------------------------------------------------------- entry point

def acoustic_features(call: Call, P: np.ndarray, freqs: np.ndarray, vad_c: Vad, vad_a: Vad | None,
                      want_ui: bool = True) -> AcousticResult:
    feats: dict = {}
    ui: dict = {}
    desc = frame_descriptors(P, freqs)
    feats["speech_seconds"] = _safe(vad_c.speech_seconds)
    feats["duration_s"] = _safe(call.duration)
    feats.update(ltas_features(P, freqs, vad_c.speech))
    f0f, f0ui = f0_features(call.caller, vad_c, want_track=want_ui)
    feats.update(f0f)
    ui.update(f0ui)
    feats.update(energy_features(vad_c))
    feats.update(rhythm_features(P, freqs, vad_c))
    feats.update(cepstral_features(P, freqs, vad_c.speech))
    feats.update(spectral_shape_features(desc, vad_c.speech))
    feats.update(floor_features(call, P, freqs, vad_c, desc))
    feats.update(cut_features(vad_c))
    bf, breaths = breath_features(vad_c, desc, call.duration)
    feats.update(bf)
    feats.update(formant_features(call.caller, vad_c, feats.get("f0_median_hz")))
    feats.update(hum_features(call.caller_raw))
    feats.update(channel_features(call, vad_c, vad_a))
    if want_ui:
        ui["breaths"] = [[round(s, 2), round(e, 2)] for s, e in breaths]
        ui["spectrogram"] = spectrogram_ui(P, freqs)
        ui["floor_db"] = _safe(vad_c.floor_db)
    return AcousticResult(features=feats, ui=ui)
