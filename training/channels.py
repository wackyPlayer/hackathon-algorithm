"""Channel simulation: the paths a caller's voice can travel before it reaches us.

The detector's job is to recognise a *voice*, not a *recording setup*. A model trained on one dataset
learns the dataset's channel as well as its voices, and then fails the moment the channel changes -- which
is exactly what happened here: a synthetic voice played into a laptop microphone was scored as human,
because a microphone in a room has none of the telephony fingerprints (sharp 3.4 kHz band edge, 300 Hz
high-pass, mu-law quantisation, dead-flat noise floor) that the training calls all shared.

So this module defines the channel families explicitly, and training applies them to *both* classes. Any
cue that survives has to be a property of the voice, because the channel no longer correlates with the label.

    telephony   PSTN / VoIP: band-limited 300-3400 Hz, mu-law companded, near-constant noise floor
    mobile      cellular: band-limited but wider, codec-style spectral holes, level drift, packet gaps
    mic_room    a loudspeaker playing into a microphone across a room: full band up to Nyquist, room
                reverb, real ambient noise at 12-30 dB SNR, AGC pumping, NO companding   <- the failure case
    voip_wide   a wideband softphone leg (Opus-like): full band, light noise, no mu-law
    handset     a phone held near a speaker: band-limited *and* reverberant, the replay attack

`apply(name, x, sr, rng)` takes a dry mono track at `sr` and returns 8 kHz audio plus a description of what
was done, so every augmented row in the feature table can say which channel produced it.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, sosfiltfilt

SR = 8000
FAMILIES = ("telephony", "mobile", "mic_room", "voip_wide", "handset")


# --------------------------------------------------------------------------- small helpers

def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12))


def set_level(x: np.ndarray, target_db: float, active: np.ndarray | None = None) -> np.ndarray:
    a = active if active is not None and active.any() else np.abs(x) > 1e-4
    cur = rms_db(x[a]) if a.any() else rms_db(x)
    return (x * (10 ** ((target_db - cur) / 20.0))).astype(np.float32)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    w = rng.standard_normal(n)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    p = lfilter(b, a, w)
    return (p / (np.std(p) + 1e-9)).astype(np.float32)


def mulaw_roundtrip(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1.0) * 127.5) / 127.5 - 1.0
    return (np.sign(q) * (np.expm1(np.abs(q) * np.log1p(mu)) / mu)).astype(np.float32)


def tilt(x: np.ndarray, db_per_oct: float) -> np.ndarray:
    if abs(db_per_oct) < 0.05:
        return x
    a = float(np.clip(0.6 * np.tanh(db_per_oct / 4.0), -0.85, 0.85))
    y = lfilter([1.0, -a], [1.0], x)
    return (y * (np.std(x) / (np.std(y) + 1e-9))).astype(np.float32)


def band(x: np.ndarray, lo: float, hi: float, order: int, sr: int = SR) -> np.ndarray:
    hi = min(hi, sr / 2 * 0.98)
    if lo <= 0:
        sos = butter(order, hi, btype="low", fs=sr, output="sos")
    else:
        sos = butter(order, [lo, hi], btype="band", fs=sr, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def room_ir(rng: np.random.Generator, sr: int, rt60: float, n_early: int = 6) -> np.ndarray:
    """Exponentially decaying noise with a few discrete early reflections -- a cheap but honest
    small-room impulse response. The early reflections are what make a replay sound 'in a room'."""
    n = max(int(rt60 * sr), 16)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * t / rt60).astype(np.float32)
    ir[0] = 1.0
    for _ in range(n_early):
        i = int(rng.uniform(0.004, 0.06) * sr)
        if 0 < i < n:
            ir[i] += rng.uniform(0.2, 0.7) * rng.choice([-1.0, 1.0])
    return (ir / (np.linalg.norm(ir) + 1e-9)).astype(np.float32)


def convolve_wet(x: np.ndarray, ir: np.ndarray, wet_db: float) -> np.ndarray:
    # fftconvolve, not np.convolve: a 2-minute call against a 0.5 s impulse response is ~10^10 multiply-adds
    # the direct way and a fraction of a second this way.
    wet = fftconvolve(x, ir)[: len(x)].astype(np.float32)
    wet = wet / (np.std(wet) + 1e-9) * (np.std(x) + 1e-9)
    g = 10 ** (wet_db / 20.0)
    return ((x + g * wet) / (1.0 + g)).astype(np.float32)


def agc(x: np.ndarray, rng: np.random.Generator, sr: int) -> np.ndarray:
    """Slow automatic gain control: the gain chases the envelope, so quiet passages are lifted and the
    natural loudness range of a talker is squashed. Laptop and headset microphones do this by default."""
    win = max(int(rng.uniform(0.3, 1.2) * sr), 1)
    sq = np.square(x, dtype=np.float64)
    c = np.cumsum(np.concatenate([[0.0], sq]))
    half = win // 2
    lo = np.clip(np.arange(len(sq)) - half, 0, len(sq))
    hi = np.clip(np.arange(len(sq)) + win - half, 0, len(sq))
    env = np.sqrt((c[hi] - c[lo]) / np.maximum(hi - lo, 1) + 1e-12)
    target = float(np.percentile(env, 80)) + 1e-9
    strength = rng.uniform(0.3, 0.8)
    g = (target / env) ** strength
    return (x * np.clip(g, 0.25, 6.0)).astype(np.float32)


def ambient(n: int, rng: np.random.Generator, level_db: float, sr: int) -> np.ndarray:
    """Room tone: pink noise that drifts, plus occasional short events (keys, a door, traffic).
    Unlike a telephony floor this is *non-stationary*, which is the cue a channel-overfitted model
    reads as 'a real person in a real place'."""
    nz = pink_noise(n, rng) * (10 ** (level_db / 20.0))
    t = np.arange(n) / sr
    drift = 10 ** (rng.uniform(1.5, 5.0) * np.sin(2 * np.pi * rng.uniform(0.02, 0.15) * t + rng.uniform(0, 6.3)) / 20.0)
    nz = nz * drift.astype(np.float32)
    for _ in range(int(rng.integers(2, 9))):
        i = int(rng.integers(0, max(n - sr, 1)))
        L = int(rng.uniform(0.08, 0.8) * sr)
        nz[i:i + L] *= rng.uniform(1.8, 5.0)
    return nz.astype(np.float32)


def resample_to(x: np.ndarray, sr: int, target: int = SR) -> np.ndarray:
    if sr == target:
        return x.astype(np.float32)
    from backend.audio import resample
    return resample(x.astype(np.float32), sr, target)


# --------------------------------------------------------------------------- the families

def _telephony(x: np.ndarray, rng, sr, desc) -> np.ndarray:
    x = resample_to(x, sr)
    x = tilt(x, rng.uniform(-3.0, 3.0))
    lo, hi, order = rng.uniform(180, 340), rng.uniform(3200, 3600), int(rng.choice([6, 8]))
    x = band(x, lo, hi, order)
    desc["band"] = f"{lo:.0f}-{hi:.0f}/{order}"
    lvl = desc["level_dbfs"] - rng.uniform(42.0, 62.0)
    nz = band(rng.standard_normal(len(x)).astype(np.float32), lo, hi, order)
    x = x + nz / (np.std(nz) + 1e-9) * (10 ** (lvl / 20.0))
    desc["floor_db"] = round(lvl, 1)
    desc["mulaw"] = True
    return mulaw_roundtrip(x)


def _mobile(x: np.ndarray, rng, sr, desc) -> np.ndarray:
    x = resample_to(x, sr)
    x = tilt(x, rng.uniform(-4.0, 2.0))
    lo, hi = rng.uniform(120, 260), rng.uniform(3400, 3850)
    x = band(x, lo, hi, int(rng.choice([4, 6])))
    desc["band"] = f"{lo:.0f}-{hi:.0f}"
    # codec-style level drift and the odd dropped packet
    t = np.arange(len(x)) / SR
    x = x * (10 ** (rng.uniform(1.0, 3.5) * np.sin(2 * np.pi * rng.uniform(0.05, 0.4) * t) / 20.0)).astype(np.float32)
    drops = int(rng.integers(0, 5))
    for _ in range(drops):
        i = int(rng.integers(0, max(len(x) - SR, 1)))
        x[i:i + int(rng.uniform(0.02, 0.09) * SR)] = 0.0
    desc["packet_drops"] = drops
    lvl = desc["level_dbfs"] - rng.uniform(34.0, 52.0)
    x = x + pink_noise(len(x), rng) * (10 ** (lvl / 20.0))
    desc["floor_db"] = round(lvl, 1)
    if rng.uniform() < 0.5:
        desc["mulaw"] = True
        x = mulaw_roundtrip(x)
    return x


def _mic_room(x: np.ndarray, rng, sr, desc) -> np.ndarray:
    """A loudspeaker playing a voice across a room into a microphone, captured by a browser at 48 kHz
    and downsampled to 8 kHz. This is the reported failure case: no telephony band edge, no companding,
    a live room behind the voice."""
    # loudspeaker: rolls off below ~120 Hz, a couple of cabinet resonances
    x = band(x, rng.uniform(90, 200), min(sr / 2 * 0.95, 14000), 2, sr=sr)
    for _ in range(int(rng.integers(1, 4))):
        f0 = rng.uniform(300, 3000)
        q = rng.uniform(2.0, 8.0)
        sos = butter(2, [max(f0 - f0 / q, 40), min(f0 + f0 / q, sr / 2 * 0.95)], btype="band", fs=sr, output="sos")
        x = (x + rng.uniform(-0.35, 0.5) * sosfiltfilt(sos, x)).astype(np.float32)
    # the room
    rt60 = rng.uniform(0.15, 0.6)
    x = convolve_wet(x, room_ir(rng, sr, rt60), rng.uniform(-12.0, -2.0))
    desc["rt60"] = round(rt60, 2)
    # microphone: gentle tilt, then the browser's 48k -> 8k anti-alias edge just under 4 kHz
    x = tilt(x, rng.uniform(-2.0, 3.0))
    x = resample_to(x, sr)
    x = set_level(x, desc["level_dbfs"])
    snr = rng.uniform(12.0, 30.0)
    lvl = desc["level_dbfs"] - snr
    x = x + ambient(len(x), rng, lvl, SR)
    desc["snr_db"] = round(snr, 1)
    desc["floor_db"] = round(lvl, 1)
    if rng.uniform() < 0.7:
        x = agc(x, rng, SR)
        desc["agc"] = True
    desc["band"] = "wideband(no telephony edge)"
    desc["mulaw"] = False
    return x


def _voip_wide(x: np.ndarray, rng, sr, desc) -> np.ndarray:
    x = band(x, rng.uniform(50, 120), min(sr / 2 * 0.95, 7500), 2, sr=sr)
    x = resample_to(x, sr)
    x = tilt(x, rng.uniform(-2.0, 2.0))
    lvl = desc["level_dbfs"] - rng.uniform(45.0, 70.0)
    x = x + rng.standard_normal(len(x)).astype(np.float32) * (10 ** (lvl / 20.0))
    desc["band"] = "wideband"
    desc["floor_db"] = round(lvl, 1)
    desc["mulaw"] = False
    return x


def _handset(x: np.ndarray, rng, sr, desc) -> np.ndarray:
    """Replay: a phone or laptop speaker held up to a handset microphone -- reverberant *and*
    band-limited, and then companded by the phone line it is fed into."""
    rt60 = rng.uniform(0.2, 0.7)
    x = convolve_wet(x, room_ir(rng, sr, rt60), rng.uniform(-9.0, 0.0))
    desc["rt60"] = round(rt60, 2)
    x = resample_to(x, sr)
    lo, hi = rng.uniform(200, 400), rng.uniform(3000, 3500)
    x = band(x, lo, hi, 6)
    desc["band"] = f"{lo:.0f}-{hi:.0f}"
    lvl = desc["level_dbfs"] - rng.uniform(22.0, 40.0)
    x = x + ambient(len(x), rng, lvl, SR)
    desc["floor_db"] = round(lvl, 1)
    if rng.uniform() < 0.6:
        x = agc(x, rng, SR)
        desc["agc"] = True
    desc["mulaw"] = True
    return mulaw_roundtrip(x)


_FN = {"telephony": _telephony, "mobile": _mobile, "mic_room": _mic_room,
       "voip_wide": _voip_wide, "handset": _handset}


def apply(name: str, x: np.ndarray, rng: np.random.Generator, sr: int = SR,
          speech_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Send `x` (mono, `sr` Hz, dry) down the named channel. Returns 8 kHz audio + a description."""
    if name not in _FN:
        raise ValueError(f"unknown channel family {name!r}; expected one of {FAMILIES}")
    x = np.asarray(x, dtype=np.float32)
    desc: dict = {"family": name, "level_dbfs": round(float(rng.uniform(-30.0, -13.0)), 1)}
    x = set_level(x, desc["level_dbfs"], speech_mask if (speech_mask is not None and len(speech_mask) == len(x)) else None)
    y = _FN[name](x, rng, sr, desc)
    y = set_level(y, desc["level_dbfs"])
    return np.clip(y, -1.0, 1.0).astype(np.float32), desc


def random_family(rng: np.random.Generator, weights: dict | None = None) -> str:
    w = weights or {"telephony": 0.30, "mobile": 0.22, "mic_room": 0.24, "voip_wide": 0.12, "handset": 0.12}
    names = list(w)
    p = np.array([w[n] for n in names], dtype=np.float64)
    return str(rng.choice(names, p=p / p.sum()))
