"""Generate synthetic stereo 8 kHz test calls (no dataset needed) for smoke tests and CI.

    python -m bench.make_clips --out clips/ --n 4

"human-like" clips: pitch with jitter/vibrato, loudness variation, breath noises, a live noise floor
with slow drift, room decay tails, fast and variable response latency, back-channels.
"bot-like" clips: flat pitch, constant loudness, digital silence between phrases, hard cuts,
band-limited spectrum, long and constant response latency.
These are caricatures used only to exercise the pipeline end to end; they are not training data.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.audio import to_wav_bytes  # noqa: E402

SR = 8000


def _voice(dur, f0, rng, jitter=0.02, vibrato=0.0, tilt=1.0, shimmer=0.1):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f0 * (1 + vibrato * np.sin(2 * np.pi * 5 * t)) * (1 + jitter * np.cumsum(rng.standard_normal(n)) / np.sqrt(np.arange(1, n + 1)))
    phase = 2 * np.pi * np.cumsum(f) / SR
    x = np.zeros(n)
    for h in range(1, 12):
        x += (1.0 / h ** tilt) * np.sin(h * phase)
    # syllabic amplitude modulation ~4.5 Hz with some randomness
    env = 0.55 + 0.45 * np.clip(np.sin(2 * np.pi * (4.5 + rng.uniform(-0.5, 0.5)) * t + rng.uniform(0, 6)), 0, None)
    env *= 1 + shimmer * rng.standard_normal(n).cumsum() / np.sqrt(np.arange(1, n + 1))
    x *= env
    # simple formant-ish colouring
    from scipy.signal import lfilter
    x = lfilter([1.0], [1.0, -1.6 * np.cos(2 * np.pi * 700 / SR), 0.8], x)
    x = lfilter([1.0], [1.0, -1.7 * np.cos(2 * np.pi * 1400 / SR), 0.85], x)
    x /= np.max(np.abs(x)) + 1e-9
    return x.astype(np.float32)


def _bandlimit(x, lo=300, hi=3400, order=4):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, [lo, hi], btype="band", fs=SR, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def make_call(kind: str, seed: int, seconds: float = 90.0):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    caller = np.zeros(n, np.float32)
    agent = np.zeros(n, np.float32)
    t = 1.0
    f0 = rng.uniform(95, 230)
    turns = []
    while t < seconds - 8:
        # agent turn (clean TTS-like tone bursts)
        d_a = rng.uniform(3, 6)
        a = _voice(d_a, 150, rng, jitter=0.002, vibrato=0.0, tilt=1.2, shimmer=0.0) * 0.3
        a = _bandlimit(a, 200, 3600)
        i = int(t * SR)
        agent[i:i + len(a)] = a[: n - i]
        t += d_a
        # response latency
        if kind == "human":
            t += max(0.15, rng.normal(0.9, 0.5))
        else:
            t += rng.normal(2.8, 0.25)
        d_c = rng.uniform(2.5, 7) if kind == "human" else rng.uniform(5, 10)
        if kind == "human":
            c = _voice(d_c, f0 * rng.uniform(0.9, 1.1), rng, jitter=0.03, vibrato=0.02, tilt=1.0, shimmer=0.15)
            c *= rng.uniform(0.25, 0.6)
            # breath before onset
            i = int(t * SR)
            br = rng.standard_normal(int(0.25 * SR)).astype(np.float32) * 0.02
            br = _bandlimit(br, 500, 3000)
            caller[max(i - len(br), 0):i] += br[: i]
            # intra-turn pause
            k = int(len(c) * rng.uniform(0.3, 0.7))
            c[k:k + int(0.35 * SR)] = 0
        else:
            c = _voice(d_c, f0, rng, jitter=0.001, vibrato=0.0, tilt=1.1, shimmer=0.0) * 0.45
            c = _bandlimit(c, 250, 3300, order=8)
        i = int(t * SR)
        caller[i:i + len(c)] += c[: n - i]
        turns.append((t, t + d_c))
        t += d_c
        if kind == "human" and rng.uniform() < 0.4:  # back-channel during next agent turn
            j = int((t + 1.0) * SR)
            bc = _voice(0.3, f0, rng, jitter=0.03) * 0.3
            caller[j:j + len(bc)] += bc[: n - j]
        t += rng.uniform(0.6, 1.5)
    if kind == "human":
        # live noise floor with slow level drift + room decay (simple exponential tail)
        floor = rng.standard_normal(n).astype(np.float32) * 0.004 * (1 + 0.5 * np.sin(2 * np.pi * 0.05 * np.arange(n) / SR))
        floor = _bandlimit(floor, 100, 3800)
        from scipy.signal import lfilter
        caller = lfilter([1.0], [1.0, -0.6], caller).astype(np.float32) * 0.8
        caller = _bandlimit(caller + floor, 250, 3700)
    else:
        # digital silence stays exactly zero; add a faint stationary comfort noise only where speech is
        pass
    return caller, agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="clips")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=90)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for k in range(a.n):
        for kind in ("human", "bot"):
            c, ag = make_call(kind, seed=100 + k, seconds=a.seconds)
            name = f"{kind}_{k:02d}.wav"
            open(os.path.join(a.out, name), "wb").write(to_wav_bytes(c, ag))
            rows.append((name, "human" if kind == "human" else "synthetic"))
    with open(os.path.join(a.out, "labels.csv"), "w") as fh:
        fh.write("file,label\n" + "".join(f"{n},{l}\n" for n, l in rows))
    print(f"wrote {len(rows)} clips to {a.out}")


if __name__ == "__main__":
    main()
