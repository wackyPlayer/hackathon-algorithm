"""WAV decoding, channel handling and resampling.

The scored endpoint receives a stereo 8 kHz WAV (channel 0 = caller, channel 1 = agent),
base64 encoded. Everything here is defensive: mono input, other sample rates, mu-law/A-law
WAVs, data-URI prefixes and whitespace inside the base64 payload are all accepted.
"""
from __future__ import annotations

import base64
import binascii
import io
import math
import re
import wave
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 8000


@dataclass
class Call:
    caller: np.ndarray          # float32 mono, DC removed, target_sr
    caller_raw: np.ndarray      # float32 mono, untouched (digital-silence / level features)
    agent: np.ndarray | None    # float32 mono or None if the file was mono
    sr: int
    duration: float
    sr_in: int
    n_channels_in: int
    fmt: str

    @property
    def has_agent(self) -> bool:
        return self.agent is not None and len(self.agent) > 0 and float(np.max(np.abs(self.agent))) > 1e-4


_DATA_URI = re.compile(r"^data:[^;]+;base64,", re.IGNORECASE)


def b64_to_bytes(s: str) -> bytes:
    s = s.strip()
    s = _DATA_URI.sub("", s)
    s = re.sub(r"\s+", "", s)
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    s = s + "=" * ((-len(s)) % 4)
    try:
        return base64.b64decode(s, validate=False)
    except (binascii.Error, ValueError):
        return base64.urlsafe_b64decode(s)


def looks_like_wav(data: bytes) -> bool:
    return len(data) > 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def decode_wav_bytes(data: bytes) -> tuple[np.ndarray, int, str]:
    """Return (samples[n, ch] float32 in [-1, 1], sample_rate, subtype)."""
    try:
        with sf.SoundFile(io.BytesIO(data)) as f:
            subtype = f.subtype
            x = f.read(dtype="float32", always_2d=True)
            return x, f.samplerate, subtype
    except Exception as exc:
        try:
            with wave.open(io.BytesIO(data)) as w:
                n_ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
                raw = w.readframes(n)
            if sw == 2:
                x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            elif sw == 1:
                x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif sw == 4:
                x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"unsupported sample width {sw}")
            return x.reshape(-1, n_ch), sr, f"PCM_{sw * 8}"
        except Exception:
            raise ValueError(f"could not decode WAV: {exc}") from exc


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    g = math.gcd(sr_in, sr_out)
    return resample_poly(x.astype(np.float64), sr_out // g, sr_in // g).astype(np.float32)


def load_call(data: bytes, target_sr: int = TARGET_SR, max_seconds: float | None = None) -> Call:
    x, sr, fmt = decode_wav_bytes(data)
    n_ch = x.shape[1]
    caller = x[:, 0]
    agent = x[:, 1] if n_ch > 1 else None
    if sr != target_sr:
        caller = resample(caller, sr, target_sr)
        agent = resample(agent, sr, target_sr) if agent is not None else None
    if max_seconds is not None:
        n = int(max_seconds * target_sr)
        caller = caller[:n]
        agent = agent[:n] if agent is not None else None
    return call_from_arrays(caller, agent, target_sr, sr_in=sr, n_channels_in=n_ch, fmt=fmt)


def call_from_arrays(caller: np.ndarray, agent: np.ndarray | None, sr: int = TARGET_SR,
                     sr_in: int | None = None, n_channels_in: int | None = None, fmt: str = "PCM_16") -> Call:
    caller_raw = np.ascontiguousarray(caller, dtype=np.float32)
    c = caller_raw - float(np.mean(caller_raw)) if len(caller_raw) else caller_raw
    a = None
    if agent is not None:
        a = np.ascontiguousarray(agent, dtype=np.float32)
        if len(a) < len(c):
            a = np.pad(a, (0, len(c) - len(a)))
        a = a[: len(c)]
        if len(a):
            a = a - float(np.mean(a))
    return Call(caller=c.astype(np.float32), caller_raw=caller_raw, agent=None if a is None else a.astype(np.float32),
                sr=sr, duration=len(c) / sr, sr_in=sr_in or sr,
                n_channels_in=n_channels_in if n_channels_in is not None else (2 if agent is not None else 1), fmt=fmt)


def to_wav_bytes(caller: np.ndarray, agent: np.ndarray | None = None, sr: int = TARGET_SR) -> bytes:
    """Encode float arrays as a 16-bit PCM WAV (stereo if agent given)."""
    if agent is None:
        data = np.asarray(caller, dtype=np.float32)[:, None]
    else:
        n = max(len(caller), len(agent))
        c = np.pad(np.asarray(caller, dtype=np.float32), (0, n - len(caller)))
        a = np.pad(np.asarray(agent, dtype=np.float32), (0, n - len(agent)))
        data = np.stack([c, a], axis=1)
    buf = io.BytesIO()
    sf.write(buf, np.clip(data, -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
