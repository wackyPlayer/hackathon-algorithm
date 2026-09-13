"""End-to-end API tests with synthetic clips (no dataset required).  Run: pytest -q"""
from __future__ import annotations

import base64
import json
import os
import sys
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.audio import to_wav_bytes  # noqa: E402
from backend.main import app  # noqa: E402
from bench.make_clips import make_call  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def clips():
    hc, ha = make_call("human", seed=1, seconds=60)
    bc, ba = make_call("bot", seed=2, seconds=60)
    return {"human": to_wav_bytes(hc, ha), "bot": to_wav_bytes(bc, ba), "mono": to_wav_bytes(hc, None)}


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_detect_json_contract(client, clips):
    t0 = time.time()
    r = client.post("/detect", json={"audio": _b64(clips["bot"])})
    dt = time.time() - t0
    assert r.status_code == 200
    j = r.json()
    assert set(j.keys()) == {"is_synthetic", "confidence"}
    assert isinstance(j["is_synthetic"], bool)
    # confidence is p(synthetic): monotone, usable for ranking and calibration, and always agreeing
    # with the boolean. It is also capped, so we never claim to be certain.
    assert 0.0 <= j["confidence"] <= 1.0
    assert j["is_synthetic"] == (j["confidence"] > 0.5)
    assert j["confidence"] <= 0.995
    assert dt < 10


def test_detect_confidence_never_saturates(client, clips):
    """A verdict reported at exactly 1.0 cannot be wrong, and this one can."""
    for name in ("bot", "human"):
        j = client.post("/detect", json={"audio": _b64(clips[name])}).json()
        assert 0.005 <= j["confidence"] <= 0.995


def test_detect_rejects_implausible_sample_rate(client, clips):
    """The WAV header is attacker-controlled and resampling cost scales with it, so a nonsense rate is
    refused up front instead of turning a short clip into minutes of CPU."""
    import struct
    raw = bytearray(clips["bot"])
    i = raw.find(b"fmt ")
    struct.pack_into("<I", raw, i + 12, 200)
    t0 = time.time()
    r = client.post("/detect", data=bytes(raw), headers={"content-type": "application/octet-stream"})
    assert r.status_code == 400
    assert time.time() - t0 < 3


def test_detect_no_evidence_does_not_accuse(client):
    """With too little caller speech to judge, the tie breaks toward human: we do not accuse on silence."""
    import io

    import numpy as np
    import soundfile as sf
    x = (np.random.default_rng(0).standard_normal(1600) * 0.01).astype("float32")
    b = io.BytesIO()
    sf.write(b, np.stack([x, x], axis=1), 8000, format="WAV", subtype="PCM_16")
    j = client.post("/detect", json={"audio": _b64(b.getvalue())}).json()
    assert j["is_synthetic"] is False


@pytest.mark.parametrize("key", ["wav", "audio_base64", "data", "clip"])
def test_detect_other_json_keys(client, clips, key):
    r = client.post("/detect", json={key: _b64(clips["human"])})
    assert r.status_code == 200 and "is_synthetic" in r.json()


def test_detect_nested_json_and_data_uri(client, clips):
    r = client.post("/detect", json={"call": {"id": "x", "audio": "data:audio/wav;base64," + _b64(clips["human"])}})
    assert r.status_code == 200


def test_detect_raw_wav_body(client, clips):
    r = client.post("/detect", content=clips["human"], headers={"Content-Type": "audio/wav"})
    assert r.status_code == 200 and "is_synthetic" in r.json()


def test_detect_bare_base64_body(client, clips):
    r = client.post("/detect", content=_b64(clips["bot"]), headers={"Content-Type": "text/plain"})
    assert r.status_code == 200


def test_detect_multipart(client, clips):
    r = client.post("/detect", files={"file": ("call.wav", clips["bot"], "audio/wav")})
    assert r.status_code == 200


def test_detect_mono_is_accepted(client, clips):
    r = client.post("/detect", json={"audio": _b64(clips["mono"])})
    assert r.status_code == 200


def test_detect_bad_input(client):
    assert client.post("/detect", json={"audio": "not-audio"}).status_code == 400
    assert client.post("/detect", content=b"").status_code == 400


def test_analyze_breakdown(client, clips):
    r = client.post("/analyze", files={"file": ("call.wav", clips["human"], "audio/wav")})
    assert r.status_code == 200
    j = r.json()
    for k in ("aspects", "attack_profile", "events", "timeline", "features", "signals", "ui", "timing"):
        assert k in j
    assert "turn_taking" in j["aspects"]
    assert j["ui"]["spectrogram"]["rows"] == 128
    assert j["timing"]["total_s"] < 10


def test_synthetic_caricatures_are_ranked(client, clips):
    """The bot-like caricature must score more synthetic than the human-like one on the interpretable
    aspect layer. The caricatures are tone generators, far outside the domain of the trained model (which is
    fit to real and TTS voices), so its probability is only checked for being a valid, evidence-weighted value."""
    rh = client.post("/analyze?ui=0", json={"audio": _b64(clips["human"])}).json()
    rb = client.post("/analyze?ui=0", json={"audio": _b64(clips["bot"])}).json()
    assert rb["signals"]["heuristic_p"] > rh["signals"]["heuristic_p"]
    assert rb["aspects"]["turn_taking"]["score"] > rh["aspects"]["turn_taking"]["score"]
    for r in (rh, rb):
        assert 0.0 <= r["p_synthetic"] <= 1.0 and 0.0 <= r["confidence"] <= 1.0
        assert r["signals"]["acoustic_p"] is None or 0.0 <= r["signals"]["acoustic_p"] <= 1.0


def test_miccheck_reports_quality(client):
    """A quiet-then-speech PCM sample gets a grade, metrics and client-flag warnings."""
    import numpy as np
    from bench.make_clips import _voice
    rng = np.random.default_rng(0)
    x = np.zeros(8000 * 5, np.float32)
    v = _voice(2.5, 140, rng, jitter=0.03, vibrato=0.02) * 0.2
    x[16000:16000 + len(v)] = v
    x += rng.standard_normal(len(x)).astype(np.float32) * 10 ** (-70 / 20)
    pcm = (x * 32767).astype("<i2").tobytes()
    r = client.post("/miccheck?ns=1&agc=1&sr=48000&device=test", content=pcm, headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 200
    j = r.json()
    assert j["quality"] in ("good", "fair", "poor")
    assert j["metrics"]["speech_seconds"] > 1.0 and j["metrics"]["snr_db"] > 20
    risks = {w["risk"] for w in j["warnings"]}
    assert "false_positive" in risks  # noise suppression + AGC flags
    assert client.post("/miccheck", content=b"\x00\x00" * 800, headers={"Content-Type": "application/octet-stream"}).status_code == 400
