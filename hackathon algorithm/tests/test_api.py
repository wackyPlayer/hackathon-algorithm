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
    assert 0.5 <= j["confidence"] <= 1.0
    assert dt < 10


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
    """The bot-like caricature must score more synthetic than the human-like one."""
    ph = client.post("/analyze?ui=0", json={"audio": _b64(clips["human"])}).json()["p_synthetic"]
    pb = client.post("/analyze?ui=0", json={"audio": _b64(clips["bot"])}).json()["p_synthetic"]
    assert pb > ph
