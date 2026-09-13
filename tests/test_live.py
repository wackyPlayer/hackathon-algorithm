"""Live-call turn taking over the WebSocket with a *noisy* microphone.

Regression: a room at -40 dBFS used to be read as continuous caller speech by a fixed -60 dBFS floor, so no
utterance ever ended, nothing was transcribed and the agent never answered. The agent's brain / voice are
disabled (canned lines, browser voice) and the transcriber is stubbed, so the test needs no network.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import backend.features.semantic as semantic  # noqa: E402
import backend.main as main_mod  # noqa: E402
from backend.config import settings  # noqa: E402
from bench.make_clips import _voice  # noqa: E402


@pytest.fixture
def offline_agent(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "elevenlabs_api_key", "")
    monkeypatch.setattr(settings, "live_answer_timeout_s", 60.0)
    monkeypatch.setattr(semantic, "transcribe_array", lambda x, model=None: "hola, buenas tardes, hablo por un cargo que no reconozco")


def _pcm(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def _stream(ws, x: np.ndarray, sent: int) -> int:
    for i in range(0, len(x) - 799, 800):
        ws.send_bytes(_pcm(x[i:i + 800]))
        sent += 800
    return sent


def _recv(ws, typ: str, limit: int = 60) -> dict:
    seen = []
    for _ in range(limit):
        m = ws.receive_json()
        seen.append(m["type"])
        if m["type"] == typ:
            return m
    raise AssertionError(f"no {typ!r} message; saw {seen}")


@pytest.mark.parametrize("seed_floor", [True, False])
def test_noisy_microphone_turn_taking(offline_agent, seed_floor):
    rng = np.random.default_rng(0)
    noise = lambda n: rng.standard_normal(n).astype(np.float32) * 10 ** (-40 / 20)   # -40 dBFS room
    with TestClient(main_mod.app) as c, c.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "start", "floor_dbfs": -40.0} if seed_floor else {"type": "start"})
        hello = _recv(ws, "hello")
        if seed_floor:
            assert hello["floor_db"] == -40.0
        greet = _recv(ws, "agent_say")
        assert greet["kind"] == "greeting" and greet["step"] == 1 and greet["audio_b64"] is None
        # 2.5 s of room noise while the greeting "plays" in the browser, then the browser reports its end
        sent = _stream(ws, noise(8000 * 25 // 10), 0)
        t = sent / 8000
        ws.send_json({"type": "agent", "event": "start", "t": t - 2.0, "text": greet["text"], "kind": greet["kind"]})
        ws.send_json({"type": "agent", "event": "end", "t": t, "text": greet["text"], "kind": greet["kind"]})
        # the caller answers: 1 s pause, 2.5 s of voice at -25 dBFS on top of the noise, 2 s of noise
        sp = _voice(2.5, 140, rng, jitter=0.03, vibrato=0.02)
        sp = sp / (np.sqrt(np.mean(sp ** 2)) + 1e-9) * 10 ** (-25 / 20)
        sent = _stream(ws, noise(8000), sent)
        sent = _stream(ws, sp.astype(np.float32) + noise(len(sp)), sent)
        sent = _stream(ws, noise(8000 * 2), sent)
        vad = _recv(ws, "vad")
        assert vad["speaking"] is True and vad["awaiting"] is True and -45 < vad["floor_db"] < -35
        said = _recv(ws, "caller_said")
        assert "hola" in said["text"]
        nxt = _recv(ws, "agent_say")
        assert nxt["kind"] == "question" and nxt["step"] == 2
        ws.send_json({"type": "stop"})
        fin = _recv(ws, "final", limit=120)
        live = fin["result"]["live"]
        assert live["utterances"] >= 1 and len(live["caller_turns"]) == 1 and -45 < live["floor_db"] < -35


def test_unintelligible_answer_is_asked_again(offline_agent, monkeypatch):
    """Speech that the transcriber cannot read gets a 'please repeat' line instead of silence."""
    monkeypatch.setattr(semantic, "transcribe_array", lambda x, model=None: "")
    rng = np.random.default_rng(1)
    noise = lambda n: rng.standard_normal(n).astype(np.float32) * 10 ** (-55 / 20)
    with TestClient(main_mod.app) as c, c.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "start", "floor_dbfs": -55.0})
        _recv(ws, "hello")
        greet = _recv(ws, "agent_say")
        sent = _stream(ws, noise(8000 * 2), 0)
        t = sent / 8000
        ws.send_json({"type": "agent", "event": "start", "t": t - 1.5, "text": greet["text"], "kind": greet["kind"]})
        ws.send_json({"type": "agent", "event": "end", "t": t, "text": greet["text"], "kind": greet["kind"]})
        sp = _voice(2.0, 120, rng, jitter=0.03)
        sp = sp / (np.sqrt(np.mean(sp ** 2)) + 1e-9) * 10 ** (-24 / 20)
        sent = _stream(ws, noise(4000), sent)
        sent = _stream(ws, sp.astype(np.float32) + noise(len(sp)), sent)
        sent = _stream(ws, noise(8000 * 2), sent)
        said = _recv(ws, "caller_said")
        assert "no se entendi" in said["text"]
        rep = _recv(ws, "agent_say")
        assert rep["kind"] == "repeat" and "repetir" in rep["text"]
        ws.send_json({"type": "stop"})
        _recv(ws, "final", limit=120)
