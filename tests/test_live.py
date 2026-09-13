"""Live-call turn taking over the WebSocket with a *noisy* microphone.

Regression: a room at -40 dBFS used to be read as continuous caller speech by a fixed -60 dBFS floor, so no
utterance ever ended, nothing was transcribed and the agent never answered. The agent's brain / voice are
disabled (canned lines, browser voice) and the transcriber is stubbed, so the tests need no network.
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
from backend.live_agent import STEPS, GeminiBrain, level_for, render_context  # noqa: E402
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


def _recv(ws, typ: str, limit: int = 80, where=None) -> dict:
    seen = []
    for _ in range(limit):
        m = ws.receive_json()
        seen.append(m["type"] + (":" + m.get("state", "") if m["type"] == "vad" else ""))
        if m["type"] == typ and (where is None or where(m)):
            return m
    raise AssertionError(f"no {typ!r} message; saw {seen}")


def _speech(rng, seconds: float, level_db: float) -> np.ndarray:
    sp = _voice(seconds, 140, rng, jitter=0.03, vibrato=0.02)
    return (sp / (np.sqrt(np.mean(sp ** 2)) + 1e-9) * 10 ** (level_db / 20)).astype(np.float32)


def _greeting(ws, noise, seconds: float = 2.5):
    _recv(ws, "hello")
    greet = _recv(ws, "agent_say")
    assert greet["kind"] == "greeting" and greet["step"] == 1 and greet["audio_b64"] is None
    sent = _stream(ws, noise(int(8000 * seconds)), 0)
    t = sent / 8000
    ws.send_json({"type": "agent", "event": "start", "t": t - 2.0, "text": greet["text"], "kind": greet["kind"]})
    ws.send_json({"type": "agent", "event": "end", "t": t, "text": greet["text"], "kind": greet["kind"]})
    return sent


@pytest.mark.parametrize("seed_floor", [True, False])
def test_noisy_microphone_turn_taking(offline_agent, seed_floor):
    rng = np.random.default_rng(0)
    noise = lambda n: rng.standard_normal(n).astype(np.float32) * 10 ** (-40 / 20)   # -40 dBFS room
    with TestClient(main_mod.app) as c, c.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "start", "floor_dbfs": -40.0} if seed_floor else {"type": "start"})
        sent = _greeting(ws, noise)
        # the caller answers: 1 s pause, 2.5 s of voice at -25 dBFS on top of the noise, 2 s of noise
        sp = _speech(rng, 2.5, -25)
        sent = _stream(ws, noise(8000), sent)
        sent = _stream(ws, sp + noise(len(sp)), sent)
        sent = _stream(ws, noise(8000 * 2), sent)
        vad = _recv(ws, "vad", where=lambda m: m["state"] == "talking" and m["speaking"])
        assert vad["awaiting"] is True and -45 < vad["floor_db"] < -35 and vad["mod_db"] >= 3
        said = _recv(ws, "caller_said")
        assert "hola" in said["text"]
        nxt = _recv(ws, "agent_say")
        assert nxt["kind"] == "question" and nxt["step"] == 2
        ws.send_json({"type": "stop"})
        fin = _recv(ws, "final", limit=150)
        live = fin["result"]["live"]
        assert live["utterances"] >= 1 and len(live["caller_turns"]) == 1 and -45 < live["floor_db"] < -35


def test_steady_noise_is_not_a_turn(offline_agent):
    """A fan / traffic / gain step raises the level without the syllabic swing of speech: it must neither open a
    caller turn nor keep one from ending, and it is reported to the browser as background noise."""
    rng = np.random.default_rng(3)
    noise = lambda n: rng.standard_normal(n).astype(np.float32) * 10 ** (-40 / 20)
    loud = lambda n: rng.standard_normal(n).astype(np.float32) * 10 ** (-26 / 20)   # +14 dB, steady
    with TestClient(main_mod.app) as c, c.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "start", "floor_dbfs": -40.0})
        sent = _greeting(ws, noise)
        sent = _stream(ws, noise(8000), sent)
        sent = _stream(ws, loud(8000 * 3), sent)                       # 3 s of steady loud noise
        sent = _stream(ws, noise(8000 * 2), sent)
        # Look at a frame from *inside* the steady stretch, not the step itself: at the transition the
        # 0.8 s window straddles both levels, so its p90-p10 is the size of the step (~14 dB) and says
        # nothing about whether the noise is steady.
        vad = _recv(ws, "vad", where=lambda m: m["state"] == "noise" and m["mod_db"] < 3)
        assert vad["speaking"] is False
        # now real speech on top of the quieter floor is still heard
        sp = _speech(rng, 2.5, -24)
        sent = _stream(ws, sp + noise(len(sp)), sent)
        sent = _stream(ws, noise(8000 * 2), sent)
        _recv(ws, "vad", where=lambda m: m["state"] == "talking" and m["speaking"])
        said = _recv(ws, "caller_said")
        assert "hola" in said["text"]
        ws.send_json({"type": "stop"})
        fin = _recv(ws, "final", limit=150)
        live = fin["result"]["live"]
        assert live["utterances"] == 1 and len(live["caller_turns"]) == 1


def test_unintelligible_answer_is_asked_again(offline_agent, monkeypatch):
    """Speech that the transcriber cannot read gets a 'please repeat' line instead of silence."""
    monkeypatch.setattr(semantic, "transcribe_array", lambda x, model=None: "")
    rng = np.random.default_rng(1)
    noise = lambda n: rng.standard_normal(n).astype(np.float32) * 10 ** (-55 / 20)
    with TestClient(main_mod.app) as c, c.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "start", "floor_dbfs": -55.0})
        sent = _greeting(ws, noise, 2.0)
        sp = _speech(rng, 2.0, -24)
        sent = _stream(ws, noise(4000), sent)
        sent = _stream(ws, sp + noise(len(sp)), sent)
        sent = _stream(ws, noise(8000 * 2), sent)
        said = _recv(ws, "caller_said")
        assert "no se entendi" in said["text"]
        rep = _recv(ws, "agent_say")
        assert rep["kind"] == "repeat" and "repetir" in rep["text"]
        ws.send_json({"type": "stop"})
        _recv(ws, "final", limit=150)


def test_call_average_escalates_the_agent(offline_agent):
    """The *average* rolling score (not the latest) drives the level; above 0.70 the challenge step is inserted."""
    sess = main_mod.LiveSession(ws=None, analyzer=None)
    base = {"confidence": 0.9, "evidence_level": 1.0, "aspects": {"a": {"label": "Turn-taking behaviour", "score": 0.9, "weight": 1.3},
                                                                 "b": {"label": "Breathing", "score": 0.4, "weight": 0.4}}}
    for p in (0.95, 0.30, 0.40):
        sess.note_result({**base, "p_synthetic": p})
    assert abs(sess.p_avg - 0.55) < 1e-6 and not sess.escalated() and level_for(sess.p_avg) == "sospechoso"
    sess.step = 4                              # silence step -> next
    sess.advance()
    assert STEPS[sess.step]["kind"] == "question"          # challenge skipped
    sess.note_result({**base, "p_synthetic": 0.99, "evidence_level": 0.1})   # too little evidence: ignored
    for p in (0.98, 0.99, 0.99, 0.99):
        sess.note_result({**base, "p_synthetic": p})
    assert sess.p_avg > 0.70 and sess.escalated() and level_for(sess.p_avg) == "alto"
    sess.step = 4
    sess.advance()
    assert STEPS[sess.step]["kind"] == "challenge" and STEPS[sess.step].get("optional") == "escalated"
    ctx = sess.detector_context()
    assert ctx["cues"][0]["label"] == "Turn-taking behaviour" and ctx["n"] == 7
    text = render_context(ctx)
    assert "alto" in text and "promedio de la llamada" in text and "nunca lo menciones" in text
    # the brain (offline -> canned) still builds the score-aware prompt
    brain = GeminiBrain()
    line = brain.line(sess.step, "siete cuatro dos nueve uno", ctx)
    assert "orden inverso" in line and "Nivel de exigencia: alto" in brain.history[-2]["parts"][0]["text"]
    assert render_context(None).endswith("normal (desconfianza habitual).")


def test_agc_compressed_microphone_is_still_heard(offline_agent):
    """A microphone with automatic gain control squashes the whole call into a few dB.

    Regression: the gate required a block to sit a fixed 8 dB above the tracked noise floor before it could
    be speech. On an AGC'd capture the entire level range is ~6 dB, so that bar is never cleared: the
    session hears 1 % of blocks, no utterance is ever finalised and the caller is scored HUMAN for want of
    any audio at all. Measured on a real call re-rendered through a room-and-microphone channel, the gate
    kept 0 % of the caller's speech before this and hears it after. The bar now scales with the dynamic
    range actually present, with a 3 dB minimum so a steady room cannot pass.
    """
    rng = np.random.default_rng(11)
    floor_db, speech_db = -31.0, -26.0            # a 5 dB span, the signature of a compressing microphone
    noise = lambda n: rng.standard_normal(n).astype(np.float32) * 10 ** (floor_db / 20)
    with TestClient(main_mod.app) as c, c.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "start", "floor_dbfs": floor_db})
        sent = _greeting(ws, noise)
        sent = _stream(ws, noise(8000), sent)
        sp = _speech(rng, 2.5, speech_db)
        sent = _stream(ws, sp + noise(len(sp)), sent)
        sent = _stream(ws, noise(8000 * 2), sent)
        vad = _recv(ws, "vad", where=lambda m: m["state"] == "talking" and m["speaking"])
        assert vad["awaiting"] is True
        said = _recv(ws, "caller_said")
        assert "hola" in said["text"]
        ws.send_json({"type": "stop"})
        fin = _recv(ws, "final", limit=150)
        assert fin["result"]["live"]["utterances"] >= 1
