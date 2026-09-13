"""Semantic check: transcript-based judging with the Claude -> Gemini fallback (no network)."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import backend.features.semantic as sem  # noqa: E402
from backend.audio import call_from_arrays  # noqa: E402
from backend.scoring import heuristics  # noqa: E402
from backend.scoring.pipeline import Analyzer  # noqa: E402
from backend.vad import stft_power, vad_from_power  # noqa: E402

AGENT = [{"start": 0.5, "end": 8.0, "text": "Veo que tiene activo el seguro Protección Total Plus. ¿Me confirma el número de póliza?"}]
CALLER = [{"start": 9.0, "end": 12.0, "text": "Claro que sí, el número de póliza es cuatro cinco ocho dos uno."}]


def test_gemini_schema_is_openapi_subset():
    g = sem._gemini_schema(sem.JUDGE_SCHEMA)
    assert "additionalProperties" not in g and "minimum" not in g["properties"]["fabrication_probability"]
    assert g["properties"]["nonexistent_probes"]["items"]["properties"]["caller_reaction"]["enum"] == ["denied", "invented", "hedged", "unclear"]


def test_judge_prefers_claude_then_gemini(monkeypatch):
    calls = []
    monkeypatch.setattr(sem, "claude_available", lambda: False)
    monkeypatch.setattr(sem, "gemini_available", lambda: True)
    monkeypatch.setattr(sem, "judge_with_gemini", lambda a, c: calls.append("gemini") or {"synthetic_probability": 0.8, "engine": "gemini-x"})
    assert sem.judge(AGENT, CALLER)["engine"] == "gemini-x" and calls == ["gemini"]
    monkeypatch.setattr(sem, "gemini_available", lambda: False)
    assert "error" in sem.judge(AGENT, CALLER)


def test_transcript_judgement_feeds_the_verdict(monkeypatch):
    """A transcript that already exists (the live call) is judged without speech recognition and the judge's
    probability is fused into p(synthetic)."""
    fake = {"fabrication_probability": 0.9, "fabrication_evidence": ["invented a policy number"],
            "nonexistent_probes": [{"agent_question": "póliza", "caller_reaction": "invented"}], "repeat_back": "not_asked",
            "llm_style_probability": 0.8, "human_markers": [], "synthetic_probability": 0.85, "rationale": "invents", "engine": "test"}
    monkeypatch.setattr(sem, "judge_available", lambda: True)
    monkeypatch.setattr(sem, "judge", lambda a, c: dict(fake))
    x = (np.random.default_rng(0).standard_normal(8000 * 12) * 0.01).astype(np.float32)
    call = call_from_arrays(x, None)
    P, _ = stft_power(call.caller)
    res = sem.semantic_analysis(call, vad_from_power(P), None, transcript={"caller": CALLER, "agent": AGENT})
    assert res["available"] and res["judge"]["engine"] == "test" and res["features"]["sem_invented_probes"] == 1.0
    assert res["features"]["text_llm_phrase_count"] >= 1          # "claro que sí"
    aspects = heuristics.aspect_scores(res["features"])
    assert aspects["semantic_fabrication"]["score"] > 0.8
    an = Analyzer()
    r0 = an.analyze(call, want_ui=False, allow_semantic=False)
    r1 = an.analyze(call, None, AGENT, False, True, True, {"caller": CALLER, "agent": AGENT})
    assert r1["signals"]["semantic_p"] == 0.85 and r1["p_synthetic"] >= r0["p_synthetic"]
    assert r1["semantic"]["judge"]["rationale"] == "invents"
