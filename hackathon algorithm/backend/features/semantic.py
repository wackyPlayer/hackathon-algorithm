"""Semantic layer: transcribe both channels (faster-whisper) and judge the dialogue.

Two products:
  text_*   cheap transcript statistics that need no LLM (disfluencies, denial phrases, verbosity)
  sem_*    a Claude judgement of the caller's replies: does the caller invent answers to questions
           about things that do not exist, repeat information back correctly, and does the wording
           read like an LLM (complete sentences, no fillers) or like a person?

Everything is optional and lazy: if faster-whisper or the Anthropic SDK / API key are missing,
`semantic_analysis` returns {"available": False, ...} and the pipeline simply skips the layer.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time

import numpy as np

from ..audio import Call, resample
from ..config import settings
from ..vad import Vad

log = logging.getLogger("detector.semantic")

_whisper = None
_whisper_lock = threading.Lock()

FILLERS = ["eh", "este", "mmm", "mm", "ehh", "em", "o sea", "pues", "bueno", "a ver", "digo", "osea", "ah", "aja", "ajá"]
DENIALS = ["no tengo", "no cuento con", "no sé", "no se", "no me suena", "no conozco", "no manejo", "no tengo idea",
           "no tengo eso", "no lo tengo", "no existe", "no hay", "no recuerdo", "desconozco", "cuál", "cual seguro",
           "no entiendo", "no entendí", "¿cómo?", "mande", "perdón"]
LLM_PHRASES = ["claro que sí", "con gusto", "por supuesto", "entiendo", "perfecto", "me parece bien", "sin problema",
               "estoy de acuerdo", "agradezco", "quedo atento", "quedo atenta", "muchas gracias por la información",
               "confirmo que", "efectivamente", "correcto,", "en efecto", "adicionalmente", "asimismo"]


def _import_faster_whisper():
    """Import faster-whisper; if PyAV's native DLLs are blocked (Windows App Control) stub PyAV out.
    PyAV is only needed to decode audio *files*; we always pass numpy arrays."""
    import sys
    import types
    try:
        import faster_whisper
        return faster_whisper
    except Exception as exc:
        msg = str(exc)
        if not any(t in msg for t in ("av", "DLL", "Application Control")):
            raise
        for k in list(sys.modules):
            if k == "av" or k.startswith("av.") or k.startswith("faster_whisper"):
                del sys.modules[k]
        av = types.ModuleType("av")
        av.__stub__ = True
        for name in ("audio", "audio.resampler", "audio.fifo", "error"):
            parts = name.split(".")
            parent = av
            for i, p in enumerate(parts):
                full = "av." + ".".join(parts[: i + 1])
                mod = sys.modules.get(full) or types.ModuleType(full)
                sys.modules[full] = mod
                setattr(parent, p, mod)
                parent = mod
        sys.modules["av.error"].InvalidDataError = type("InvalidDataError", (Exception,), {})
        sys.modules["av"] = av
        log.warning("PyAV unavailable (%s); faster-whisper loaded with a PyAV stub (numpy input only)", msg.splitlines()[-1][:80])
        import faster_whisper
        return faster_whisper


def whisper_available() -> bool:
    try:
        _import_faster_whisper()
        return True
    except Exception:
        return False


def claude_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except Exception:
        return False
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN") or _has_profile())


def _has_profile() -> bool:
    return os.path.isdir(os.path.expanduser("~/.config/anthropic"))


_whisper_models: dict = {}


def get_whisper(model_name: str | None = None):
    """Lazily load (and cache) a faster-whisper model by size, e.g. 'base' for the live agent."""
    name = model_name or settings.whisper_model
    with _whisper_lock:
        if name not in _whisper_models:
            fw = _import_faster_whisper()
            t0 = time.time()
            _whisper_models[name] = fw.WhisperModel(name, device=settings.whisper_device,
                                                    compute_type=settings.whisper_compute_type)
            log.info("loaded faster-whisper %s in %.1fs", name, time.time() - t0)
        return _whisper_models[name]


def _get_whisper():
    return get_whisper(settings.whisper_model)


def transcribe_array(x8k: np.ndarray, model_name: str | None = None) -> str:
    """Transcribe one short 8 kHz utterance (live agent). Returns '' on silence/failure."""
    if len(x8k) < 2400:
        return ""
    try:
        model = get_whisper(model_name)
        segs, _ = model.transcribe(resample(x8k, 8000, 16000), language="es", beam_size=1, vad_filter=False,
                                   condition_on_previous_text=False)
        return " ".join(sg.text.strip() for sg in segs).strip()
    except Exception as exc:  # pragma: no cover
        log.warning("live transcription failed: %s", exc)
        return ""


def transcribe(x8k: np.ndarray, segments: list, max_seconds: float) -> list:
    """Transcribe only the speech segments of one channel; returns [{start, end, text}]."""
    model = _get_whisper()
    out = []
    budget = max_seconds
    for s, e in segments:
        if budget <= 0:
            break
        a, b = int(s * 8000), int(min(e, s + budget) * 8000)
        if b - a < 2400:
            continue
        chunk = resample(x8k[a:b], 8000, 16000)
        try:
            segs, _ = model.transcribe(chunk, language="es", beam_size=1, vad_filter=False,
                                       condition_on_previous_text=False)
            text = " ".join(sg.text.strip() for sg in segs).strip()
        except Exception as exc:  # pragma: no cover
            log.warning("whisper failed on segment %.1f-%.1f: %s", s, e, exc)
            continue
        budget -= (b - a) / 8000
        if text:
            out.append({"start": round(s, 2), "end": round(e, 2), "text": text})
    return out


def text_features(caller_turns: list) -> dict:
    out: dict = {}
    texts = [t["text"] for t in caller_turns]
    if not texts:
        return out
    joined = " ".join(texts).lower()
    words = re.findall(r"[a-záéíóúñü]+", joined)
    n_words = max(len(words), 1)
    dur = sum(t["end"] - t["start"] for t in caller_turns) or 1.0
    out["text_words"] = float(n_words)
    out["text_words_per_turn"] = float(n_words / max(len(texts), 1))
    out["text_words_per_sec"] = float(n_words / dur)
    out["text_ttr"] = float(len(set(words)) / n_words)
    out["text_filler_rate"] = float(sum(joined.count(f" {f} ") for f in FILLERS) / n_words * 100)
    out["text_denial_count"] = float(sum(joined.count(p) for p in DENIALS))
    out["text_llm_phrase_count"] = float(sum(joined.count(p) for p in LLM_PHRASES))
    reps = sum(1 for a, b in zip(words[:-1], words[1:]) if a == b)
    out["text_repeat_word_rate"] = float(reps / n_words * 100)
    out["text_question_frac"] = float(np.mean(["?" in t for t in texts]))
    out["text_mean_sentence_len"] = float(np.mean([len(re.findall(r"[a-záéíóúñü]+", t)) for t in texts]))
    return out


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "fabrication_probability": {"type": "number", "minimum": 0, "maximum": 1},
        "fabrication_evidence": {"type": "array", "items": {"type": "string"}},
        "nonexistent_probes": {"type": "array", "items": {"type": "object", "properties": {
            "agent_question": {"type": "string"},
            "caller_reaction": {"type": "string", "enum": ["denied", "invented", "hedged", "unclear"]},
        }, "required": ["agent_question", "caller_reaction"], "additionalProperties": False}},
        "repeat_back": {"type": "string", "enum": ["correct", "partial", "wrong", "not_asked", "unclear"]},
        "llm_style_probability": {"type": "number", "minimum": 0, "maximum": 1},
        "human_markers": {"type": "array", "items": {"type": "string"}},
        "synthetic_probability": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
    "required": ["fabrication_probability", "fabrication_evidence", "nonexistent_probes", "repeat_back",
                 "llm_style_probability", "human_markers", "synthetic_probability", "rationale"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = """You are an expert fraud analyst for a bank's phone channel in Mexico. You read transcripts of
calls between the bank's AI customer-service agent (AGENT) and a caller (CALLER), in Mexican Spanish.
The caller is either a real person or an autonomous AI caller (speech recognition -> language model -> synthetic voice).

The agent deliberately asks the caller to repeat information back and sometimes asks about things that do not
exist (products, policies, folios the customer never had). A real person typically says they do not have that,
do not know, or asks what the agent means. A language model tends to invent a plausible answer (numbers, names,
confirmations) and to answer every question in complete, polite, well-formed sentences without fillers,
self-corrections, hesitations or emotion.

The transcript comes from automatic speech recognition: ignore obvious recognition errors and missing fillers,
judge the content and the pragmatics of the replies. Be well calibrated: 0.5 means you genuinely cannot tell.
Return only the JSON object described by the schema."""


def _format_dialogue(agent_turns: list, caller_turns: list) -> str:
    rows = [(t["start"], "AGENT", t["text"]) for t in agent_turns] + [(t["start"], "CALLER", t["text"]) for t in caller_turns]
    rows.sort(key=lambda r: r[0])
    return "\n".join(f"[{s:6.1f}s] {who}: {txt}" for s, who, txt in rows)


def judge_with_claude(agent_turns: list, caller_turns: list) -> dict:
    import anthropic

    client = anthropic.Anthropic(timeout=settings.claude_timeout_s, max_retries=1)
    dialogue = _format_dialogue(agent_turns, caller_turns)
    user = ("Transcript of the call (timestamps in seconds from the start of the recording):\n\n" + dialogue +
            "\n\nAnalyse the CALLER only and fill the JSON schema.")
    kwargs = dict(
        model=settings.claude_model,
        max_tokens=2000,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
    )
    try:
        resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    except TypeError:
        resp = client.messages.create(**kwargs)
    if resp.stop_reason == "refusal":
        return {"error": "refusal"}
    text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
    return json.loads(text)


def semantic_analysis(call: Call, vad_c: Vad, vad_a: Vad | None, agent_text_turns: list | None = None,
                      use_llm: bool = True) -> dict:
    """Run STT (+ optional Claude judge). Never raises."""
    res: dict = {"available": False, "features": {}}
    if not whisper_available():
        res["error"] = "faster-whisper not installed"
        return res
    t0 = time.time()
    try:
        caller_turns = transcribe(call.caller, vad_c.turns, settings.semantic_max_seconds)
        if agent_text_turns is not None:
            agent_turns = agent_text_turns
        elif vad_a is not None and call.agent is not None:
            agent_turns = transcribe(call.agent, vad_a.turns, settings.semantic_max_seconds)
        else:
            agent_turns = []
    except Exception as exc:
        log.exception("transcription failed: %s", exc)
        res["error"] = f"transcription failed: {exc}"
        return res
    res["stt_seconds"] = round(time.time() - t0, 2)
    res["transcript"] = {"caller": caller_turns, "agent": agent_turns}
    res["features"].update(text_features(caller_turns))
    res["available"] = True
    if use_llm and caller_turns and claude_available():
        t1 = time.time()
        try:
            j = judge_with_claude(agent_turns, caller_turns)
            if "error" not in j:
                res["judge"] = j
                res["features"]["sem_fabrication_p"] = float(j["fabrication_probability"])
                res["features"]["sem_llm_style_p"] = float(j["llm_style_probability"])
                res["features"]["sem_synthetic_p"] = float(j["synthetic_probability"])
                res["features"]["sem_human_markers"] = float(len(j.get("human_markers", [])))
                res["features"]["sem_invented_probes"] = float(
                    sum(1 for p in j.get("nonexistent_probes", []) if p.get("caller_reaction") == "invented"))
                res["features"]["sem_denied_probes"] = float(
                    sum(1 for p in j.get("nonexistent_probes", []) if p.get("caller_reaction") == "denied"))
            else:
                res["judge_error"] = j["error"]
        except Exception as exc:
            log.warning("claude judge failed: %s", exc)
            res["judge_error"] = str(exc)
        res["llm_seconds"] = round(time.time() - t1, 2)
    elif use_llm and not claude_available():
        res["judge_error"] = "anthropic SDK or credentials not available"
    return res
