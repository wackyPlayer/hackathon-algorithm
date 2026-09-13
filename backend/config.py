"""Runtime configuration (environment-driven, no external deps)."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from <project>/.env into the environment (existing variables win)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except OSError:
        pass


_load_dotenv()


def _b(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _f(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v not in (None, "") else default


def _s(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


@dataclass
class Settings:
    # --- audio ---
    sample_rate: int = 8000
    max_seconds: float = _f("MAX_SECONDS", 300.0)          # cap on analysed audio per clip

    # --- decision ---
    decision_threshold: float = _f("DECISION_THRESHOLD", 0.5)
    # What /detect puts in "confidence". The brief says confidence breaks ties and rewards well-calibrated
    # systems, and both of those need a number that is monotone in the thing being decided: p(synthetic)
    # ranks every call on one axis and can be scored for calibration directly, whereas p(verdict correct)
    # folds the scale in half (0.02 and 0.98 both come back as 0.98), which destroys the ranking.
    #   p_synthetic  (default)  is_synthetic == (confidence >= 0.5) always holds
    #   p_correct               max(p, 1-p), the old behaviour, if a harness asks for it
    confidence_semantics: str = _s("CONFIDENCE_SEMANTICS", "p_synthetic")
    # Never report absolute certainty: a saturated linear model returns 1.0 on inputs it has never seen,
    # and a confidently wrong answer is worse than a hedged one.
    confidence_cap: float = _f("CONFIDENCE_CAP", 0.995)
    # below this much caller speech the verdict is pulled toward 0.5 (not enough evidence)
    min_speech_seconds: float = _f("MIN_SPEECH_SECONDS", 1.5)
    full_confidence_speech_seconds: float = _f("FULL_CONFIDENCE_SPEECH_SECONDS", 8.0)

    # --- trained model ---
    model_path: str = _s("MODEL_PATH", "models/detector.joblib")

    # --- optional self-supervised embeddings (torch + transformers) ---
    enable_embeddings: bool = _b("ENABLE_EMBEDDINGS", False)
    embedding_model: str = _s("EMBEDDING_MODEL", "microsoft/wavlm-base-plus")
    embedding_layer: int = int(_f("EMBEDDING_LAYER", 6))
    embedding_max_seconds: float = _f("EMBEDDING_MAX_SECONDS", 20.0)

    # --- optional semantic layer (faster-whisper + Claude) ---
    # off | uncertain | always
    semantic_mode: str = _s("SEMANTIC_MODE", "off")
    semantic_low: float = _f("SEMANTIC_LOW", 0.30)
    semantic_high: float = _f("SEMANTIC_HIGH", 0.70)
    semantic_max_seconds: float = _f("SEMANTIC_MAX_SECONDS", 90.0)
    whisper_model: str = _s("WHISPER_MODEL", "small")
    whisper_device: str = _s("WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = _s("WHISPER_COMPUTE_TYPE", "int8")
    claude_model: str = _s("CLAUDE_MODEL", "claude-opus-5")
    claude_timeout_s: float = _f("CLAUDE_TIMEOUT_S", 25.0)

    # --- fusion weights (logit space) when no stacking model was trained ---
    w_fast: float = _f("W_FAST", 1.0)
    w_embed: float = _f("W_EMBED", 1.0)
    w_semantic: float = _f("W_SEMANTIC", 0.8)

    # --- live mode ---
    live_update_interval_s: float = _f("LIVE_UPDATE_INTERVAL_S", 2.0)
    # live agent: Gemini writes the agent's lines, ElevenLabs speaks them, faster-whisper hears the caller
    gemini_api_key: str = _s("GEMINI_API_KEY", "")
    gemini_model: str = _s("GEMINI_MODEL", "gemini-3.1-flash-lite")      # ~1.3 s per line, no thinking
    gemini_fallback_model: str = _s("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash")
    elevenlabs_api_key: str = _s("ELEVENLABS_API_KEY", "")
    elevenlabs_voice_id: str = _s("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
    elevenlabs_model: str = _s("ELEVENLABS_MODEL", "eleven_multilingual_v2")
    live_whisper_model: str = _s("LIVE_WHISPER_MODEL", "base")
    live_end_silence_s: float = _f("LIVE_END_SILENCE_S", 1.0)      # silence that ends a caller utterance
    live_min_utterance_s: float = _f("LIVE_MIN_UTTERANCE_S", 0.4)
    live_answer_timeout_s: float = _f("LIVE_ANSWER_TIMEOUT_S", 9.0)   # agent moves on if the caller says nothing
    live_max_utterance_s: float = _f("LIVE_MAX_UTTERANCE_S", 20.0)    # a caller turn is cut here (monologue or noise)
    live_speech_rise_db: float = _f("LIVE_SPEECH_RISE_DB", 8.0)       # speech = this far above the tracked noise floor
    live_talk_mod_db: float = _f("LIVE_TALK_MOD_DB", 3.0)            # talking swings > this over 0.8 s; steady = background noise
    live_peak_margin_db: float = _f("LIVE_PEAK_MARGIN_DB", 18.0)      # blocks quieter than the caller's own peaks by more are background
    # Spectral flatness below this means the block has harmonic structure, i.e. somebody (or something) is
    # speaking. This is the gate that decides speech-vs-noise, NOT the level swing: a fan and a synthetic
    # voice are both steady in level, and only one of them is a caller.
    live_voiced_flatness: float = _f("LIVE_VOICED_FLATNESS", 0.35)
    # How much trailing audio a rolling verdict looks at. The analysis costs ~175 MB per minute of audio
    # (scipy's STFT builds a complex128 intermediate), and the live call re-runs it every 2 s, so an
    # unbounded window reached 1.4 GB per pass on an 8-minute call. The model is already fully confident
    # by ~20 s of caller speech, so 90 s is generous and keeps memory flat. The closing verdict still
    # sees the whole call (capped by MAX_SECONDS).
    live_window_s: float = _f("LIVE_WINDOW_S", 90.0)
    live_escalate_p: float = _f("LIVE_ESCALATE_P", 0.70)             # call-average p(synthetic) that escalates the agent's questions

    # --- misc ---
    log_level: str = _s("LOG_LEVEL", "INFO")
    include_spectrogram: bool = _b("INCLUDE_SPECTROGRAM", True)


settings = Settings()
