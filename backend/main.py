"""FastAPI server: the scored POST /detect endpoint, a verbose POST /analyze, a live WebSocket, and the dashboard.

Run:  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 2
"""
from __future__ import annotations

import asyncio
import base64
import csv
import json
import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .audio import b64_to_bytes, call_from_arrays, load_call, looks_like_wav, to_wav_bytes
from .config import settings
from .live_agent import STEPS, ElevenLabsVoice, GeminiBrain, elevenlabs_available, gemini_available
from .scoring.pipeline import Analyzer

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("detector.api")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(ROOT, "frontend")
STATE: dict = {"analyzer": None, "started": time.time(), "requests": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    an = Analyzer()
    an.warmup()
    STATE["analyzer"] = an
    log.info("analyzer ready in %.1fs (mode=%s, semantic=%s, gemini=%s, elevenlabs=%s)", time.time() - t0, an.mode,
             settings.semantic_mode, gemini_available(), elevenlabs_available())
    # pre-load the live agent's speech recogniser so the first caller turn is not delayed by a model download
    from .features.semantic import get_whisper, whisper_available
    if whisper_available():
        import threading

        def _warm():
            try:
                get_whisper(settings.live_whisper_model)
            except Exception as exc:  # pragma: no cover
                log.warning("live whisper warm-up failed: %s", exc)
        threading.Thread(target=_warm, name="whisper-warmup", daemon=True).start()
    yield


app = FastAPI(title="Altur Voice Shield", version="1.1.0", lifespan=lifespan)
# Open CORS so the dashboard and the API work from any origin (shared tunnel link, other teams' tools, judges).
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["*"])
if os.path.isdir(FRONTEND):
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/share")
async def share(request: Request):
    """Links other people can use: the public tunnel URL (written by share.ps1 / share.sh or SHARE_URL) and LAN URLs."""
    public = os.getenv("SHARE_URL", "").strip()
    if not public:
        p = os.path.join(ROOT, "share_url.txt")
        if os.path.exists(p):
            try:
                public = open(p, encoding="utf-8").read().strip()
            except OSError:
                public = ""
    port = request.url.port or (443 if request.url.scheme == "https" else 80)
    lan = []
    try:
        import socket
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if not ip.startswith("127.") and not ip.startswith("169.254."):
                lan.append(f"http://{ip}:{port}")
    except Exception:
        pass
    return {"public_url": public or None, "lan_urls": lan, "https_required_for_mic": True}


# ----------------------------------------------------------------------------- input parsing

def _find_audio_in_json(obj, depth: int = 0) -> bytes | None:
    """Depth-first search for a base64 WAV (or raw base64 audio) inside arbitrary JSON."""
    if depth > 6:
        return None
    if isinstance(obj, str):
        if len(obj) < 64:
            return None
        try:
            data = b64_to_bytes(obj)
        except Exception:
            return None
        return data if len(data) > 44 else None
    if isinstance(obj, dict):
        preferred = [k for k in obj if any(t in k.lower() for t in ("audio", "wav", "clip", "data", "file", "b64", "base64", "content"))]
        for k in preferred + [k for k in obj if k not in preferred]:
            r = _find_audio_in_json(obj[k], depth + 1)
            if r is not None:
                return r
    if isinstance(obj, list):
        for it in obj:
            r = _find_audio_in_json(it, depth + 1)
            if r is not None:
                return r
    return None


async def read_audio_bytes(request: Request) -> bytes:
    ct = request.headers.get("content-type", "").lower()
    if "multipart/form-data" in ct:
        form = await request.form()
        for v in form.values():
            if hasattr(v, "read"):
                data = await v.read()
                if data:
                    return data
        for v in form.values():
            if isinstance(v, str):
                r = _find_audio_in_json(v)
                if r:
                    return r
        raise HTTPException(400, "multipart form without an audio file")
    body = await request.body()
    if not body:
        raise HTTPException(400, "empty body")
    if looks_like_wav(body):
        return body
    stripped = body.lstrip()
    if stripped[:1] in (b"{", b"["):
        try:
            obj = json.loads(body)
        except Exception as exc:
            raise HTTPException(400, f"invalid JSON: {exc}")
        r = _find_audio_in_json(obj)
        if r is None:
            raise HTTPException(400, "JSON body does not contain a base64-encoded WAV (expected e.g. {\"audio\": \"<base64>\"})")
        return r
    try:  # raw base64 text body
        data = b64_to_bytes(body.decode("ascii", errors="ignore"))
    except Exception as exc:
        raise HTTPException(400, f"body is neither WAV, JSON nor base64: {exc}")
    if len(data) < 44:
        raise HTTPException(400, "decoded payload too small to be audio")
    return data


def _load(data: bytes):
    try:
        return load_call(data, target_sr=settings.sample_rate, max_seconds=settings.max_seconds)
    except Exception as exc:
        raise HTTPException(400, f"could not decode audio: {exc}")


# ----------------------------------------------------------------------------- endpoints

@app.post("/detect")
async def detect(request: Request):
    """Scored endpoint. Body: stereo WAV (8 kHz) base64-encoded, e.g. {"audio": "<base64>"}.
    Returns {"is_synthetic": bool, "confidence": float} where confidence is the probability that
    the returned verdict is correct (>= 0.5)."""
    STATE["requests"] += 1
    data = await read_audio_bytes(request)
    call = _load(data)
    t0 = time.time()
    try:
        res = await run_in_threadpool(STATE["analyzer"].analyze, call, None, None, False, True, False)
    except Exception as exc:
        log.exception("analysis failed; returning neutral verdict: %s", exc)
        return JSONResponse({"is_synthetic": False, "confidence": 0.5})
    log.info("/detect dur=%.1fs speech=%.1fs -> %s p=%.3f in %.2fs", call.duration, res["speech_seconds"],
             res["verdict"], res["p_synthetic"], time.time() - t0)
    return JSONResponse({"is_synthetic": bool(res["is_synthetic"]), "confidence": float(res["confidence"])})


@app.post("/analyze")
async def analyze(request: Request, semantic: int = 0, ui: int = 1, sample: str = ""):
    """Verbose analysis for the dashboard: aspects, attack profile, timeline, features, spectrogram.
    `sample=<name>` analyses a file from SAMPLES_DIR instead of the request body (demo convenience)."""
    STATE["requests"] += 1
    if sample:
        data = open(_sample_path(sample), "rb").read()
    else:
        data = await read_audio_bytes(request)
    call = _load(data)
    res = await run_in_threadpool(STATE["analyzer"].analyze, call, None, None, bool(ui), True, bool(semantic))
    if sample:
        res["sample"] = {"name": sample, **SAMPLE_LABELS.get(os.path.splitext(sample)[0], {})}
    return JSONResponse(res)


@app.get("/health")
async def health():
    an: Analyzer | None = STATE["analyzer"]
    from .features.semantic import claude_available, whisper_available
    det = an.detector if an else None
    return {
        "status": "ok" if an else "starting",
        "mode": an.mode if an else None,
        "model_path": settings.model_path if det else None,
        "model_meta": {k: v for k, v in (det.meta.items() if det else []) if not isinstance(v, (list, dict))},
        "semantic_mode": settings.semantic_mode,
        "whisper_available": whisper_available(),
        "claude_available": claude_available(),
        "gemini_available": gemini_available(),
        "gemini_model": settings.gemini_model,
        "elevenlabs_available": elevenlabs_available(),
        "live_whisper_model": settings.live_whisper_model,
        "embeddings_enabled": settings.enable_embeddings,
        "uptime_s": round(time.time() - STATE["started"], 1),
        "requests": STATE["requests"],
        "confidence_semantics": "probability that the returned is_synthetic verdict is correct (0.5-1.0)",
    }


@app.api_route("/", methods=["GET", "HEAD"])
async def index():
    p = os.path.join(FRONTEND, "index.html")
    if not os.path.exists(p):
        return JSONResponse({"service": "altur-voice-shield", "endpoints": ["/detect", "/analyze", "/health", "/ws/live"]})
    return FileResponse(p)


# ----------------------------------------------------------------------------- microphone check (live call)

@app.post("/miccheck")
async def miccheck(request: Request, ns: int = 0, agc: int = 0, ec: int = 0, sr: int = 0, device: str = ""):
    """Microphone quality check before a live call. Body: raw int16 8 kHz PCM (application/octet-stream) or a
    WAV / base64 JSON like /detect. Query: ns/agc/ec = the browser's noiseSuppression / autoGainControl /
    echoCancellation flags as actually applied, sr = device sample rate, device = its label.
    Returns {quality: good|fair|poor, summary, warnings[{level, risk, text}], metrics}."""
    from .miccheck import mic_check_report
    from .scoring.pipeline import extract
    body = await request.body()
    ct = request.headers.get("content-type", "").lower()
    if body and not looks_like_wav(body) and ("octet-stream" in ct or not ct):
        x = np.frombuffer(body[: len(body) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
        call = call_from_arrays(x, None)
    else:
        call = _load(await read_audio_bytes(request))
    if call.duration < 0.5:
        raise HTTPException(400, "sample too short (send at least 2 s of audio)")
    ex = await run_in_threadpool(extract, call, None, False)
    client = {"noise_suppression": bool(ns), "auto_gain": bool(agc), "echo_cancellation": bool(ec),
              "sample_rate": sr or None, "device": device[:80]}
    rep = mic_check_report(ex.features, client)
    rep["duration_seconds"] = round(call.duration, 2)
    log.info("/miccheck %.1fs: quality=%s snr=%s floor=%s warnings=%d", call.duration, rep["quality"],
             rep["metrics"]["snr_db"], rep["metrics"]["floor_dbfs"], len(rep["warnings"]))
    return JSONResponse(rep)


# ----------------------------------------------------------------------------- demo samples (optional)

SAMPLES_DIR = os.getenv("SAMPLES_DIR", "")
SAMPLE_LABELS: dict = {}
if SAMPLES_DIR:
    for cand in (os.getenv("SAMPLES_MANIFEST", ""), os.path.join(SAMPLES_DIR, "manifest.csv"),
                 os.path.join(os.path.dirname(SAMPLES_DIR.rstrip("/\\")), "manifest.csv")):
        if cand and os.path.exists(cand):
            for r in csv.DictReader(open(cand, encoding="utf-8")):
                SAMPLE_LABELS[r.get("anon_id") or r.get("file", "").replace(".wav", "")] = \
                    {"label": r.get("label"), "split": r.get("split"), "duration_s": r.get("duration_s")}
            break


def _sample_path(name: str) -> str:
    base = os.path.basename(name)
    if not SAMPLES_DIR or not base.lower().endswith(".wav"):
        raise HTTPException(404, "samples are not enabled (set SAMPLES_DIR) or bad name")
    p = os.path.join(SAMPLES_DIR, base)
    if not os.path.exists(p):
        raise HTTPException(404, f"sample {base} not found")
    return p


@app.get("/samples")
async def samples(limit: int = 400):
    if not SAMPLES_DIR or not os.path.isdir(SAMPLES_DIR):
        return {"enabled": False, "samples": []}
    names = sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(".wav"))[:limit]
    return {"enabled": True, "samples": [{"name": n, **SAMPLE_LABELS.get(os.path.splitext(n)[0], {})} for n in names]}


# ----------------------------------------------------------------------------- live call (WebSocket)

class LiveSession:
    """One live call: caller PCM (int16 mono 8 kHz) from the browser, agent lines from Gemini/ElevenLabs.

    The browser only captures the microphone and plays the agent's audio; the conversation state machine,
    the caller's end-of-utterance detection and the speech recognition all live here."""

    def __init__(self, ws: WebSocket, analyzer: Analyzer):
        self.ws, self.an = ws, analyzer
        self.chunks: list = []
        self.n = 0
        self.agent_events: list = []       # {"event": start|end, "t": float, "text": str, "kind": str}
        self.last_analyzed = 0
        self.started = time.time()
        # agent state machine
        self.brain = GeminiBrain()
        self.voice = ElevenLabsVoice()
        self.step = 0                      # index into STEPS of the line being spoken / answered
        self.await_answer = False
        self.agent_speaking = False
        self.busy = False
        self.done = False
        self.interrupted_step = -1
        self.timer: asyncio.Task | None = None
        self.pipeline: asyncio.Task | None = None
        # caller utterance detection (100 ms blocks)
        self.floor_db = -60.0
        self.speaking = False
        self.utt_start: float | None = None
        self.last_speech: float | None = None
        self.caller_turns: list = []

    # ---------------------------------------------------------------- audio in
    def seconds(self) -> float:
        return self.n / 8000.0

    def audio(self) -> np.ndarray:
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)

    def add(self, data: bytes) -> list:
        """Append PCM; return finalized caller utterances [(start_s, end_s)] detected in this chunk."""
        x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        self.chunks.append(x)
        finished = []
        for i in range(0, len(x), 800):
            blk = x[i:i + 800]
            if len(blk) < 400:
                break
            t = (self.n + i) / 8000.0
            db = 20 * np.log10(float(np.sqrt(np.mean(blk ** 2))) + 1e-6)
            if not self.speaking and db < self.floor_db + 6:
                self.floor_db = 0.95 * self.floor_db + 0.05 * db
            thr = max(self.floor_db + 9.0, -50.0)
            if db > thr:
                if not self.speaking:
                    self.speaking, self.utt_start = True, t
                self.last_speech = t
            elif self.speaking and self.last_speech is not None and t - self.last_speech >= settings.live_end_silence_s:
                self.speaking = False
                if self.last_speech - self.utt_start >= settings.live_min_utterance_s:
                    finished.append((self.utt_start, self.last_speech + 0.15))
        self.n += len(x)
        return finished

    # ---------------------------------------------------------------- agent timeline helpers
    def agent_segments(self) -> list:
        segs, open_t = [], None
        for ev in self.agent_events:
            if ev["event"] == "start":
                open_t = ev["t"]
            elif ev["event"] == "end" and open_t is not None:
                segs.append((open_t, max(ev["t"], open_t + 0.05)))
                open_t = None
        if open_t is not None:
            segs.append((open_t, self.seconds()))
        return segs

    def agent_text_turns(self) -> list:
        turns, open_ev = [], None
        for ev in self.agent_events:
            if ev["event"] == "start":
                open_ev = ev
            elif ev["event"] == "end" and open_ev is not None:
                turns.append({"start": round(open_ev["t"], 2), "end": round(ev["t"], 2), "text": open_ev.get("text", "")})
                open_ev = None
        return turns

    def agent_track(self) -> np.ndarray:
        """Channel-1 stand-in for the downloadable WAV: soft noise bursts where the agent spoke."""
        a = np.zeros(self.n, np.float32)
        rng = np.random.default_rng(1)
        for s, e in self.agent_segments():
            i, j = int(s * 8000), min(int(e * 8000), self.n)
            if j > i:
                a[i:j] = rng.standard_normal(j - i).astype(np.float32) * 0.03
        return a

    # ---------------------------------------------------------------- messaging
    async def send(self, payload: dict) -> None:
        try:
            await self.ws.send_text(json.dumps(payload))
        except Exception as exc:  # pragma: no cover
            log.debug("ws send failed: %s", exc)

    async def status(self, text: str) -> None:
        await self.send({"type": "status", "text": text})

    # ---------------------------------------------------------------- conversation flow
    def cancel_timer(self) -> None:
        # never cancel the task we are running in (the timer itself calls speak_step after a timeout / silence)
        if self.timer is not None and not self.timer.done() and self.timer is not asyncio.current_task():
            self.timer.cancel()
        self.timer = None

    async def speak_step(self, caller_text: str) -> None:
        """Generate (Gemini) + synthesize (ElevenLabs) the current step's line and hand it to the browser."""
        if self.done or self.step >= len(STEPS):
            return
        self.busy = True
        self.await_answer = False
        self.cancel_timer()
        step = STEPS[self.step]
        try:
            await self.status("thinking…")
            text = await run_in_threadpool(self.brain.line, self.step, caller_text)
            await self.status("synthesizing…")
            audio = await run_in_threadpool(self.voice.synthesize, text)
            await self.send({"type": "agent_say", "step": self.step + 1, "steps": len(STEPS), "kind": step["kind"],
                             "text": text, "audio_b64": base64.b64encode(audio).decode("ascii") if audio else None,
                             "silence_after": step.get("silence_after", 0), "end": bool(step.get("end")),
                             "brain": "gemini" if (self.brain.enabled and not self.brain.last_error) else "canned",
                             "voice": "elevenlabs" if audio else "browser"})
            await self.status("")
        except Exception as exc:  # pragma: no cover
            log.exception("speak_step failed: %s", exc)
            await self.send({"type": "error", "message": f"agent failed: {exc}"})
        finally:
            self.busy = False

    async def on_agent_event(self, ev: dict) -> None:
        self.agent_events.append(ev)
        if ev["event"] == "start":
            self.agent_speaking = True
            return
        self.agent_speaking = False
        if ev.get("kind") == "interrupt" or self.done:
            return
        step = STEPS[min(self.step, len(STEPS) - 1)]
        if step.get("end"):
            self.done = True
            await self.send({"type": "agent_done"})
        elif step.get("silence_after"):
            async def later():
                await asyncio.sleep(step["silence_after"])
                self.step += 1
                await self.speak_step("")
            self.cancel_timer()
            self.timer = asyncio.create_task(later())
        else:
            self.await_answer = True
            self.cancel_timer()
            self.timer = asyncio.create_task(self.answer_timeout())

    async def answer_timeout(self) -> None:
        await asyncio.sleep(settings.live_answer_timeout_s)
        while self.speaking or self.busy:         # caller mid-sentence or a transcription in flight: wait
            await asyncio.sleep(0.3)
        if self.await_answer and not self.busy and not self.done:
            log.info("live: no answer within %.0fs, agent moves on", settings.live_answer_timeout_s)
            await self.send({"type": "caller_said", "text": "(no se escuchó respuesta)", "t": round(self.seconds(), 2)})
            self.step += 1
            await self.speak_step("")

    async def on_utterance(self, s: float, e: float) -> None:
        if not self.await_answer or self.busy or self.done:
            return
        x = self.audio()[int(s * 8000):int(e * 8000)]
        # hold the turn while transcribing so the no-answer timer cannot advance the flow underneath us
        self.busy = True
        self.await_answer = False
        self.cancel_timer()
        await self.status("transcribing…")
        from .features.semantic import transcribe_array
        t0 = time.time()
        text = await run_in_threadpool(transcribe_array, x, settings.live_whisper_model)
        log.info("live stt %.1fs of audio in %.1fs: %s", e - s, time.time() - t0, text[:80])
        self.busy = False
        if self.done:
            return
        if not text.strip():
            await self.status("")
            self.await_answer = True
            self.timer = asyncio.create_task(self.answer_timeout())
            return
        self.caller_turns.append({"start": round(s, 2), "end": round(e, 2), "text": text})
        await self.send({"type": "caller_said", "text": text, "t": round(s, 2)})
        self.step += 1
        await self.speak_step(text)

    async def maybe_interrupt(self) -> None:
        """Deliberately talk over the caller once, during the step that asks for a long answer."""
        if self.done or self.busy or not self.await_answer or self.agent_speaking or not self.speaking:
            return
        step = STEPS[min(self.step, len(STEPS) - 1)]
        after = step.get("interrupt_after")
        if not after or self.interrupted_step == self.step or self.utt_start is None:
            return
        if self.seconds() - self.utt_start < after:
            return
        self.interrupted_step = self.step
        text = step["interrupt_text"]
        audio = await run_in_threadpool(self.voice.synthesize, text)
        await self.send({"type": "agent_say", "step": self.step + 1, "steps": len(STEPS), "kind": "interrupt", "text": text,
                         "audio_b64": base64.b64encode(audio).decode("ascii") if audio else None, "silence_after": 0,
                         "end": False, "brain": "canned", "voice": "elevenlabs" if audio else "browser"})

    async def analyze(self, final: bool):
        x = self.audio()
        if len(x) < 8000:
            return None
        call = call_from_arrays(x, None)
        segs = self.agent_segments()
        res = await run_in_threadpool(self.an.analyze, call, segs, self.agent_text_turns(), final, final, False)
        res["live"] = {"seconds": round(self.seconds(), 1), "agent_turns": len(segs), "step": self.step,
                       "caller_turns": self.caller_turns if final else len(self.caller_turns),
                       "brain": "gemini" if self.brain.enabled else "canned", "voice": "elevenlabs" if self.voice.enabled else "browser"}
        return res


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    sess = LiveSession(ws, STATE["analyzer"])
    interval = settings.live_update_interval_s
    inflight: dict = {"task": None}

    async def update_task():
        try:
            res = await sess.analyze(final=False)
            if res:
                res.pop("features", None)
                res.pop("ui", None)
                await sess.send({"type": "update", "result": res})
        except Exception as exc:  # pragma: no cover
            log.warning("live update failed: %s", exc)

    def spawn_pipeline(coro):
        if sess.pipeline is not None and not sess.pipeline.done():
            coro.close()
            return
        sess.pipeline = asyncio.create_task(coro)

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes"):
                for s, e in sess.add(msg["bytes"]):
                    spawn_pipeline(sess.on_utterance(s, e))
                if sess.speaking:
                    spawn_pipeline(sess.maybe_interrupt())
                if sess.n - sess.last_analyzed >= interval * 8000 and (inflight["task"] is None or inflight["task"].done()):
                    sess.last_analyzed = sess.n
                    inflight["task"] = asyncio.create_task(update_task())
            elif msg.get("text"):
                try:
                    m = json.loads(msg["text"])
                except Exception:
                    continue
                t = m.get("type")
                if t == "start":
                    await sess.send({"type": "hello", "gemini": sess.brain.enabled, "elevenlabs": sess.voice.enabled,
                                     "steps": [s["kind"] for s in STEPS]})
                    spawn_pipeline(sess.speak_step(""))
                elif t == "agent":
                    await sess.on_agent_event({"event": m.get("event"), "t": float(m.get("t", sess.seconds())),
                                               "text": m.get("text", ""), "kind": m.get("kind", "")})
                elif t == "stop":
                    sess.done = True
                    sess.cancel_timer()
                    for task in (inflight["task"], sess.pipeline):
                        if task is not None and not task.done():
                            try:
                                await asyncio.wait_for(task, timeout=30)
                            except Exception:
                                pass
                    res = await sess.analyze(final=True)
                    payload = {"type": "final", "result": res}
                    if res is not None:
                        payload["wav_b64"] = base64.b64encode(to_wav_bytes(sess.audio(), sess.agent_track())).decode("ascii")
                    await sess.send(payload)
                elif t == "ping":
                    await sess.send({"type": "pong", "seconds": sess.seconds()})
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover
        log.exception("live session error: %s", exc)
        await sess.send({"type": "error", "message": str(exc)})
    finally:
        sess.cancel_timer()
