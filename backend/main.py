"""FastAPI server: the scored POST /detect endpoint, a verbose POST /analyze, a live WebSocket, and the dashboard.

Run:  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 2
"""
from __future__ import annotations

import base64
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
    log.info("analyzer ready in %.1fs (mode=%s, semantic=%s)", time.time() - t0, an.mode, settings.semantic_mode)
    yield


app = FastAPI(title="Altur Voice Shield", version="1.0.0", lifespan=lifespan)
if os.path.isdir(FRONTEND):
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


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
        path = _sample_path(sample)
        data = open(path, "rb").read()
    else:
        data = await read_audio_bytes(request)
    call = _load(data)
    res = await run_in_threadpool(STATE["analyzer"].analyze, call, None, None, bool(ui), True, bool(semantic))
    if sample:
        res["sample"] = {"name": sample, **SAMPLE_LABELS.get(os.path.splitext(sample)[0], {})}
    return JSONResponse(res)


# ----------------------------------------------------------------------------- demo samples (optional)

SAMPLES_DIR = os.getenv("SAMPLES_DIR", "")
SAMPLE_LABELS: dict = {}
if SAMPLES_DIR:
    for cand in (os.getenv("SAMPLES_MANIFEST", ""), os.path.join(SAMPLES_DIR, "manifest.csv"),
                 os.path.join(os.path.dirname(SAMPLES_DIR.rstrip("/\\")), "manifest.csv")):
        if cand and os.path.exists(cand):
            import csv
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
        "embeddings_enabled": settings.enable_embeddings,
        "uptime_s": round(time.time() - STATE["started"], 1),
        "requests": STATE["requests"],
        "confidence_semantics": "probability that the returned is_synthetic verdict is correct (0.5-1.0)",
    }


@app.get("/")
async def index():
    p = os.path.join(FRONTEND, "index.html")
    if not os.path.exists(p):
        return JSONResponse({"service": "altur-voice-shield", "endpoints": ["/detect", "/analyze", "/health", "/ws/live"]})
    return FileResponse(p)


# ----------------------------------------------------------------------------- live call (WebSocket)

class LiveSession:
    """Accumulates caller PCM (int16 mono 8 kHz) and the agent's spoken timeline from the browser."""

    def __init__(self):
        self.chunks: list = []
        self.n = 0
        self.agent_events: list = []       # {"event": start|end, "t": float, "text": str, "kind": str}
        self.last_analyzed = 0
        self.started = time.time()

    def add(self, data: bytes) -> None:
        x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        self.chunks.append(x)
        self.n += len(x)

    def audio(self) -> np.ndarray:
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)

    def seconds(self) -> float:
        return self.n / 8000.0

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


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    sess = LiveSession()
    an: Analyzer = STATE["analyzer"]
    interval = settings.live_update_interval_s

    async def run_analysis(final: bool):
        x = sess.audio()
        if len(x) < 8000:
            return None
        call = call_from_arrays(x, None)
        segs = sess.agent_segments()
        res = await run_in_threadpool(an.analyze, call, segs, sess.agent_text_turns(), final, final, False)
        res["live"] = {"seconds": round(sess.seconds(), 1), "agent_turns": len(segs)}
        return res

    import asyncio
    inflight: dict = {"task": None}

    async def update_task():
        try:
            res = await run_analysis(final=False)
            if res:
                res.pop("features", None)
                await ws.send_text(json.dumps({"type": "update", "result": res}))
        except Exception as exc:  # pragma: no cover
            log.warning("live update failed: %s", exc)

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes"):
                sess.add(msg["bytes"])
                # keep reading (so protocol pings are answered); run at most one analysis at a time
                if sess.n - sess.last_analyzed >= interval * 8000 and (inflight["task"] is None or inflight["task"].done()):
                    sess.last_analyzed = sess.n
                    inflight["task"] = asyncio.create_task(update_task())
            elif msg.get("text"):
                try:
                    m = json.loads(msg["text"])
                except Exception:
                    continue
                t = m.get("type")
                if t == "agent":
                    sess.agent_events.append({"event": m.get("event"), "t": float(m.get("t", sess.seconds())),
                                              "text": m.get("text", ""), "kind": m.get("kind", "")})
                elif t == "stop":
                    if inflight["task"] is not None and not inflight["task"].done():
                        await inflight["task"]
                    res = await run_analysis(final=True)
                    payload = {"type": "final", "result": res}
                    if res is not None:
                        payload["wav_b64"] = base64.b64encode(to_wav_bytes(sess.audio(), sess.agent_track())).decode("ascii")
                    await ws.send_text(json.dumps(payload))
                elif t == "ping":
                    await ws.send_text(json.dumps({"type": "pong", "seconds": sess.seconds()}))
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover
        log.exception("live session error: %s", exc)
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
