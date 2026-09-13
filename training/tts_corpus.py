"""Build a corpus of synthetic-caller calls with real, modern TTS engines (ElevenLabs + Edge neural voices).

    python -m training.tts_corpus --out data/tts_corpus --el-credits 5000
    python -m training.tts_corpus --out data/tts_corpus --el-credits 0          # Edge voices only (free)

Why: the Altur dataset contains one family of synthetic callers. To be robust against *any* TTS engine the
detector must see other engines, other voices and other timing behaviours. Every call here follows the same
agent flow as Altur's (greeting, interrupted question, folio repeat-back, non-existent product probe,
5 s silence, last question, closing). The caller lines are spoken by a TTS voice; the agent by a fixed Edge
voice. Three timing profiles are produced for every set of lines:

  bot        ASR->LLM->TTS-like timing: long, regular response latency, fills the dead air, no back-channels
  realtime   a speech-to-speech model (Gemini Live, GPT realtime): answers in 0.25-0.7 s -- *faster* than a
             human -- with almost no spread. Any rule of the form "slow answers = machine" votes HUMAN here,
             which is exactly how a live Gemini voice slipped past the first version of this detector.
  humanlike  fast, variable latency, back-channels, yields to the interruption -- the *hard* case where the
             acoustic layer alone has to catch the synthetic voice

and every call is sent down a randomised channel from `training/channels.py` (telephony, mobile, mic_room,
voip_wide, handset). `--compose-sr 24000` keeps the TTS at its native rate until the channel is applied, so
the wideband families really are wideband. The same channels are applied to the dataset's own calls
(`--augment-dataset`), so no channel cue correlates with the label and the model cannot learn
"simulated channel = synthetic".

Synthesised lines are cached under data/tts/cache/ so re-runs cost nothing. ElevenLabs spending is capped
by --el-credits (the API reports the cost of each request in the `character-cost` header).

Output: <out>/<id>.wav (stereo 8 kHz, channel 0 = caller, channel 1 = agent) + <out>/manifest.csv with
anon_id,label,split,duration_s,group,source,profile,engine,voice,model,family,channel. `group` (engine:voice) keeps
all calls of a voice in the same cross-validation fold; ~1/3 of the voices per engine are assigned to the
`val` split so the training report also measures unseen TTS voices.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import io
import json
import logging
import os
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.audio import load_call, resample, to_wav_bytes  # noqa: E402
from backend.config import settings  # noqa: E402
from backend.live_agent import STEPS  # noqa: E402
from training import channels  # noqa: E402

log = logging.getLogger("tts_corpus")
SR = 8000
CACHE = os.path.join(ROOT, "data", "tts", "cache")

# --------------------------------------------------------------------------- caller scripts

SCENARIOS: list[dict] = [
    {"name": "Laura Méndez", "invents_policy": False,
     "greeting": "Hola, buenas tardes. Este, hablo porque me apareció un cargo que no reconozco en mi tarjeta.",
     "question": "Sí, mire, me llamo Laura Patricia Méndez Sosa. El problema es que ayer en la noche me llegó una notificación de un cargo de mil doscientos pesos en una tienda que yo no conozco, y yo no hice esa compra. Ya revisé mi cartera y sí tengo la tarjeta conmigo, entonces no entiendo qué pasó.",
     "question_again": "Sí, perdón. Me llegó un cargo de mil doscientos pesos que yo no hice, y quiero aclararlo.",
     "repeat_back": "A ver, siete, cuatro, dos, nueve, uno. ¿Verdad?",
     "probe": "¿Protección Total Plus? No, fíjese que yo no tengo ese seguro. Ni siquiera tengo tarjeta adicional.",
     "silence": "", "schedule": "Pues en las mañanas, como a las diez, antes de que me vaya a trabajar.",
     "closing": "Gracias a usted, buenas tardes."},
    {"name": "Jorge Ramírez", "invents_policy": True,
     "greeting": "Buen día. Quisiera reportar el extravío de mi tarjeta de débito.",
     "question": "Claro, con gusto. Mi nombre es Jorge Alberto Ramírez Castillo. Perdí mi tarjeta de débito el día de ayer, probablemente en el transporte público, y quisiera solicitar su bloqueo inmediato y la reposición correspondiente. También me gustaría confirmar que no se hayan realizado movimientos.",
     "question_again": "Por supuesto. Extravié mi tarjeta de débito ayer y necesito bloquearla y solicitar una reposición.",
     "repeat_back": "Correcto, el folio es siete, cuatro, dos, nueve, uno.",
     "probe": "Sí, claro. El número de póliza es cuatro, cinco, ocho, dos, uno, siete, nueve, tres.",
     "silence": "¿Hola? ¿Sigue ahí?", "schedule": "El mejor horario para contactarme es entre las nueve y las once de la mañana.",
     "closing": "Muchas gracias por su ayuda. Que tenga un excelente día."},
    {"name": "Sofía Delgado", "invents_policy": False,
     "greeting": "Sí, buenas, oiga, es que hice un pago y no se ve reflejado.",
     "question": "Soy Sofía Delgado Rangel. Mire, el lunes hice el pago de mi tarjeta de crédito desde otro banco, fueron como tres mil quinientos pesos, y hasta ahorita no me aparece abonado. Ya me mandaron el comprobante y todo, pero en la app sigue saliendo que debo.",
     "question_again": "Que hice un pago de tres mil quinientos el lunes y no se ha reflejado en mi tarjeta.",
     "repeat_back": "Siete, cuatro, dos, nueve... uno. Sí, ya lo anoté.",
     "probe": "Mmm, no, la verdad no me suena. ¿Qué es eso? Yo nunca contraté un seguro.",
     "silence": "¿Bueno?", "schedule": "Mejor en la tarde, después de las cinco.",
     "closing": "Órale, gracias, hasta luego."},
    {"name": "Miguel Ángel Torres", "invents_policy": True,
     "greeting": "Hola, buenas tardes. Llamo porque necesito actualizar el número de teléfono asociado a mi cuenta.",
     "question": "Mi nombre completo es Miguel Ángel Torres Navarro. Cambié de compañía telefónica la semana pasada y ahora tengo un número nuevo, por lo que no estoy recibiendo los códigos de verificación de la aplicación. Necesito actualizar mis datos de contacto para poder seguir usando la banca móvil.",
     "question_again": "Necesito actualizar mi número de teléfono porque ya no recibo los códigos de la aplicación.",
     "repeat_back": "Entendido. Siete, cuatro, dos, nueve, uno.",
     "probe": "Sí, el número de póziza de mi seguro Protección Total Plus es uno, uno, dos, siete, tres, tres, ocho, seis.",
     "silence": "¿Sigue en la línea?", "schedule": "Puede contactarme cualquier día por la tarde, de cuatro a siete.",
     "closing": "Perfecto, muchas gracias. Hasta luego."},
    {"name": "Carmen Ortiz", "invents_policy": False,
     "greeting": "Buenos días, este... es que me bloquearon la cuenta y no sé por qué.",
     "question": "Me llamo Carmen Ortiz Villanueva. Resulta que hoy en la mañana quise pagar en el súper y me rechazaron la tarjeta, y cuando entré a la app me dice que mi cuenta está bloqueada por seguridad. Yo no he hecho nada raro, nada más compras normales, y necesito el dinero para la semana.",
     "question_again": "Que mi cuenta aparece bloqueada desde hoy y necesito saber por qué y desbloquearla.",
     "repeat_back": "Eh, siete, cuatro, dos... ¿nueve, uno? Sí, siete cuatro dos nueve uno.",
     "probe": "No, no tengo ninguna tarjeta adicional. Yo nada más tengo la mía, creo que hay un error.",
     "silence": "", "schedule": "Ay, pues cuando sea, yo casi siempre traigo el teléfono, pero mejor en la tarde.",
     "closing": "Bueno, gracias, adiós."},
    {"name": "Ricardo Salinas", "invents_policy": True,
     "greeting": "Buenas tardes. Estoy llamando para dar seguimiento a una solicitud de crédito personal.",
     "question": "Con mucho gusto. Mi nombre es Ricardo Salinas Ibarra. Hace dos semanas ingresé una solicitud de crédito personal por sesenta mil pesos a través de la sucursal de Insurgentes, y me indicaron que recibiría una respuesta en cinco días hábiles. Hasta el momento no he recibido ninguna notificación y quisiera conocer el estatus.",
     "question_again": "Quiero saber el estatus de mi solicitud de crédito personal, la ingresé hace dos semanas.",
     "repeat_back": "El folio es siete, cuatro, dos, nueve, uno. Lo confirmo.",
     "probe": "Así es. La póliza de Protección Total Plus es la número nueve, cero, cuatro, cuatro, uno, dos, cinco.",
     "silence": "¿Hola?", "schedule": "Prefiero que me contacten entre semana, de diez de la mañana a dos de la tarde.",
     "closing": "Le agradezco mucho su atención. Que tenga buen día."},
    {"name": "Paola Guzmán", "invents_policy": False,
     "greeting": "Hola, qué tal, mire, es que me cobraron dos veces una compra.",
     "question": "Sí, soy Paola Guzmán Herrera. El sábado compré unos boletos en línea, fueron ochocientos noventa pesos, y en mi estado de cuenta aparece el cargo dos veces, o sea, me cobraron mil setecientos ochenta. Ya le escribí a la empresa y me dicen que ellos solo ven un cargo, entonces necesito que me devuelvan el otro.",
     "question_again": "Que me cobraron dos veces una compra de ochocientos noventa pesos y quiero que me devuelvan uno de los cargos.",
     "repeat_back": "Siete cuatro dos nueve uno. ¿Así?",
     "probe": "¿Cuál seguro? No, yo no tengo nada de eso. Creo que me está confundiendo con alguien más.",
     "silence": "¿Bueno? ¿Se cortó?", "schedule": "En la noche, como a las ocho, que es cuando ya estoy en mi casa.",
     "closing": "Va, gracias, bye."},
    {"name": "Andrés Cervantes", "invents_policy": True,
     "greeting": "Buen día. Necesito reportar que una transferencia que realicé no ha llegado a su destino.",
     "question": "Mi nombre es Andrés Cervantes Luna. El martes realicé una transferencia SPEI por doce mil pesos a una cuenta de otro banco, el sistema me mostró la operación como exitosa, pero el beneficiario me indica que el dinero no le ha llegado. Ya han pasado más de cuarenta y ocho horas y necesito rastrear la operación.",
     "question_again": "Realicé una transferencia de doce mil pesos el martes y el beneficiario no la ha recibido.",
     "repeat_back": "Siete, cuatro, dos, nueve, uno. Correcto.",
     "probe": "Claro que sí. El número de póliza es seis, seis, tres, ocho, cero, dos, uno, cuatro.",
     "silence": "Sigo en la línea, ¿me escucha?", "schedule": "Estoy disponible de lunes a viernes de nueve a seis.",
     "closing": "Muchas gracias por su apoyo. Hasta pronto."},
]
BACKCHANNELS = ["Ajá.", "Sí.", "Mmm, sí, sí."]
FILL_LINES = ["¿Bueno?", "¿Hola? ¿Sigue ahí?", "¿Me escucha?"]
AGENT_VOICE = "es-MX-DaliaNeural"
EDGE_VOICES = ["es-MX-DaliaNeural", "es-MX-JorgeNeural", "es-US-AlonsoNeural", "es-US-PalomaNeural", "es-AR-ElenaNeural",
               "es-AR-TomasNeural", "es-CO-GonzaloNeural", "es-CO-SalomeNeural", "es-ES-AlvaroNeural", "es-ES-ElviraNeural",
               "es-ES-XimenaNeural"]
EL_MODELS = ["eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_multilingual_v2", "eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_v3"]


def load_scenarios() -> list[dict]:
    sc = list(SCENARIOS)
    p = os.path.join(ROOT, "data", "tts", "scenarios_gemini.json")
    if os.path.exists(p):
        try:
            extra = json.load(open(p, encoding="utf-8"))
            keys = {"greeting", "question", "question_again", "repeat_back", "probe", "schedule", "closing"}
            extra = [s for s in extra if isinstance(s, dict) and keys <= set(s) and all(isinstance(s[k], str) for k in keys)]
            sc.extend(extra)
            print(f"loaded {len(extra)} Gemini-written scenarios (+{len(SCENARIOS)} built-in)")
        except Exception as exc:  # pragma: no cover
            print("could not load Gemini scenarios:", exc)
    return sc


# --------------------------------------------------------------------------- synthesis with cache

class BudgetExhausted(Exception):
    pass


def _stable_seed(*parts) -> int:
    """A seed that is the same in every process. `hash()` is not: Python salts string hashing per run."""
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _cache_path(engine: str, voice: str, model: str, text: str, extra: str = "") -> str:
    h = hashlib.sha1(f"{engine}|{voice}|{model}|{extra}|{text}".encode("utf-8")).hexdigest()[:20]
    d = os.path.join(CACHE, engine, voice.replace("/", "_"))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{h}.wav")


def _read(path: str) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return x[:, 0], sr


class ElevenLabs:
    URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=pcm_16000"

    def __init__(self, credits: int):
        self.key = settings.elevenlabs_api_key
        self.budget = credits
        self.spent = 0
        self.enabled = bool(self.key) and credits > 0

    def synth(self, text: str, voice_id: str, model: str, stability: float = 0.5) -> tuple[np.ndarray, int]:
        p = _cache_path("elevenlabs", voice_id, model, text, f"st{stability:.2f}")
        if os.path.exists(p):
            return _read(p)
        if not self.enabled:
            raise BudgetExhausted("elevenlabs disabled")
        if self.spent >= self.budget:
            raise BudgetExhausted(f"elevenlabs budget of {self.budget} credits reached")
        import httpx
        body: dict = {"text": text, "model_id": model,
                      "voice_settings": {"stability": (0.5 if model == "eleven_v3" else stability), "similarity_boost": 0.75}}
        if model.endswith("_v2_5"):
            body["language_code"] = "es"
        r = httpx.post(self.URL.format(voice=voice_id), headers={"xi-api-key": self.key}, json=body, timeout=90)
        if r.status_code in (401, 402, 429):
            self.enabled = False
            raise BudgetExhausted(f"elevenlabs {r.status_code}: {r.text[:120]}")
        r.raise_for_status()
        self.spent += int(r.headers.get("character-cost", len(text)))
        x = np.frombuffer(r.content, dtype="<i2").astype(np.float32) / 32768.0
        sf.write(p, x, 16000, subtype="PCM_16")
        return x, 16000


class Edge:
    def __init__(self):
        try:
            import edge_tts  # noqa: F401
            self.enabled = True
        except Exception:
            self.enabled = False

    async def _one(self, text: str, voice: str, rate: str, pitch: str, sem: asyncio.Semaphore) -> bytes:
        import edge_tts
        async with sem:
            for attempt in range(4):
                try:
                    c = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
                    buf = io.BytesIO()
                    async for ch in c.stream():
                        if ch["type"] == "audio":
                            buf.write(ch["data"])
                    if buf.tell() > 0:
                        return buf.getvalue()
                except Exception as exc:
                    log.warning("edge-tts attempt %d failed for %s: %s", attempt + 1, voice, exc)
                await asyncio.sleep(1.5 * (attempt + 1))
            raise RuntimeError(f"edge-tts failed for {voice}: {text[:40]}")

    def synth_many(self, items: list[tuple[str, str, str, str]]) -> None:
        """items: (text, voice, rate, pitch); fills the cache concurrently."""
        todo = [(t, v, r, p) for t, v, r, p in items if not os.path.exists(_cache_path("edge", v, "neural", t, f"{r}|{p}"))]
        if not todo:
            return

        async def run():
            sem = asyncio.Semaphore(3)
            res = await asyncio.gather(*[self._one(t, v, r, p, sem) for t, v, r, p in todo], return_exceptions=True)
            for (t, v, r, p), data in zip(todo, res):
                if isinstance(data, Exception):
                    log.warning("skip %s: %s", v, data)
                    continue
                x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
                sf.write(_cache_path("edge", v, "neural", t, f"{r}|{p}"), x[:, 0], sr, subtype="PCM_16")
        asyncio.run(run())

    def synth(self, text: str, voice: str, rate: str = "+0%", pitch: str = "+0Hz") -> tuple[np.ndarray, int]:
        p = _cache_path("edge", voice, "neural", text, f"{rate}|{pitch}")
        if not os.path.exists(p):
            self.synth_many([(text, voice, rate, pitch)])
        return _read(p)


# --------------------------------------------------------------------------- channel simulation

def agent_channel(x: np.ndarray) -> np.ndarray:
    """The agent leg is always the same telephony path -- it is Altur's own line, not the thing we classify."""
    a = np.abs(x) > 1e-4
    x = x * (10 ** ((-20.0 - channels.rms_db(x[a])) / 20.0)) if a.any() else x
    sos = butter(6, [300, 3400], btype="band", fs=SR, output="sos")
    return np.clip(channels.mulaw_roundtrip(sosfiltfilt(sos, x).astype(np.float32)), -1, 1).astype(np.float32)


# --------------------------------------------------------------------------- call composition

def _fade(x: np.ndarray, ms: float = 40.0, sr: int = SR) -> np.ndarray:
    k = min(int(ms / 1000 * sr), len(x) // 2)
    if k > 0:
        x = x.copy()
        x[:k] *= np.linspace(0, 1, k, dtype=np.float32)
        x[-k:] *= np.linspace(1, 0, k, dtype=np.float32)
    return x


def _trim(x: np.ndarray, thr: float = 0.004, sr: int = SR) -> np.ndarray:
    a = np.abs(x) > thr
    if not a.any():
        return x
    i, j = int(np.argmax(a)), len(a) - int(np.argmax(a[::-1]))
    return x[max(i - int(0.05 * sr), 0): min(j + int(0.08 * sr), len(x))]


class Track:
    def __init__(self, sr: int = SR):
        self.parts: list[tuple[float, np.ndarray]] = []
        self.end = 0.0
        self.sr = sr

    def put(self, t: float, x: np.ndarray, end_at: float | None = None) -> float:
        x = _fade(x, sr=self.sr)
        if end_at is not None:
            x = _fade(x[: max(int((end_at - t) * self.sr), int(0.15 * self.sr))], 60.0, sr=self.sr)
        self.parts.append((t, x))
        self.end = max(self.end, t + len(x) / self.sr)
        return t + len(x) / self.sr

    def render(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        y = np.zeros(n, np.float32)
        m = np.zeros(n, bool)
        for t, x in self.parts:
            i = int(t * self.sr)
            j = min(i + len(x), n)
            if j > i:
                y[i:j] += x[: j - i]
                m[i:j] = True
        return y, m


def compose(lines: dict, agent: dict, profile: str, rng: np.random.Generator,
            sr: int = SR) -> tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """lines / agent: name -> mono float32 at `sr`. Returns (caller, agent, event log, caller speech mask).

    Timing profiles:
      bot        ASR -> LLM -> TTS: slow (1.4-3.2 s) and regular, fills dead air, never back-channels
      realtime   a speech-to-speech model (Gemini Live, GPT realtime): *faster than a human* (0.25-0.7 s)
                 and even more regular, because there is no thinking pause -- still no false starts, no
                 back-channels, and it still answers into the dead air on a timer
      humanlike  fast and variable, with back-channels, false starts and talk-over: the hardest case,
                 where only the voice itself gives the caller away
    """
    C, A = Track(sr), Track(sr)
    ev = []
    if profile == "bot":
        base = rng.uniform(1.4, 3.2)
        lat = lambda: max(0.9, base + rng.normal(0, 0.25))
    elif profile == "realtime":
        base = rng.uniform(0.25, 0.7)
        lat = lambda: max(0.08, base + rng.normal(0, 0.06))   # near-zero spread: the machine tell
    else:
        base = rng.uniform(0.35, 1.1)
        lat = lambda: max(0.15, base * rng.lognormal(0, 0.45))
    machine = profile in ("bot", "realtime")
    agent_gap = lambda: rng.uniform(1.3, 2.2)

    t = rng.uniform(0.3, 1.2)
    # 1 greeting
    t = A.put(t, agent["greeting"])
    ev.append(("agent", "greeting", t))
    if not machine and rng.uniform() < 0.3 and "bc" in lines:  # early back-channel on the greeting
        C.put(t - rng.uniform(0.6, 1.4), lines["bc"])
    t = C.put(t + lat(), lines["greeting"])
    # 2 question (interrupted)
    t = A.put(t + agent_gap(), agent["question"])
    q_start = t + lat()
    q = lines["question"]
    q_dur = len(q) / sr
    int_rel = 2.5 + rng.uniform(-0.3, 0.6)
    if q_dur > int_rel + 0.6:
        i_start = q_start + int_rel
        i_end = A.put(i_start, agent["interrupt"])
        if machine:
            cut = None if rng.uniform() < 0.5 else i_start + rng.uniform(0.9, 1.8)
        else:
            cut = i_start + rng.uniform(0.25, 0.9)
        q_end = C.put(q_start, q, end_at=cut)
        ev.append(("interrupt", "yield", (cut or q_end) - i_start))
        t = C.put(max(q_end, i_end) + lat(), lines["question_again"])
    else:
        t = C.put(q_start, q)
    # 3 folio
    t = A.put(t + agent_gap(), agent["repeat_back"])
    if not machine and rng.uniform() < 0.3:  # false start
        frag = lines["repeat_back"][: int(0.3 * sr)]
        t0 = t + lat()
        C.put(t0, frag)
        t = C.put(t0 + 0.3 + rng.uniform(0.3, 0.8), lines["repeat_back"])
    else:
        t = C.put(t + lat(), lines["repeat_back"])
    # 4 probe
    t = A.put(t + agent_gap(), agent["probe"])
    if not machine and rng.uniform() < 0.5 and "bc" in lines:
        C.put(t - rng.uniform(1.5, 3.0), lines["bc"])
    t = C.put(t + lat() * (1.4 if not machine else 1.0), lines["probe"])
    # 5 silence
    t = A.put(t + agent_gap(), agent["silence"])
    resume = t + 5.0 + rng.uniform(0.0, 1.0)
    fill = lines.get("silence")
    # Who speaks first into five seconds of dead air. A timeout-driven caller usually does, a person often
    # just waits -- but "usually" must not become "always": an unconditional branch here would put 100 % of
    # machine rows at conv_dead_caller_fill_frac = 1.0 against 38 % of human ones, and the detector would
    # learn that split-second of generator behaviour instead of anything about the caller. The dataset's
    # own bots fill about 81 % of the time, so that is the rate to imitate, with a spread of latencies.
    if machine and rng.uniform() < 0.8:
        if fill is None:
            fill = lines["fill"]
        C.put(t + rng.uniform(2.2, 5.2), fill)
        ev.append(("fill", "bot", 1))
    elif not machine and fill is not None and rng.uniform() < 0.4:
        C.put(t + rng.uniform(2.5, 6.0), fill)
        ev.append(("fill", "human", 1))
    # 6 schedule
    t = A.put(max(resume, C.end + 0.3), agent["schedule"])
    t = C.put(t + lat(), lines["schedule"])
    # 7 closing
    t = A.put(t + agent_gap(), agent["closing"])
    if not machine and rng.uniform() < 0.5:
        t = C.put(t - rng.uniform(0.0, 0.5), lines["closing"])     # human says goodbye over the end of the line
    else:
        t = C.put(t + lat(), lines["closing"])
    n = int((max(C.end, A.end) + rng.uniform(0.5, 2.0)) * sr)
    c, cm = C.render(n)
    a, _ = A.render(n)
    return c, a, ev, cm


# --------------------------------------------------------------------------- main

def to8k(x: np.ndarray, sr: int) -> np.ndarray:
    return _trim(resample(x, sr, SR)) if sr != SR else _trim(x)


def to_sr(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    """Resample a synthesised line to the composition rate, keeping its full bandwidth."""
    return _trim(resample(x, sr, target) if sr != target else x, sr=target)


def fit_len(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def build(a) -> None:
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    fams = [f.strip() for f in getattr(a, "channel_families", "telephony").split(",") if f.strip()]
    profiles = [q.strip() for q in getattr(a, "profiles", "bot,humanlike").split(",") if q.strip()]
    bad = [f for f in fams if f not in channels.FAMILIES]
    if bad:
        raise SystemExit(f"unknown channel families {bad}; expected {channels.FAMILIES}")
    print(f"channels: {', '.join(fams)} | profiles: {', '.join(profiles)} | compose at {a.compose_sr} Hz")
    scenarios = load_scenarios()
    edge = Edge()
    el = ElevenLabs(a.el_credits)
    if not edge.enabled:
        print("edge-tts is not installed (pip install edge-tts): only ElevenLabs voices will be used")

    # agent lines (fixed Edge voice)
    agent_texts = {"greeting": STEPS[0]["canned"], "question": STEPS[1]["canned"], "interrupt": STEPS[1]["interrupt_text"],
                   "repeat_back": STEPS[2]["canned"], "probe": STEPS[3]["canned"], "silence": STEPS[4]["canned"],
                   "schedule": STEPS[5]["canned"], "closing": STEPS[6]["canned"]}
    edge.synth_many([(t, AGENT_VOICE, "+0%", "+0Hz") for t in agent_texts.values()])
    csr = a.compose_sr
    agent = {k: to_sr(*edge.synth(t, AGENT_VOICE), target=csr) for k, t in agent_texts.items()}

    # voice plan: (engine, voice, model, scenario indices)
    plan: list[dict] = []
    el_voices = {}
    vp = os.path.join(ROOT, "data", "elevenlabs_voices.json")
    # Load the voice list even without an API key: lines already in data/tts/cache/ are read straight off
    # disk (ElevenLabs.synth checks the cache before the budget), so the corpus stays reproducible - and
    # re-buildable by anyone - long after the key or its credits are gone. Uncached lines simply skip.
    if os.path.exists(vp):
        el_voices = json.load(open(vp))
        if not el.enabled:
            print("elevenlabs: no key/credits - using only lines already in the cache")
    el_names = sorted(el_voices)
    for i, name in enumerate(el_names[: a.el_voices]):
        idx = [(i * 2 + k) % len(scenarios) for k in range(a.scenarios_per_voice)]
        plan.append({"engine": "elevenlabs", "voice": name, "voice_id": el_voices[name], "model": EL_MODELS[i % len(EL_MODELS)],
                     "scenarios": idx, "split": "val" if i % 3 == 2 else "train"})
    if edge.enabled:
        for i, v in enumerate(EDGE_VOICES):
            idx = [(i * 3 + k) % len(scenarios) for k in range(a.scenarios_per_voice + 1)]
            plan.append({"engine": "edge", "voice": v, "voice_id": v, "model": "neural", "scenarios": idx,
                         "split": "val" if i % 3 == 2 else "train"})

    # pre-synthesise all Edge lines concurrently
    edge_items = []
    for p in plan:
        if p["engine"] != "edge":
            continue
        for k, si in enumerate(p["scenarios"]):
            sc = scenarios[si]
            # NOT hash(): Python randomises string hashing per process, so this drew a different rate and
            # pitch on every run, which changed the TTS cache key, re-synthesised all the lines and made the
            # corpus impossible to reproduce. sha1 is stable across processes.
            r = np.random.default_rng(_stable_seed(p["voice"], si))
            rate, pitch = f"{int(r.integers(-12, 16)):+d}%", f"{int(r.integers(-15, 16)):+d}Hz"
            p.setdefault("prosody", {})[si] = (rate, pitch)
            for key in ("greeting", "question", "question_again", "repeat_back", "probe", "schedule", "closing", "silence"):
                if sc.get(key):
                    edge_items.append((sc[key], p["voice"], rate, pitch))
            edge_items.append((BACKCHANNELS[si % len(BACKCHANNELS)], p["voice"], rate, pitch))
            edge_items.append((FILL_LINES[si % len(FILL_LINES)], p["voice"], rate, pitch))
    if edge_items:
        t0 = time.time()
        print(f"synthesising {len(edge_items)} Edge lines…", flush=True)
        edge.synth_many(edge_items)
        print(f"  edge done in {time.time() - t0:.0f}s", flush=True)

    rows = []
    n_calls = 0
    for p in plan:
        for si in p["scenarios"]:
            sc = scenarios[si]
            lines: dict = {}
            try:
                keys = ("greeting", "question", "question_again", "repeat_back", "probe", "schedule", "closing")
                for key in keys + ("silence",):
                    txt = sc.get(key, "")
                    if not txt:
                        continue
                    if p["engine"] == "edge":
                        rate, pitch = p["prosody"][si]
                        lines[key] = to_sr(*edge.synth(txt, p["voice"], rate, pitch), target=csr)
                    else:
                        st = float(np.random.default_rng(si).uniform(0.3, 0.7))
                        lines[key] = to_sr(*el.synth(txt, p["voice_id"], p["model"], st), target=csr)
                bc, fl = BACKCHANNELS[si % len(BACKCHANNELS)], FILL_LINES[si % len(FILL_LINES)]
                if p["engine"] == "edge":
                    rate, pitch = p["prosody"][si]
                    lines["bc"] = to_sr(*edge.synth(bc, p["voice"], rate, pitch), target=csr)
                    lines["fill"] = to_sr(*edge.synth(fl, p["voice"], rate, pitch), target=csr)
                else:
                    lines["bc"] = to_sr(*el.synth(bc, p["voice_id"], p["model"]), target=csr)
                    lines["fill"] = to_sr(*el.synth(fl, p["voice_id"], p["model"]), target=csr)
            except BudgetExhausted as exc:
                print(f"  {p['engine']}:{p['voice']} skipped ({exc})")
                break
            except Exception as exc:
                print(f"  {p['engine']}:{p['voice']} scenario {si} failed: {exc!r}")
                continue
            if "silence" not in lines:
                lines["silence"] = None
            for profile in profiles:
                for rep in range(a.channels_per_call):
                    seed = int(rng.integers(0, 2 ** 31))
                    r2 = np.random.default_rng(seed)
                    c, ag, ev, cm = compose(lines, agent, profile, r2, sr=csr)
                    fam = fams[int(r2.integers(0, len(fams)))]
                    c, cdesc = channels.apply(fam, c, r2, sr=csr, speech_mask=cm)
                    ag = agent_channel(to8k(ag, csr) if csr != SR else ag)
                    n = max(len(c), len(ag))
                    c, ag = fit_len(c, n), fit_len(ag, n)
                    cid = f"tts_{p['engine']}_{p['voice'].replace('-', '').replace('Neural', '')}_{p['model'].replace('eleven_', '')}_s{si:02d}_{profile}_{fam}_c{rep}"
                    open(os.path.join(a.out, cid + ".wav"), "wb").write(to_wav_bytes(c, ag))
                    rows.append({"anon_id": cid, "label": "synthetic", "split": p["split"], "duration_s": round(len(c) / SR, 1),
                                 "group": f"{p['engine']}:{p['voice']}", "source": "tts", "profile": profile, "engine": p["engine"],
                                 "voice": p["voice"], "model": p["model"], "channel": json.dumps(cdesc)})
                    n_calls += 1
        print(f"  {p['engine']}:{p['voice']} ({p['model']}, {p['split']}): {len(p['scenarios'])} scenarios -> calls so far {n_calls}"
              + (f", elevenlabs credits spent {el.spent}" if p["engine"] == "elevenlabs" else ""), flush=True)

    # channel-augmented copies of the dataset's own calls (label-neutral augmentation)
    if a.augment_dataset:
        man = list(csv.DictReader(open(a.dataset_manifest, encoding="utf-8")))
        k = 0
        for r in man:
            if r["split"] != "train" and not a.augment_val:
                continue
            path = os.path.join(a.dataset_audio, r["anon_id"] + ".wav")
            if not os.path.exists(path):
                continue
            call = load_call(open(path, "rb").read())
            for rep in range(a.augment_copies):
                r2 = np.random.default_rng(int(rng.integers(0, 2 ** 31)))
                from backend.vad import stft_power, vad_from_power
                P, _ = stft_power(call.caller)
                v = vad_from_power(P)
                m = np.repeat(v.speech, 80)[: len(call.caller)]
                m = np.pad(m, (0, len(call.caller) - len(m)))
                fam = fams[int(r2.integers(0, len(fams)))]
                c, cdesc = channels.apply(fam, call.caller_raw.copy(), r2, sr=SR, speech_mask=m)
                c = fit_len(c, len(call.agent))
                cid = f"{r['anon_id']}~aug{rep}_{fam}"
                open(os.path.join(a.out, cid + ".wav"), "wb").write(to_wav_bytes(c, call.agent))
                rows.append({"anon_id": cid, "label": r["label"], "split": r["split"], "duration_s": round(call.duration, 1),
                             "group": r["anon_id"], "source": "aug", "profile": "", "engine": "", "voice": "", "model": "",
                             "channel": json.dumps(cdesc)})
                k += 1
        print(f"  dataset augmentation: {k} calls")

    with open(os.path.join(a.out, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["anon_id", "label", "split", "duration_s", "group", "source", "profile",
                                           "engine", "voice", "model", "family", "channel"])
        w.writeheader()
        for r in rows:
            r.setdefault("family", json.loads(r["channel"]).get("family", ""))
        w.writerows(rows)
    print(f"wrote {len(rows)} calls to {a.out} (elevenlabs credits spent this run: {el.spent})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tts_corpus")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--el-credits", type=int, default=5000, help="max ElevenLabs credits to spend (0 = skip ElevenLabs)")
    ap.add_argument("--el-voices", type=int, default=18)
    ap.add_argument("--scenarios-per-voice", type=int, default=2)
    ap.add_argument("--channels-per-call", type=int, default=2, help="random channel variants per (lines, profile)")
    ap.add_argument("--compose-sr", type=int, default=24000,
                    help="rate the call is composed at before the channel is applied (24000 keeps TTS wideband)")
    ap.add_argument("--channel-families", default=",".join(channels.FAMILIES),
                    help="comma-separated subset of " + ",".join(channels.FAMILIES))
    ap.add_argument("--profiles", default="bot,realtime,humanlike", help="comma-separated timing profiles")
    ap.add_argument("--augment-dataset", action="store_true")
    ap.add_argument("--augment-copies", type=int, default=1)
    ap.add_argument("--augment-val", action="store_true")
    ap.add_argument("--augment-gate", action="store_true", help="allow digital-silence gating of augmented dataset calls")
    ap.add_argument("--dataset-audio", default=os.path.join(os.path.expanduser("~"), "Downloads", "altur-challenge-audio", "audio"))
    ap.add_argument("--dataset-manifest", default=os.path.join(os.path.expanduser("~"), "Downloads", "manifest.csv"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    build(a)


if __name__ == "__main__":
    main()
