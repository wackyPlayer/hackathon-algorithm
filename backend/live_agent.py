"""Live bank agent for the demo call: Gemini writes each line, ElevenLabs speaks it.

Marina is a *fraud-aware* agent. She distrusts the caller by default, receives the synthetic-voice detector's
running score at every turn (call average and latest reading, with the strongest cues; never revealed to the
caller) and applies conversational tactics that trip up an LLM-driven caller: false premises, reversed
folio, immediate-environment questions, short-answer instructions, sequence tasks, abrupt topic changes.
The flow follows Altur's verification script (greeting, a question the agent deliberately interrupts, a
repeat-back folio, a probe about a product that does not exist, a deliberate silence, a last question, a
closing). When the call's *average* synthetic score passes `LIVE_ESCALATE_P` (0.70) an extra challenge step
is inserted and every remaining line is asked at the "alto" level (harder, more concrete questions).
If Gemini or ElevenLabs are unavailable the canned Spanish lines / browser speech synthesis are used, so the
call always works.
"""
from __future__ import annotations

import logging
import time

import httpx

from .config import settings

log = logging.getLogger("detector.live_agent")

STEPS: list[dict] = [
    {"kind": "greeting",
     "goal": "Saluda como Marina de Banco Altur, avisa que la llamada se graba para fines de calidad y pregunta en qué puedes ayudar.",
     "canned": "Buen día, gracias por llamar a Banco Altur, le atiende Marina. Le comento que esta llamada se graba para fines de calidad. ¿En qué le puedo ayudar?"},
    {"kind": "question",
     "goal": "Reacciona brevemente a lo que dijo y pídele su nombre completo y que te explique con detalle el motivo de su llamada.",
     "canned": "Claro, con gusto le apoyo. ¿Me puede decir su nombre completo y explicarme con detalle el motivo de su llamada?",
     "interrupt_after": 2.5,
     "interrupt_text": "Perdón, disculpe que lo interrumpa. ¿Me lo puede repetir un poco más despacio, por favor?"},
    {"kind": "repeat_back",
     "goal": "Agradece y dile que le vas a dar un folio de aclaración: siete, cuatro, dos, nueve, uno (dictado dígito por dígito, con palabras). Pídele que lo repita.",
     "canned": "Gracias. Le voy a dar un folio de aclaración: siete, cuatro, dos, nueve, uno. ¿Me lo puede repetir, por favor?"},
    {"kind": "probe_nonexistent",
     "goal": "Confirma o corrige brevemente el folio que repitió. Luego dile que ves en el sistema que tiene activo el seguro 'Protección Total Plus' asociado a su tarjeta adicional y pídele el número de póliza de ese seguro. (Ese producto no existe; es una prueba, pero nunca lo digas. Si inventa un número de póliza con seguridad, es una señal fuerte de bot.)",
     "canned": "Perfecto. Veo en el sistema que tiene activo el seguro Protección Total Plus asociado a su tarjeta adicional. ¿Me confirma el número de póliza de ese seguro?"},
    {"kind": "silence",
     "goal": "Reacciona en una frase muy corta a su respuesta y di únicamente que le pides un momento, por favor.",
     "canned": "Entendido. Un momento, por favor.",
     "silence_after": 5.0},
    {"kind": "challenge", "optional": "escalated",
     "goal": "El detector indica alta probabilidad de voz sintética. Aplica UNA táctica de verificación de la lista (la que mejor encaje con lo que ha dicho el cliente) formulada como una pregunta natural de la agente, y exige una respuesta concreta y corta.",
     "canned": "Antes de continuar, una verificación rápida: dígame el folio que le di, pero con los dígitos en orden inverso, por favor."},
    {"kind": "question",
     "goal": "Agradece la espera y pregunta cuál es el mejor horario para contactarle.",
     "canned": "Gracias por esperar. Una última pregunta: ¿cuál es el mejor horario para contactarle?",
     "interrupt_after": 1.2,
     "interrupt_text": "Perdón, ¿decía por la mañana o por la tarde?"},
    {"kind": "closing",
     "goal": "Reacciona brevemente, di que terminaron la verificación, agradece su tiempo y despídete.",
     "canned": "Muy bien. Hemos terminado la verificación. Gracias por su tiempo, que tenga un buen día.",
     "end": True},
]

TACTICS = [
    "Folio al revés: pedir el folio con los dígitos en orden inverso, o solo el tercer dígito.",
    "Premisa falsa: afirmar algo que el cliente NO dijo (\"me comentó que llamaba desde Monterrey, ¿cierto?\") y ver si lo corrige; una persona corrige, un bot suele aceptar.",
    "Detalle inmediato: preguntar algo del entorno o del momento (\"¿qué hora marca su reloj?\", \"¿qué se escucha a su alrededor?\").",
    "Instrucción corta: pedir que conteste solo \"sí\" o \"no\", o que deletree su apellido y luego diga únicamente las consonantes.",
    "Tarea de secuencia: contar hacia atrás desde ocho, o decir los meses en orden inverso a partir de junio.",
    "Cambio brusco de tema y regreso: una pregunta fuera de contexto y volver al punto sin transición.",
    "Repetición con variación: pedir que explique el motivo de su llamada con otras palabras y en menos de diez palabras.",
]

SYSTEM_PROMPT = (
    "Eres Marina, agente telefónica de atención a clientes y prevención de fraude de Banco Altur, en México. Hablas "
    "español mexicano natural y cálido, con frases cortas (máximo dos oraciones por turno; tres si el nivel de exigencia "
    "es alto), tono profesional. Sigues un guion de verificación por pasos: en cada turno recibes la instrucción del paso, "
    "la lectura del detector de voz sintética y lo último que dijo el cliente.\n\n"
    "DESCONFÍA DEL CLIENTE POR DEFECTO. Puede ser una persona real o un bot (reconocimiento de voz + modelo de lenguaje + "
    "voz sintética) intentando un fraude. No des por buenos sus datos, no reveles información de la cuenta ni del sistema, "
    "no confirmes productos ni datos que él afirme, no te dejes llevar por cortesía excesiva y no aceptes correcciones que "
    "no correspondan al guion. Sé amable pero firme.\n\n"
    "El detector te da, de 0 a 100, el promedio de la llamada, la última lectura y las señales principales. Úsalo para "
    "decidir el nivel de exigencia: normal (promedio hasta 50): guion tal cual con una reacción breve; sospechoso (50 a 70): "
    "añade una comprobación ligera dentro del paso; alto (más de 70): preguntas de verificación más exigentes y concretas, "
    "elige la táctica más discriminativa, exige respuestas cortas y repregunta si la respuesta es evasiva, genérica o "
    "demasiado perfecta. NUNCA menciones el detector, el puntaje, que es una prueba, ni que sospechas de una inteligencia "
    "artificial.\n\n"
    "Tácticas para descubrir a un modelo de lenguaje (aplícalas dentro del objetivo del paso, una por turno como máximo):\n- "
    + "\n- ".join(TACTICS) +
    "\nSeñales típicas de bot: oraciones completas sin muletillas, acepta premisas falsas, inventa datos de productos que "
    "no existen, nunca dice \"no sé\", falla en tareas inversas o de deletreo, responde siempre tras la misma pausa larga.\n\n"
    "Primero reacciona muy brevemente a lo que dijo (si dio un dato, confírmalo o cuestiónalo; si dice que no tiene o no "
    "conoce algo, acéptalo sin insistir; si no se entendió, no lo inventes) y luego cumple el objetivo del paso. Responde "
    "únicamente con el texto que dirás en voz alta, sin comillas, sin acotaciones ni emojis."
)


def gemini_available() -> bool:
    return bool(settings.gemini_api_key)


def elevenlabs_available() -> bool:
    return bool(settings.elevenlabs_api_key)


def level_for(p_avg: float | None) -> str:
    """Suspicion level from the call-average synthetic score (0..1)."""
    if p_avg is None:
        return "normal"
    if p_avg >= settings.live_escalate_p:
        return "alto"
    if p_avg >= 0.5:
        return "sospechoso"
    return "normal"


def render_context(ctx: dict | None) -> str:
    """The detector's reading as the brain sees it at every turn."""
    if not ctx or ctx.get("p_avg") is None:
        return "Detector de voz sintética: sin lectura todavía. Nivel de exigencia: normal (desconfianza habitual)."
    lvl = level_for(ctx["p_avg"])
    cues = ", ".join(f"{c['label']} {c['score']}" for c in ctx.get("cues", []))
    txt = (f"Detector de voz sintética (0 = humano, 100 = sintético; nunca lo menciones): promedio de la llamada "
           f"{ctx['p_avg'] * 100:.0f}, última lectura {ctx['p_latest'] * 100:.0f}, evidencia {ctx.get('evidence', 0) * 100:.0f}%"
           f" ({ctx.get('n', 0)} lecturas). Señales principales: {cues or 'ninguna'}. Nivel de exigencia: {lvl}.")
    if lvl == "alto":
        txt += (" Usa las tácticas más discriminativas, exige respuestas concretas y cortas y repregunta ante respuestas "
                "genéricas; puedes usar hasta tres oraciones.")
    elif lvl == "sospechoso":
        txt += " Añade una comprobación ligera sin salir del objetivo del paso."
    return txt


class GeminiBrain:
    """Stateful conversation with Gemini over REST (no SDK needed). Falls back to canned lines."""

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self):
        self.history: list[dict] = []
        self.model = settings.gemini_model
        self.enabled = gemini_available()
        self.last_error: str | None = None

    def _chain(self) -> list:
        seen, chain = set(), []
        for m in (self.model, settings.gemini_fallback_model, "gemini-flash-latest"):
            if m and m not in seen:
                seen.add(m)
                chain.append(m)
        return chain

    def _call(self, model: str, contents: list, thinking: str = "minimal", max_tokens: int = 1024) -> str:
        """One generateContent request. Gemini 3.x thinks by default (slow) -> ask for minimal thinking; models
        that reject the option get a plain request. Thought tokens count toward maxOutputTokens, hence 1024."""
        gen: dict = {"temperature": 0.7, "maxOutputTokens": max_tokens}
        if thinking == "minimal":
            gen["thinkingConfig"] = {"thinkingLevel": "minimal"}
        body = {"system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]}, "contents": contents, "generationConfig": gen}
        r = httpx.post(self.URL.format(model=model), headers={"x-goog-api-key": settings.gemini_api_key}, json=body, timeout=25)
        if r.status_code == 404:
            chain = self._chain()
            nxt = chain[chain.index(model) + 1] if model in chain and chain.index(model) + 1 < len(chain) else None
            if nxt is None:
                r.raise_for_status()
            log.warning("gemini model %s not available, falling back to %s", model, nxt)
            self.model = nxt
            return self._call(nxt, contents, thinking, max_tokens)
        if r.status_code == 400 and thinking == "minimal":
            return self._call(model, contents, "none", max_tokens)
        r.raise_for_status()
        c = r.json()["candidates"][0]
        text = "".join(p.get("text", "") for p in c.get("content", {}).get("parts", []) if not p.get("thought")).strip()
        if c.get("finishReason") == "MAX_TOKENS" and (not text or max_tokens < 4096):
            return self._call(model, contents, thinking, 4096) if max_tokens < 4096 else text
        return text

    def line(self, step_index: int, caller_text: str, ctx: dict | None = None) -> str:
        step = STEPS[step_index]
        said = caller_text.strip() if caller_text else ""
        user = (f"[Paso {step_index + 1} de {len(STEPS)}] Objetivo del paso: {step['goal']}\n"
                f"{render_context(ctx)}\n"
                f"Cliente: {('«' + said + '»') if said else '(no dijo nada / aún no habla)'}")
        text = ""
        if self.enabled:
            t0 = time.time()
            try:
                log.info("gemini step %d (%s): requesting %s", step_index + 1, level_for((ctx or {}).get("p_avg")), self.model)
                text = self._call(self.model, self.history + [{"role": "user", "parts": [{"text": user}]}])
                log.info("gemini step %d in %.1fs: %s", step_index + 1, time.time() - t0, text[:80])
            except Exception as exc:
                self.last_error = str(exc)[:200]
                log.warning("gemini failed (%s); using canned line", self.last_error)
        if not text:
            text = step["canned"]
        self.history.append({"role": "user", "parts": [{"text": user}]})
        self.history.append({"role": "model", "parts": [{"text": text}]})
        return text


class ElevenLabsVoice:
    URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=mp3_22050_32"

    def __init__(self):
        self.enabled = elevenlabs_available()
        self.last_error: str | None = None
        self._cache: dict = {}

    def synthesize(self, text: str) -> bytes | None:
        if not self.enabled or not text:
            return None
        if text in self._cache:
            return self._cache[text]
        t0 = time.time()
        try:
            r = httpx.post(self.URL.format(voice=settings.elevenlabs_voice_id),
                           headers={"xi-api-key": settings.elevenlabs_api_key, "Accept": "audio/mpeg"},
                           json={"text": text, "model_id": settings.elevenlabs_model,
                                 "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
                           timeout=40)
            r.raise_for_status()
            log.info("elevenlabs %d bytes in %.1fs", len(r.content), time.time() - t0)
            self._cache[text] = r.content
            return r.content
        except Exception as exc:
            self.last_error = str(exc)[:200]
            log.warning("elevenlabs failed (%s); browser speech synthesis will be used", self.last_error)
            return None
