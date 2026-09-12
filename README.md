# Altur Voice Shield — synthetic caller detection for bank phone lines

HackMTY 2026 · Altur challenge *"Defend the Bank Against Voice Deepfakes"*.

Given a recorded phone conversation (stereo 8 kHz WAV, channel 0 = caller, channel 1 = Altur's agent) the
system decides whether the caller is a real person or a synthetic voice, exposes the scored `POST /detect`
endpoint, and ships an analysis dashboard with a live-call mode.

## Results (speaker-disjoint validation split, 71 calls the model never saw)

| Input | Accuracy | AUC | EER | Brier | Errors |
|---|---|---|---|---|---|
| Full calls (61–273 s) | **98.6 %** | 0.998 | 1.5 % | 0.013 | 1 miss, 0 false alarms |
| First 60 s only | **100 %** | 1.000 | 0 % | 0.000 | none |
| First 30 s only | 98.6 % | 0.975 | 1.5 % | 0.015 | 1 miss, 0 false alarms |

Grouped 5-fold cross-validation on the train split (282 calls, folds never share a caller): AUC 1.000.
Latency of `/detect` on a laptop CPU: ~1.3 s for a full 2.5-minute call, ~0.3 s for a 20 s clip
(see `bench/benchmark.py`; the number that matters for the judges' "how fast" criterion is that **30–60 s
of audio is enough for the full accuracy**).

## How it works — three signal families, one calibrated decision

```
WAV ─► decode/resample ─► shared STFT grid (50 ms / 10 ms) ─► VAD on both channels
        │
        ├─ Acoustic (caller channel, ~150 features)      ─┐
        ├─ Conversational (turn timeline of both channels) ─┼─► 185-dim vector ─► L2 logistic regression
        └─ Semantic (optional: Whisper + Claude judge)   ─┘        (C = 0.03, class-balanced)
                                                                            │
                        evidence weighting (seconds of caller speech) ◄─────┘
                                            │
                              {"is_synthetic": bool, "confidence": p(verdict correct)}
```

**1. Acoustic** (`backend/features/acoustic.py`) — pure numpy/scipy, no neural net needed:
long-term spectrum shape (out-of-band energy, roll-off frequency and steepness = the *"sharp cut"*),
prosody (pitch variability, contour smoothness, jitter, shimmer, harmonics-to-noise), loudness dynamics,
syllable rate and its variability, modulation spectrum, LFCC/MFCC statistics, noise-floor level /
stationarity / digital silence, onset-offset ramps and hard cuts, breath events, LPC formants vs. pitch
(pitch-shift / voice-changer cue), mains hum, and cross-channel cues (acoustic echo of the agent, caller
channel behaviour while the agent talks).

**2. Conversational** (`backend/features/conversational.py`) — the challenge's own hint: the agent
interrupts, falls silent and talks over the caller. From the two VAD timelines we measure response latency
(mean / median / min / spread / fraction over 2 s), how fast the caller yields when interrupted and whether
it restarts, back-channels ("ajá", "sí") during agent speech, who fills dead air and after how long,
turn-length statistics, intra-turn pauses and false starts, plus an overall regularity index.

**3. Semantic** (`backend/features/semantic.py`, optional) — both channels are transcribed with
faster-whisper (Spanish) and Claude (`claude-opus-5`, structured JSON output, low effort) judges the caller's
replies: does it invent an answer when the agent asks about a product that does not exist, does it repeat
information back correctly, does the wording read like an LLM (complete sentences, no fillers, no
self-corrections). Cheap transcript statistics (filler rate, denial phrases, formulaic politeness) are
computed even without an API key. Because CPU transcription costs ~1 s per second of speech, this layer
runs as a **cascade**: only when the fast model is unsure (`SEMANTIC_MODE=uncertain`, 0.30 < p < 0.70).

**Decision.** All features go through a single, heavily regularised logistic regression (chosen against
gradient boosting by grouped cross-validation; the LR generalises better to unseen callers and is fully
explainable). A logit-space fusion adds the optional embedding head and the semantic probability. The final
probability is pulled toward 0.5 when there is less than ~8 s of caller speech (not enough evidence).

### What the model actually learned (train split, human vs synthetic callers)

| Cue | Human | Synthetic | Why |
|---|---|---|---|
| Energy above 3.6 kHz vs voice band | −25 dB | −35 dB | the bot's audio path is band-limited more sharply (roll-off 3.85 kHz vs 3.51 kHz) |
| Median response latency after the agent stops | 1.2 s | 2.5 s | ASR → LLM → TTS pipeline delay |
| Fastest response in the call | 0.6 s | 1.7 s | a bot never answers *fast* |
| Dead-air windows filled by the caller | 14 % | 81 % | timeout-driven turn taking |
| Noise-floor level spread (p90 − p10) | 7.3 dB | 2.3 dB | injected audio has a stationary/digital floor |
| Loudness range across phrases | 19 dB | 11 dB | TTS is level-normalised, humans move the handset |
| Syllable-rate variability (CV) | 0.16 | 0.09 | machine rhythm is regular |

The dashboard shows the per-group logit contribution of every decision ("Why the model decided").

## Threat coverage

| Attack | Where it shows up | Status |
|---|---|---|
| Cloned / TTS voice | acoustic texture, prosody, band shape, breathing | primary target, trained |
| Autonomous LLM caller (ASR→LLM→TTS) | response latency, dead-air filling, no back-channels, invented answers | primary target, trained (this is what the dataset's synthetic callers are) |
| Digital injection (no microphone) | digital silence, stationary floor, no channel behaviour while agent speaks, out-of-band energy | features present, trained |
| Replay through a handset | reverberant decay tails + loudspeaker ripples/hum **combined with** synthetic-voice cues | heuristic attack profile only (no labelled replay data in the set) |
| Voice changer / pitch shift | pitch-vs-formant mismatch, vocoder texture | heuristic attack profile only |
| A different human who sounds similar | needs an enrolled voiceprint of the real customer | **out of scope of `/detect`** (the challenge labels such a caller as human); add speaker verification against the account's enrolled embedding as a next step |

`POST /analyze` returns an *attack profile*: how the synthetic probability splits across these attack
types, plus ten interpretable aspect ratings (0 = human-like, 100 = synthetic-like) with the cues behind them.

## API

```
POST /detect            scored endpoint
  body: {"audio": "<base64 of the stereo 8 kHz WAV>"}
        any key containing audio/wav/clip/data/file/b64 is accepted, nested JSON too,
        as are raw WAV bytes, a bare base64 body, a data: URI and multipart uploads
  200:  {"is_synthetic": true, "confidence": 0.93}
        confidence = probability that the returned verdict is correct (0.5 … 1.0)
  400:  audio could not be decoded (a pipeline failure still returns 200 with confidence 0.5)

POST /analyze[?semantic=1&ui=1]   everything: p_synthetic, aspects, attack_profile, events, timeline,
                                  features, contributions, spectrogram, transcript + judge (if enabled)
POST /analyze?sample=<name>.wav   analyse a file from SAMPLES_DIR (dashboard dropdown)
GET  /health                      model, metrics, optional layers, uptime
WS   /ws/live                     live call: binary int16 8 kHz PCM frames in, rolling verdict out
GET  /                            dashboard
```

Things to confirm with the Altur engineers on site (the brief does not specify them): the JSON key that
carries the base64 clip (any reasonable key works, see above) and whether `confidence` should be
*p(verdict correct)* (what we return) or *p(synthetic)*; switch with one line in `backend/main.py` if needed.

## Run

```powershell
# Windows (creates .venv and installs requirements on first run)
.\run.ps1                                    # http://localhost:8000
.\run.ps1 -Port 8010 -Workers 4 -Samples C:\path\to\audio
# or, for the dashboard with the dataset dropdown:
.venv\Scripts\python.exe dev_server.py --port 8010
```

```bash
./run.sh                                      # Linux / macOS
PORT=8010 WORKERS=4 SEMANTIC_MODE=uncertain ANTHROPIC_API_KEY=sk-ant-... ./run.sh
```

Environment (see `.env.example`): `MODEL_PATH`, `SEMANTIC_MODE=off|uncertain|always`, `WHISPER_MODEL`,
`ANTHROPIC_API_KEY`, `ENABLE_EMBEDDINGS`, `SAMPLES_DIR`, `DECISION_THRESHOLD`, `MAX_SECONDS`.

Tests: `.venv\Scripts\python.exe -m pytest -q` (14 end-to-end API tests on generated clips, no dataset needed).

Judge-style benchmark against a running server:

```bash
python -m bench.benchmark --url http://localhost:8000/detect --audio audio/ --manifest manifest.csv --split val
python -m bench.benchmark ... --clip 30          # send only the first 30 s of every call
```

## Dashboard

* **Analyze recording** — drop a WAV (or pick a dataset call): verdict, confidence, ten aspect ratings with
  their cues, attack profile, spectrogram with both speaker lanes, response-latency labels, interruption /
  silence-fill / back-channel markers, breath marks, pitch track, model contributions, raw JSON. Batch mode
  runs many files through `/detect` and tabulates verdicts and latency.
* **Live call with the agent** — the browser plays a bank agent (speech synthesis, Spanish) that runs the
  same flow as Altur's agent: greeting, a question it deliberately interrupts, a repeat-back folio, a probe
  about a non-existent insurance product, a 5-second silence, a closing. Your microphone is streamed to
  the server at 8 kHz; the verdict, aspects and turn events update every 2 s; "End call" returns the full
  report and the recorded WAV.

## Training

```bash
python -m training.build_features --audio audio/ --manifest manifest.csv --out data/features.csv --clip 30 --clip 60
python -m training.train --features data/features.csv --out models/detector.joblib
# optional SSL embedding head (needs torch + transformers): add --embeddings to build_features and train
```

`build_features` extracts the same features the server uses (no train/serve skew) for every call and for
30 s / 60 s prefixes of it, so the model also sees short excerpts. `train` selects the model by grouped
cross-validation, evaluates on the untouched speaker-disjoint `val` split, decides whether Platt calibration
helps on unseen speakers (it did not: raw probabilities had lower log-loss, so they are deployed), refits on
train + val and writes `models/detector.joblib` plus `models/train_report.json`. Count-like features that
scale with clip length are excluded so the model does not learn "long call = human".

## Judge-day checklist

1. `.\run.ps1 -Workers 4` (16 threads on this laptop analyse ~4 calls/s); the first request after start is
   already warm.
2. Expose the port (`cloudflared tunnel --url http://localhost:8000` or ngrok) and test with
   `bench/benchmark.py --url https://<tunnel>/detect --dir some_clips/`.
3. Keep `SEMANTIC_MODE=off` unless a GPU is available (Whisper on CPU adds ~1 s per second of speech).
4. Confirm the request JSON key and the confidence semantics with the Altur engineers (see API).
5. Show the dashboard on a val call and on a live call; open "Why the model decided".

## Layout

```
backend/            FastAPI app (main.py), audio decoding, VAD, features/, scoring/ (heuristics, model, pipeline)
frontend/           dashboard (index.html, app.js, style.css) — served by the backend
training/           build_features.py, train.py
bench/              benchmark.py (judge-style client), make_clips.py (synthetic smoke-test clips)
tests/              pytest end-to-end API tests
models/             detector.joblib + train_report.json
```

## Honest limitations

* The dataset contains one agent flow and one family of synthetic callers; the hidden set uses unseen
  callers and voices, which is what the speaker-disjoint validation estimates. Engines that keep human-like
  latency and full-band audio would fall back on the acoustic-texture and semantic layers.
* VAD is energy-based (fast, codec-agnostic); the provided `turns/*.json` were used only to sanity-check it.
* The replay and voice-changer scores are heuristic (no labelled examples); the similar-human case needs
  speaker verification against an enrolled voiceprint, which the `/detect` contract cannot express.
* The synthetic smoke-test clips in `bench/make_clips.py` are caricatures for CI only, never training data.
