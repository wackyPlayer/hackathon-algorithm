# Calliope — autenticación de voz con IA (synthetic caller detection for bank phone lines)

HackMTY 2026 · Altur challenge *"Defend the Bank Against Voice Deepfakes"*.

Given a recorded phone conversation (stereo 8 kHz WAV, channel 0 = caller, channel 1 = Altur's agent) the
system decides whether the caller is a real person or a synthetic voice, exposes the scored `POST /detect`
endpoint, and ships an analysis dashboard with a live-call mode.

## Results

Speaker-disjoint validation split of the Altur dataset (71 calls the model never saw), plus a corpus of
synthetic callers made with **other TTS engines** (ElevenLabs flash / turbo / multilingual / v3 and Edge neural
voices, 132 calls, see *Other TTS engines* below) that the previous model mostly missed:

| Input | Accuracy | AUC | Errors |
|---|---|---|---|
| Dataset val, full calls (61–273 s) | **100 %** | 1.000 | none |
| Dataset val, first 60 s | 98.6 % | 1.000 | 0 misses, 1 false alarm |
| Dataset val, first 30 s | 98.6 % | 1.000 | 0 misses, 1 false alarm (a clip with 5 s of speech) |
| Other-engine TTS, voices never seen (36 val calls) | **100 % detected** | – | none |
| Other-engine TTS, *engine* never seen (leave-one-engine-out) | 89–94 % detected | – | 0 false alarms |

Before this round the same detector caught only **45 %** of the other-engine TTS calls (54 % with bot-like
timing, 35 % with human-like timing): it had learned the dataset's one bot pipeline, not "synthetic voice".
Adding the TTS corpus (through a randomised telephony-channel simulation, plus channel-augmented copies of
the dataset's own calls so that the simulated channel cannot become a shortcut) and new per-turn consistency /
intra-turn pause features fixed that without losing dataset accuracy. Grouped 5-fold cross-validation on the
training rows (folds never share a caller or a TTS voice): AUC 0.998.

Latency of `/detect` end-to-end over HTTP on a laptop CPU (16 threads, one worker): ~1.3 s for a full
2.5-minute call, 0.3 s for a 20 s clip. **30–60 s of audio is enough for the full accuracy**; the live mode
gives a rolling verdict every 2 s.

The deployed `models/detector.joblib` is refit on train + val (dataset and TTS corpus), so `bench/benchmark.py`
and `bench/eval_corpus.py` on those files reproduce the endpoint contract and latency but are *not*
generalisation estimates; the table above (from `training/train.py` with val untouched, and the
leave-one-engine-out runs) is.

## How it works — three signal families, one calibrated decision

```
WAV ─► decode/resample ─► shared STFT grid (50 ms / 10 ms) ─► VAD on both channels
        │
        ├─ Acoustic (caller channel, ~170 features)      ─┐
        ├─ Conversational (turn timeline of both channels) ─┼─► 204-dim vector ─► L2 logistic regression
        └─ Semantic (optional: Whisper + Claude judge)   ─┘        (C = 0.03, class-balanced, Platt-calibrated)
                                                                    + acoustic-only head (voice texture, shown as a signal)
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
channel behaviour while the agent talks). Cues aimed specifically at neural TTS: **per-turn consistency** of
level, spectral tilt, high-band balance and centroid (an engine renders every utterance alike; a person moves
the handset and changes effort between turns), the noise floor **inside** a turn versus **between** turns
(injected TTS carries the engine's own silence inside a turn and the channel floor between turns), the
frame-to-frame swing of the high band, and the regularity of inter-syllable intervals.

**2. Conversational** (`backend/features/conversational.py`) — the challenge's own hint: the agent
interrupts, falls silent and talks over the caller. From the two VAD timelines we measure response latency
(mean / median / min / spread / fraction over 2 s), how fast the caller yields when interrupted and whether
it restarts, back-channels ("ajá", "sí") during agent speech, who fills dead air and after how long,
turn-length statistics, intra-turn pauses and false starts, plus an overall regularity index.

**3. Semantic** (`backend/features/semantic.py`) — what the caller actually *says*. Both channels are
transcribed with faster-whisper (Spanish) and a judge (Claude `claude-opus-5` when the Anthropic SDK and
credentials exist, otherwise **Gemini** through `GEMINI_API_KEY`, same JSON schema) reads the dialogue: does
the caller invent an answer when the agent asks about a product that does not exist ("a person says *I don't
have that*, a language model tends to invent an answer"), does it repeat information back correctly, does the
wording read like an LLM (complete sentences, no fillers, no self-corrections). Cheap transcript statistics
(filler rate, denial phrases, formulaic politeness) are computed even without a judge. In the analysis tab it
is opt-in (CPU transcription costs ~1 s per second of speech; `SEMANTIC_MODE=uncertain` runs it as a cascade
when the fast model is unsure); in the **live call** the caller is transcribed turn by turn anyway, so the
judge runs on the finished transcript at "End call" and its probability is fused into the final verdict.

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

### Other TTS engines: the synthetic-caller corpus

The dataset contains one family of synthetic callers, so a model fit to it alone learns *that* pipeline.
`training/tts_corpus.py` builds calls with real, current TTS engines and the same agent flow (greeting,
interrupted question, folio repeat-back, non-existent product probe, 5 s silence, last question, closing):

* **caller voices**: 11 ElevenLabs premade voices (`eleven_flash_v2_5`, `eleven_turbo_v2_5`,
  `eleven_multilingual_v2`, `eleven_v3`) and 11 Edge neural Spanish voices (free), speaking 23 scenario scripts
  (8 hand-written, 15 written by Gemini) with random rate / pitch / stability; the agent is a fixed Edge voice;
* **two timing profiles per set of lines**: `bot` (long, regular response latency, fills the dead air, no
  back-channels) and `humanlike` (fast variable latency, back-channels, yields to the interruption, false
  starts) — the hard case where only the voice itself gives the caller away;
* **randomised telephony channel**: level, spectral tilt, band-limit, G.711 mu-law companding, one of four
  noise-floor behaviours (digital silence, dither, stationary, drifting microphone floor with bursts) with SNRs
  matched to the real calls (38–57 dB), optional short reverb;
* **label-neutral augmentation**: every training call of the dataset also goes through the same channel
  simulation (`--augment-dataset`), so "simulated channel" cannot become the cue;
* ~1/3 of the voices per engine are held out as `val` (voice-disjoint); `train.py --exclude-group-prefix
  elevenlabs:` / `edge:` gives the leave-one-engine-out numbers.

Every synthesised line is cached under `data/tts/cache/`; the ElevenLabs spend is capped with `--el-credits`
(this corpus cost ~5.5 k characters). `bench/eval_corpus.py` runs the deployed pipeline on any labelled
directory and reports detection rate, false alarms and the feature groups behind every error, per timing
profile / engine / model.

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
POST /miccheck[?ns=1&agc=1&sr=..] microphone quality report for a short PCM/WAV sample (quality, warnings, metrics)
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

Environment (see `.env.example`; a `.env` file next to `README.md` is loaded automatically): `MODEL_PATH`,
`SEMANTIC_MODE=off|uncertain|always`, `WHISPER_MODEL`, `ANTHROPIC_API_KEY` (judge; Gemini is used otherwise),
`ENABLE_EMBEDDINGS`, `SAMPLES_DIR`, `DECISION_THRESHOLD`, `MAX_SECONDS`; live agent: `GEMINI_API_KEY`,
`GEMINI_MODEL`, `GEMINI_FALLBACK_MODEL`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `ELEVENLABS_MODEL`,
`LIVE_WHISPER_MODEL`, `LIVE_ANSWER_TIMEOUT_S`, `LIVE_TALK_MOD_DB`, `LIVE_PEAK_MARGIN_DB`, `LIVE_ESCALATE_P`.

Tests: `.venv\Scripts\python.exe -m pytest -q` (14 end-to-end API tests on generated clips, no dataset needed).

Judge-style benchmark against a running server:

```bash
python -m bench.benchmark --url http://127.0.0.1:8000/detect --audio audio/ --manifest manifest.csv --split val
python -m bench.benchmark ... --clip 30          # send only the first 30 s of every call
python -m bench.benchmark ... --concurrency 4    # parallel requests (run the server with --workers 4)
```

Use `127.0.0.1`, not `localhost`, for local clients on Windows: Python resolves `localhost` to IPv6 first
and the fallback adds ~2 s per request that has nothing to do with the detector.

## Dashboard

* **Analyze recording** — drop a WAV (or pick a dataset call). The verdict and its confidence dominate the
  page; under the probability bar the *Human* / *Synthetic* labels are ringed green / red until the verdict is
  known, then the losing one turns gray. Below them the **three security checks of the brief** — *Voice*,
  *Conversation* (how the caller handles the agent interrupting, falling silent and talking over them: people
  recover instantly and messily, machines consistently) and *Semantics* (what the caller says when asked to
  repeat information or about things that do not exist) — each with its score, the strongest cue behind it and
  what it measures; *All indicators* expands the ten underlying indicators (0 = human-like, 100 =
  synthetic-like) and each opens to its measurements. Then the semantic transcript and judgement (when run),
  the spectrogram with both speaker lanes, response-latency labels, interruption / silence-fill / back-channel
  markers, breath marks and pitch track; attack profile, model signals, feature contributions and the raw JSON
  are collapsed sections. Batch mode runs many files through `/detect` and tabulates verdicts and latency.
* **Live call with the agent** — you are the caller; the bank agent "Marina" is generated live on the server
  (`backend/live_agent.py` + the `/ws/live` session in `backend/main.py`):
  **Gemini** writes every line from what you just said (`GEMINI_API_KEY`, default model `gemini-3.1-flash-lite`,
  ~1.3 s per line; falls back to `gemini-3.6-flash` with minimal thinking, then to canned Spanish lines),
  **ElevenLabs** speaks it (`ELEVENLABS_API_KEY` + `ELEVENLABS_VOICE_ID`; without a key the browser's Spanish
  speech synthesis is used) and **faster-whisper** (`LIVE_WHISPER_MODEL=base`, pre-loaded at start-up) hears
  you. The flow is the same as Altur's agent: greeting, a question the agent deliberately interrupts after
  2.5 s of your speech, a folio you repeat back, a probe about an insurance product that does not exist, a
  5-second silence, a last question the agent talks over, a closing. End-of-turn detection, the interruption and the no-answer
  timeout (9 s) run server-side on the 8 kHz stream; the browser only captures the microphone and plays the
  agent's audio, reporting when it starts and stops so the turn timeline is exact. The verdict, aspects and
  events update every 2 s; "End call" returns the full report, the transcript and the recorded WAV.
  **Hearing the caller.** The noise floor is the 10th percentile of the last 15 s of 100 ms block levels
  (seeded from the microphone check), never a fixed level: a −40 dBFS room used to read as continuous speech,
  so no turn ever ended and the agent never answered. A block above the floor is *talking* only if the level
  swings the way speech does (p90 − p10 over the last 0.8 s above 3 dB, `LIVE_TALK_MOD_DB`); a raised but
  steady level is *background noise* (a fan, traffic, a gain step): it cannot open a turn or keep one from
  ending, and after 2 s it becomes the new floor. The "hearing:" line under the captions shows which of the
  three the server currently takes the input for, with level, swing and floor. Turns are capped at 20 s,
  caller and agent actions run under one lock (an answer is queued, never dropped), the browser guards the
  agent's end-of-audio events with watchdogs, and speech the transcriber cannot read gets a "¿me lo puede
  repetir?" instead of silence. Captions show the agent's current line and your last transcription.
  **A distrustful agent.** Marina's prompt tells her to distrust the caller (a possible ASR → LLM → TTS bot),
  never to confirm products or data the caller claims, and gives her tactics that trip up a language model
  (false premise, reversed folio, immediate-environment detail, short-answer instructions, sequence tasks,
  abrupt topic changes). Every turn she receives the detector's reading: the **call average** of the rolling
  synthetic score (not the latest reading), the latest reading, evidence and the strongest cues, never
  revealed to the caller. Up to 50 % she follows the script; 50–70 % adds a light check; above 70 %
  (`LIVE_ESCALATE_P`) an extra challenge step is inserted and the remaining questions are asked at the
  "alto" level (harder, more concrete, re-asked when the answer is generic). The live panel shows the call
  average and the agent's level.
* **Microphone check** — *Test microphone* (and, automatically, the first *Start call*) records 2 s of silence
  and a spoken sentence, sends it to `POST /miccheck` and grades the input **good / fair / poor** with plain
  warnings and the risk they carry: digital silence or a perfectly constant floor from a noise gate, browser
  noise suppression or automatic gain control, clipping and narrow-band (Bluetooth) input all make a real voice
  look like a synthetic pipeline (*possible false positive*); a very noisy line or mains hum masks the cues
  (*possible false negative* / unreliable verdict). The report is computed from the same features the detector
  uses, so it describes what the model will actually see.
* **Look** — minimal: the Calliope logo palette (shield blue `#2f6be4` on the light gray ground, dark gray
  text, white cards), the *Switzer* typeface (Fontshare, bundled in `frontend/fonts/`) for everything, thin
  borders and rounded cards, secondary information in smaller gray text. A **dark mode** toggle sits in the
  header (☾ / ☀, remembered in the browser; the system preference is the default). The spectrogram maps a
  fixed 55 dB window below the loudest bins (not silence-floor-to-peak), which keeps harmonics readable
  instead of saturating speech.

## Sharing the dashboard and API with other people

```powershell
.\run.ps1 -Port 8010          # host: start the server (or python dev_server.py)
.\share.ps1 -Port 8010        # host: public HTTPS link via a Cloudflare quick tunnel (installs cloudflared with winget)
```

`share.ps1` / `share.sh` print a `https://….trycloudflare.com` link and write it to `share_url.txt`; the
**Share link** button in the header then shows and copies it (`GET /share` also lists the LAN address). The
public link exposes everything a non-host user needs: the dashboard and its static files, `POST /detect`,
`POST /analyze`, `GET /samples`, the `/ws/live` WebSocket and `GET /health`. CORS is open on every route and
`HEAD /` is answered, so external tools and uptime checks work too. HTTPS matters: browsers only allow the
microphone on `https://` or `localhost`, so the live call for other people needs the tunnel link, while the
analysis endpoints also work over the plain LAN address.

## Training

```bash
# 1. synthetic-caller corpus with other TTS engines (ELEVENLABS_API_KEY in .env; Edge voices need only pip install edge-tts)
python -m training.tts_corpus --out data/tts_corpus --el-credits 5500 --el-voices 12 --scenarios-per-voice 1 --augment-dataset
# 2. features for the dataset (full calls + 30 s / 60 s prefixes) and the corpus
python -m training.build_features --audio audio/ --manifest manifest.csv --out data/features.csv --clip 30 --clip 60 --extra data/tts_corpus data/tts_corpus/manifest.csv
# 3. train (logistic regression, acoustic head kept as a displayed signal); prints val metrics per source
python -m training.train --features data/features.csv --out models/detector.joblib --acoustic-head --fusion none --only-lr --force lr_C0.03
python -m training.train ... --exclude-group-prefix elevenlabs:     # leave-one-engine-out estimate
# 4. behaviour check of the deployed pipeline on any labelled directory
python -m bench.eval_corpus --audio data/tts_corpus --manifest data/tts_corpus/manifest.csv --by profile engine
# optional SSL embedding head (needs torch + transformers): add --embeddings to build_features and train
```

`build_features` extracts the same features the server uses (no train/serve skew) for every call and for
30 s / 60 s prefixes of it, so the model also sees short excerpts; `--extra` adds labelled corpora whose
manifest carries a `group` (cross-validation group, e.g. `engine:voice`) and a `source` column. `train`
selects the model by grouped cross-validation (`--only-lr` skips the slow gradient-boosting candidates, which
tied the logistic regression on val but cannot be explained), evaluates on the untouched speaker-disjoint
`val` split per source, decides whether Platt calibration helps on unseen speakers (with the TTS corpus it
does, and it is deployed), optionally trains the acoustic-only head and prints val metrics for the `max` /
`mean` / `stack` fusion rules (none beat the main model, so `--fusion none` is deployed and the head is only
shown as a signal), refits on train + val and writes `models/detector.joblib` plus `models/train_report.json`.
Count-like features that scale with clip length are excluded so the model does not learn "long call = human".

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
  callers and voices, which is what the speaker-disjoint validation estimates. Robustness to *other* TTS
  engines is estimated with the ElevenLabs / Edge corpus (leave-one-engine-out 89–94 % detection); the corpus
  goes through a simulated telephony channel, not a real carrier, and the caller scripts are read by TTS rather
  than produced by an LLM, so an unseen engine on a real line may still land below those numbers.
* The microphone check grades the input path, not the person: it can only warn that a verdict may be biased.
* VAD is energy-based (fast, codec-agnostic); the provided `turns/*.json` were used only to sanity-check it.
* The replay and voice-changer scores are heuristic (no labelled examples); the similar-human case needs
  speaker verification against an enrolled voiceprint, which the `/detect` contract cannot express.
* The synthetic smoke-test clips in `bench/make_clips.py` are caricatures for CI only, never training data.
