# Calliope — synthetic-caller detection for bank phone lines

HackMTY 2026 · Altur, *"Defend the Bank Against Voice Deepfakes"*.

Given a recorded phone conversation (stereo 8 kHz WAV, channel 0 = caller, channel 1 = Altur's agent), the
system decides whether the **caller** is a real person or a synthetic voice, and exposes the scored endpoint:

```
POST /detect     {"audio": "<base64 of the stereo 8 kHz WAV>"}
  ->  {"is_synthetic": true, "confidence": 0.93}
```

`confidence` is **p(the caller is synthetic)**, so it is monotone: it ranks calls for tie-breaking, it can be
scored for calibration directly, and `is_synthetic == (confidence > 0.5)` always holds. (Set
`CONFIDENCE_SEMANTICS=p_correct` if your harness wants `max(p, 1-p)` instead.)

---

## The short version

The first version of this detector scored **100 % on the provided validation split** and then called a live
Gemini voice, played into a laptop microphone, **human**.

That gap is the whole story. It had not learned what a synthetic voice sounds like; it had learned what the
dataset's *recording path* sounds like. In the provided data the human callers arrive full-band to 4000 Hz
and the synthetic ones roll off at ~3530 Hz, so a single band-limit feature (`ltas_high_ratio_db`) separated
the classes with AUC 0.967 on its own. A microphone in a room is wideband, so every such cue pointed at
"human" at once.

We proved it causally rather than guessing: adding **only** room noise and a short reverb tail to the caller
channel of calls the model was *trained on* — the voice samples otherwise untouched — moved them from

| call | p(synthetic) clean | p(synthetic) + room |
|---|---|---|
| call_01c3806808d6 | 0.983 | **0.017** |
| call_02c249d2f89d | 0.994 | **0.066** |
| call_0294f969f98b | 0.993 | 0.587 |

while human calls moved by at most +0.014. The voice had not changed at all. Only the room had.

So this round is about making the detector read the **caller** and not the **line**.

---

## What we changed

**1. The channel is now an explicit, randomised variable — applied to both classes.**
`training/channels.py` models five paths a voice can take: `telephony` (PSTN, 300–3400 Hz, µ-law),
`mobile` (wider band, codec drift, packet loss), `mic_room` (a loudspeaker into a microphone across a room:
full band, reverb, 12–30 dB SNR ambient noise, AGC — *the failure case*), `voip_wide`, and `handset` (the
replay attack: reverberant *and* band-limited). Every training call, human and synthetic alike, is sent down
a randomly chosen path. Because the channel no longer correlates with the label, no channel cue can be used
as a shortcut and the model has to find something in the voice.

**2. Cepstral mean normalisation.** A transmission path is a convolution, and convolution is addition in the
cepstral domain — so the per-call cepstral *mean* is roughly "this speaker" plus "this microphone and codec".
Those features carried ~33 % of the old model's weight. We now also emit post-CMN dynamics (`lfcc_d*`,
`mfcc_d*`, `*_dyn_ratio`): how fast the spectral envelope *moves*.

We checked this rather than assuming it, and the obvious test was misleading. Counting how many features
shift when the channel changes says the deltas shift *more* (32/33 vs 23/33) — but that counts a common-mode
move, which tells you nothing about whether a classifier can still use them. Training on one channel and
testing on the other is the test that matters, and there the ordering reverses: the delta block transfers at
**AUC 0.817** from `mic_room` to `telephony` against **0.553** — near chance — for the raw means, and it wins
in 19 of 20 paired seeds. Removing the raw means costs about one call out of 438 either way, so both stay in;
the point is which block is *load-bearing* when the line changes.

**3. Consistency instead of slowness.** The brief is explicit — *"Humans recover from these instantly and
messily. Machines recover consistently, and consistency is a signal."* The old model measured the wrong half
of that. Every synthetic caller in the dataset ran recognition → LLM → speech and took 2–3 s to answer, so it
learned **slow = machine** — and a modern speech-to-speech agent answers in ~300 ms, *faster* than a human,
which makes those features vote for HUMAN.

How much of the Gemini failure this accounted for we cannot say: the channel effect above is proven
causally, this one is not, and the two arrived together. What is measured is that it matters. On generated
callers that answer in 0.25–0.7 s, the old model detected 79 % through a handset and 88 % through a
microphone; the new one detects 97 % of both. On the interpretable layer the gap is starker still — the
turn-taking aspect flagged 14 % of such callers and now flags 55 %, at an unchanged false-alarm rate on real
held-out customers.

The new conversational features are scale-free by construction: `conv_resp_log_std`, `conv_resp_mad_norm`,
`conv_resp_entropy_norm`, `conv_resp_range_norm`, `conv_int_yield_log_std`, `conv_consistency`, and
`conv_resp_predictability` — how much of the caller's response delay is explained by a straight line through
the agent's turn length, because a pipeline's latency is a constant plus processing time and a person's is
not. None of them ask how *fast* the caller is. The corpus gained a matching `realtime` caller profile
(answers in 0.25–0.7 s with near-zero spread) so the model is actually trained against that threat.

**4. No more self-grading.** The previously shipped model was refit on train + val and then reported its
accuracy *on val*. `training/train.py` now carves 20 % of the dataset's training calls into a held-out side
(grouped, so a call's clips and channel-augmented copies never straddle the split) and `--refit fit` ships
the model fitted on the training side only — every number below describes the weights that actually ship.

**5. Corpus-construction artifacts are detected and dropped.** Adding self-made synthetic calls creates its
own trap: our generated calls share a fixed agent recording and a scripted flow, so features like
"fraction of the call the agent speaks" identify *our generator* rather than a synthetic caller — and those
were the #1 and #2 weights. `--drop-corpus-artifacts` compares, per feature, how well it separates
human-from-synthetic within the real dataset against how well it separates real-synthetic from
corpus-synthetic (same label, different origin) and drops the ones that know the generator better than the
class. It removed 19 features at no cost in accuracy — and it says so out loud when it has nothing to
compare against, because a filter that silently does nothing makes an ablation look like a shipped
configuration when it is not.

One of those artifacts we had built ourselves: the corpus generator originally had machines answer into the
five seconds of dead air **every** time, against 38 % for its humans, so `conv_dead_caller_fill_frac` sat at
exactly 1.0 for 98 % of generated machine calls. That is a property of our `if` statement, not of synthetic
callers. It now fires 80 % of the time, matching the rate the real dataset's bots actually show (~81 %).

---

## Results

All numbers are on data the shipped model never saw.

<!--RESULTS-->

The shipped model is fitted on 2601 rows and never saw any call below (`--holdout-frac 0.2 --refit fit`).

### Held out from training

| Test set | calls | accuracy | AUC | errors |
|---|---:|---:|---:|---|
| 20 % of the training calls, held out (full calls) | 57 | **100.0 %** | 1.0000 | none |
| the provided `val` split (full calls) | 71 | **100.0 %** | 1.0000 | none |
| the `val` split, first 60 s only | 71 | **100.0 %** | 1.0000 | none |
| the `val` split, first 30 s only | 71 | **97.2 %** | 0.9730 | 1 missed, 1 false alarm |
| held-out calls re-sent down a different channel | 142 | **97.2 %** | 0.9956 | 4 false alarms |
| **everything held out, pooled** | 456 | **98.9 %** | 0.9983 | 5 false alarms |

### By call condition — the robustness that was missing

Same held-out calls, grouped by the path the caller's voice travelled. `original` is the dataset's own recordings; the rest are re-renderings of held-out calls through a channel the model was not fitted on for that call.

| Channel | calls | accuracy | AUC | errors |
|---|---:|---:|---:|---|
| replay through a handset | 76 | **100.0 %** | 1.0000 | none |
| **loudspeaker → room → microphone** | 65 | **93.8 %** | 0.9844 | 4 false alarms |
| mobile / cellular | 67 | **98.5 %** | 1.0000 | 1 false alarm |
| original recording | 128 | **100.0 %** | 1.0000 | none |
| telephony (PSTN, µ-law) | 67 | **100.0 %** | 1.0000 | none |
| wideband VoIP | 53 | **100.0 %** | 1.0000 | none |

### By caller behaviour

| Caller | calls | accuracy | AUC | errors |
|---|---:|---:|---:|---|
| the dataset's own callers | 384 | **98.7 %** | 0.9987 | 5 false alarms |
| pipeline bot (recognition → LLM → speech, answers in 1.4–3.2 s) | 24 | **100.0 %** | — | none |
| synthetic voice with human timing (back-channels, false starts) | 24 | **100.0 %** | — | none |
| **speech-to-speech bot (answers in 0.25–0.7 s, faster than a human)** | 24 | **100.0 %** | — | none |

### The same callers, down five different lines

A paired test: one set of generated synthetic callers, each rendered through every channel, so a difference
between rows is the transmission path and nothing else. These share voices and scripts with the training
corpus — they measure **channel** robustness, not unseen voices; for that see the two sections below. The
*before* column is the model this round replaced.

| Channel | calls | before | after |
|---|---:|---:|---:|
| telephony | 99 | 100.0 % | **100.0 %** |
| mobile | 99 | 100.0 % | **100.0 %** |
| voip_wide | 99 | 99.0 % | **99.0 %** |
| mic_room | 99 | 89.9 % | **96.0 %** |
| handset | 99 | 83.8 % | **97.0 %** |

### An engine we have never heard

The corpus uses two TTS engines. Pulling one out of training entirely — every voice of it — and scoring only
that engine is the honest estimate of what happens when a fraudster shows up with a generator we have never
seen. The dataset side is untouched in both arms.

| Engine held out of training | its calls | detected | provided `val` split | held-out dataset calls |
|---|---:|---:|---:|---:|
| ElevenLabs (11 voices) | 66 | **92.4 %** | 100 % | 100 % |
| Edge neural (11 voices) | 198 | **85.9 %** | 100 % | 100 % |

For contrast, a model trained with **no** generated speech at all detects the same ElevenLabs calls at
56.8 %. Training on one engine buys ~+35 points on a different one — the generalisation is real, and it is
also clearly not free: 86–92 % is the number to quote, not the 99 % from familiar conditions.

Unseen *voices* of a familiar engine (the corpus holds a third of its voices out of training) are detected at
**100 %** (72 calls).

### Latency

Measured end to end through `/detect` on real calls, single request, laptop CPU.

| Audio sent | median | p90 | audio per second of compute |
|---|---:|---:|---:|
| 5 s | 0.20 s | 0.36 s | 38× |
| 10 s | 0.36 s | 0.49 s | 43× |
| **20 s** | **0.73 s** | 0.93 s | 37× |
| 30 s | 0.99 s | 1.16 s | 40× |
| 60 s | 1.67 s | 1.97 s | 48× |
| full call (median 136 s) | 3.26 s | 4.42 s | 51× |

The answer to "how fast can it decide with reasonable confidence" is **about 20 seconds of call**, which
costs well under a second: below that the caller has usually not said enough, and the endpoint abstains at
0.5 rather than guessing. 160 responses across those lengths were checked for the contract: every one
well-formed, zero violations of `is_synthetic == (confidence > 0.5)`, and nothing ever reported at 1.0.

### The live call

Two bugs worth recording, both found by measurement rather than inspection.

*It went deaf on a compressing microphone.* A block counted as speech only once it sat a fixed 8 dB above the
tracked noise floor. A microphone with automatic gain control squashes an entire call into ~6 dB, so that bar
is unreachable: on a real call re-rendered through a room-and-microphone channel the session accepted **1 %**
of blocks and finalised no utterance at all — the detector then scored a caller it had never heard, which is
a false negative that has nothing to do with the model. The threshold now scales to the dynamic range
actually present (3 dB floor), and that call goes from 0 % of its speech heard to all of it. Regression test
in `tests/test_live.py`.

*It used gigabytes.* Every rolling verdict re-analysed the whole call from the start, and the analysis costs
roughly 175 MB per minute of audio — scipy's STFT builds a complex128 intermediate about fourteen times the
size of the spectrogram it returns. By minute eight that is **1.4 GB per verdict, every two seconds**, and
still climbing. Rolling verdicts now read a bounded 90 s trailing window (with the agent's turn times shifted
to match) and the capture buffer grows by doubling instead of re-concatenating every 100 ms block:

| call length | before | after |
|---|---:|---:|
| 1 min | 178 MB | 178 MB |
| 4 min | 703 MB | **267 MB** |
| 8 min | 1407 MB | **267 MB** |
| 10 min | ~1.8 GB | **267 MB** |

The closing verdict still sees the whole call. On the page, the AudioWorklet's blob URL was never revoked
(one leak per microphone check) and finished agent clips kept their base64 audio; both are released now.

---

## How it works

```
stereo WAV ─► decode / resample ─► one STFT grid (50 ms / 10 ms) ─► VAD on both channels
        │
        ├─ Acoustic (caller channel)        ─┐
        ├─ Conversational (both timelines)  ─┼─► feature vector ─► L2 logistic regression
        └─ Semantic (optional, cascaded)    ─┘        │
                                                      ▼
                          evidence weighting ──► {"is_synthetic", "confidence"}
```

**Acoustic** (`backend/features/acoustic.py`, pure numpy/scipy): prosody (pitch variability, contour
smoothness, jitter, shimmer, harmonics-to-noise), syllable rate and its variability, the modulation
spectrum, LFCC/MFCC statistics with cepstral mean normalisation, breath events, LPC formants against pitch
(the voice-changer cue), onset/offset ramps, and the long-term spectrum. The channel descriptors are still
computed — they are genuinely informative about *how* a call arrived, and the dashboard shows them — but
augmentation means they cannot stand in for the voice.

**Conversational** (`backend/features/conversational.py`): the brief's own hint. The agent interrupts, falls
silent and talks over the caller; we measure how *repeatably* the caller handles each of those, plus
back-channels, false starts, who fills dead air, and intra-turn pauses. All from the two VAD timelines, so
it costs microseconds and no transcription.

**Semantic** (`backend/features/semantic.py`, optional): both channels are transcribed with faster-whisper
and an LLM judges the caller's replies — does it invent an answer when the agent asks about a product that
does not exist, does it repeat the folio back correctly, does the wording read like a language model. This
runs as a **cascade**: only when the fast layers are unsure (`SEMANTIC_MODE=uncertain`, 0.30 < p < 0.70), so
it never costs latency on a confident call.

**Decision.** One heavily regularised logistic regression over all of it — chosen over gradient boosting by
grouped cross-validation, and fully explainable: the dashboard shows the per-group logit contribution behind
every verdict. The final probability is pulled toward 0.5 when there is less than ~8 s of caller speech, and
a clip with no usable speech resolves to **human** — with no evidence we do not accuse a customer.

---

## Threat coverage

| Attack | Where it shows up | Status |
|---|---|---|
| Cloned / TTS voice | acoustic texture, prosody, cepstral dynamics, breathing | trained, 2 engines + 23 voices |
| Pipeline caller (ASR→LLM→TTS) | slow *and regular* turn taking, dead-air filling, no back-channels | trained (the dataset's own bots) |
| **Realtime speech-to-speech caller** | latency consistency, no false starts, no back-channels, invented answers | trained via the `realtime` profile |
| **Synthetic voice through a microphone** | voice-intrinsic cues only; channel cues neutralised by augmentation | trained via `mic_room` |
| Replay through a handset | reverberant decay + band-limiting together with synthetic-voice cues | trained via `handset` |
| Digital injection (no microphone) | digital silence, stationary floor, no channel behaviour while the agent talks | features present, trained |
| A different human who sounds similar | needs an enrolled voiceprint of the real customer | **out of scope** of `/detect` — add speaker verification against the account's enrolled embedding |

`POST /analyze` returns an *attack profile*: how the synthetic probability splits across these, plus ten
interpretable aspect ratings with the measurements behind them.

---

## API

```
POST /detect            the scored endpoint
  body: {"audio": "<base64 stereo 8 kHz WAV>"}   (any key containing audio/wav/clip/data/file/b64 works,
                                                  as do raw WAV bytes, a bare base64 body and multipart)
  200:  {"is_synthetic": true, "confidence": 0.93}
  400:  the audio could not be decoded

POST /analyze[?semantic=1&ui=1]   everything: p_synthetic, aspects, attack profile, events, timeline,
                                  features, contributions, spectrogram, transcript + judge
POST /analyze?sample=<name>.wav   analyse a file from SAMPLES_DIR (the dashboard dropdown)
GET  /health                      model, metrics, optional layers, uptime
GET  /share                       which links work, and whether each one can run the live call
POST /miccheck                    microphone quality report for a short PCM sample
WS   /ws/live                     live call: int16 8 kHz PCM in, rolling verdict out
GET  /                            dashboard
```

CORS is open on every route, and no input shape returns a 500 — malformed audio is a 400 and an internal
failure is a neutral 200, so a benchmark harness never stalls on us.

## Run

```powershell
.\run.ps1                                     # http://localhost:8000
.\run.ps1 -Port 8010 -Workers 4               # more throughput for the benchmark
.\run.ps1 -Port 8010 -Https                   # https://<LAN ip>:8010 — needed for other devices' microphones
```

```bash
./run.sh                                      # Linux / macOS
PORT=8010 WORKERS=4 ./run.sh
HTTPS=1 PORT=8010 ./run.sh
```

Tests: `.venv\Scripts\python.exe -m pytest -q`.

Benchmark against a running server:

```bash
python -m bench.benchmark --url http://127.0.0.1:8000/detect --audio audio/ --manifest manifest.csv --split val
python -m bench.benchmark ... --clip 30            # only the first 30 s of every call
python -m bench.benchmark ... --concurrency 4      # parallel (run the server with --workers 4)
```

Use `127.0.0.1`, not `localhost`, for Python clients on Windows: it resolves to IPv6 first and the fallback
adds ~2 s per request that has nothing to do with the detector.

## Sharing it with other people (Cloudflare Tunnel)

The dashboard and the API work over plain HTTP on the LAN. **The live call does not**, and this is the usual
reason sharing appears broken: browsers only hand out a microphone in a *secure context*, meaning `https://`
or `localhost`. On `http://10.0.0.5:8010` the microphone API is simply absent.

```powershell
.un.ps1 -Port 8020            # 1. start the detector
.\share.ps1 -Port 8020          # 2. public HTTPS link via Cloudflare Tunnel, in a second window
```

`share.ps1` prints a `https://….trycloudflare.com` link, writes it to `share_url.txt` (which `/share` and the
dashboard's **Share link** button then serve), and keeps running until Ctrl+C. Verified end to end: `/health`
200 and `/detect` returning correct verdicts on real calls through the public URL.

Three things this setup had to learn the hard way, all of which it now handles:

**QUIC is usually blocked.** A quick tunnel prefers QUIC over UDP 7844, which university and corporate wifi
routinely drop — cloudflared's own precheck on this network reports `UDP Connectivity: QUIC connection
failed` / `suggested_protocol=http2`. It still prints a public URL, and every request to it returns **530**.
`share.ps1` therefore passes `--protocol http2` (TCP 443) by default; `-Protocol auto` restores the old
behaviour.

**A printed URL is not a working URL.** cloudflared prints the link before it has a connection, so the script
used to hand over dead links. It now waits for a registered connection *and* fetches `/health` through the
link before declaring success, and distinguishes the two failures: a 530 means the tunnel really is not
connected and is a hard error, whereas a name that does not resolve is usually just this machine's DNS being
slow on a brand-new hostname (campus resolvers cache negative answers) — the link works for everyone else,
so it warns instead of throwing it away.

**Quick tunnels are explicitly best-effort.** Cloudflare says so on every start: *"account-less Tunnels have
no uptime guarantee… If you intend to use Tunnels in production you should use a pre-created named tunnel."*
The hostname also changes on every restart, so anything holding the old link breaks. For judging, use a
**named tunnel** — same command, stable hostname:

```powershell
cloudflared tunnel login                                   # once per machine, opens a browser
.\share.ps1 -Port 8020 -Hostname detector.yourdomain.com    # creates the tunnel + DNS route, then runs it
```

That needs a domain on your Cloudflare account; `share.ps1` creates the tunnel (`-TunnelName`, default
`calliope`), points the hostname at it and runs it. Start it *before* judging so DNS has propagated.

**If the network blocks tunnels entirely**, serve TLS yourself and share the LAN link:

```powershell
.un.ps1 -Port 8020 -Https     # self-signed; each device accepts the warning once, then the mic works
```

`backend/tls.py` puts every LAN address of the machine in the certificate's SAN, so one certificate covers
every device on the wifi. `/share` reports which URLs exist and whether each can run the live call, and the
dashboard says so up front instead of failing silently over plain HTTP.

Nothing Cloudflare writes can reach the repository: `cloudflared.log`, `share_url.txt` and `certs/` (which
holds a private key) are all in `.gitignore`.

## Training

```bash
# 1. synthetic callers from other engines + channel-augmented copies of the dataset (both classes)
python -m training.tts_corpus --out data/corpus_v3 --el-credits 0 \
    --profiles bot,realtime,humanlike --augment-dataset --augment-copies 2 --augment-val

# 2. one feature table (the same code the server runs, so there is no train/serve skew)
python -m training.build_features --audio audio/ --manifest manifest.csv \
    --extra data/corpus_v3 data/corpus_v3/manifest.csv --out data/features_v3.csv --clip 30 --clip 60

# 3. train: 20 % of the training calls held out, shipped weights fitted only on the rest
python -m training.train --features data/features_v3.csv --out models/detector.joblib \
    --only-lr --holdout-frac 0.2 --refit fit --drop-corpus-artifacts
```

Useful flags: `--exclude-group-prefix elevenlabs:` holds an entire TTS engine out of training and reports it
separately (the leave-one-engine-out number); `--drop-feature-prefix floor_` removes a feature family to
measure what the model is leaning on; `--refit all` trades the honest report for a little more training data.

## Dashboard

* **Analyze recording** — drop a WAV or pick a dataset call: verdict, confidence, the three checks from the
  brief (voice / conversation / semantics), ten aspect ratings with their measurements, attack profile,
  spectrogram with both speaker lanes, response-latency labels, interruption and silence-fill markers,
  breath marks, pitch track, model contributions, raw JSON. Batch mode runs many files through `/detect`.
* **Live call with the agent** — you are the caller; the bank agent "Marina" is generated on the server
  (Gemini writes each line, ElevenLabs speaks it, faster-whisper hears you) and runs the brief's flow:
  greeting, a question she interrupts, a folio to repeat back, a probe about an insurance product that does
  not exist, a 5 s silence, a closing. The verdict updates every 2 s.

## Honest limitations

* The `mic_room` / `handset` channels are **simulated**, not recorded. They model the physics (room impulse
  response, ambient noise at realistic SNR, loudspeaker colouration, AGC, the 48 kHz → 8 kHz anti-alias
  edge), and the microphone failure they were built to reproduce is real and was measured — but a corpus of
  genuine re-recorded calls would be better evidence, and is the first thing we would collect next.
* The synthetic corpus uses two TTS engines (ElevenLabs, Edge). Leave-one-engine-out measures what happens on
  an engine never seen, but two is a small number of engines — and both are conventional TTS. No
  speech-to-speech model's actual audio is in the corpus, only its *timing*: the `realtime` profile is an
  ElevenLabs or Edge voice on a fast, regular schedule, not a recording of Gemini Live.
* Every error the shipped model makes on held-out data is a **false alarm**, not a miss (5 of 456), and four
  of the five are human callers re-rendered through the simulated microphone channel. That is the safer
  direction to fail in, but it is the expensive one for a bank, and it is what we would attack next.
* The held-out dataset slice is 57 calls and the provided `val` split is 71, so several cells above move by
  1.4–1.8 points per call. Read "100 %" as "no errors in 57 calls", not as a rate.
* The semantic layer needs an API key and adds seconds, so it is cascaded and off by default. Every number
  above is from the acoustic + conversational layers alone.
* Speaker *verification* (is this the customer this account belongs to?) is out of scope here and is the
  natural next layer: a synthetic voice and an impostor human are different problems.
