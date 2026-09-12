/* Altur Voice Shield dashboard - vanilla JS, no build step. */
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x)) ? "—" : Number(x).toFixed(d);
const pct = (x) => (x === null || x === undefined) ? "—" : Math.round(x * 100) + "%";
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ------------------------------------------------------------------ tabs + health + share
$$(".tab").forEach(b => b.addEventListener("click", () => {
  $$(".tab").forEach(t => t.classList.toggle("active", t === b));
  $$(".tabpane").forEach(p => p.classList.toggle("active", p.id === "tab-" + b.dataset.tab));
}));

let HEALTH = null;
async function loadHealth() {
  try {
    const r = await fetch("/health");
    HEALTH = await r.json();
    const el = $("#health");
    el.textContent = `${HEALTH.status} · ${HEALTH.mode}` + (HEALTH.model_meta && HEALTH.model_meta.val_auc ? ` · val AUC ${fmt(HEALTH.model_meta.val_auc, 3)}` : "") +
      ` · agent: ${HEALTH.gemini_available ? "gemini" : "scripted"} + ${HEALTH.elevenlabs_available ? "elevenlabs" : "browser voice"}`;
    el.className = "pill " + (HEALTH.status === "ok" ? "ok" : "bad");
    $("#health-json").textContent = JSON.stringify(HEALTH, null, 2);
    if (!HEALTH.whisper_available) { $("#opt-semantic").disabled = true; $("#opt-semantic").parentElement.title = "faster-whisper not installed on the server"; }
  } catch (e) {
    $("#health").textContent = "server unreachable"; $("#health").className = "pill bad";
  }
}
loadHealth();
setInterval(loadHealth, 15000);

async function loadShare() {
  try {
    const j = await (await fetch("/share")).json();
    const btn = $("#share-btn");
    const url = j.public_url || (j.lan_urls && j.lan_urls[0]) || "";
    if (!url) return;
    btn.classList.remove("hidden");
    btn.title = j.public_url ? "public HTTPS link (works for anyone, including the live call)" : "LAN link (same network; the live call needs HTTPS or localhost for the microphone)";
    btn.onclick = async () => {
      const lines = [];
      if (j.public_url) lines.push("Public: " + j.public_url);
      (j.lan_urls || []).forEach(u => lines.push("LAN: " + u));
      try { await navigator.clipboard.writeText(url); btn.textContent = "Copied!"; setTimeout(() => btn.textContent = "Share link", 1500); } catch (e) { /* clipboard blocked */ }
      $("#share-box").classList.remove("hidden");
      $("#share-box").innerHTML = lines.map(l => `<div>${esc(l)}</div>`).join("") + (j.public_url ? "" : `<div class="muted small">Run <code>share.ps1</code> (or <code>share.sh</code>) on the host to get a public HTTPS link.</div>`);
    };
  } catch (e) { /* no share info */ }
}
loadShare();
setInterval(loadShare, 20000);

// ------------------------------------------------------------------ analyze tab
const drop = $("#drop"), fileInput = $("#file");
drop.addEventListener("click", () => fileInput.click());
drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", e => { e.preventDefault(); drop.classList.remove("over"); handleFiles(e.dataTransfer.files); });
fileInput.addEventListener("change", () => handleFiles(fileInput.files));

async function handleFiles(files) {
  if (!files || !files.length) return;
  if ($("#opt-batch").checked || files.length > 1) return runBatch(files);
  const f = files[0];
  $("#progress").textContent = `Analysing ${f.name} (${(f.size / 1024).toFixed(0)} kB)…`;
  const t0 = performance.now();
  try {
    const fd = new FormData(); fd.append("file", f);
    const sem = $("#opt-semantic").checked ? 1 : 0;
    const r = await fetch(`/analyze?semantic=${sem}&ui=1`, { method: "POST", body: fd });
    if (!r.ok) throw new Error(await r.text());
    const res = await r.json();
    $("#progress").textContent = `Done in ${((performance.now() - t0) / 1000).toFixed(2)} s (server ${fmt(res.timing.total_s)} s).`;
    renderResult(res, $("#result"), f.name);
  } catch (e) {
    $("#progress").textContent = "Error: " + e.message;
  }
}

async function runBatch(files) {
  const box = $("#batch"); box.classList.remove("hidden");
  box.innerHTML = `<h2>Batch via /detect</h2><div class="batch"><table><thead><tr><th>file</th><th>is_synthetic</th><th>confidence</th><th>latency</th></tr></thead><tbody></tbody></table></div>`;
  const tb = $("tbody", box);
  let n = 0, syn = 0;
  for (const f of files) {
    const t0 = performance.now();
    const b64 = await fileToBase64(f);
    let row;
    try {
      const r = await fetch("/detect", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ audio: b64 }) });
      const j = await r.json();
      row = `<td>${esc(f.name)}</td><td class="${j.is_synthetic ? "pos" : "neg"}">${j.is_synthetic}</td><td>${fmt(j.confidence, 3)}</td><td>${((performance.now() - t0) / 1000).toFixed(2)} s</td>`;
      n++; if (j.is_synthetic) syn++;
    } catch (e) { row = `<td>${esc(f.name)}</td><td colspan="3">error: ${esc(e.message)}</td>`; }
    const tr = document.createElement("tr"); tr.innerHTML = row; tb.appendChild(tr);
  }
  $("#progress").textContent = `${n} files: ${syn} synthetic, ${n - syn} human.`;
}

// dataset samples (server started with SAMPLES_DIR)
async function loadSamples() {
  try {
    const j = await (await fetch("/samples")).json();
    if (!j.enabled || !j.samples.length) return;
    const sel = $("#sample");
    sel.innerHTML = j.samples.map(s => `<option value="${esc(s.name)}">${esc(s.name)}${s.label ? " · " + s.label : ""}${s.split ? " (" + s.split + ")" : ""}</option>`).join("");
    $("#samples-box").classList.remove("hidden");
  } catch (e) { /* samples disabled */ }
}
loadSamples();
$("#sample-go").addEventListener("click", async () => {
  const name = $("#sample").value; if (!name) return;
  $("#progress").textContent = `Analysing ${name}…`;
  const t0 = performance.now();
  try {
    const sem = $("#opt-semantic").checked ? 1 : 0;
    const r = await fetch(`/analyze?sample=${encodeURIComponent(name)}&semantic=${sem}&ui=1`, { method: "POST" });
    if (!r.ok) throw new Error(await r.text());
    const res = await r.json();
    const truth = res.sample && res.sample.label ? ` · ground truth: ${res.sample.label.toUpperCase()}` : "";
    $("#progress").textContent = `Done in ${((performance.now() - t0) / 1000).toFixed(2)} s (server ${fmt(res.timing.total_s)} s)${truth}.`;
    renderResult(res, $("#result"), name + truth);
  } catch (e) { $("#progress").textContent = "Error: " + e.message; }
});

function fileToBase64(file) {
  return new Promise((res, rej) => { const fr = new FileReader(); fr.onload = () => res(fr.result.split(",")[1]); fr.onerror = rej; fr.readAsDataURL(file); });
}

// ------------------------------------------------------------------ result renderer
function scoreColor(s) {
  return s === null || s === undefined ? "#3b3630" : (s < 0.4 ? "var(--human)" : (s > 0.6 ? "var(--synthetic)" : "var(--warn)"));
}

function renderResult(res, root, title) {
  const tpl = $("#tpl-result").content.cloneNode(true);
  const f = (k) => tpl.querySelector(`[data-f="${k}"]`);
  const p = res.p_synthetic;
  const v = f("verdict");
  const unsure = res.evidence_level < 0.5;
  v.textContent = unsure ? "INSUFFICIENT EVIDENCE" : res.verdict;
  v.className = "verdict-big " + (unsure ? "unsure" : res.verdict.toLowerCase());
  f("confidence").textContent = pct(res.confidence);
  f("p").textContent = fmt(p, 3);
  f("meta").textContent = `${title ? title + " · " : ""}${fmt(res.duration_seconds, 1)} s of audio, ${fmt(res.speech_seconds, 1)} s of caller speech · evidence ${pct(res.evidence_level)} · ${res.mode} · ${fmt(res.timing.total_s)} s` +
    (res.input && !res.input.has_agent_channel ? " · no agent channel (turn-taking signals unavailable)" : "");
  // flat rectangular probability bar: fill = p(synthetic), colour by verdict
  const fill = f("pfill");
  fill.style.width = (p * 100) + "%";
  fill.style.background = unsure ? "var(--warn)" : (res.is_synthetic ? "var(--synthetic)" : "var(--human)");
  tpl.querySelector(".pmark").style.left = (p * 100) + "%";

  // aspects
  const asp = f("aspects");
  for (const [k, a] of Object.entries(res.aspects)) {
    const d = document.createElement("div");
    const s = a.score;
    d.className = "aspect";
    d.innerHTML = `<div class="aspect-head"><span class="aspect-name">${esc(a.label)}</span><span class="muted small">${a.n_terms} cues</span><span class="aspect-score">${s === null ? "n/a" : Math.round(s * 100)}</span></div>
      <div class="bar"><i style="width:${s === null ? 0 : s * 100}%;background:${scoreColor(s)}"></i></div>
      <div class="aspect-ev">${(a.evidence || []).map(e => `<div><span>${esc(e.desc)} <code>${esc(e.feature)}</code></span><span>${fmt(e.value, 3)} <b class="read-${e.read}">${e.read}</b></span></div>`).join("") || "<div>no cues available for this clip</div>"}</div>`;
    d.addEventListener("click", () => d.classList.toggle("open"));
    asp.appendChild(d);
  }
  // attack profile
  const at = f("attacks");
  const attacks = Object.entries(res.attack_profile).sort((a, b) => b[1].p - a[1].p);
  for (const [k, a] of attacks) {
    const d = document.createElement("div"); d.className = "attack";
    d.innerHTML = `<div class="row2"><span>${esc(a.label)}</span><span><b>${pct(a.p)}</b> <span class="muted small">(cue strength ${Math.round(a.raw * 100)})</span></span></div><div class="bar"><i style="width:${a.p * 100}%;background:var(--synthetic)"></i></div>`;
    at.appendChild(d);
  }
  // signals
  const sg = f("signals");
  const S = res.signals;
  const rows = [["fast model (features)", S.fast_model_p], ["heuristic aspects", S.heuristic_p], ["SSL embedding head", S.embedding_p], ["semantic judge", S.semantic_p], ["fused", S.fused_p], ["final (evidence-weighted)", res.p_synthetic]];
  sg.innerHTML = rows.map(([k, v]) => `<span class="k">${k}</span><span>${v === null || v === undefined ? "not run" : fmt(v, 3)}</span>`).join("");

  // spectrogram + events
  const canvas = f("spec");
  requestAnimationFrame(() => drawSpectrogram(canvas, res));
  f("events").innerHTML = eventChips(res.events);

  // contributions
  const c = res.contributions;
  if (c && c.groups) {
    const g = Object.entries(c.groups).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    f("contrib").innerHTML = `<div class="groups">${g.map(([k, v]) => `<span class="${v > 0 ? "pos" : "neg"}">${esc(k)} ${v > 0 ? "+" : ""}${fmt(v, 2)}</span>`).join("")}</div>
      <table class="contrib"><thead><tr><th>feature</th><th class="num">value</th><th class="num">contribution</th></tr></thead><tbody>
      ${c.top_features.map(t => `<tr><td><code>${esc(t.feature)}</code></td><td class="num">${fmt(t.value, 3)}</td><td class="num ${t.contribution > 0 ? "pos" : "neg"}">${t.contribution > 0 ? "+" : ""}${fmt(t.contribution, 2)}</td></tr>`).join("")}
      </tbody></table><div class="muted small">positive pushes toward synthetic, negative toward human (intercept ${fmt(c.intercept, 2)})</div>`;
  } else {
    f("contrib-card").classList.add("hidden");
  }
  // semantic / transcript
  const liveTurns = res.live && Array.isArray(res.live.caller_turns) ? res.live.caller_turns : null;
  if (res.semantic || liveTurns) {
    f("sem-card").classList.remove("hidden");
    const s = res.semantic || {};
    let html = "";
    const tr = (s.available && s.transcript) ? s.transcript : (liveTurns ? { agent: [], caller: liveTurns } : null);
    if (tr) {
      const turns = [...tr.agent.map(t => ({ ...t, who: "agent" })), ...tr.caller.map(t => ({ ...t, who: "caller" }))].sort((a, b) => a.start - b.start);
      html += `<div class="transcript">${turns.map(t => `<div class="t ${t.who}"><span class="muted small">${fmt(t.start, 1)}s ${t.who.toUpperCase()}</span> ${esc(t.text)}</div>`).join("")}</div>`;
      if (s.stt_seconds) html += `<div class="muted small">transcription ${fmt(s.stt_seconds, 1)} s${s.llm_seconds ? ", judge " + fmt(s.llm_seconds, 1) + " s" : ""}</div>`;
    }
    if (s.judge) {
      const j = s.judge;
      html += `<h2 class="mt">Judge</h2><p><b>synthetic ${pct(j.synthetic_probability)}</b> · fabrication ${pct(j.fabrication_probability)} · LLM style ${pct(j.llm_style_probability)} · repeat-back: ${esc(j.repeat_back)}</p><p>${esc(j.rationale)}</p>`;
      if (j.nonexistent_probes && j.nonexistent_probes.length) html += `<ul>${j.nonexistent_probes.map(q => `<li><i>${esc(q.agent_question)}</i> → <b>${esc(q.caller_reaction)}</b></li>`).join("")}</ul>`;
      if (j.human_markers && j.human_markers.length) html += `<div class="muted small">human markers: ${esc(j.human_markers.join("; "))}</div>`;
    } else if (s.judge_error) {
      html += `<div class="muted">judge not run: ${esc(s.judge_error)}</div>`;
    } else if (s.error) {
      html += `<div class="muted">${esc(s.error)}</div>`;
    }
    f("sem").innerHTML = html;
  }
  f("raw").textContent = JSON.stringify({ ...res, ui: res.ui ? "(omitted)" : undefined }, null, 1);
  root.innerHTML = "";
  root.appendChild(tpl);
}

function eventChips(events) {
  const lab = { response: e => `↳ response ${fmt(e.latency)} s @${fmt(e.t, 1)}`, agent_interrupt: e => `agent interrupts @${fmt(e.t, 1)}: yields after ${fmt(e.yield)} s${e.restart ? ", restarts" : ""}`,
    caller_interrupt: e => `caller interrupts @${fmt(e.t, 1)} (${fmt(e.dur)} s)`, backchannel: e => `back-channel @${fmt(e.t, 1)}`,
    silence_fill: e => `caller fills silence after ${fmt(e.after)} s @${fmt(e.t, 1)}`, agent_fills_silence: e => `agent fills silence after ${fmt(e.after)} s`, false_start: e => `false start @${fmt(e.t, 1)}` };
  return (events || []).slice(0, 80).map(e => `<span>${(lab[e.type] || (x => x.type))(e)}</span>`).join("") || "<span>no turn events</span>";
}

// flat, warm colormap
function cmap(v) {
  const stops = [[10, 9, 8], [46, 30, 42], [96, 46, 66], [150, 66, 66], [196, 104, 60], [222, 156, 60], [236, 205, 110], [240, 232, 200]];
  const x = Math.max(0, Math.min(1, v / 255)) * (stops.length - 1);
  const i = Math.floor(x), t = x - i, a = stops[i], b = stops[Math.min(i + 1, stops.length - 1)];
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

function drawSpectrogram(canvas, res) {
  const ui = res.ui || {}, sp = ui.spectrogram, tl = res.timeline || {};
  const W = canvas.clientWidth || 900, H = 300;
  canvas.width = W * devicePixelRatio; canvas.height = H * devicePixelRatio;
  const ctx = canvas.getContext("2d"); ctx.scale(devicePixelRatio, devicePixelRatio);
  ctx.fillStyle = "#0a0908"; ctx.fillRect(0, 0, W, H);
  const dur = res.duration_seconds || 1;
  const specH = 190, laneY = specH + 8, laneH = 18, evY = laneY + laneH * 2 + 10;
  const X = t => (t / dur) * W;
  if (sp && sp.data) {
    const bytes = Uint8Array.from(atob(sp.data), c => c.charCodeAt(0));
    const off = document.createElement("canvas"); off.width = sp.cols; off.height = sp.rows;
    const octx = off.getContext("2d"); const img = octx.createImageData(sp.cols, sp.rows);
    for (let i = 0; i < bytes.length; i++) { const [r, g, b] = cmap(bytes[i]); img.data[i * 4] = r; img.data[i * 4 + 1] = g; img.data[i * 4 + 2] = b; img.data[i * 4 + 3] = 255; }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, W, specH);
    if (ui.f0) {
      ctx.strokeStyle = "#4cc3d9"; ctx.lineWidth = 1.5; ctx.beginPath(); let pen = false;
      for (const [t, f] of ui.f0) {
        if (f === null) { pen = false; continue; }
        const x = X(t), y = specH - (f / (sp.fmax || 4000)) * specH;
        if (!pen) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.fillStyle = "rgba(148,163,184,.55)";
    for (const [s, e] of (ui.breaths || [])) ctx.fillRect(X(s), specH - 12, Math.max(2, X(e) - X(s)), 12);
    ctx.fillStyle = "#a39a89"; ctx.font = "11px 'Monster Friend', monospace";
    ctx.fillText("4 kHz", 4, 12); ctx.fillText("0", 4, specH - 4);
  }
  ctx.fillStyle = "#24211e"; ctx.fillRect(0, laneY, W, laneH); ctx.fillRect(0, laneY + laneH + 2, W, laneH);
  ctx.fillStyle = "#6b8fd6"; for (const [s, e] of (tl.agent || [])) ctx.fillRect(X(s), laneY + 2, Math.max(1, X(e) - X(s)), laneH - 4);
  ctx.fillStyle = "#d69a5c"; for (const [s, e] of (tl.caller_phrases || tl.caller || [])) ctx.fillRect(X(s), laneY + laneH + 4, Math.max(1, X(e) - X(s)), laneH - 4);
  ctx.fillStyle = "#a39a89"; ctx.font = "10px 'Monster Friend', monospace"; ctx.fillText("agent", 3, laneY + 12); ctx.fillText("caller", 3, laneY + laneH + 14);
  const colors = { response: "#a78bfa", agent_interrupt: "#e07aa8", silence_fill: "#e0b34a", backchannel: "#5fb98f", caller_interrupt: "#e07aa8", false_start: "#94a3b8" };
  for (const e of (res.events || [])) {
    const c = colors[e.type]; if (!c) continue;
    ctx.fillStyle = c; ctx.fillRect(X(e.t) - 1, evY, 3, 14);
    if (e.type === "response") { ctx.fillStyle = "#a78bfa"; ctx.font = "10px 'Monster Friend', monospace"; ctx.fillText(fmt(e.latency, 1), X(e.t) + 3, evY + 11); }
  }
  ctx.fillStyle = "#a39a89"; ctx.font = "10px 'Monster Friend', monospace";
  const step = dur > 120 ? 30 : (dur > 40 ? 10 : 5);
  for (let t = 0; t <= dur; t += step) { ctx.fillRect(X(t), H - 14, 1, 4); ctx.fillText(t + "s", X(t) + 2, H - 4); }
}

// ------------------------------------------------------------------ live call (server-driven agent: Gemini + ElevenLabs)
const WORKLET_SRC = `
class PcmCapture extends AudioWorkletProcessor {
  constructor() { super(); this.buf = []; this.ratio = sampleRate / 8000; this.pos = 0; this.out = []; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0]; if (!ch) return true;
    const k = Math.max(1, Math.round(this.ratio / 2));
    for (let i = 0; i < ch.length; i++) {
      let s = 0; for (let j = 0; j < k; j++) s += ch[Math.max(0, i - j)]; s /= k;
      this.buf.push(s);
    }
    while (this.pos + 1 < this.buf.length) {
      const i = Math.floor(this.pos), t = this.pos - i;
      this.out.push(this.buf[i] * (1 - t) + this.buf[i + 1] * t);
      this.pos += this.ratio;
    }
    const drop = Math.floor(this.pos); this.buf.splice(0, drop); this.pos -= drop;
    while (this.out.length >= 800) {
      const chunk = this.out.splice(0, 800); const pcm = new Int16Array(800); let e = 0;
      for (let i = 0; i < 800; i++) { const v = Math.max(-1, Math.min(1, chunk[i])); pcm[i] = v * 32767; e += v * v; }
      this.port.postMessage({ pcm: pcm.buffer, rms: Math.sqrt(e / 800) }, [pcm.buffer]);
    }
    return true;
  }
}
registerProcessor("pcm-capture", PcmCapture);`;

class LiveCall {
  constructor() {
    this.ws = null; this.ctx = null; this.sent = 0; this.speaking = false; this.lastSpeech = 0; this.floor = -60;
    this.history = []; this.running = false; this.voice = null; this.queue = []; this.playing = false; this.stopping = false;
  }
  t() { return this.sent / 8000; }
  log(text, kind, cls = "") {
    const d = document.createElement("div"); d.className = "line " + cls;
    d.innerHTML = `<span class="kind">${esc(kind)}</span>${esc(text)}`; $("#agent-log").appendChild(d); d.scrollIntoView({ block: "nearest" });
  }
  status(t) { $("#live-status").textContent = t; }
  async start() {
    $("#live-result").innerHTML = ""; $("#agent-log").innerHTML = ""; $("#live-events").innerHTML = ""; this.history = []; this.queue = []; this.playing = false; this.stopping = false;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error("microphone API unavailable: open the page over https:// or on localhost");
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: false } });
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    await this.ctx.resume();
    const src = this.ctx.createMediaStreamSource(stream);
    const url = URL.createObjectURL(new Blob([WORKLET_SRC], { type: "application/javascript" }));
    await this.ctx.audioWorklet.addModule(url);
    this.node = new AudioWorkletNode(this.ctx, "pcm-capture");
    this.stream = stream;
    src.connect(this.node);
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${proto}://${location.host}/ws/live`);
    await new Promise((res, rej) => { this.ws.onopen = res; this.ws.onerror = () => rej(new Error("websocket connection failed")); });
    this.ws.onmessage = ev => this.onMessage(JSON.parse(ev.data));
    this.ws.onclose = () => { if (this.running) { this.status("connection closed"); this.cleanup(); } };
    this.node.port.onmessage = ev => this.onPcm(ev.data);
    this.running = true;
    $("#live-start").disabled = true; $("#live-stop").disabled = false; this.status("connecting the agent…");
    this.pickVoice();
    this.ws.send(JSON.stringify({ type: "start" }));
  }
  pickVoice() {
    const vs = window.speechSynthesis ? speechSynthesis.getVoices() : [];
    this.voice = vs.find(v => /es[-_]MX/i.test(v.lang)) || vs.find(v => /es[-_]US/i.test(v.lang)) || vs.find(v => /^es/i.test(v.lang)) || vs[0] || null;
  }
  onPcm({ pcm, rms }) {
    if (!this.running || !this.ws || this.ws.readyState !== 1) return;
    this.ws.send(pcm); this.sent += 800;
    const db = 20 * Math.log10(rms + 1e-6);
    if (!this.speaking && db < this.floor + 6) this.floor = this.floor * 0.95 + db * 0.05;
    const thr = Math.max(this.floor + 9, -50);
    const now = this.t();
    if (db > thr) { this.speaking = true; this.lastSpeech = now; }
    else if (this.speaking && now - this.lastSpeech > 0.4) { this.speaking = false; }
    $("#mic-level").style.width = Math.max(0, Math.min(100, (db + 60) * 1.8)) + "%";
    $("#mic-level").style.background = this.speaking ? "var(--caller)" : "var(--human)";
  }
  sendAgent(ev, text, kind) {
    if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify({ type: "agent", event: ev, t: this.t(), text, kind }));
  }
  // play one agent line: ElevenLabs mp3 from the server, or browser speech synthesis as fallback
  playLine(m) {
    return new Promise(res => {
      const kind = m.kind, text = m.text;
      let started = false;
      const start = () => { if (!started) { started = true; this.sendAgent("start", text, kind); } };
      const end = () => { if (!started) start(); this.sendAgent("end", text, kind); res(); };
      if (m.audio_b64) {
        const a = new Audio("data:audio/mpeg;base64," + m.audio_b64);
        a.onplaying = start; a.onended = end; a.onerror = () => { this.log("(audio playback failed, using browser voice)", "note", "note"); this.speak(text, kind, res); };
        a.play().catch(() => { this.log("(autoplay blocked, using browser voice)", "note", "note"); this.speak(text, kind, res); });
      } else {
        this.speak(text, kind, res);
      }
    });
  }
  speak(text, kind, done) {
    let started = false;
    const start = () => { if (!started) { started = true; this.sendAgent("start", text, kind); } };
    const end = () => { start(); this.sendAgent("end", text, kind); done(); };
    if (window.speechSynthesis && this.voice) {
      const u = new SpeechSynthesisUtterance(text); u.voice = this.voice; u.lang = this.voice.lang; u.rate = 1.0;
      u.onstart = start; u.onend = end; u.onerror = end;
      speechSynthesis.speak(u);
      setTimeout(start, 250);
    } else {
      const dur = Math.max(1.2, text.length / 14);
      start();
      const o = this.ctx.createOscillator(), g = this.ctx.createGain(); g.gain.value = 0.05; o.frequency.value = 440; o.connect(g); g.connect(this.ctx.destination); o.start();
      setTimeout(() => { o.stop(); end(); }, dur * 1000);
    }
  }
  async enqueue(m) {
    if (m.kind === "interrupt") { this.log(m.text, "agent interrupts", ""); await this.playLine(m); return; }
    this.queue.push(m);
    if (this.playing) return;
    this.playing = true;
    while (this.queue.length && this.running) {
      const line = this.queue.shift();
      this.log(line.text, `agent · ${line.kind}${line.brain === "gemini" ? " · gemini" : ""}${line.voice === "elevenlabs" ? " · elevenlabs" : ""}`);
      await this.playLine(line);
      if (line.silence_after) this.log(`(agent stays silent for ${line.silence_after} s)`, "silence", "note");
    }
    this.playing = false;
  }
  onMessage(m) {
    if (m.type === "hello") { this.status(`call in progress · agent ${m.gemini ? "Gemini" : "scripted"} · voice ${m.elevenlabs ? "ElevenLabs" : "browser"}`); }
    else if (m.type === "agent_say") this.enqueue(m);
    else if (m.type === "caller_said") this.log(m.text, "you", "you");
    else if (m.type === "status") { if (m.text) this.status(m.text); }
    else if (m.type === "agent_done") { this.status("agent finished the flow — press End call for the final verdict"); this.log("(the agent hung up)", "note", "note"); }
    else if (m.type === "update" && m.result) this.renderUpdate(m.result);
    else if (m.type === "final") this.renderFinal(m);
    else if (m.type === "error") this.status("server error: " + m.message);
  }
  renderUpdate(r) {
    this.history.push(r.p_synthetic);
    const v = $("#live-verdict"); const unsure = r.evidence_level < 0.5;
    v.textContent = unsure ? "…" : r.verdict; v.className = "verdict-big " + (unsure ? "unsure" : r.verdict.toLowerCase());
    $("#live-conf").textContent = `${pct(r.confidence)} confidence · p=${fmt(r.p_synthetic, 2)} · ${fmt(r.speech_seconds, 1)} s speech · evidence ${pct(r.evidence_level)}`;
    const c = $("#live-spark"), ctx = c.getContext("2d"); ctx.clearRect(0, 0, c.width, c.height);
    ctx.fillStyle = "#0a0908"; ctx.fillRect(0, 0, c.width, c.height); ctx.strokeStyle = "#4a443c"; ctx.beginPath(); ctx.moveTo(0, c.height / 2); ctx.lineTo(c.width, c.height / 2); ctx.stroke();
    ctx.strokeStyle = "#ece4d3"; ctx.lineWidth = 2; ctx.beginPath();
    this.history.forEach((p, i) => { const x = (i / Math.max(1, this.history.length - 1)) * c.width, y = c.height - p * c.height; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
    $("#live-aspects").innerHTML = Object.values(r.aspects).map(a => `<div><span>${esc(a.label)}</span><div class="bar"><i style="width:${(a.score || 0) * 100}%;background:${scoreColor(a.score)}"></i></div></div>`).join("");
    $("#live-events").innerHTML = eventChips(r.events);
  }
  renderFinal(m) {
    this.status("final verdict ready");
    if (m.result) {
      renderResult(m.result, $("#live-result"), "live call");
      if (m.wav_b64) {
        const a = document.createElement("a"); a.href = "data:audio/wav;base64," + m.wav_b64; a.download = "live_call.wav"; a.textContent = "Download the recorded call (WAV, channel 1 = agent activity)"; a.className = "muted";
        $("#live-result").prepend(a);
      }
    } else $("#live-result").innerHTML = '<div class="card muted">Not enough audio was captured.</div>';
    this.cleanup();
  }
  stop() {
    if (!this.running || this.stopping) return;
    this.stopping = true;
    if (window.speechSynthesis) speechSynthesis.cancel();
    this.status("computing final verdict…"); $("#live-stop").disabled = true;
    if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify({ type: "stop" }));
    else this.cleanup();
  }
  cleanup() {
    this.running = false;
    try { this.node && this.node.disconnect(); this.stream && this.stream.getTracks().forEach(t => t.stop()); this.ctx && this.ctx.close(); } catch (e) { /* ignore */ }
    setTimeout(() => this.ws && this.ws.close(), 500);
    $("#live-start").disabled = false; $("#live-stop").disabled = true;
  }
}

const live = new LiveCall();
if (window.speechSynthesis) speechSynthesis.onvoiceschanged = () => live.pickVoice();
$("#live-start").addEventListener("click", async () => { try { await live.start(); } catch (e) { live.status("could not start: " + e.message); live.cleanup(); } });
$("#live-stop").addEventListener("click", () => live.stop());
