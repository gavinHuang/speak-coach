const $ = (s) => document.querySelector(s);
const state = { register: "work", mode: "read", refId: null, refKey: null, cfg: null, result: null, task: null };

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const cls = (v) => (v == null ? "" : v >= 80 ? "good" : v >= 60 ? "ok" : "bad");
const opts = () => ({ accent: $("#accent").value, gender: $("#gender").value, level: $("#level").value });

async function api(path, body) {
  const r = await fetch(path, body instanceof FormData ? { method: "POST", body } :
    body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

async function busy(btn, fn) {
  const label = btn.innerHTML;
  btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>${label}`;
  try { return await fn(); } catch (e) { alert(e.message); } finally { btn.disabled = false; btn.innerHTML = label; }
}

// ---------- setup ----------
async function init() {
  state.cfg = await api("/api/config");
  $("#coach-model").textContent = state.cfg.llm.split("/").pop();
  try {
    const saved = JSON.parse(localStorage.getItem("speakcoach") || "{}");
    for (const k of ["accent", "gender", "level"]) if (saved[k]) $("#" + k).value = saved[k];
  } catch {}
  for (const k of ["accent", "gender", "level"]) $("#" + k).onchange = () => {
    try { localStorage.setItem("speakcoach", JSON.stringify(opts())); } catch {}
    state.refId = null;
  };
  setRegister("work");
}

function setRegister(r) {
  state.register = r; state.refId = null;
  document.querySelectorAll("#registers button").forEach((b) => b.classList.toggle("on", b.dataset.r === r));
  $("#reg-notes ul").innerHTML = state.cfg.registers[r].notes.map((n) => `<li>${esc(n)}</li>`).join("");
  $("#reg-notes summary").textContent = `How "${state.cfg.registers[r].label}" should sound`;
}
document.querySelectorAll("#registers button").forEach((b) => (b.onclick = () => setRegister(b.dataset.r)));
document.querySelectorAll("#modes button").forEach((b) => (b.onclick = () => {
  state.mode = b.dataset.m;
  document.querySelectorAll("#modes button").forEach((x) => x.classList.toggle("on", x === b));
  $("#read-pane").hidden = state.mode !== "read";
  $("#free-pane").hidden = state.mode !== "free";
}));
$("#target").oninput = () => { state.refId = null; if (state.clip && $("#target").value.trim() !== state.clip.text) clearClip(); };

// ---------- reference audio ----------
async function speak(text, register, audioEl) {
  const o = opts();
  const r = await api("/api/tts", { text, register, accent: o.accent, gender: o.gender });
  audioEl.src = r.url; audioEl.hidden = false; audioEl.play();
  return r.id;
}
$("#btn-ref").onclick = (e) => busy(e.currentTarget, async () => {
  const text = $("#target").value.trim();
  if (!text) return;
  if (state.clip && state.clip.text === text) { const a = $("#ref-audio"); a.src = clipUrl(state.clip); a.hidden = false; a.play(); return; }
  state.refId = await speak(text, state.register, $("#ref-audio"));
  state.refKey = text + state.register + JSON.stringify(opts());
});

function suggestionCard(text, note, tag, register) {
  const d = document.createElement("div");
  d.className = "sugg";
  d.innerHTML = `<div class="txt">${tag ? `<div class="tag">${esc(tag)}</div>` : ""}<div>${esc(text)}</div><div class="note">${esc(note || "")}</div></div>
    <button class="play">▶</button><button class="use">Practise</button><audio hidden></audio>`;
  d.querySelector(".play").onclick = (e) => busy(e.currentTarget, () => speak(text, register, d.querySelector("audio")));
  d.querySelector(".use").onclick = () => {
    $("#target").value = text; setRegister(register); window.scrollTo({ top: 0, behavior: "smooth" });
  };
  return d;
}

$("#btn-suggest").onclick = (e) => busy(e.currentTarget, async () => {
  const r = await api("/api/sentences", { register: state.register, topic: $("#topic").value, level: opts().level });
  const box = $("#suggestions"); box.innerHTML = "";
  const items = Array.isArray(r.items) ? r.items : [];
  if (!items.length) box.textContent = r.items?.raw || "No suggestions returned.";
  items.forEach((it) => box.appendChild(suggestionCard(it.text, it.note, state.cfg.registers[state.register].label, state.register)));
});

$("#btn-rewrite").onclick = (e) => busy(e.currentTarget, async () => {
  const text = $("#target").value.trim();
  if (!text) return;
  const r = await api("/api/rewrite", { text });
  const box = $("#suggestions"); box.innerHTML = "";
  for (const k of ["work", "casual", "intimate"]) {
    if (r[k]) box.appendChild(suggestionCard(r[k].text, r[k].note, state.cfg.registers[k].label, k));
  }
  if (!box.children.length) box.textContent = r.raw || "No rewrite returned.";
});

$("#btn-task").onclick = (e) => busy(e.currentTarget, async () => {
  const t = await api("/api/task", { register: state.register, level: opts().level });
  state.task = t;
  $("#free-prompt").value = t.task || "";
  $("#task-box").innerHTML = `<div class="sugg"><div class="txt"><div class="tag">${esc(t.scene || "")}</div>
    <div>${esc(t.task || t.raw || "")}</div>
    ${(t.useful_phrases || []).length ? `<div class="note">Useful: ${t.useful_phrases.map(esc).join(" · ")}</div>` : ""}</div></div>`;
});

// ---------- recording ----------
let rec = null, chunks = [], t0 = 0, timer = null;
$("#btn-rec").onclick = async () => {
  if (rec && rec.state === "recording") { rec.stop(); return; }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: true } });
  } catch (e) { alert("Microphone not available: " + e.message); return; }
  const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"].find((m) => MediaRecorder.isTypeSupported(m)) || "";
  rec = new MediaRecorder(stream, mime ? { mimeType: mime } : {});
  chunks = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  rec.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    clearInterval(timer);
    $("#btn-rec").classList.remove("live"); $("#btn-rec").textContent = "● Record";
    submit(new Blob(chunks, { type: rec.mimeType }), rec.mimeType.includes("mp4") ? "m4a" : "webm");
  };
  rec.start();
  t0 = Date.now();
  $("#btn-rec").classList.add("live"); $("#btn-rec").textContent = "■ Stop";
  timer = setInterval(() => ($("#rec-status").textContent = `Recording… ${((Date.now() - t0) / 1000).toFixed(0)} s`), 250);
};

async function submit(blob, ext) {
  const fd = new FormData();
  fd.append("audio", blob, "rec." + ext);
  fd.append("mode", state.mode);
  fd.append("register", state.register);
  fd.append("accent", opts().accent);
  if (state.mode === "read") {
    const text = $("#target").value.trim();
    fd.append("target_text", text);
    const key = text + state.register + JSON.stringify(opts());
    if (state.clip && state.clip.text === text) {
      state.refId = "lib:" + state.clip.id; state.refKey = key;
    } else if (!state.refId || state.refKey !== key) {
      $("#rec-status").innerHTML = `<span class="spinner"></span>Preparing reference voice…`;
      const r = await api("/api/tts", { text, register: state.register, accent: opts().accent, gender: opts().gender });
      state.refId = r.id; state.refKey = key;
      $("#ref-audio").src = r.url; $("#ref-audio").hidden = false;
    }
    fd.append("ref_id", state.refId);
  } else {
    fd.append("prompt", $("#free-prompt").value);
  }
  $("#rec-status").innerHTML = `<span class="spinner"></span>Analysing pronunciation, intonation and fluency…`;
  $("#btn-rec").disabled = true;
  try {
    const r = await api("/api/analyze", fd);
    $("#rec-status").textContent = "Done. Record again any time.";
    render(r);
    loadCoach(r.id);
  } catch (e) {
    $("#rec-status").textContent = "Error: " + e.message;
  } finally { $("#btn-rec").disabled = false; }
}

// ---------- results ----------
function render(r) {
  state.result = r;
  $("#results").hidden = false;
  const s = r.scores;
  const tile = (k, v, sub = "") => `<div class="tile"><div class="v ${cls(v)}">${v ?? "–"}</div><div class="k">${k}</div>${sub}</div>`;
  $("#tiles").innerHTML = tile("Overall", s.overall) + tile("Pronunciation", s.pronunciation) +
    tile("Fluency", s.fluency) + tile("Intonation", s.intonation) + (s.accuracy != null ? tile("Accuracy", s.accuracy) : "") +
    `<div class="tile" id="cefr-tile"><div class="v">…</div><div class="k">CEFR (coach)</div></div>`;

  $("#my-audio").src = r.audio_url;
  const box = $("#words"); box.innerHTML = "";
  r.words.forEach((w, i) => {
    const sp = document.createElement("span");
    sp.className = "w " + (w.filler ? "filler" : cls(w.score)) + (w.stress_issue ? " stress" : "");
    sp.textContent = w.text;
    sp.title = w.filler ? "filler" : `${w.score ?? "?"} /${w.ipa}/`;
    sp.onclick = () => { document.querySelectorAll(".w.sel").forEach((x) => x.classList.remove("sel")); sp.classList.add("sel"); wordDetail(i); };
    box.appendChild(sp);
    box.appendChild(document.createTextNode(" "));
  });
  $("#word-detail").hidden = true;
  $("#asr-line").innerHTML = r.mode === "read"
    ? `Speech recogniser heard: “${esc(r.transcript)}”` + (r.accuracy?.extra_words?.length ? ` · extra words: ${esc(r.accuracy.extra_words.join(", "))}` : "")
    : "";

  drawPitch(r);
  const ends = r.intonation.sentences || [];
  $("#endings").innerHTML = ends.length ? `<table><tr><th>Sentence ending</th><th>Expected</th><th>You</th></tr>${ends.map((e) =>
    `<tr><td>${esc(e.text)}</td><td>${e.expected}</td><td><span class="pill ${e.status === "ok" ? "good" : e.status === "warn" ? "ok" : "bad"}">${e.got}${e.delta_st != null ? ` (${e.delta_st > 0 ? "+" : ""}${e.delta_st} st)` : ""}</span></td></tr>`).join("")}</table>` : "";
  const ref = r.reference;
  let em = "";
  if (ref) {
    const max = Math.max(1, ...ref.words.flatMap((w) => [Math.abs(w.you), Math.abs(w.ref)]));
    const h = (v) => Math.max(3, ((v + max) / (2 * max)) * 62);
    em = `<h2 style="margin-top:14px">Word emphasis vs reference</h2>
      <p class="hint">Taller bar = more prominent (louder, higher, longer). Emphasis match ${fmtCorr(ref.emphasis_match)}, pitch-shape match ${fmtCorr(ref.pitch_shape_match)}${ref.pace_ratio ? `, you took ${ref.pace_ratio}× the reference time` : ""}.</p>
      <div class="bars">${ref.words.map((w) => `<div class="bar"><div class="pair"><div class="y" style="height:${h(w.you)}px"></div><div class="r" style="height:${h(w.ref)}px"></div></div>${esc(w.text.replace(/[.,!?]/g, ""))}</div>`).join("")}</div>
      ${ref.missed_emphasis.length ? `<p>The reference stresses <b>${ref.missed_emphasis.map(esc).join(", ")}</b> — you didn't.</p>` : ""}
      ${ref.extra_emphasis.length ? `<p>You stressed <b>${ref.extra_emphasis.map(esc).join(", ")}</b>, which the reference keeps light.</p>` : ""}`;
  }
  $("#emphasis").innerHTML = em;

  const f = r.fluency || {};
  const rows = [
    ["Speaking rate", f.wpm != null ? `${f.wpm} words/min` : "–"],
    ["Articulation rate", f.articulation_syl_per_s != null ? `${f.articulation_syl_per_s} syllables/s (excl. pauses)` : "–"],
    ["Pauses", `${f.pauses ?? 0} (${f.long_pauses ?? 0} longer than 1 s), ${f.pause_time_pct ?? 0}% of time`],
    ["Pauses inside a phrase", f.mid_phrase_pauses ?? 0],
    ["Mean length of run", f.mean_length_of_run_syl != null ? `${f.mean_length_of_run_syl} syllables between pauses` : "–"],
    ["Fillers (um, uh…)", `${f.fillers ?? 0} (${f.fillers_per_min ?? 0}/min)`],
    ["Discourse markers", Object.entries(f.discourse_markers || {}).map(([k, v]) => `${k} ×${v}`).join(", ") || "none"],
    ["Repetitions", f.repetitions ?? 0],
    ["Pitch variation", r.intonation.pitch_sd_st != null ? `${r.intonation.pitch_sd_st} st SD, range ${r.intonation.pitch_range_st} st` : "–"],
  ];
  $("#fluency").innerHTML = `<table>${rows.map(([k, v]) => `<tr><td>${k}</td><td>${esc(v)}</td></tr>`).join("")}</table>`;
  $("#checks").innerHTML = `<table>${r.register_checks.map((c) => `<tr><td>${c.metric}</td><td><span class="pill ${c.ok ? "good" : "bad"}">${esc(c.value)}</span></td><td class="hint">${esc(c.target)}${c.ok ? "" : "<br>" + esc(c.msg)}</td></tr>`).join("")}</table>
    <details><summary>Register notes</summary><ul>${r.register_notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul></details>`;

  $("#sounds-card").hidden = !r.sound_problems.length;
  $("#sounds").innerHTML = r.sound_problems.map((p) => `<div class="issue"><div class="area ipa">/${esc(p.phone)}/ × ${p.count}${p.heard_as.length ? ` — sounded like /${p.heard_as.map(esc).join("/, /")}/` : ""}</div>
    <div>${esc(p.tip)}</div><div class="hint">in: ${p.examples.map(esc).join(", ")}</div></div>`).join("");
  $("#coach-body").innerHTML = `<p class="hint"><span class="spinner"></span>The coach is listening to the analysis…</p>`;
  $("#results").scrollIntoView({ behavior: "smooth" });
  realExamples(r);
}

const fmtCorr = (c) => (c == null ? "n/a" : `${Math.round(Math.max(0, c) * 100)}%`);

function playSegment(audioEl, t0, t1) {
  audioEl.currentTime = Math.max(0, t0 - 0.05);
  const stop = () => { if (audioEl.currentTime >= t1 + 0.05) { audioEl.pause(); audioEl.removeEventListener("timeupdate", stop); } };
  audioEl.addEventListener("timeupdate", stop);
  audioEl.play();
}

function wordDetail(i) {
  const r = state.result, w = r.words[i], d = $("#word-detail");
  d.hidden = false;
  const refW = r.reference?.words.find((x) => x.i === i);
  d.innerHTML = `<div class="row between"><div><b style="font-size:20px">${esc(w.text)}</b> &nbsp; expected <b>/${esc(w.ipa)}/</b> &nbsp; heard <b>/${esc(w.heard)}/</b></div>
    <div>${w.start != null ? `<button id="pw">▶ You</button>` : ""} ${refW && r.ref_url ? `<button id="pr">▶ Reference</button>` : ""}</div></div>
    <div class="phones">${w.phones.map((p) => `<div class="ph ${cls(p.score)}">${esc(p.p)}${p.stress === 1 ? "ˈ" : ""}<small>${p.heard !== p.p ? "heard " + esc(p.heard) : "&nbsp;"}</small><small>${p.score}</small></div>`).join("")}</div>
    ${w.asr ? `<p>Speech recogniser understood this word as <b>“${esc(w.asr)}”</b> — a listener might too.</p>` : ""}
    ${w.stress_issue ? `<p class="ok">Word stress: ${esc(w.stress_issue)}.</p>` : ""}
    ${w.phones.filter((p) => p.score < 60).map((p) => PHONE_HINT(p)).join("")}`;
  if (w.start != null) $("#pw").onclick = () => playSegment($("#my-audio"), w.start, w.end);
  if (refW && r.ref_url) $("#pr").onclick = () => { const a = $("#ref-audio"); if (!a.src.endsWith(r.ref_url)) a.src = r.ref_url; playSegment(a, refW.ref_start, refW.ref_end); };
}
const PHONE_HINT = (p) => {
  const prob = state.result.sound_problems.find((x) => x.phone === p.p);
  return prob?.tip ? `<p class="hint">/${esc(p.p)}/: ${esc(prob.tip)}</p>` : "";
};

function drawPitch(r) {
  const c = $("#pitch"), ctx = c.getContext("2d");
  const W = (c.width = c.clientWidth * devicePixelRatio), H = (c.height = 220 * devicePixelRatio);
  ctx.scale(devicePixelRatio, devicePixelRatio);
  const w = W / devicePixelRatio, h = 220, pad = 28;
  const css = getComputedStyle(document.documentElement);
  const col = (n) => css.getPropertyValue(n).trim();
  ctx.clearRect(0, 0, w, h);
  const ts = r.pitch.t, t0 = r.words.find((x) => x.start != null)?.start ?? ts[0], t1 = [...r.words].reverse().find((x) => x.end != null)?.end ?? ts[ts.length - 1];
  const x = (t) => pad + ((t - t0) / Math.max(0.1, t1 - t0)) * (w - 2 * pad);
  const y = (st) => h / 2 - (st / 10) * (h / 2 - 24);
  ctx.strokeStyle = col("--line"); ctx.lineWidth = 1;
  for (const s of [-6, 0, 6]) { ctx.beginPath(); ctx.moveTo(pad, y(s)); ctx.lineTo(w - pad, y(s)); ctx.stroke(); }
  ctx.fillStyle = col("--muted"); ctx.font = "11px -apple-system, sans-serif";
  ctx.fillText("+6", 2, y(6) + 4); ctx.fillText("0", 8, y(0) + 4); ctx.fillText("−6", 2, y(-6) + 4);
  const line = (t, st, color, width) => {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath(); let pen = false;
    for (let i = 0; i < t.length; i++) {
      if (st[i] == null || t[i] < t0 - 0.1 || t[i] > t1 + 0.1) { pen = false; continue; }
      pen ? ctx.lineTo(x(t[i]), y(st[i])) : ctx.moveTo(x(t[i]), y(st[i])); pen = true;
    }
    ctx.stroke();
  };
  if (r.reference?.contour) line(r.reference.contour.t, r.reference.contour.st, col("--ref"), 2);
  line(ts, r.pitch.st, col("--you"), 2.5);
  ctx.fillStyle = col("--muted");
  let lastX = -99;
  for (const wd of r.words) {
    if (wd.start == null) continue;
    const xx = x(wd.start);
    if (xx - lastX < 26) continue;
    ctx.fillText(wd.text.replace(/[.,!?]/g, ""), xx, h - 6); lastX = xx + ctx.measureText(wd.text).width;
  }
}
window.addEventListener("resize", () => state.result && drawPitch(state.result));

// ---------- coach (LLM) ----------
async function loadCoach(id) {
  try {
    const c = await api(`/api/feedback/${id}`, {});
    renderCoach(c);
  } catch (e) { $("#coach-body").innerHTML = `<p class="bad">Coach unavailable: ${esc(e.message)}</p>`; }
}

function renderCoach(c) {
  if (c.raw) { $("#coach-body").innerHTML = `<pre style="white-space:pre-wrap">${esc(c.raw)}</pre>`; return; }
  const names = { fluency_coherence: "Fluency & coherence", lexical_resource: "Vocabulary", grammar: "Grammar", pronunciation: "Pronunciation", register_fit: "Register fit" };
  const t = $("#cefr-tile"); if (t) t.querySelector(".v").textContent = c.cefr || "–";
  const reg = state.result.register;
  $("#coach-body").innerHTML = `
    <div class="bands">${Object.entries(c.bands || {}).filter(([, v]) => v != null).map(([k, v]) => `<div class="band">${names[k] || k} <b>${v}</b>/9</div>`).join("")}</div>
    <p>${esc(c.summary)}</p>
    ${(c.strengths || []).length ? `<p><b>Strengths:</b> ${c.strengths.map(esc).join(" · ")}</p>` : ""}
    ${(c.issues || []).map((i) => `<div class="issue"><div class="area">${esc(i.area)}</div><div>${esc(i.what)}</div><div class="hint">Fix: ${esc(i.fix)}</div></div>`).join("")}
    ${c.corrected ? `<p><b>Corrected:</b></p><blockquote>${esc(c.corrected)}</blockquote>` : ""}
    ${c.natural_version ? `<div class="row between"><b>How a native would say it (${esc(state.cfg.registers[reg].label)})</b><span><button id="nat-play">▶</button> <button id="nat-use">Practise this</button></span></div><blockquote>${esc(c.natural_version)}</blockquote><audio id="nat-audio" hidden></audio>` : ""}
    ${c.tone_coaching ? `<p><b>Voice &amp; tone:</b> ${esc(c.tone_coaching)}</p>` : ""}
    ${(c.drills || []).length ? `<p><b>Drills:</b></p><ul>${c.drills.map((d) => `<li>${esc(d)}</li>`).join("")}</ul>` : ""}`;
  if (c.natural_version) {
    $("#nat-play").onclick = (e) => busy(e.currentTarget, () => speak(c.natural_version, reg, $("#nat-audio")));
    $("#nat-use").onclick = () => {
      $("#modes button[data-m=read]").click(); $("#target").value = c.natural_version; state.refId = null; window.scrollTo({ top: 0, behavior: "smooth" });
    };
  }
}

// ---------- history ----------
function showPage(id) {
  for (const p of ["practice", "library", "history"]) $("#" + p).hidden = p !== id;
  window.scrollTo({ top: 0 });
}
$("#tab-practice").onclick = () => showPage("practice");
$("#tab-history").onclick = async () => {
  showPage("history");
  const items = await api("/api/history");
  $("#hist-table tbody").innerHTML = items.map((it) => `<tr class="click" data-id="${it.id}"><td>${new Date(it.created * 1000).toLocaleString()}</td><td>${it.register}</td><td>${esc(it.text)}</td>
    ${["overall", "pronunciation", "fluency", "intonation"].map((k) => `<td class="${cls(it.scores[k])}">${it.scores[k] ?? "–"}</td>`).join("")}<td>${it.cefr || ""}</td></tr>`).join("");
  document.querySelectorAll("#hist-table tr.click").forEach((tr) => (tr.onclick = async () => {
    const r = await api(`/api/sessions/${tr.dataset.id}`);
    $("#back").click(); render(r); r.coach ? renderCoach(r.coach) : loadCoach(r.id);
  }));
  drawTrend(items.slice().reverse());
};
$("#back").onclick = () => showPage("practice");

function drawTrend(items) {
  const c = $("#trend"), ctx = c.getContext("2d");
  c.width = c.clientWidth * devicePixelRatio; c.height = 180 * devicePixelRatio; ctx.scale(devicePixelRatio, devicePixelRatio);
  const w = c.clientWidth, h = 180, pad = 24;
  const css = getComputedStyle(document.documentElement);
  if (items.length < 2) { ctx.fillStyle = css.getPropertyValue("--muted"); ctx.fillText("Record a few sessions to see your trend.", pad, h / 2); return; }
  const x = (i) => pad + (i / (items.length - 1)) * (w - 2 * pad), y = (v) => h - pad - (v / 100) * (h - 2 * pad);
  const series = [["pronunciation", "--you"], ["fluency", "--good"], ["intonation", "--ref"]];
  for (const [k, colv] of series) {
    ctx.strokeStyle = css.getPropertyValue(colv).trim(); ctx.lineWidth = 2; ctx.beginPath();
    items.forEach((it, i) => { const v = it.scores[k]; if (v != null) (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))); });
    ctx.stroke();
  }
  ctx.fillStyle = css.getPropertyValue("--muted"); ctx.font = "11px -apple-system, sans-serif";
  ctx.fillText("blue: pronunciation · green: fluency · orange: intonation", pad, 12);
}


// ---------- real-speech library ----------
const clipUrl = (c) => `/api/library/clip/${c.id}/audio`;
let lib = { offset: 0, stats: null, polling: null };
const LIB_FILTERS = ["q", "register", "source", "tone", "act", "gender", "accent", "length", "pace", "order"];

function clipMeta(c) {
  return [c.source_label, c.speaker, c.accent && c.source !== "meld" ? c.accent : "", c.emotion && c.emotion !== "neutral" ? c.emotion : "",
    c.style && c.style !== "default" ? c.style : "", `${c.duration}s`, `${c.wpm} wpm`].filter(Boolean).map(esc).join(" · ");
}

function clipCard(c, compact = false) {
  const d = document.createElement("div");
  d.className = "clip";
  d.innerHTML = `<div class="txt"><div class="line">${esc(c.text)}</div>
      ${c.context && !compact ? `<div class="ctx">before: ${esc(c.context.slice(-180))}</div>` : ""}
      <div class="meta"><span class="tag ${c.register}">${c.register}</span>${c.tone ? `<span class="tag">${esc(c.tone)}</span>` : ""}${c.act ? `<span class="tag">${esc(c.act)}</span>` : ""} ${clipMeta(c)}</div></div>
    <div class="btns"><button class="play">▶</button>${compact ? "" : `<button class="fav">${c.favorite ? "★" : "☆"}</button>`}<button class="primary shadow">Shadow</button></div>`;
  const audio = new Audio();
  d.querySelector(".play").onclick = () => { audio.src = clipUrl(c); audio.play(); };
  const fav = d.querySelector(".fav");
  if (fav) fav.onclick = async () => { const r = await api(`/api/library/clip/${c.id}/favorite`, {}); fav.textContent = r.favorite ? "★" : "☆"; };
  d.querySelector(".shadow").onclick = () => useClip(c);
  return d;
}

function useClip(c) {
  state.clip = c;
  showPage("practice");
  $("#modes button[data-m=read]").click();
  if (["work", "casual", "intimate"].includes(c.register)) setRegister(c.register);
  $("#target").value = c.text;
  state.refId = "lib:" + c.id;
  const b = $("#clip-banner");
  b.hidden = false;
  b.innerHTML = `<button id="cb-play">▶</button><div class="grow"><b>Shadowing a real recording</b> · ${clipMeta(c)}
    ${c.context ? `<div class="hint">before: ${esc(c.context.slice(-200))}</div>` : ""}</div><button id="cb-x" class="ghost">✕</button>`;
  $("#cb-play").onclick = () => { const a = $("#ref-audio"); a.src = clipUrl(c); a.hidden = false; a.play(); };
  $("#cb-x").onclick = () => clearClip();
  const a = $("#ref-audio"); a.src = clipUrl(c); a.hidden = false;
}
function clearClip() { state.clip = null; state.refId = null; $("#clip-banner").hidden = true; }

function libQuery(extra = {}) {
  const p = new URLSearchParams();
  for (const k of LIB_FILTERS) { const v = $("#lf-" + k).value; if (v) p.set(k, v); }
  if ($("#lf-fav").checked) p.set("favorites", "true");
  for (const [k, v] of Object.entries(extra)) p.set(k, v);
  return p;
}

async function libSearch(append = false, extra = {}) {
  if (!append) lib.offset = 0;
  const p = libQuery({ ...extra, offset: lib.offset, limit: 30 });
  const r = await api("/api/library/search?" + p);
  const box = $("#lib-results");
  if (!append) box.innerHTML = r.total ? `<p class="hint">${r.total} clips</p>` : `<p class="hint">No clips match yet. Add some below.</p>`;
  r.items.forEach((c) => box.appendChild(clipCard(c)));
  lib.offset += r.items.length;
  $("#lf-more").hidden = lib.offset >= r.total;
}
$("#lf-go").onclick = () => libSearch();
$("#lf-q").onkeydown = (e) => e.key === "Enter" && libSearch();
$("#lf-random").onclick = () => { $("#lf-order").value = "random"; libSearch(); };
$("#lf-more").onclick = () => libSearch(true);

function fillSelect(id, items, label) {
  const sel = $(id), cur = sel.value;
  sel.innerHTML = `<option value="">${label}</option>` + items.filter((x) => x.v).map((x) => `<option value="${esc(x.v)}">${esc(x.t ?? x.v)}${x.n != null ? ` (${x.n})` : ""}</option>`).join("");
  sel.value = cur;
}

async function libStats() {
  const s = await api("/api/library/stats");
  lib.stats = s;
  const reg = Object.fromEntries(s.by_register.map((x) => [x.register, x.n]));
  $("#lib-stats").textContent = `${s.total} clips · ${s.hours} h · work ${reg.work || 0} · casual ${reg.casual || 0} · intimate ${reg.intimate || 0}` +
    (s.total > s.labeled ? ` · ${s.total - s.labeled} being labelled` : "");
  const srcs = s.by_source.map((x) => ({ v: x.source === "your media" ? "media" : x.source, t: x.source === "your media" ? "Your media" : s.catalog[x.source]?.label || x.source, n: x.n }));
  fillSelect("#lf-source", srcs, "Any source");
  fillSelect("#lf-tone", s.tones.map((x) => ({ v: x.tone, n: x.n })), "Any tone");
  fillSelect("#lf-act", s.acts.map((x) => ({ v: x.act, n: x.n })), "Any purpose");
  fillSelect("#lf-accent", s.accents.map((x) => ({ v: x.accent, n: x.n })), "Any accent");
  const have = Object.fromEntries(s.by_source.map((x) => [x.source, x.n]));
  $("#catalog").innerHTML = Object.entries(s.catalog).map(([k, c]) => `<div class="src"><div class="txt"><b>${esc(c.label)}</b> <span class="hint">· ${esc(c.kind)} · ${esc(c.accent)} · ${esc(c.license)}</span>
      <div class="hint">${esc(c.about)} Download: ${esc(c.size)}. In library: ${have[k] || 0}.</div></div>
      <label class="hint">add <input type="number" min="10" step="50" value="${k === "meld" ? 1000 : 300}" data-src="${k}"> clips</label><button data-add="${k}">Add</button></div>`).join("");
  document.querySelectorAll("[data-add]").forEach((b) => (b.onclick = async () => {
    const n = +document.querySelector(`input[data-src=${b.dataset.add}]`).value || 300;
    await api("/api/library/jobs", { kind: "ingest", source: b.dataset.add, max: n });
    pollJobs();
  }));
}

$("#imp-go").onclick = (e) => busy(e.currentTarget, async () => {
  await api("/api/library/jobs", { kind: "import", src: $("#imp-src").value, title: $("#imp-title").value || null });
  $("#imp-src").value = ""; $("#imp-title").value = "";
  pollJobs();
});

async function pollJobs() {
  const jobs = await api("/api/library/jobs");
  $("#jobs").innerHTML = jobs.map((j) => {
    const what = j.kind === "ingest" ? (lib.stats?.catalog[j.args.source]?.label || j.args.source) : (j.args.title || j.args.src);
    const live = ["queued", "running"].includes(j.status);
    const why = Object.entries(j.reasons || {}).map(([k, v]) => `${k} ${v}`).join(", ");
    return `<div class="job">${live ? '<span class="spinner"></span>' : ""}<div class="grow"><b>${esc(what)}</b> · ${j.status} · ${esc(j.message)}${why ? `<div class="hint">rejected: ${esc(why)}</div>` : ""}</div>
      ${live ? `<button data-stop="${j.id}" class="ghost">Stop</button>` : ""}</div>`;
  }).join("");
  document.querySelectorAll("[data-stop]").forEach((b) => (b.onclick = () => api(`/api/library/jobs/${b.dataset.stop}/stop`, {}).then(pollJobs)));
  const running = jobs.some((j) => ["queued", "running"].includes(j.status));
  clearTimeout(lib.polling);
  if (running) lib.polling = setTimeout(() => { pollJobs(); libStats(); }, 3000);
}

$("#tab-library").onclick = async () => {
  showPage("library");
  await libStats();
  if (!$("#lib-results").children.length) libSearch();
  pollJobs();
};

async function realExamples(r) {
  const box = $("#real-list"), card = $("#real-examples");
  const stop = new Set("the a an and or but to of in on at for is are was were be been i you he she it we they my your me this that with".split(" "));
  const words = (r.target_text || r.transcript).toLowerCase().replace(/[^a-z' ]/g, " ").split(/\s+/).filter((w) => w.length > 2 && !stop.has(w)).slice(0, 8);
  let items = [];
  try {
    if (words.length) items = (await api("/api/library/search?" + new URLSearchParams({ q: words.join(" "), any_words: "true", register: r.register, limit: 4 }))).items;
    if (items.length < 3) items = items.concat((await api("/api/library/search?" + new URLSearchParams({ register: r.register, order: "random", limit: 4 - items.length }))).items);
  } catch { items = []; }
  items = items.filter((c) => !state.clip || c.id !== state.clip.id);
  card.hidden = !items.length;
  box.innerHTML = "";
  items.forEach((c) => box.appendChild(clipCard(c, true)));
}

init();
