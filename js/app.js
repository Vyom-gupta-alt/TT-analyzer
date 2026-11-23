import { decodeAudio, detectHits, findRallies, analyzeStrokes, classify, strokeContext, compareStrokes } from "./engine.js";
import { trackWindows } from "./pose.js";

const $ = (id) => document.getElementById(id);
const LABEL = {
  serve_pendulum: "Pendulum serve", serve_reverse_pendulum: "Reverse pendulum serve", serve_tomahawk: "Tomahawk serve",
  serve_backhand: "Backhand serve", receive_short_flick: "Short flick", receive_long_flick: "Long flick",
  receive_push: "Push receive", forehand_topspin: "Forehand topspin", backhand_topspin: "Backhand topspin",
  forehand_push: "Forehand push", backhand_push: "Backhand push", block: "Block", counter_drive: "Counter-drive",
  chop: "Chop", lob: "Lob", smash: "Smash",
};
const label = (t) => LABEL[t] || t.replace(/_/g, " ");
const family = (t) => t.startsWith("serve_") ? "serve" : t.startsWith("receive_") ? "receive"
  : ["forehand_topspin", "backhand_topspin", "smash", "counter_drive"].includes(t) ? "attack" : "control";
const FAM = { serve: "Serve", receive: "Receive", attack: "Attack", control: "Control" };
const TABS = [
  ["all", "All", null], ["serve", "Serve", ["serve_pendulum", "serve_reverse_pendulum", "serve_tomahawk", "serve_backhand"]],
  ["receive", "Receive", ["receive_short_flick", "receive_long_flick", "receive_push"]],
  ["push", "Push", ["forehand_push", "backhand_push", "receive_push"]],
  ["fh", "Forehand loop", ["forehand_topspin"]], ["bh", "Backhand loop", ["backhand_topspin"]],
  ["attack", "Smash & drive", ["smash", "counter_drive"]], ["def", "Block & defence", ["block", "chop", "lob"]],
];
const PHASE = { preparation: "Backswing", contact: "Contact", follow_through: "Follow-through" };
const ANG = [["elbow", "Elbow"], ["knee_flexion", "Knee bend"], ["shoulder_abduction", "Arm raise"], ["trunk_lean", "Lean"]];
const EDGES = [[11,12],[11,13],[13,15],[12,14],[14,16],[15,19],[16,20],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],[27,31],[28,32]];
const PLAY = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
const PAUSE = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 5h4v14H7zM13 5h4v14h-4z"/></svg>';
const fmt = (s) => { const m = Math.floor(s / 60), x = s - m * 60; return `${m}:${x < 10 ? "0" : ""}${x.toFixed(1)}`; };
const pct = (x, of) => `${(x / of * 100).toFixed(3)}%`;
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const show = (id) => ["landing", "processing", "matchView", "compareView"].forEach((v) => { $(v).hidden = v !== id; });
$("play").innerHTML = PLAY;

// ---------------------------------------------------------------- state
const S = { url: null, name: "", duration: 0, fps: 30, rallies: [], strokes: [], OFF: [], TOTAL: 0, keyframes: [],
  mode: "rallies", lines: false, loop: false, sel: null, tab: "all", posing: false, poseError: null };
const vid = $("vid"), cv = $("cv");

// ---------------------------------------------------------------- upload
const input = $("fileInput"), drop = $("drop");
input.onchange = () => input.files[0] && start(input.files[0]);
["dragenter", "dragover"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (ev) => { const f = ev.dataTransfer.files?.[0]; if (f) start(f); });
$("sampleBtn").onclick = async () => {
  $("sampleBtn").textContent = "Loading the sample match…";
  try {
    const blob = await (await fetch("sample/match.mp4")).blob();
    start(new File([blob], "Sample match (animated).mp4", { type: "video/mp4" }));
  } catch { $("sampleBtn").textContent = "Couldn't load the sample. Check your connection and try again."; }
};

function step(name, state, note = "") {
  const li = document.querySelector(`#steps li[data-s="${name}"]`);
  li.className = state; li.querySelector("small").textContent = note;
  if (state === "done") li.querySelector("i").textContent = "✓";
}
const meter = (f) => { $("meter").style.width = `${Math.round(f * 100)}%`; };
function fail(msg) { $("procErr").hidden = false; $("procErr").innerHTML = msg; $("procTitle").textContent = "Something went wrong"; }

async function start(file) {
  S.name = file.name;
  S.url && URL.revokeObjectURL(S.url);
  S.url = URL.createObjectURL(file);
  $("fileChip").hidden = false; $("fileChip").innerHTML = `<b>${escapeHtml(file.name)}</b>`;
  $("procFile").textContent = file.name;
  show("processing"); meter(0.02);
  step("audio", "now");
  let samples;
  try {
    samples = await decodeAudio(file);
  } catch (e) {
    return fail(file.size > 1.2e9
      ? "This video is too big for the browser to read in one go. Trim it into shorter parts, or record at 720p, and try again."
      : "Couldn't read the sound from this video. Is it an MP4 or MOV file with sound? Videos without sound can't be cut into rallies.");
  }
  S.duration = samples.length / 22050;
  step("audio", "done"); meter(0.15); step("rallies", "now");
  await new Promise((r) => setTimeout(r, 30));
  const hits = detectHits(samples);
  S.rallies = findRallies(hits, S.duration);
  samples = null;
  step("rallies", "done", `${S.rallies.length} rallies`); meter(0.25);
  if (!S.rallies.length) {
    return fail("No rallies were found. The app listens for the ball hitting bats and the table, so the video needs clear sound of the ball. Try a video recorded closer to the table, without music or commentary over it.");
  }
  await openMatch();
  runPose();
}

async function openMatch() {
  vid.src = S.url;
  await new Promise((r) => { if (vid.readyState >= 1) r(); else vid.addEventListener("loadedmetadata", r, { once: true }); });
  S.duration = vid.duration || S.duration;
  grabber.src = S.url;
  show("matchView");
  refreshAll();
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- pose (strokes)
async function runPose() {
  S.posing = true; step("pose", "now");
  const pv = $("poseVideo"); pv.src = S.url;
  setPoseNote(0);
  try {
    const windows = S.rallies.map((r) => [r.start, r.end]);
    const fps = +(new URLSearchParams(location.search).get("posefps") || 30);
    const seqs = await trackWindows(pv, windows, (f) => setPoseNote(f), { fps });
    seqs.forEach((seq, i) => {
      const r = S.rallies[i];
      const found = analyzeStrokes(seq, r.hit_times).sort((a, b) => a.time - b.time);
      r.strokes = found.map((s, k) => {
        const [type, confidence] = classify(s.metrics, strokeContext(s, r, k === 0));
        return { ...s, stroke_type: type, confidence, rally_index: r.index, key: `${r.index}:${k}`, box: boxOf(s.track.image) };
      });
    });
    S.posing = false;
    $("poseNote").hidden = true;
  } catch (e) {
    console.error(e);
    S.posing = false; S.poseError = e;
    $("poseNote").hidden = false;
    $("poseNote").innerHTML = "<b>Stroke analysis couldn't run in this browser.</b> The rallies above still work. Try Chrome or Edge on a computer for the stroke breakdown.";
  }
  refreshAll();
}
function setPoseNote(f) {
  $("poseNote").hidden = false;
  $("poseNote").innerHTML = `<span class="busy"><b>Watching your strokes… ${Math.round(f * 100)}%</b></span> The rallies are ready to watch now; strokes appear below when this finishes. On most laptops this takes a minute or two per 10 minutes of rallies.`;
}
function boxOf(image) {
  let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
  image.forEach((f) => f.slice(11).forEach(([x, y, v]) => { if ((v ?? 1) > 0.3) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); } }));
  y0 -= 0.06 * (y1 - y0) + 0.05;
  return [Math.max(0, x0), Math.max(0, y0), Math.min(1, x1), Math.min(1, y1)];
}

// ---------------------------------------------------------------- frame grabs
const grabber = document.createElement("video");
grabber.muted = true; grabber.preload = "auto"; grabber.playsInline = true;
const grabCanvas = document.createElement("canvas");
let grabCache = new Map(), grabQueue = Promise.resolve();
function grab(src, t, crop, w = 360) {
  const key = `${src}|${t.toFixed(2)}|${crop ? crop.join(",") : "full"}|${w}`;
  if (grabCache.has(key)) return grabCache.get(key);
  const p = (grabQueue = grabQueue.then(() => new Promise((res) => {
    const go = () => {
      const done = () => {
        try {
          const vw = grabber.videoWidth, vh = grabber.videoHeight;
          const [sx, sy, sw, sh] = crop ? crop.map((v, i) => v * (i % 2 ? vh : vw)) : [0, 0, vw, vh];
          grabCanvas.width = w; grabCanvas.height = Math.round(w * sh / sw);
          grabCanvas.getContext("2d").drawImage(grabber, sx, sy, sw, sh, 0, 0, grabCanvas.width, grabCanvas.height);
          res(grabCanvas.toDataURL("image/jpeg", 0.82));
        } catch { res(null); }
      };
      grabber.addEventListener("seeked", done, { once: true });
      setTimeout(done, 4000);
      grabber.currentTime = t;
    };
    if (grabber.getAttribute("data-src") !== src) { grabber.setAttribute("data-src", src); grabber.src = src; }
    grabber.readyState >= 1 ? go() : grabber.addEventListener("loadedmetadata", go, { once: true });
  })));
  grabCache.set(key, p);
  return p;
}
const setBg = (el, p) => p.then((u) => { if (u && el) el.style.backgroundImage = `url(${u})`; });
function boxCrop(b, aspect = 4 / 5, pad = 1.25, vw = 16, vh = 9) {
  const cx = (b[0] + b[2]) / 2 * vw, cy = (b[1] + b[3]) / 2 * vh;
  let h = (b[3] - b[1]) * vh * pad, w = h * aspect;
  const bw = (b[2] - b[0]) * vw * pad;
  if (w < bw) { w = bw; h = w / aspect; }
  if (h > vh) { h = vh; w = h * aspect; }
  if (w > vw) { w = vw; h = w / aspect; }
  return [Math.max(0, Math.min(vw - w, cx - w / 2)) / vw, Math.max(0, Math.min(vh - h, cy - h / 2)) / vh, w / vw, h / vh];
}
const vAspect = () => [vid.videoWidth || 16, vid.videoHeight || 9];

// ---------------------------------------------------------------- render
function refreshAll() {
  S.strokes = S.rallies.flatMap((r) => r.strokes);
  S.keyframes = [...new Set(S.strokes.flatMap((s) => s.keyframes.map((k) => +k.time.toFixed(3))))].sort((a, b) => a - b);
  const kept = S.rallies.reduce((a, r) => a + r.duration, 0);
  $("sRallies").textContent = S.rallies.length;
  $("sStrokes").textContent = S.posing ? "…" : S.strokes.length;
  $("sLongest").innerHTML = `${S.rallies.reduce((a, r) => Math.max(a, r.hit_times.length), 0)}<small>hits</small>`;
  const cut = Math.max(0, Math.round((1 - kept / S.duration) * 100));
  $("sCut").innerHTML = `${cut}<small>%</small>`; $("cutBar").style.width = `${cut}%`;
  $("cutText").textContent = `${fmt(S.duration)} of video becomes ${fmt(kept)} of rallies.`;
  buildTracks(); buildStrip(); buildTabs(); renderList();
  $("strokeHint").textContent = S.posing ? "Strokes appear here when the analysis finishes." : "Pick a skill, then a stroke to see how you hit it.";
}
function toVirtual(t) { const i = S.rallies.findIndex((r) => t >= r.start && t <= r.end); return i < 0 ? null : S.OFF[i] + t - S.rallies[i].start; }
function fromVirtual(v) { for (let i = S.rallies.length - 1; i >= 0; i--) if (v >= S.OFF[i]) return S.rallies[i].start + v - S.OFF[i]; return 0; }
function buildTracks() {
  let acc = 0; S.OFF = S.rallies.map((r) => { const o = acc; acc += r.end - r.start; return o; }); S.TOTAL = acc || 1;
  const d = S.duration || 1;
  const tick = (s, x) => `<div class="tick${S.sel?.key === s.key ? " sel" : ""}" data-k="${s.key}" title="${label(s.stroke_type)} · ${fmt(s.time)}" style="left:${x};background:var(--${family(s.stroke_type)})"></div>`;
  $("trackF").innerHTML = S.rallies.map((r) => `<div class="blk" style="left:${pct(r.start, d)};width:${pct(r.end - r.start, d)}"><em>${r.index + 1}</em></div>`).join("")
    + S.strokes.map((s) => tick(s, pct(s.time, d))).join("") + `<div class="head" id="headF"></div>`;
  $("trackR").innerHTML = S.rallies.map((r, i) => `<div class="blk" style="left:calc(${pct(S.OFF[i], S.TOTAL)} + 2px);width:calc(${pct(r.end - r.start, S.TOTAL)} - 4px)"><em>${r.index + 1}</em></div>`).join("")
    + S.strokes.map((s) => { const v = toVirtual(s.time); return v === null ? "" : tick(s, pct(v, S.TOTAL)); }).join("") + `<div class="head" id="headR"></div>`;
}
const seekOn = (el, f) => el.addEventListener("click", (e) => { const r = el.getBoundingClientRect(); vid.currentTime = f((e.clientX - r.left) / r.width); });
seekOn($("trackF"), (f) => f * S.duration); seekOn($("trackR"), (f) => fromVirtual(f * S.TOTAL));
(function buildSpeeds() {
  const box = $("speeds");
  [0.25, 0.5, 1, 1.5, 2].forEach((s) => {
    const b = document.createElement("button"); b.textContent = `${s}×`; if (s === 1) b.className = "on";
    b.onclick = () => { vid.playbackRate = s; [...box.children].forEach((c) => c.classList.toggle("on", c === b)); };
    box.appendChild(b);
  });
})();
function buildStrip() {
  const strip = $("strip"); strip.innerHTML = "";
  S.rallies.forEach((r) => {
    const b = document.createElement("button");
    b.className = "rcard"; b.dataset.i = r.index;
    b.innerHTML = `<div class="thumb"><span class="n">Rally ${r.index + 1}</span><span class="d">${r.duration.toFixed(1)}s</span></div>
      <div class="rmeta"><div class="dots">${r.strokes.map((s) => `<i title="${label(s.stroke_type)}" style="background:var(--${family(s.stroke_type)})"></i>`).join("")}</div>
      <small class="num">${r.hit_times.length} ball contacts · starts ${fmt(r.start)}</small></div>`;
    b.onclick = () => { vid.currentTime = r.start; vid.play().catch(() => {}); window.scrollTo({ top: 0, behavior: "smooth" }); };
    strip.appendChild(b);
    setBg(b.querySelector(".thumb"), grab(S.url, Math.min(r.serve_contact_time + 1.0, r.end - 0.1), null, 480));
  });
}
function buildTabs() {
  $("tabs").innerHTML = TABS.map(([id, name, types]) => {
    const n = types ? S.strokes.filter((s) => types.includes(s.stroke_type)).length : S.strokes.length;
    return `<button role="tab" data-id="${id}" class="${id === S.tab ? "on" : ""}" aria-selected="${id === S.tab}">${name}<span class="c">${n}</span></button>`;
  }).join("");
  $("tabs").querySelectorAll("button").forEach((b) => b.onclick = () => { S.tab = b.dataset.id; buildTabs(); renderList(); });
}
function renderList() {
  const types = TABS.find((x) => x[0] === S.tab)[2];
  const shown = types ? S.strokes.filter((s) => types.includes(s.stroke_type)) : S.strokes;
  const list = $("list");
  list.innerHTML = shown.length ? "" : `<div class="empty">${S.posing ? "Watching your strokes…" : "No strokes like this in this match."}</div>`;
  const [vw, vh] = vAspect();
  shown.forEach((s) => {
    const b = document.createElement("button");
    b.className = "srow" + (S.sel?.key === s.key ? " on" : "");
    b.innerHTML = `<span class="mini"></span><span><b>${label(s.stroke_type)}</b><small class="num">Rally ${s.rally_index + 1} · ${fmt(s.time)}</small></span>
      <span class="pill" style="background:var(--${family(s.stroke_type)})">${FAM[family(s.stroke_type)]}</span>`;
    b.onclick = () => select(s);
    list.appendChild(b);
    setBg(b.querySelector(".mini"), grab(S.url, s.time, boxCrop(s.box, 1, 1.25, vw, vh), 160));
  });
  if (!S.sel) { $("detail").className = "empty"; $("detail").textContent = S.posing ? "Strokes appear here soon." : "Pick a stroke on the left."; }
}
function select(s) {
  S.sel = s; $("loop").disabled = false;
  document.querySelectorAll(".tick").forEach((m) => m.classList.toggle("sel", m.dataset.k === s.key));
  renderList(); renderDetail();
  vid.currentTime = Math.max(s.start - 0.3, 0); vid.play().catch(() => {});
}
function renderDetail() {
  const s = S.sel, m = s.metrics, el = $("detail"), [vw, vh] = vAspect();
  el.className = "detail card";
  const cm = (x) => Math.round(x * 100);
  el.innerHTML = `
    <div class="dhead"><div><div class="eyebrow">Rally ${s.rally_index + 1} · ${fmt(s.time)} · ${Math.round(s.confidence * 100)}% sure</div>
      <h3>${label(s.stroke_type)}</h3></div><span class="spacer"></span>
      <button class="btn" id="saveRef">Save as reference</button>
      <button class="btn primary" id="toCompare">Compare →</button></div>
    <div class="phases">${s.keyframes.map((k) => `<button class="phase ${k.phase === "contact" ? "contact" : ""}" data-t="${k.time}">
      <div class="shot"><span>${PHASE[k.phase]}</span></div>
      <dl>${ANG.map(([key, n]) => `<dt>${n}</dt><dd>${Math.round(k.angles[key])}°</dd>`).join("")}</dl></button>`).join("")}</div>
    <div class="facts">
      <div class="fact"><span>Racket speed</span><b>${m.racket_speed_peak.toFixed(1)}<small>m/s</small></b></div>
      <div class="fact"><span>Swing path</span><b>${Math.abs(Math.round(m.swing_angle_at_contact))}°<small>${m.swing_angle_at_contact >= 0 ? "up" : "down"}</small></b></div>
      <div class="fact"><span>Contact in front</span><b>${cm(m.contact_forward_of_com)}<small>cm</small></b></div>
      <div class="fact"><span>Contact height</span><b>${cm(m.contact_height_vs_com)}<small>cm</small></b></div>
      <div class="fact"><span>Backswing</span><b>${m.backswing_duration.toFixed(2)}<small>s</small></b></div>
    </div>
    <p class="fine">Tap a picture to jump the video to that moment. Racket speed is measured relative to the hips, so it shows arm and body rotation, not footwork.</p>`;
  el.querySelectorAll(".phase").forEach((b) => {
    const t = +b.dataset.t;
    setBg(b.querySelector(".shot"), grab(S.url, t, boxCrop(s.box, 4 / 5, 1.25, vw, vh), 360));
    b.onclick = () => { vid.pause(); vid.currentTime = t; window.scrollTo({ top: 0, behavior: "smooth" }); };
  });
  $("toCompare").onclick = () => openCompare(s);
  $("saveRef").onclick = () => saveReference(s);
}

// ---------------------------------------------------------------- playback
function toggle() {
  if (vid.paused) {
    if (S.mode === "rallies" && !S.rallies.some((r) => vid.currentTime >= r.start && vid.currentTime < r.end))
      vid.currentTime = (S.rallies.find((r) => r.start > vid.currentTime) || S.rallies[0]).start;
    vid.play().catch(() => {});
  } else vid.pause();
}
const stepF = (d) => { vid.pause(); vid.currentTime = Math.min(Math.max(vid.currentTime + d / S.fps, 0), S.duration); };
function stepK(d) {
  vid.pause(); const now = vid.currentTime;
  const t = d > 0 ? S.keyframes.find((k) => k > now + 0.01) : [...S.keyframes].reverse().find((k) => k < now - 0.01);
  if (t !== undefined) vid.currentTime = t;
}
$("play").onclick = toggle; vid.onclick = toggle;
$("fPrev").onclick = () => stepF(-1); $("fNext").onclick = () => stepF(1);
$("kPrev").onclick = () => stepK(-1); $("kNext").onclick = () => stepK(1);
function setMode(m) {
  S.mode = m; $("mR").classList.toggle("on", m === "rallies"); $("mF").classList.toggle("on", m === "full");
  $("laneR").classList.toggle("on", m === "rallies"); $("laneF").classList.toggle("on", m === "full");
  $("modeTag").textContent = m === "rallies" ? "● Rallies only" : "Full match";
}
$("mR").onclick = () => setMode("rallies"); $("mF").onclick = () => setMode("full");
const setLines = (v) => { S.lines = v; $("lines").classList.toggle("on", v); };
const setLoop = (v) => { S.loop = v && !!S.sel; $("loop").classList.toggle("on", S.loop); };
$("lines").onclick = () => setLines(!S.lines); $("loop").onclick = () => setLoop(!S.loop);
vid.addEventListener("play", () => { $("play").innerHTML = PAUSE; $("play").setAttribute("aria-label", "Pause"); });
vid.addEventListener("pause", () => { $("play").innerHTML = PLAY; $("play").setAttribute("aria-label", "Play"); });
window.addEventListener("keydown", (e) => {
  if ($("matchView").hidden || ["INPUT", "TEXTAREA"].includes(e.target.tagName)) return;
  if (e.code === "Space") { e.preventDefault(); toggle(); }
  else if (e.key === "ArrowRight") { e.preventDefault(); e.shiftKey ? stepK(1) : stepF(1); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); e.shiftKey ? stepK(-1) : stepF(-1); }
  else if (e.key === "b") setLines(!S.lines); else if (e.key === "l") setLoop(!S.loop);
});

function sample(times, frames, t) {
  const n = times.length;
  if (!n || t < times[0] || t > times[n - 1]) return null;
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (times[m] <= t) lo = m; else hi = m; }
  const sp = times[hi] - times[lo], u = sp > 0 ? (t - times[lo]) / sp : 0;
  return frames[lo].map((p, j) => [p[0] + (frames[hi][j][0] - p[0]) * u, p[1] + (frames[hi][j][1] - p[1]) * u]);
}
function fit(canvas) {
  const ctx = canvas.getContext("2d"), dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}
function drawLines(ctx, pts, map, color) {
  ctx.save(); ctx.lineCap = "round"; ctx.lineWidth = 3; ctx.strokeStyle = color; ctx.shadowColor = "rgba(0,0,0,.6)"; ctx.shadowBlur = 4;
  for (const [a, b] of EDGES) { const p = map(pts[a]), q = map(pts[b]); ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke(); }
  ctx.fillStyle = "#fff"; for (let i = 11; i < 29; i++) { const p = map(pts[i]); ctx.beginPath(); ctx.arc(p[0], p[1], 3, 0, 6.283); ctx.fill(); }
  ctx.restore();
}
function contentRect(v, w, h) {
  const vw = v.videoWidth || 16, vh = v.videoHeight || 9, s = Math.min(w / vw, h / vh);
  return { x: (w - vw * s) / 2, y: (h - vh * s) / 2, w: vw * s, h: vh * s };
}
const phaseTimes = (s) => { const t = s.track.times, p = s.track.phases; return { prep_start: t[p.prep_start], backswing_end: t[p.backswing_end], contact: t[p.contact], follow_end: t[p.follow_end] }; };
const phaseName = (t, p) => t < p.prep_start ? "Ready" : Math.abs(t - p.contact) < 0.05 ? "Contact" : t < p.contact ? "Backswing" : t <= p.follow_end ? "Follow-through" : "Recovery";

function tick() {
  if (!$("matchView").hidden) {
    const t = vid.currentTime;
    if (!vid.paused) {
      if (S.loop && S.sel && t > S.sel.end + 0.3) vid.currentTime = Math.max(S.sel.start - 0.3, 0);
      else if (S.mode === "rallies" && !S.rallies.some((r) => t >= r.start - 0.05 && t < r.end)) {
        const nx = S.rallies.find((r) => r.start > t); if (nx) vid.currentTime = nx.start; else vid.pause();
      }
    }
    $("clock").textContent = `${fmt(t)} / ${fmt(S.duration)}`;
    const hf = $("headF"), hr = $("headR");
    if (hf) hf.style.left = pct(t, S.duration || 1);
    const v = toVirtual(t); if (hr) { hr.hidden = v === null; if (v !== null) hr.style.left = pct(v, S.TOTAL); }
    const rally = S.rallies.find((r) => t >= r.start && t <= r.end);
    $("rallyTag").hidden = !rally; if (rally) $("rallyTag").textContent = `Rally ${rally.index + 1}`;
    document.querySelectorAll(".rcard").forEach((b) => b.classList.toggle("on", !!rally && +b.dataset.i === rally.index));
    const s = S.strokes.find((x) => t >= x.start - 0.15 && t <= x.end + 0.15);
    $("lt").classList.toggle("on", !!s);
    if (s) {
      $("ltA").textContent = phaseName(t, phaseTimes(s)); $("ltB").textContent = label(s.stroke_type);
      $("ltC").textContent = `Rally ${s.rally_index + 1} · ${s.metrics.racket_speed_peak.toFixed(1)} m/s racket speed`;
    }
    const { ctx, w, h } = fit(cv);
    if (S.lines && s) {
      const pts = sample(s.track.times, s.track.image, t), r = contentRect(vid, w, h);
      if (pts) drawLines(ctx, pts, ([x, y]) => [r.x + x * r.w, r.y + y * r.h], css("--ball"));
    }
  } else if (!$("compareView").hidden) cmpTick();
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- references
let REFS = null;
async function loadRefs() {
  if (REFS) return REFS;
  REFS = [];
  try {
    const raw = await (await fetch("sample/references.json")).json();
    raw.forEach((r, i) => addRef(r, { id: `sample${i}`, src: new URL("sample/match.mp4", location.href).href }));
  } catch { /* offline: only saved references */ }
  savedRefs().forEach((r, i) => addRef(r, { id: `saved${i}`, src: r.src === S.name ? S.url : null, saved: true }));
  return REFS;
}
function addRef(r, extra) {
  const seq = { times: r.times, world: r.world, image: r.image, valid: r.times.map(() => true) };
  const found = analyzeStrokes(seq, [r.contact], { smooth: 1, handedness: r.handedness || "right" });
  if (!found.length) return;
  const st = found.reduce((a, b) => (Math.abs(b.time - r.contact) < Math.abs(a.time - r.contact) ? b : a));
  const [type] = classify(st.metrics, {});
  REFS.push({ ...st, stroke_type: r.stroke_type || type, player: r.player, box: boxOf(st.track.image), ...extra });
}
function savedRefs() { try { return JSON.parse(localStorage.getItem("tt-refs") || "[]"); } catch { return []; } }
function saveReference(s) {
  const name = (window.prompt?.("Whose stroke is this? (e.g. Coach, Me at my best)") ?? null) || "Saved reference";
  const r = { player: name, stroke_type: s.stroke_type, contact: s.time, handedness: s.handedness, src: S.name,
    times: s.track.times, world: s.track.world.map((f) => f.map((p) => p.map((v) => +v.toFixed(3)))),
    image: s.track.image.map((f) => f.map((p) => p.map((v) => +v.toFixed(3)))) };
  try {
    const all = savedRefs(); all.push(r); localStorage.setItem("tt-refs", JSON.stringify(all.slice(-20)));
    REFS = null; $("saveRef").textContent = "Saved ✓";
  } catch { $("saveRef").textContent = "Couldn't save (browser storage is full or blocked)"; }
}

// ---------------------------------------------------------------- compare
const cmp = {};
$("back").onclick = () => { cmp.uv?.pause(); show("matchView"); };
async function openCompare(s) {
  vid.pause(); cmp.s = s; show("compareView"); window.scrollTo(0, 0);
  $("cmpTitle").textContent = `${label(s.stroke_type)} · rally ${s.rally_index + 1}`;
  $("refs").innerHTML = `<p class="fine">Loading references…</p>`;
  const refs = await loadRefs();
  const ranked = refs.map((p) => ({ p, m: compareStrokes(s, p) })).sort((a, b) => b.m.similarity - a.m.similarity).slice(0, 8);
  if (!ranked.length) { $("refs").innerHTML = `<p class="fine">No references yet. Save a stroke as a reference first.</p>`; $("cmpMain").innerHTML = ""; return; }
  const same = ranked.filter((x) => x.p.stroke_type === s.stroke_type);
  cmp.pick = same[0] || ranked[0];
  const paint = () => {
    $("refs").innerHTML = ranked.map(({ p, m }) => `<button class="ref ${p === cmp.pick.p ? "on" : ""}"><span class="mini"></span>
      <span><b>${label(p.stroke_type)}</b><small class="num">${escapeHtml(p.player)} · ${Math.round(m.similarity * 100)}% similar</small>
      <div class="match-bar"><i style="width:${Math.round(m.similarity * 100)}%"></i></div></span></button>`).join("");
    $("refs").querySelectorAll(".ref").forEach((b, i) => {
      const { p } = ranked[i];
      if (p.src) setBg(b.querySelector(".mini"), grab(p.src, p.time, boxCrop(p.box, 1, 1.25), 160));
      b.onclick = () => { cmp.pick = ranked[i]; paint(); renderCompare(); };
    });
  };
  paint(); renderCompare();
}
function renderCompare() {
  const s = cmp.s, { p: ref, m } = cmp.pick, u = phaseTimes(s), rp = phaseTimes(ref);
  Object.assign(cmp, { u, rp, uStart: u.prep_start - 0.3, uEnd: u.follow_end + 0.3, ref });
  const sim = Math.round(m.similarity * 100), C = 2 * Math.PI * 54;
  const rows = Object.values(m.metric_diffs).filter((d) => Math.abs(d.user) > 0.5 || Math.abs(d.pro) > 0.5)
    .sort((a, b) => Number(a.within_tolerance) - Number(b.within_tolerance));
  $("cmpMain").innerHTML = `
    <div class="pair">
      <div class="pane"><canvas id="uc" class="view"></canvas><span class="tag" style="background:var(--you)">You</span><span class="phase-now" id="uPh">Ready</span></div>
      <div class="pane"><canvas id="rc" class="view"></canvas><span class="tag" style="background:var(--ref)">${escapeHtml(ref.player)}</span><span class="phase-now" id="rPh">Ready</span></div>
      <video id="uv" muted playsinline preload="auto" hidden></video><video id="rv" muted playsinline preload="auto" hidden></video>
    </div>
    <div class="controls">
      <button class="btn primary" id="cPlay" aria-label="Play">${PLAY}</button>
      <button class="btn" data-j="backswing_end">Backswing</button><button class="btn" data-j="contact">Contact</button><button class="btn" data-j="follow_end">Follow-through</button>
      <span class="spacer"></span>
      <div class="seg" id="cSpeeds">${[0.25, 0.5, 1].map((r) => `<button data-r="${r}" class="${r === 0.5 ? "on" : ""}">${r}×</button>`).join("")}</div>
    </div>
    <input type="range" id="scrub" min="${cmp.uStart}" max="${cmp.uEnd}" step="0.001" value="${u.prep_start}" aria-label="Scrub through the stroke">
    <p class="fine">Both videos move in step phase by phase: your backswing lines up with theirs, contact with contact.</p>
    <div class="card score">
      <svg class="ring" viewBox="0 0 132 132" role="img" aria-label="${sim}% similar">
        <circle cx="66" cy="66" r="54" fill="none" stroke="${css("--line")}" stroke-width="12"/>
        <circle cx="66" cy="66" r="54" fill="none" stroke="${css("--ref")}" stroke-width="12" stroke-linecap="round"
          stroke-dasharray="${(C * sim / 100).toFixed(1)} ${C.toFixed(1)}" transform="rotate(-90 66 66)"/>
        <text x="66" y="74" text-anchor="middle" font-size="34">${sim}%</text></svg>
      <div><div class="eyebrow">How similar the movement is</div>
        <p class="fine" style="margin-top:4px">Whole-body movement compared frame by frame after lining the two strokes up in time.</p>
        ${m.cues.length ? `<ul class="cues">${m.cues.slice(0, 3).map((c) => `<li><i>!</i><span>${c.replace(" vs ", " vs them ")}</span></li>`).join("")}</ul>`
          : `<p style="margin:8px 0 0;color:var(--good);font-weight:600">Everything is close to this reference.</p>`}</div>
    </div>
    <div class="card"><div class="eyebrow" style="margin-bottom:6px">Side by side</div>
      <div class="metrics">${rows.map(metricRow).join("")}</div></div>`;
  cmp.uv = $("uv"); cmp.rv = $("rv");
  cmp.uv.src = S.url; if (ref.src) cmp.rv.src = ref.src;
  const init = () => { cmp.uv.currentTime = u.prep_start; };
  cmp.uv.readyState >= 1 ? init() : cmp.uv.addEventListener("loadedmetadata", init, { once: true });
  cmp.uv.playbackRate = 0.5; cmp.rate = 0.5;
  cmp.uv.addEventListener("play", () => { $("cPlay").innerHTML = PAUSE; });
  cmp.uv.addEventListener("pause", () => { $("cPlay").innerHTML = PLAY; });
  $("cPlay").onclick = () => cmp.uv.paused ? cmp.uv.play().catch(() => {}) : cmp.uv.pause();
  document.querySelectorAll("[data-j]").forEach((b) => b.onclick = () => { cmp.uv.pause(); cmp.uv.currentTime = u[b.dataset.j]; });
  $("cSpeeds").querySelectorAll("button").forEach((b) => b.onclick = () => {
    cmp.uv.playbackRate = +b.dataset.r; $("cSpeeds").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
  });
  $("scrub").oninput = (e) => { cmp.uv.pause(); cmp.uv.currentTime = +e.target.value; };
}
function metricRow(d) {
  const span = Math.max(Math.abs(d.user), Math.abs(d.pro), 1e-6) * 1.25;
  const lo = Math.min(0, d.user, d.pro) - span * 0.05, hi = Math.max(d.user, d.pro) + span * 0.15;
  const x = (v) => `${((v - lo) / (hi - lo) * 100).toFixed(1)}%`;
  const f = (v) => d.unit === "°" ? `${Math.round(v)}°` : `${v.toFixed(2)} ${d.unit}`;
  const a = Math.min(d.user, d.pro), b = Math.max(d.user, d.pro);
  return `<div class="mrow"><div class="lbl"><b>${d.label}</b><small>you ${f(d.user)} · them ${f(d.pro)}</small></div>
    <div class="scale"><div class="rail"></div>
      <div class="gap" style="left:${x(a)};width:calc(${x(b)} - ${x(a)});background:${d.within_tolerance ? "rgba(79,214,138,.35)" : "rgba(255,176,32,.4)"}"></div>
      <div class="dot" style="left:${x(d.pro)};background:var(--ref)"></div><div class="dot" style="left:${x(d.user)};background:var(--you)"></div></div>
    <div class="verdict ${d.within_tolerance ? "good" : "work"}">${d.within_tolerance ? "Close" : "Work on"}</div></div>`;
}
const ANCH = ["prep_start", "backswing_end", "contact", "follow_end"];
function mapPhase(t, a, b) {
  const xa = ANCH.map((k) => a[k]), xb = ANCH.map((k) => b[k]);
  if (t <= xa[0]) return xb[0] + (t - xa[0]);
  if (t >= xa[3]) return xb[3] + (t - xa[3]);
  for (let i = 0; i < 3; i++) if (t <= xa[i + 1]) { const sp = xa[i + 1] - xa[i], u = sp > 1e-6 ? (t - xa[i]) / sp : 1; return xb[i] + u * (xb[i + 1] - xb[i]); }
  return xb[3];
}
function cmpTick() {
  const { uv, rv } = cmp; if (!uv) return;
  if (!uv.paused && uv.currentTime >= cmp.uEnd) uv.currentTime = cmp.uStart;
  const t = uv.currentTime, tr = mapPhase(t, cmp.u, cmp.rp);
  if (cmp.ref.src && Math.abs(rv.currentTime - tr) > 0.02) rv.currentTime = tr;
  $("scrub").value = Math.min(Math.max(t, cmp.uStart), cmp.uEnd);
  $("uPh").textContent = phaseName(t, cmp.u); $("rPh").textContent = phaseName(tr, cmp.rp);
  paintPane($("uc"), uv, cmp.s.box);
  if (cmp.ref.src) paintPane($("rc"), rv, cmp.ref.box);
  else {        // a saved reference from another video: show its body lines
    const { ctx, w, h } = fit($("rc")); ctx.fillStyle = "#0e1524"; ctx.fillRect(0, 0, w, h);
    const pts = sample(cmp.ref.track.times, cmp.ref.track.image, tr), [x0, y0, cw, ch] = boxCrop(cmp.ref.box, 4 / 5, 1.15);
    if (pts) drawLines(ctx, pts, ([x, y]) => [(x - x0) / cw * w, (y - y0) / ch * h], css("--ref"));
  }
}
function paintPane(canvas, video, box) {
  if (video.readyState < 2) return;
  const { ctx, w, h } = fit(canvas), vw = video.videoWidth, vh = video.videoHeight;
  const crop = boxCrop(box, 4 / 5, 1.15, vw, vh);
  ctx.drawImage(video, crop[0] * vw, crop[1] * vh, crop[2] * vw, crop[3] * vh, 0, 0, w, h);
}

// ---------------------------------------------------------------- exports
function downloadBlob(blob, name) {
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
$("exportJson").onclick = () => {
  const data = { video: S.name, duration: S.duration, rallies: S.rallies.map((r) => ({ ...r, strokes: r.strokes.map(({ track, ...rest }) => rest) })) };
  downloadBlob(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }), `${S.name.replace(/\.[^.]+$/, "")}-analysis.json`);
};
$("exportVideo").onclick = async () => {
  const btn = $("exportVideo"), note = $("exportNote");
  const v = document.createElement("video");
  v.src = S.url; v.playsInline = true; v.style.cssText = "position:fixed;left:0;bottom:0;width:2px;height:2px;opacity:0.01";
  document.body.appendChild(v);
  const stream = (v.captureStream || v.mozCaptureStream)?.call(v);
  if (!stream || !window.MediaRecorder) { note.hidden = false; note.textContent = "This browser can't record video. Try Chrome or Edge."; v.remove(); return; }
  btn.disabled = true; note.hidden = false;
  const chunks = [], rec = new MediaRecorder(stream, { mimeType: MediaRecorder.isTypeSupported("video/webm;codecs=vp9,opus") ? "video/webm;codecs=vp9,opus" : "video/webm" });
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  const total = S.rallies.reduce((a, r) => a + r.duration, 0); let done = 0;
  for (const r of S.rallies) {
    await new Promise((res) => { v.addEventListener("seeked", res, { once: true }); v.currentTime = r.start; });
    if (rec.state === "inactive") rec.start(1000); else rec.resume();
    await v.play();
    await new Promise((res) => {
      const t = setInterval(() => {
        note.textContent = `Recording the rally-only video… ${Math.round(((done + v.currentTime - r.start) / total) * 100)}% (keep this tab open)`;
        if (v.currentTime >= r.end || v.ended) { clearInterval(t); res(); }
      }, 100);
    });
    v.pause(); rec.pause(); done += r.duration;
  }
  await new Promise((res) => { rec.onstop = res; rec.stop(); });
  v.remove();
  downloadBlob(new Blob(chunks, { type: "video/webm" }), `${S.name.replace(/\.[^.]+$/, "")}-rallies.webm`);
  btn.disabled = false; note.textContent = "Done. The rally-only video is in your downloads.";
};

function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
