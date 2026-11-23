// TT Analyzer browser engine: the Python pipeline (ml_pipeline/ttvision) ported to
// JavaScript so a match can be analysed entirely on the viewer's computer.
//
//   detectHits(samples, sr)        ball-contact clicks in the 2.5-4 kHz band
//   findRallies(hits, duration)    hits -> rally clips (1 s before the toss -> point over)
//   analyzeStrokes(seq, hints)     pose sequence -> strokes with phases, angles, metrics
//   classify(metrics, ctx)         rule-based stroke label
//   compareStrokes(a, b)           DTW + cosine similarity, metric diffs and coaching cues

// ------------------------------------------------------------------ audio
export const AUDIO = {
  sr: 22050, hop: 128, win: 512,
  bandLow: 2500, bandHigh: 4000,
  minBandRatio: 0.18, minRise: 1.0, minLevelDb: 9, minGap: 0.08,
};

function biquadBandpass(x, sr, f0, q) {
  // RBJ cookbook band-pass (0 dB peak gain), applied in place on a copy
  const w0 = 2 * Math.PI * f0 / sr, alpha = Math.sin(w0) / (2 * q), c = Math.cos(w0);
  const a0 = 1 + alpha;
  const b0 = alpha / a0, b2 = -alpha / a0, a1 = -2 * c / a0, a2 = (1 - alpha) / a0;
  const y = new Float32Array(x.length);
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < x.length; i++) {
    const v = b0 * x[i] + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1; x1 = x[i]; y2 = y1; y1 = v; y[i] = v;
  }
  return y;
}
function highpass(x, sr, fc) {
  const w0 = 2 * Math.PI * fc / sr, alpha = Math.sin(w0) / (2 * 0.707), c = Math.cos(w0), a0 = 1 + alpha;
  const b0 = (1 + c) / 2 / a0, b1 = -(1 + c) / a0, b2 = (1 + c) / 2 / a0, a1 = -2 * c / a0, a2 = (1 - alpha) / a0;
  const y = new Float32Array(x.length);
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < x.length; i++) {
    const v = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1; x1 = x[i]; y2 = y1; y1 = v; y[i] = v;
  }
  return y;
}
function frameEnergy(x, hop, win) {
  const n = Math.max(0, Math.floor((x.length - win) / hop) + 1);
  const sq = new Float64Array(x.length + 1);
  for (let i = 0; i < x.length; i++) sq[i + 1] = sq[i] + x[i] * x[i];
  const e = new Float64Array(n);
  for (let f = 0; f < n; f++) e[f] = sq[f * hop + win] - sq[f * hop];
  return e;
}
function rollingMeanStd(x, half) {
  const n = x.length, s1 = new Float64Array(n + 1), s2 = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) { s1[i + 1] = s1[i] + x[i]; s2[i + 1] = s2[i] + x[i] * x[i]; }
  const mean = new Float64Array(n), std = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, i - half), b = Math.min(n, i + half + 1), k = b - a;
    const m = (s1[b] - s1[a]) / k;
    mean[i] = m; std[i] = Math.sqrt(Math.max(0, (s2[b] - s2[a]) / k - m * m));
  }
  return { mean, std };
}

/** Ball-contact clicks: [{time, strength, bandRatio}] */
export function detectHits(samples, sr = AUDIO.sr, cfg = AUDIO) {
  if (samples.length < cfg.win * 4) return [];
  const f0 = Math.sqrt(cfg.bandLow * cfg.bandHigh), q = f0 / (cfg.bandHigh - cfg.bandLow);
  // two cascaded band-passes give a steeper band edge
  const band = frameEnergy(biquadBandpass(biquadBandpass(samples, sr, f0, q), sr, f0, q), cfg.hop, cfg.win);
  const total = frameEnergy(highpass(samples, sr, 80), cfg.hop, cfg.win);
  const n = band.length, EPS = 1e-10;
  const logE = new Float64Array(n);
  for (let i = 0; i < n; i++) logE[i] = Math.log(band[i] + EPS);
  const onset = new Float64Array(n);
  for (let i = 2; i < n; i++) onset[i] = Math.max(0, logE[i] - Math.max(logE[i - 1], logE[i - 2]));
  const fps = sr / cfg.hop;
  const th = rollingMeanStd(onset, Math.round(0.3 * fps));
  const bg = rollingMeanStd(logE, Math.round(0.75 * fps));
  const minGap = Math.max(1, Math.round(cfg.minGap * fps));
  const hits = [];
  let last = -1e9;
  for (let i = 1; i < n - 1; i++) {
    const thr = th.mean[i] + 4 * Math.max(th.std[i], 0.05) + cfg.minRise;
    if (onset[i] <= thr || onset[i] < onset[i - 1] || onset[i] < onset[i + 1]) continue;
    let ratio = 0, level = -1e9;
    for (let k = i; k < Math.min(n, i + 3); k++) {
      ratio = Math.max(ratio, band[k] / (total[k] + EPS));
      level = Math.max(level, (logE[k] - bg.mean[i]) * 10 / Math.LN10);
    }
    if (ratio < cfg.minBandRatio || level < cfg.minLevelDb) continue;
    const time = (i * cfg.hop + cfg.win / 2) / sr, strength = onset[i] - thr;
    if (i - last < minGap) {
      if (hits.length && strength > hits[hits.length - 1].strength) { hits[hits.length - 1] = { time, strength, bandRatio: ratio }; last = i; }
      continue;
    }
    hits.push({ time, strength, bandRatio: ratio });
    last = i;
  }
  return hits;
}

/** Decode a video/audio File to mono Float32 at sr. */
export async function decodeAudio(file, sr = AUDIO.sr) {
  const buf = await file.arrayBuffer();
  const ctx = new OfflineAudioContext(1, 1, sr);
  const audio = await ctx.decodeAudioData(buf);
  if (audio.numberOfChannels === 1) return audio.getChannelData(0);
  const out = new Float32Array(audio.length);
  for (let c = 0; c < audio.numberOfChannels; c++) {
    const d = audio.getChannelData(c);
    for (let i = 0; i < d.length; i++) out[i] += d[i] / audio.numberOfChannels;
  }
  return out;
}

// ------------------------------------------------------------------ rallies
export const RALLY = { preRoll: 1.0, postRoll: 0.6, minHits: 3, maxGap: 1.6, maxRally: 60, toss: 0.7,
  dribbleMax: 0.35, dribbleDecay: 0.92, minConfidence: 0.35 };

export function findRallies(hits, duration, cfg = RALLY) {
  const times = hits.map((h) => h.time).sort((a, b) => a - b);
  const groups = [];
  let g = [];
  for (const t of times) {
    if (g.length && (t - g[g.length - 1] > cfg.maxGap || t - g[0] > cfg.maxRally)) { groups.push(g); g = []; }
    g.push(t);
  }
  if (g.length) groups.push(g);

  const rallies = [];
  for (let grp of groups) {
    // the server bouncing the ball before serving: >= 3 steady, slow intervals at the start
    const iv = grp.slice(1).map((t, i) => t - grp[i]);
    let run = 0;
    while (run + 1 < iv.length && iv[run] >= 0.4 && Math.abs(iv[run + 1] - iv[run]) <= 0.12 * iv[run]) run++;
    if (run >= 2) grp = grp.slice(run + 2);
    // a dead ball dribbling on the table: shortening intervals at the end
    const iv2 = grp.slice(1).map((t, i) => t - grp[i]);
    let k = iv2.length - 1, decay = 0;
    while (k >= 1 && iv2[k] < cfg.dribbleMax && iv2[k] < iv2[k - 1] * cfg.dribbleDecay) { decay++; k--; }
    const dribbled = decay >= 2;
    if (dribbled) grp = grp.slice(0, k + 2);
    if (grp.length < cfg.minHits) continue;

    const contact = grp[0], lastHit = grp[grp.length - 1];
    const iv3 = grp.slice(1).map((t, i) => t - grp[i]);
    const inGroup = hits.filter((h) => h.time >= contact - 1e-3 && h.time <= lastHit + 1e-3);
    const parts = [
      Math.min(1, (grp.length - 1) / 4),
      iv3.length ? iv3.filter((x) => x >= 0.1 && x <= 1.2).length / iv3.length : 0,
      Math.min(1, (inGroup.reduce((a, h) => a + h.bandRatio, 0) / Math.max(inGroup.length, 1)) / (2 * AUDIO.minBandRatio)),
    ];
    const confidence = (parts[0] + parts[1] + parts[2] * 0.5) / 2.5;
    if (confidence < cfg.minConfidence) continue;
    const toss = Math.max(0, contact - cfg.toss);
    rallies.push({
      start: Math.max(0, toss - cfg.preRoll), end: Math.min(duration || Infinity, lastHit + 0.3 + cfg.postRoll),
      serve_toss_time: toss, serve_contact_time: contact, last_hit_time: lastHit,
      end_reason: dribbled ? "dribble" : "audio_gap", confidence, hit_times: grp, strokes: [],
    });
  }
  rallies.sort((a, b) => a.start - b.start);
  for (let i = 1; i < rallies.length; i++) {
    const p = rallies[i - 1], r = rallies[i];
    if (r.start < p.end) { p.end = Math.max(p.last_hit_time, r.start); if (r.start < p.end) r.start = p.end; }
  }
  rallies.forEach((r, i) => { r.index = i; r.duration = r.end - r.start; });
  return rallies;
}

// ------------------------------------------------------------------ vectors
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);
const unit = (a) => { const n = norm(a) || 1e-9; return [a[0] / n, a[1] / n, a[2] / n]; };
const deg = (r) => r * 180 / Math.PI;
const angleAt = (a, b, c) => deg(Math.acos(Math.max(-1, Math.min(1, dot(unit(sub(a, b)), unit(sub(c, b)))))));
const UP = [0, -1, 0];

// MediaPipe BlazePose indices
const NOSE = 0, LS = 11, RS = 12, LE = 13, RE = 14, LW = 15, RW = 16, LI = 19, RI = 20,
  LH = 23, RH = 24, LK = 25, RK = 26, LA = 27, RA = 28;
const armFor = (hand) => hand === "left"
  ? { sh: LS, el: LE, wr: LW, ix: LI, hip: LH, side: "left" }
  : { sh: RS, el: RE, wr: RW, ix: RI, hip: RH, side: "right" };

function gradient(series, times) {
  const n = series.length, out = new Array(n);
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1);
    const dt = times[b] - times[a] || 1e-6;
    out[i] = mul(sub(series[b], series[a]), 1 / dt);
  }
  return out;
}

function unwrapDeg(arr) {
  const out = [arr[0]];
  for (let i = 1; i < arr.length; i++) {
    let d = arr[i] - arr[i - 1];
    while (d > 180) d -= 360;
    while (d < -180) d += 360;
    out.push(out[i - 1] + d);
  }
  return out;
}

const SEGMENTS = [
  [[NOSE], 0.081], [[LS, RS, LH, RH], 0.497], [[LS, LE], 0.028], [[RS, RE], 0.028], [[LE, LW], 0.022], [[RE, RW], 0.022],
  [[LH, LK], 0.1], [[RH, RK], 0.1], [[LK, LA], 0.061], [[RK, RA], 0.061],
];
function com(w) {
  let c = [0, 0, 0];
  for (const [idx, m] of SEGMENTS) {
    let p = [0, 0, 0];
    idx.forEach((i) => { p = add(p, w[i]); });
    c = add(c, mul(p, m / idx.length));
  }
  return c;
}

/** Clean a pose sequence: fill gaps, smooth. seq = {times, world[T][33][3], image[T][33][3], valid[T]} */
export function cleanSequence(seq, smoothWin = 5) {
  const T = seq.times.length;
  const valid = seq.valid.map((v, i) => v && [LS, RS, LH, RH].reduce((a, j) => a + (seq.image[i][j][2] ?? 1), 0) / 4 >= 0.3);
  const idx = valid.map((v, i) => (v ? i : -1)).filter((i) => i >= 0);
  if (!idx.length) return null;
  const world = [];
  for (let i = 0; i < T; i++) {
    if (valid[i]) { world.push(seq.world[i].map((p) => p.slice())); continue; }
    let a = -1, b = -1;
    for (const j of idx) { if (j < i) a = j; else { b = j; break; } }
    if (a < 0) a = b; if (b < 0) b = a;
    const u = a === b ? 0 : (i - a) / (b - a);
    world.push(seq.world[a].map((p, j) => add(p, mul(sub(seq.world[b][j], p), u))));
  }
  if (smoothWin <= 1) return world;
  const half = smoothWin >> 1;
  return world.map((_, i) => world[0].map((__, j) => {
    let s = [0, 0, 0], k = 0;
    for (let d = -half; d <= half; d++) { const t = Math.min(T - 1, Math.max(0, i + d)); s = add(s, world[t][j]); k++; }
    return mul(s, 1 / k);
  }));
}

function racketTrack(world, arm, ext = 0.18) {
  return world.map((w) => {
    const fa = unit(sub(w[arm.wr], w[arm.el])), hand = unit(sub(w[arm.ix], w[arm.wr]));
    return add(w[arm.wr], mul(unit(add(mul(fa, 0.5), mul(hand, 0.5))), ext));
  });
}

function bodyFrames(world, arm) {
  const other = arm.hip === RH ? LH : RH;
  const lat = world.map((w) => { const v = sub(w[arm.hip], w[other]); v[1] = 0; return unit(v); });
  let fwd = lat.map((l) => unit(cross(UP, l)));
  const facing = world.reduce((a, w, i) => a + dot(sub(w[NOSE], mul(add(w[LS], w[RS]), 0.5)), fwd[i]), 0);
  if (facing < 0) fwd = fwd.map((f) => mul(f, -1));
  return { fwd, lat };
}

export function jointAngles(world, arm) {
  const other = armFor(arm.side === "right" ? "left" : "right");
  const yaw = (a, b) => unwrapDeg(world.map((w) => { const v = sub(w[b], w[a]); return deg(Math.atan2(v[2], v[0])); }));
  const hipYaw = yaw(LH, RH), shYaw = yaw(LS, RS);
  const res = { elbow: [], shoulder_abduction: [], free_elbow: [], knee_flexion: [], hip_rotation: [],
    shoulder_rotation: [], x_factor: [], trunk_lean: [] };
  world.forEach((w, i) => {
    res.elbow.push(angleAt(w[arm.sh], w[arm.el], w[arm.wr]));
    res.shoulder_abduction.push(angleAt(w[arm.hip], w[arm.sh], w[arm.el]));
    res.free_elbow.push(angleAt(w[other.sh], w[other.el], w[other.wr]));
    const kl = angleAt(w[LH], w[LK], w[LA]), kr = angleAt(w[RH], w[RK], w[RA]);
    res.knee_flexion.push(180 - (kl + kr) / 2);
    res.hip_rotation.push(hipYaw[i] - hipYaw[0]);
    res.shoulder_rotation.push(shYaw[i] - shYaw[0]);
    res.x_factor.push(shYaw[i] - hipYaw[i]);
    const spine = unit(sub(mul(add(w[LS], w[RS]), 0.5), mul(add(w[LH], w[RH]), 0.5)));
    res.trunk_lean.push(deg(Math.acos(Math.max(-1, Math.min(1, dot(spine, UP))))));
  });
  return res;
}

const idxAt = (times, t) => { let i = 0; while (i < times.length - 1 && times[i] < t) i++; return i; };

export function segmentPhases(times, speed, hint) {
  const n = times.length;
  let contact;
  if (hint != null) {
    const c0 = idxAt(times, hint), lo = idxAt(times, times[c0] - 0.08), hi = idxAt(times, times[c0] + 0.08);
    contact = lo; for (let i = lo; i <= hi; i++) if (speed[i] > speed[contact]) contact = i;
  } else { contact = 0; for (let i = 1; i < n; i++) if (speed[i] > speed[contact]) contact = i; }
  const peak = Math.max(speed[contact], 1e-6);
  const lo = idxAt(times, times[contact] - 0.6);
  let k = contact;
  while (k > lo && !(speed[k - 1] >= speed[k] && speed[k] < 0.5 * peak)) k--;
  const backswing = Math.min(k, Math.max(contact - 1, 0));
  const preLo = idxAt(times, times[backswing] - 0.8);
  let prep = preLo;
  if (backswing > preLo) {
    let pk = preLo; for (let i = preLo; i < backswing; i++) if (speed[i] > speed[pk]) pk = i;
    for (let i = pk - 1; i >= preLo; i--) if (speed[i] < 0.3 * speed[pk]) { prep = i; break; }
  }
  let follow = n - 1;
  for (let i = contact + 1; i < n; i++) if (speed[i] < 0.25 * peak) { follow = i; break; }
  follow = Math.max(contact, Math.min(follow, idxAt(times, times[contact] + 0.6)));
  return { prep_start: prep, backswing_end: backswing, contact, follow_end: follow };
}

function strokePeaks(times, speed, fwdSpeed, minSpeed = 1.5, sep = 0.35) {
  const peaks = [];
  for (let i = 1; i < speed.length - 1; i++)
    if (speed[i] >= speed[i - 1] && speed[i] > speed[i + 1] && speed[i] >= minSpeed && fwdSpeed[i] > 0.2 * speed[i]) peaks.push(i);
  peaks.sort((a, b) => speed[b] - speed[a]);
  const chosen = [];
  for (const p of peaks) if (chosen.every((q) => Math.abs(times[p] - times[q]) >= sep)) chosen.push(p);
  return chosen.sort((a, b) => a - b);
}

/** One analysed stroke from a cleaned world track. */
function analyzeWindow(seq, world, lo, hi, arm, contactTime) {
  const times = seq.times.slice(lo, hi), w = world.slice(lo, hi);
  const racket = racketTrack(w, arm), vel = gradient(racket, times), speed = vel.map(norm);
  const ph = segmentPhases(times, speed, contactTime);
  const ang = jointAngles(w, arm), { fwd, lat } = bodyFrames(w, arm);
  const rb = w.map((wf, i) => { const rel = sub(racket[i], com(wf)); return [dot(rel, fwd[i]), dot(rel, lat[i]), dot(rel, UP)]; });
  const c = ph.contact, b = ph.backswing_end, f = ph.follow_end, seg = [ph.prep_start, f + 1];
  const vf = dot(vel[c], fwd[c]), vl = dot(vel[c], lat[c]), vu = dot(vel[c], UP);
  const slice = (a) => a.slice(seg[0], seg[1]);
  const ptp = (a) => Math.max(...a) - Math.min(...a);
  const r3 = (x) => Math.round(x * 1000) / 1000;
  const metrics = {
    elbow_at_backswing: ang.elbow[b], elbow_at_contact: ang.elbow[c], elbow_at_follow_through: ang.elbow[f],
    knee_flexion_at_backswing: ang.knee_flexion[b], knee_flexion_at_contact: ang.knee_flexion[c],
    knee_flexion_max: Math.max(...slice(ang.knee_flexion)),
    hip_rotation_range: ptp(slice(ang.hip_rotation)), shoulder_rotation_range: ptp(slice(ang.shoulder_rotation)),
    x_factor_max: Math.max(...slice(ang.x_factor).map((x) => Math.abs(x - ang.x_factor[ph.prep_start]))),
    trunk_lean_at_contact: ang.trunk_lean[c], racket_speed_peak: speed[c],
    swing_angle_at_contact: deg(Math.atan2(vu, Math.max(Math.hypot(vf, vl), 1e-6))),
    swing_lateral_velocity: vl, swing_forward_velocity: vf,
    contact_forward_of_com: rb[c][0], contact_lateral_of_com: rb[c][1], contact_height_vs_com: rb[c][2],
    elbow_height_vs_shoulder: dot(sub(w[c][arm.el], w[c][arm.sh]), UP),
    backswing_duration: times[c] - times[ph.prep_start], follow_through_duration: times[f] - times[c],
    backswing_depth: rb[c][0] - rb[b][0],
  };
  Object.keys(metrics).forEach((k) => { metrics[k] = r3(metrics[k]); });
  const keyframes = [["preparation", b], ["contact", c], ["follow_through", f]].map(([phase, i]) => ({
    phase, time: times[i], frame_index: i,
    angles: Object.fromEntries(Object.entries(ang).map(([k, v]) => [k, Math.round(v[i] * 10) / 10])),
  }));
  return {
    time: times[c], start: times[ph.prep_start], end: times[f], handedness: arm.side, keyframes, metrics,
    track: { times, world: w, image: seq.image.slice(lo, hi), angles: ang, racket_body: rb, speed, phases: ph },
  };
}

/** Find and break down strokes in one pose sequence (e.g. one rally). */
export function analyzeStrokes(seq, hints = [], opts = {}) {
  if (seq.valid.filter(Boolean).length < 5) return [];
  const world = cleanSequence(seq, opts.smooth ?? 5);
  if (!world) return [];
  let hand = opts.handedness;
  if (hand !== "left" && hand !== "right") {
    const p95 = (h) => { const s = gradient(racketTrack(world, armFor(h)), seq.times).map(norm).sort((a, b) => a - b); return s[Math.floor(s.length * 0.95)] || 0; };
    hand = p95("right") >= p95("left") ? "right" : "left";
  }
  const arm = armFor(hand);
  const racket = racketTrack(world, arm), vel = gradient(racket, seq.times), speed = vel.map(norm);
  const { fwd } = bodyFrames(world, arm);
  let peaks = strokePeaks(seq.times, speed, vel.map((v, i) => dot(v, fwd[i])), opts.minSpeed ?? 1.5);
  if (hints.length) {
    const near = peaks.filter((p) => hints.some((h) => Math.abs(h - seq.times[p]) <= 0.12));
    if (near.length) peaks = near;
  }
  const out = [];
  for (const p of peaks) {
    const t = seq.times[p];
    const lo = seq.times.findIndex((x) => x >= t - 0.7);
    let hi = seq.times.findIndex((x) => x > t + 0.5); if (hi < 0) hi = seq.times.length;
    if (hi - lo >= 5) out.push(analyzeWindow(seq, world, lo, hi, arm, t));
  }
  return out;
}

// ------------------------------------------------------------------ classification
export function classify(m, ctx = {}) {
  const speed = m.racket_speed_peak, angle = m.swing_angle_at_contact, lateral = m.contact_lateral_of_com;
  const height = m.contact_height_vs_com, depth = m.backswing_depth, forehand = lateral >= -0.02;
  const conf = (margin, scale) => Math.max(0.35, Math.min(0.95, 0.5 + margin / scale));
  if (ctx.isServe) {
    if (!forehand) return ["serve_backhand", conf(-lateral, 0.2)];
    if (m.elbow_height_vs_shoulder > -0.12) return ["serve_tomahawk", conf(m.elbow_height_vs_shoulder + 0.12, 0.2)];
    return m.swing_lateral_velocity < 0 ? ["serve_pendulum", conf(-m.swing_lateral_velocity, 2)]
      : ["serve_reverse_pendulum", conf(m.swing_lateral_velocity, 2)];
  }
  if (ctx.isReceive) {
    if (angle > 15 && speed >= 3.5) return [depth < 0.25 ? "receive_short_flick" : "receive_long_flick", conf(angle - 15, 30)];
    if (angle <= 5 && speed < 5) return ["receive_push", conf(5 - angle, 30)];
  }
  if (height > 0.45 && angle < -10 && speed >= 7) return ["smash", conf(speed - 7, 4)];
  if (angle > 40 && height > 0.2) return ["lob", conf(angle - 40, 30)];
  if (angle < -10) return speed >= 4.5 ? ["chop", conf(-angle - 10, 30)]
    : [forehand ? "forehand_push" : "backhand_push", conf(4.5 - speed, 3)];
  if (speed < 3.5 && depth < 0.15) return ["block", conf(3.5 - speed, 2)];
  if (angle > 20 && speed >= 4.5) return [forehand ? "forehand_topspin" : "backhand_topspin", conf(angle - 20, 25)];
  if (speed >= 4.5) return ["counter_drive", conf(20 - Math.abs(angle - 5), 30)];
  if (angle > 10) return [forehand ? "forehand_topspin" : "backhand_topspin", 0.4];
  return ["block", 0.4];
}

export function strokeContext(stroke, rally, first) {
  const isServe = Math.abs(stroke.time - rally.serve_contact_time) <= 0.15;
  const early = rally.hit_times.filter((t) => t <= stroke.time + 0.12).length <= 4;
  return { isServe, isReceive: first && !isServe && early };
}

// ------------------------------------------------------------------ comparison
const CMP_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
const CMP_ANGLES = ["elbow", "shoulder_abduction", "knee_flexion", "x_factor", "trunk_lean"];
export const METRIC_CUES = {
  knee_flexion_at_backswing: ["Knee bend at backswing", "°", 6, "Bend your knees more as you load", "Stand a little taller in the backswing"],
  elbow_at_contact: ["Elbow angle at contact", "°", 8, "Open the elbow more at contact", "Keep the elbow more bent at contact"],
  elbow_at_backswing: ["Elbow angle at backswing", "°", 10, "Extend the arm more in the backswing", "Keep the arm more compact in the backswing"],
  hip_rotation_range: ["Hip rotation", "°", 8, "Turn the hips more", "Less hip rotation; stay balanced"],
  shoulder_rotation_range: ["Shoulder rotation", "°", 8, "Rotate the shoulders more", "Shorter shoulder turn"],
  x_factor_max: ["Trunk twist (shoulders vs hips)", "°", 6, "Coil the upper body more against the hips", "Less trunk twist"],
  trunk_lean_at_contact: ["Forward lean at contact", "°", 6, "Lean forward more into the ball", "Stay more upright"],
  swing_angle_at_contact: ["Racket path angle at contact", "°", 8, "Swing more upward through the ball", "Swing more forward, less upward"],
  racket_speed_peak: ["Racket speed (rel. to hips)", "m/s", 0.8, "Accelerate more into contact", "Control the swing speed"],
  contact_forward_of_com: ["Contact in front of body", "m", 0.06, "Meet the ball further in front", "Let the ball come closer"],
  knee_flexion_at_contact: ["Knee bend at contact", "°", 6, "Stay lower through contact", "Drive up a bit more through contact"],
};

function resample(rows, n) {
  if (rows.length === 1) return Array.from({ length: n }, () => rows[0].slice());
  return Array.from({ length: n }, (_, i) => {
    const x = i * (rows.length - 1) / (n - 1), a = Math.floor(x), b = Math.min(rows.length - 1, a + 1), u = x - a;
    return rows[a].map((v, j) => v + (rows[b][j] - v) * u);
  });
}
function phaseResample(rows, ph, L = 60) {
  const nPre = Math.max(2, Math.round(L * 0.6));
  return resample(rows.slice(ph.prep_start, ph.contact + 1), nPre)
    .concat(resample(rows.slice(ph.contact, ph.follow_end + 1), L - nPre + 1).slice(1));
}
function torso(world) {
  let s = 0;
  world.forEach((w) => { s += norm(sub(mul(add(w[LS], w[RS]), 0.5), mul(add(w[LH], w[RH]), 0.5))); });
  return s / world.length || 1;
}
function features(st) {
  const { world, angles, racket_body: rb, phases } = st.track;
  const arm = armFor(st.handedness), { fwd, lat } = bodyFrames(world, arm), t = torso(world);
  const rows = world.map((w, i) => {
    const c = com(w), row = [];
    CMP_JOINTS.forEach((j) => { const r = sub(w[j], c); row.push(dot(r, fwd[i]) / t, dot(r, lat[i]) / t, dot(r, UP) / t); });
    CMP_ANGLES.forEach((k) => row.push(angles[k][i] * Math.PI / 180));
    row.push(rb[i][0] / t, rb[i][1] / t, rb[i][2] / t);
    return row;
  });
  return phaseResample(rows, phases);
}
export function dtw(cost, band = 0.2) {
  const n = cost.length, m = cost[0].length, w = Math.max(Math.floor(band * Math.max(n, m)), Math.abs(n - m)) + 1;
  const acc = Array.from({ length: n + 1 }, () => new Float64Array(m + 1).fill(Infinity));
  acc[0][0] = 0;
  for (let i = 1; i <= n; i++) {
    const c = Math.round(i * m / n);
    for (let j = Math.max(1, c - w); j <= Math.min(m, c + w); j++)
      acc[i][j] = cost[i - 1][j - 1] + Math.min(acc[i - 1][j], acc[i][j - 1], acc[i - 1][j - 1]);
  }
  let i = n, j = m; const path = [[i - 1, j - 1]];
  while (i > 1 || j > 1) {
    const opts = [[acc[i - 1][j - 1], i - 1, j - 1], [acc[i - 1][j], i - 1, j], [acc[i][j - 1], i, j - 1]].filter((o) => o[1] >= 1 && o[2] >= 1);
    opts.sort((a, b) => a[0] - b[0]); [, i, j] = opts[0]; path.push([i - 1, j - 1]);
  }
  path.reverse();
  return { cost: path.reduce((a, [p, q]) => a + cost[p][q], 0) / path.length, path };
}
export function compareStrokes(user, ref) {
  const fu = features(user), fr = features(ref);
  const nrm = (r) => { const n = Math.hypot(...r) || 1e-9; return r.map((v) => v / n); };
  const nu = fu.map(nrm), nr = fr.map(nrm);
  const cost = nu.map((a) => nr.map((b) => 1 - a.reduce((s, v, k) => s + v * b[k], 0)));
  const { cost: c, path } = dtw(cost);
  const tu = torso(user.track.world), tr = torso(ref.track.world);
  const ru = phaseResample(user.track.racket_body, user.track.phases).map((r) => r.map((v) => v / tu));
  const rr = phaseResample(ref.track.racket_body, ref.track.phases).map((r) => r.map((v) => v / tr));
  const racketErr = path.reduce((a, [p, q]) => a + Math.hypot(ru[p][0] - rr[q][0], ru[p][1] - rr[q][1], ru[p][2] - rr[q][2]), 0) / path.length;
  const diffs = {}, ranked = [];
  for (const [k, [label, unit, tol, low, high]] of Object.entries(METRIC_CUES)) {
    const u = user.metrics[k], p = ref.metrics[k];
    if (u == null || p == null) continue;
    const delta = u - p;
    diffs[k] = { label, unit, user: u, pro: p, delta, within_tolerance: Math.abs(delta) <= tol };
    if (Math.abs(delta) > tol) {
      const f = (v) => unit === "°" ? `${v.toFixed(0)}°` : `${v.toFixed(2)} ${unit}`;
      ranked.push([Math.abs(delta) / tol, `${delta < 0 ? low : high} (${label.toLowerCase()}: ${f(u)} vs ${f(p)})`]);
    }
  }
  ranked.sort((a, b) => b[0] - a[0]);
  return { similarity: 1 - c, racket_path_error: racketErr, metric_diffs: diffs, cues: ranked.slice(0, 4).map((r) => r[1]) };
}
