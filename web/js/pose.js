// In-browser pose tracking with MediaPipe PoseLandmarker (WASM / WebGL).
// Steps a hidden copy of the video through each rally frame by frame and runs
// the model on each one, following the player nearest the camera.

const VERSION = "0.10.14";
const BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}`;
const MODEL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task";

let landmarker = null;
let lastTs = 0;

export async function loadPose() {
  if (landmarker) return landmarker;
  const { FilesetResolver, PoseLandmarker } = await import(`${BASE}/vision_bundle.mjs`);
  const files = await FilesetResolver.forVisionTasks(`${BASE}/wasm`);
  const make = (delegate) => PoseLandmarker.createFromOptions(files, {
    baseOptions: { modelAssetPath: MODEL, delegate },
    runningMode: "VIDEO", numPoses: 2,
    minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  try { landmarker = await make("GPU"); } catch { landmarker = await make("CPU"); }
  return landmarker;
}

function area(img) {
  let x0 = 1, y0 = 1, x1 = 0, y1 = 0, n = 0;
  for (const p of img) if (p[2] > 0.3) { n++; x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]); x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]); }
  return n < 4 ? 0 : (x1 - x0) * (y1 - y0);
}
const centre = (img) => [(img[23][0] + img[24][0]) / 2, (img[23][1] + img[24][1]) / 2];

function pickPlayer(people, prev) {
  if (!people.length) return null;
  if (prev) {
    const pc = centre(prev.image);
    const d = people.map((p) => Math.hypot(centre(p.image)[0] - pc[0], centre(p.image)[1] - pc[1]));
    const i = d.indexOf(Math.min(...d));
    if (d[i] < 0.12) return people[i];
  }
  return people.reduce((a, b) => (area(b.image) > area(a.image) ? b : a));     // nearest = biggest
}

function seek(video, t) {
  return new Promise((res) => {
    if (Math.abs(video.currentTime - t) < 1e-3 && video.readyState >= 2) return res();
    video.addEventListener("seeked", () => res(), { once: true });
    video.currentTime = t;
  });
}

/**
 * Track the near player through each [start, end] window.
 * Steps through the video frame by frame (seek, wait, detect), so a slow
 * computer takes longer but never skips frames.
 * Returns one sequence per window: {times, world, image, valid}.
 */
export async function trackWindows(video, windows, onProgress, { fps = 30, signal } = {}) {
  const lm = await loadPose();
  const total = windows.reduce((a, [s, e]) => a + (e - s), 0) || 1;
  let done = 0;
  const out = [];
  video.muted = true;
  video.pause();
  for (const [start, end] of windows) {
    if (signal?.aborted) break;
    const seq = { times: [], world: [], image: [], valid: [] };
    let prev = null;
    for (let t = start; t < end; t += 1 / fps) {
      if (signal?.aborted) break;
      await seek(video, t);
      lastTs = Math.max(lastTs + 1, Math.round(performance.now()));
      let res = null;
      try { res = lm.detectForVideo(video, lastTs); } catch { res = null; }
      const people = (res?.landmarks || []).map((l2, i) => ({
        image: l2.map((p) => [p.x, p.y, p.visibility ?? 1]),
        world: res.worldLandmarks[i].map((p) => [p.x, p.y, p.z]),
      }));
      const me = pickPlayer(people, prev);
      if (me) prev = me;
      seq.times.push(video.currentTime);
      seq.world.push(me ? me.world : Array.from({ length: 33 }, () => [0, 0, 0]));
      seq.image.push(me ? me.image : Array.from({ length: 33 }, () => [0, 0, 0]));
      seq.valid.push(!!me);
      onProgress?.(Math.min(1, (done + (t - start)) / total));
    }
    done += end - start;
    out.push(seq);
  }
  return out;
}
