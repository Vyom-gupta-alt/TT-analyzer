"""Build the data bundle for the browser preview (preview/index.html).

    python preview/build_preview.py OUT_DIR

Renders the animated sample match, runs the real rally detector on it, and
analyses both players' strokes. Pose comes from the animation itself (exact
joint positions), because the preview is about showing the app, not about
how well a pose model reads cartoon players.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml_pipeline"))

from ttvision.config import PoseConfig  # noqa: E402
from ttvision.matching.pro_matcher import ProMatcher, ProLibrary, ProStroke  # noqa: E402
from ttvision.pipeline import stroke_context  # noqa: E402
from ttvision.pose.analyzer import PoseAnalyzer, PoseSequence  # noqa: E402
from ttvision.rally.detector import RallyDetector  # noqa: E402
from ttvision.sample_match import W, H, Camera, plan_match, player_pose, render_sample_match  # noqa: E402
from ttvision.strokes.classifier import StrokeClassifier  # noqa: E402

KEEP = ["elbow", "knee_flexion", "hip_rotation", "shoulder_rotation", "x_factor", "trunk_lean"]


def rnd(x, n=3):
    if isinstance(x, float):
        return round(x, n)
    if isinstance(x, (list, tuple)):
        return [rnd(v, n) for v in x]
    if isinstance(x, dict):
        return {k: rnd(v, n) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return rnd(x.tolist(), n)
    if isinstance(x, np.floating):
        return round(float(x), n)
    return x


def track(who: str, times: np.ndarray, cam: Camera, rallies) -> PoseSequence:
    world, image = [], []
    for t in times:
        wcm, local = player_pose(rallies, who, float(t))
        uv = cam.project(wcm)
        world.append(local)
        image.append(np.column_stack([uv[:, 0] / W, uv[:, 1] / H, np.ones(33)]))
    return PoseSequence(times, np.asarray(world, np.float32), np.asarray(image, np.float32), np.ones(len(times), bool))


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    video = out / "match.mp4"
    if not video.exists():
        render_sample_match(video, progress=lambda f: print(f"\rrendering {f:5.0%}", end="", flush=True))
        print()
    plan = plan_match(7)
    cam = Camera()
    rallies = RallyDetector().detect(video)
    analyzer = PoseAnalyzer(PoseConfig(smoothing_window=1, handedness="right"))
    clf = StrokeClassifier()

    skeletons, tracks = {}, {}
    for r in rallies:
        times = np.arange(r.start, r.end, 1 / 60)
        seq = track("near", times, cam, plan)
        found = sorted(analyzer.analyze_sequence(seq, r.hit_times), key=lambda a: a.contact_time)
        for k, a in enumerate(found):
            label, conf = clf.classify(a, stroke_context(a, r, k == 0))
            ev = a.to_event(label, conf)
            ev.rally_index = r.index
            r.strokes.append(ev)
            key = f"{r.index}:{k}"
            skeletons[key] = {"times": a.times, "landmarks": a.image[:, :, :2], "phases": a.phases.as_dict()}
            tracks[key] = a

    # references: the yellow player's strokes, analysed the same way
    refs = []
    for r in rallies:
        times = np.arange(r.start, r.end, 1 / 60)
        seq = track("far", times, cam, plan)
        for a in analyzer.analyze_sequence(seq, r.hit_times):
            label, _ = clf.classify(a, stroke_context(a, r, False))
            refs.append(ProStroke(id=f"y{len(refs)}", player="Yellow player", stroke_type=label.value,
                                  context="sample", analysis=a))
    matcher = ProMatcher(ProLibrary(refs))
    matches = {}
    for key, a in tracks.items():
        for p in refs:
            m = matcher.compare(a, p).to_dict()
            m.pop("path", None)
            matches[f"{key}|{p.id}"] = {k: m[k] for k in ("similarity", "racket_path_error", "metric_diffs", "cues")}

    def box(a):
        pts = a.image[:, 11:33, :2].reshape(-1, 2)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        lo[1] -= 0.06 * (hi[1] - lo[1]) + 0.05      # room for the head
        return [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]

    def phase_times(a):
        t = a.times
        return {k: float(t[v]) for k, v in a.phases.as_dict().items()}

    data = {
        "video": {"filename": "Sample match", "duration": float(plan and max(e[0] for r in plan for e in r.events) + 3.0),
                  "fps": 30},
        "rallies": [r.to_dict() for r in rallies],
        "skeletons": skeletons,
        "userPhases": {k: phase_times(a) for k, a in tracks.items()},
        "userBoxes": {k: box(a) for k, a in tracks.items()},
        "refs": [{"id": p.id, "player": p.player, "stroke_type": p.stroke_type, "time": p.analysis.contact_time,
                  "phases": phase_times(p.analysis), "handedness": p.analysis.handedness,
                  "box": box(p.analysis)} for p in refs],
        "matches": matches,
    }
    (out / "demo.json").write_text(json.dumps(rnd(data), separators=(",", ":")))
    print(f"{len(rallies)} rallies, {sum(len(r.strokes) for r in rallies)} strokes, {len(refs)} references")
    for r in rallies:
        print(f"  rally {r.index + 1}: {r.start:.2f}-{r.end:.2f}", [s.stroke_type.value for s in r.strokes])
    print("  refs:", [p.stroke_type for p in refs])


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "preview" / "build")
