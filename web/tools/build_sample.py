"""Build web/sample/: the animated sample match and the reference strokes.

    python web/tools/build_sample.py

References are the yellow (far) player's strokes from the sample match: their
body positions straight from the animation, around each contact. The website
analyses them with the same engine as the viewer's own strokes.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml_pipeline"))
from ttvision.sample_match import W, H, Camera, plan_match, player_pose, render_sample_match  # noqa: E402

OUT = ROOT / "web" / "sample"


def main(prebuilt: str | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    video = OUT / "match.mp4"
    if prebuilt:
        shutil.copy(prebuilt, video)
    elif not video.exists():
        render_sample_match(video)
    plan, cam = plan_match(7), Camera()
    refs = []
    for r in plan:
        for s in r.shots:
            if s.player != "far":
                continue
            times = np.arange(s.contact - 0.9, s.contact + 0.7, 1 / 30)
            world, image = [], []
            for t in times:
                wcm, local = player_pose(plan, "far", float(t))
                uv = cam.project(wcm)
                world.append(np.round(local, 3).tolist())
                image.append(np.round(np.column_stack([uv[:, 0] / W, uv[:, 1] / H, np.ones(33)]), 3).tolist())
            refs.append({"player": "Yellow player (sample)", "contact": round(s.contact, 3),
                         "times": np.round(times, 3).tolist(), "world": world, "image": image})
    (OUT / "references.json").write_text(json.dumps(refs, separators=(",", ":")))
    print(f"{len(refs)} reference strokes -> {OUT / 'references.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
