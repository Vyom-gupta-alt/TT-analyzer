"""Render an animated sample match (video + audio) for demos and tests.

The scene is drawn with a real pinhole camera behind one end of the table,
broadcast style: arena floor, surround barriers, a regulation table
(274 x 152.5 cm, 76 cm high, 15.25 cm net) and two players. The players are
solid, shaded figures animated by the synthetic stroke model, and the ball's
flight is solved so it meets each racket exactly at contact. Every bat hit
and table bounce gets a click on the audio track.

This is animation, not a recording: it exists so the app can be tried before
any real footage is uploaded, and so the pipeline can be tested end to end.
"""

from __future__ import annotations

import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .demo import synthetic_swing
from .pose import kinematics as K

W, H, FPS = 1280, 720, 30
SR = 22050
TABLE_Z = 76.0
HALF_L, HALF_W = 137.0, 76.25
NET_H = 15.25
BALL_R = 2.0

# stroke shapes for the synthetic swing model
STROKES = {
    "serve": dict(up_amp=0.1, forward_amp=0.6, lateral_amp=-0.5, back_amp=0.25),
    "fh_loop": dict(up_amp=0.7, forward_amp=1.1, knee_bend_deg=36),
    "bh_loop": dict(up_amp=0.55, forward_amp=0.9, knee_bend_deg=30, wrist_side=-0.15, lateral_amp=0.35),
    "fh_push": dict(up_amp=-0.25, forward_amp=0.5, back_amp=0.15, knee_bend_deg=30),
    "bh_push": dict(up_amp=-0.25, forward_amp=0.5, back_amp=0.15, knee_bend_deg=30, wrist_side=-0.15,
                    lateral_amp=0.1),
    "block": dict(up_amp=0.1, forward_amp=0.35, back_amp=0.08, knee_bend_deg=28),
}
CONTACT = 0.7          # contact time inside each synthetic swing
# the two players play differently, so comparing them shows real differences:
# red (near) stands tall with a flatter, shorter swing; yellow (far) loads low and brushes up more
PLAYER_STYLE = {
    "near": dict(knee=-16.0, up=0.75, back=0.85),
    "far": dict(knee=8.0, up=1.15, back=1.2),
}


# ------------------------------------------------------------------ camera
class Camera:
    def __init__(self, pos=(0.0, -820.0, 420.0), target=(0.0, 10.0, 70.0), focal=1950.0):
        self.c = np.array(pos, float)
        fwd = np.array(target, float) - self.c
        self.fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(self.fwd, [0, 0, 1.0])
        self.right = right / np.linalg.norm(right)
        self.up = np.cross(self.right, self.fwd)
        self.f = focal

    def depth(self, p) -> np.ndarray:
        return (np.asarray(p, float) - self.c) @ self.fwd

    def project(self, p) -> np.ndarray:
        d = np.asarray(p, float) - self.c
        z = d @ self.fwd
        return np.stack([W / 2 + self.f * (d @ self.right) / z, H / 2 - self.f * (d @ self.up) / z], axis=-1)

    def px(self, cm: float, at) -> float:
        return self.f * cm / max(float(self.depth(at)), 1.0)


# ------------------------------------------------------------------ helpers
def _poly(img, cam, pts, color, aa=True):
    import cv2
    p = np.round(cam.project(np.array(pts, float)) * 16).astype(np.int32)
    cv2.fillPoly(img, [p], color, cv2.LINE_AA if aa else cv2.LINE_8, shift=4)


def _line(img, a, b, color, thick):
    import cv2
    pa, pb = np.round(a * 16).astype(int), np.round(b * 16).astype(int)
    cv2.line(img, tuple(pa), tuple(pb), color, max(1, int(round(thick))), cv2.LINE_AA, shift=4)


def _smooth(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


# ------------------------------------------------------------------ static scene
def _background(cam: Camera) -> np.ndarray:
    import cv2
    img = np.zeros((H, W, 3), np.uint8)
    # back wall: dark navy gradient with a sponsor-free band
    for y in range(H):
        t = y / H
        img[y] = (int(46 + 20 * t), int(30 + 14 * t), int(24 + 10 * t))
    # floor (court mat): deep red-brown, like an ITTF competition floor
    floor = [(-420, -700, 0), (420, -700, 0), (420, 560, 0), (-420, 560, 0)]
    _poly(img, cam, floor, (38, 44, 112))
    # surround barriers (dark blue boards, 75 cm high) on the far side and the sides
    for x0, y0, x1, y1 in ((-420, 560, 420, 560), (-420, -200, -420, 560), (420, -200, 420, 560)):
        n = 7
        for k in range(n):
            a0, a1 = k / n, (k + 1) / n
            p0 = (x0 + (x1 - x0) * a0, y0 + (y1 - y0) * a0)
            p1 = (x0 + (x1 - x0) * a1, y0 + (y1 - y0) * a1)
            body = (110, 48, 20) if k % 2 else (96, 40, 16)
            _poly(img, cam, [(*p0, 0), (*p1, 0), (*p1, 75), (*p0, 75)], body)
            _poly(img, cam, [(*p0, 30), (*p1, 30), (*p1, 44), (*p0, 44)], (170, 120, 60) if k % 2 else (60, 150, 230))
        _poly(img, cam, [(x0, y0, 72), (x1, y1, 72), (x1, y1, 75), (x0, y0, 75)], (150, 80, 40))
    # court boundary tape
    for a, b in (((-400, -660), (400, -660)), ((400, -660), (400, 540)), ((-400, -660), (-400, 540))):
        _poly(img, cam, [(a[0] - 2, a[1] - 2, 0.1), (b[0] + 2, b[1] - 2, 0.1), (b[0] + 2, b[1] + 2, 0.1),
                         (a[0] - 2, a[1] + 2, 0.1)], (70, 80, 150))
    # soft light pool on the floor around the table
    glow = np.zeros((H, W), np.float32)
    centre = cam.project([0, 0, 0])
    cv2.ellipse(glow, tuple(int(v) for v in centre), (520, 210), 0, 0, 360, 1.0, -1)
    glow = cv2.GaussianBlur(glow, (0, 0), 90)
    img = np.clip(img.astype(np.float32) * (1 + 0.35 * glow[..., None]), 0, 255).astype(np.uint8)
    return img


def _table_layer(cam: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Table, legs, net and the table's floor shadow, as an image + alpha mask."""
    import cv2
    img = np.zeros((H, W, 3), np.uint8)
    mask = np.zeros((H, W), np.uint8)

    def both(fn, pts, color):
        _poly(img, cam, pts, color)
        _poly(mask, cam, pts, 255)

    # floor shadow (drawn into the alpha only as darkening, handled by caller)
    shadow = np.zeros((H, W), np.uint8)
    _poly(shadow, cam, [(-HALF_W - 6, -HALF_L - 6, 0), (HALF_W + 6, -HALF_L - 6, 0),
                        (HALF_W + 6, HALF_L + 6, 0), (-HALF_W - 6, HALF_L + 6, 0)], 255)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 14)

    # legs and frame
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = sx * (HALF_W - 18), sy * (HALF_L - 30)
            both(None, [(x - 2.5, y, 0), (x + 2.5, y, 0), (x + 2.5, y, TABLE_Z - 3), (x - 2.5, y, TABLE_Z - 3)], (40, 40, 44))
    both(None, [(-HALF_W + 10, -HALF_L + 10, TABLE_Z - 10), (HALF_W - 10, -HALF_L + 10, TABLE_Z - 10),
                (HALF_W - 10, -HALF_L + 10, TABLE_Z - 3), (-HALF_W + 10, -HALF_L + 10, TABLE_Z - 3)], (30, 30, 34))
    # table edge (thickness) on the near side and sides
    top = TABLE_Z
    both(None, [(-HALF_W, -HALF_L, top - 2.5), (HALF_W, -HALF_L, top - 2.5), (HALF_W, -HALF_L, top), (-HALF_W, -HALF_L, top)], (70, 38, 16))
    for sx in (-1, 1):
        both(None, [(sx * HALF_W, -HALF_L, top - 2.5), (sx * HALF_W, HALF_L, top - 2.5),
                    (sx * HALF_W, HALF_L, top), (sx * HALF_W, -HALF_L, top)], (80, 44, 20))
    # playing surface: ITTF blue with a subtle light falloff
    both(None, [(-HALF_W, -HALF_L, top), (HALF_W, -HALF_L, top), (HALF_W, HALF_L, top), (-HALF_W, HALF_L, top)], (150, 82, 28))
    # white lines: 2 cm edges, 3 mm centre line
    e = 2.0
    for pts in ([(-HALF_W, -HALF_L, top), (HALF_W, -HALF_L, top), (HALF_W, -HALF_L + e, top), (-HALF_W, -HALF_L + e, top)],
                [(-HALF_W, HALF_L - e, top), (HALF_W, HALF_L - e, top), (HALF_W, HALF_L, top), (-HALF_W, HALF_L, top)],
                [(-HALF_W, -HALF_L, top), (-HALF_W + e, -HALF_L, top), (-HALF_W + e, HALF_L, top), (-HALF_W, HALF_L, top)],
                [(HALF_W - e, -HALF_L, top), (HALF_W, -HALF_L, top), (HALF_W, HALF_L, top), (HALF_W - e, HALF_L, top)],
                [(-0.3, -HALF_L, top), (0.3, -HALF_L, top), (0.3, HALF_L, top), (-0.3, HALF_L, top)]):
        both(None, pts, (238, 240, 240))
    # light sheen across the surface
    sheen = np.zeros((H, W), np.float32)
    _poly(sheen, cam, [(-HALF_W, -40, top), (HALF_W, -40, top), (HALF_W, 60, top), (-HALF_W, 60, top)], 1.0)
    sheen = cv2.GaussianBlur(sheen, (0, 0), 40) * (mask > 0)
    img = np.clip(img.astype(np.float32) + 22 * sheen[..., None], 0, 255).astype(np.uint8)

    # net: dark translucent mesh with a white top band and posts
    net = [(-HALF_W - 15, 0, top), (HALF_W + 15, 0, top), (HALF_W + 15, 0, top + NET_H), (-HALF_W - 15, 0, top + NET_H)]
    net_img = img.copy()
    _poly(net_img, cam, net, (60, 55, 50))
    net_mask = np.zeros((H, W), np.uint8)
    _poly(net_mask, cam, net, 255)
    a = (net_mask > 0)[..., None]
    img = np.where(a, (0.6 * net_img + 0.4 * img).astype(np.uint8), img)
    mask = np.maximum(mask, (net_mask * 0.55).astype(np.uint8))
    both(None, [(-HALF_W - 15, 0, top + NET_H - 1.5), (HALF_W + 15, 0, top + NET_H - 1.5),
                (HALF_W + 15, 0, top + NET_H), (-HALF_W - 15, 0, top + NET_H)], (235, 235, 235))
    for sx in (-1, 1):
        both(None, [(sx * (HALF_W + 13), 0, top - 2), (sx * (HALF_W + 16), 0, top - 2),
                    (sx * (HALF_W + 16), 0, top + NET_H + 1), (sx * (HALF_W + 13), 0, top + NET_H + 1)], (60, 60, 64))
    return img, mask, shadow


# ------------------------------------------------------------------ players
SKIN, HAIR = (150, 170, 215), (30, 32, 40)


@dataclass
class PlayerStyle:
    shirt: tuple
    shorts: tuple
    shoes: tuple = (235, 235, 235)


NEAR_STYLE = PlayerStyle(shirt=(40, 40, 200), shorts=(30, 30, 30))       # red shirt
FAR_STYLE = PlayerStyle(shirt=(40, 200, 235), shorts=(70, 40, 20))       # yellow shirt, navy shorts


def to_world(local: np.ndarray, base_xy, facing: float) -> np.ndarray:
    """Synthetic-swing coords (m; x right, y down, -z forward) -> scene cm."""
    right = np.array([facing, 0.0])          # facing +y => right is +x
    fwd = np.array([0.0, facing])
    lat, fw, up = local[..., 0], -local[..., 2], 0.9 - local[..., 1]
    xy = (lat[..., None] * right + fw[..., None] * fwd) * 100 + np.asarray(base_xy)
    return np.concatenate([xy, (up * 100)[..., None]], axis=-1)


def draw_player(img, cam: Camera, w: np.ndarray, style: PlayerStyle, hand: str = "right", back: bool = False):
    """Draw a solid figure from world joints (33, 3) in cm, far parts first."""
    import cv2
    prims = []   # (depth, kind, data)

    def seg(a, b, cm, color):
        pa, pb = w[a], w[b]
        prims.append((float(cam.depth((pa + pb) / 2)), "seg", (pa, pb, cm, color)))

    # floor shadow
    feet = (w[K.L_ANKLE] + w[K.R_ANKLE]) / 2
    feet[2] = 0
    sh = np.zeros((H, W), np.uint8)
    c = cam.project(feet)
    cv2.ellipse(sh, (int(c[0]), int(c[1])), (int(cam.px(40, feet)), int(cam.px(14, feet))), 0, 0, 360, 255, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), 6)
    img[:] = (img * (1 - 0.45 * sh[..., None] / 255)).astype(np.uint8)

    seg(K.L_HIP, K.L_KNEE, 15, style.shorts); seg(K.R_HIP, K.R_KNEE, 15, style.shorts)
    seg(K.L_KNEE, K.L_ANKLE, 10, SKIN); seg(K.R_KNEE, K.R_ANKLE, 10, SKIN)
    seg(K.L_ANKLE, 31, 9, style.shoes); seg(K.R_ANKLE, 32, 9, style.shoes)
    seg(K.L_SHOULDER, K.L_ELBOW, 9, style.shirt); seg(K.R_SHOULDER, K.R_ELBOW, 9, style.shirt)
    seg(K.L_ELBOW, K.L_WRIST, 7, SKIN); seg(K.R_ELBOW, K.R_WRIST, 7, SKIN)
    torso = [w[K.L_SHOULDER], w[K.R_SHOULDER], w[K.R_HIP], w[K.L_HIP]]
    prims.append((float(cam.depth(np.mean(torso, axis=0))), "torso", torso))
    neck = (w[K.L_SHOULDER] + w[K.R_SHOULDER]) / 2
    head = neck + np.array([0, 0, 22.0])
    prims.append((float(cam.depth(head)), "head", head))
    # racket: blade beyond the hand along the forearm
    arm = K.Arm.for_hand(hand)
    fa = w[arm.wrist] - w[arm.elbow]
    fa = fa / (np.linalg.norm(fa) + 1e-6)
    blade = w[arm.wrist] + fa * 14
    prims.append((float(cam.depth(blade)) - 1, "racket", (w[arm.wrist], blade)))

    for _, kind, d in sorted(prims, key=lambda p: -p[0]):
        if kind == "seg":
            pa, pb, cm, color = d
            a, b = cam.project(pa), cam.project(pb)
            t = cam.px(cm, (pa + pb) / 2)
            _line(img, a, b, tuple(int(v * 0.7) for v in color), t + 2)
            _line(img, a, b, color, t)
        elif kind == "torso":
            ls, rs, rh, lh = d
            outward = (rs - ls) * 0.12
            pts = [ls - outward, rs + outward, rh + (rh - lh) * 0.08, lh - (rh - lh) * 0.08]
            _poly(img, cam, pts, style.shirt)
            mid_a = cam.project((ls + rs) / 2)
            mid_b = cam.project((lh + rh) / 2)
            _line(img, mid_a, mid_b, tuple(int(v * 0.8) for v in style.shirt), cam.px(6, ls))
        elif kind == "head":
            p = cam.project(d)
            r = cam.px(10.5, d)
            nb = cam.project(neck)
            _line(img, nb, p, SKIN, cam.px(8, neck))
            cv2.circle(img, (int(p[0] * 16), int(p[1] * 16)), int(r * 16), HAIR if back else SKIN, -1,
                       cv2.LINE_AA, shift=4)
            if not back:     # facing the camera: hair only on top
                cv2.ellipse(img, (int(p[0] * 16), int((p[1] - r * 0.25) * 16)), (int(r * 16), int(r * 0.85 * 16)),
                            0, 180, 360, HAIR, -1, cv2.LINE_AA, shift=4)
        elif kind == "racket":
            wr, bl = d
            a, b = cam.project(wr), cam.project(bl)
            _line(img, a, b, (60, 90, 140), cam.px(3, wr))          # handle
            r = cam.px(8, bl)
            ang = np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))
            cv2.ellipse(img, (int(b[0] * 16), int(b[1] * 16)), (int(r * 16), int(r * 0.85 * 16)), ang, 0, 360,
                        (20, 20, 190), -1, cv2.LINE_AA, shift=4)
            cv2.ellipse(img, (int(b[0] * 16), int(b[1] * 16)), (int(r * 16), int(r * 0.85 * 16)), ang, 0, 360,
                        (15, 15, 20), max(1, int(r * 0.15)), cv2.LINE_AA, shift=4)


# ------------------------------------------------------------------ choreography
@dataclass
class Shot:
    player: str          # "near" | "far"
    stroke: str          # key of STROKES
    contact: float       # seconds
    hit: np.ndarray = None           # world cm, filled in
    base: np.ndarray = None          # player base xy at contact


@dataclass
class Rally:
    shots: list
    ending: str                      # "long" | "net"
    ball_path: list = field(default_factory=list)   # (t0, t1, p0, p1, apex)
    events: list = field(default_factory=list)       # (t, kind) kind: bat | table | floor


_SWINGS: dict = {}


def swing(stroke: str, who: str = "near", hand="right"):
    key = (stroke, who, hand)
    if key not in _SWINGS:
        kw = dict(STROKES[stroke])
        st = PLAYER_STYLE[who]
        kw["knee_bend_deg"] = max(8.0, kw.get("knee_bend_deg", 30.0) + st["knee"])
        if kw.get("up_amp", 0.6) > 0:
            kw["up_amp"] = kw.get("up_amp", 0.6) * st["up"]
        kw["back_amp"] = kw.get("back_amp", 0.4) * st["back"]
        seq = synthetic_swing(fps=60.0, hand=hand, **kw)
        racket = K.racket_position(seq.world.astype(float), K.Arm.for_hand(hand), 0.16)
        _SWINGS[key] = (seq, racket)
    return _SWINGS[key]


READY = None


def ready_pose():
    global READY
    if READY is None:
        READY = {w: swing("block", w)[0].world[0].astype(float) for w in PLAYER_STYLE}
    return READY


def plan_match(seed: int = 7) -> list[Rally]:
    rng = np.random.default_rng(seed)
    scripts = [
        (["serve", "fh_push", "fh_loop", "block", "fh_loop", "bh_push"], "long"),
        (["serve", "bh_push", "bh_loop", "fh_loop", "block"], "net"),
        (["serve", "fh_push", "fh_push", "bh_loop", "fh_loop", "block", "fh_loop"], "long"),
        (["serve", "bh_push", "fh_loop", "fh_loop"], "long"),
    ]
    rallies, t = [], 3.0
    for strokes, ending in scripts:
        shots, c = [], t
        for i, s in enumerate(strokes):
            player = "near" if i % 2 == 0 else "far"
            shots.append(Shot(player, s, c))
            c += 0.95 if i == 0 else float(rng.uniform(0.85, 1.0))
        rallies.append(Rally(shots, ending))
        t = c + float(rng.uniform(5.5, 7.0))          # dead time: ball fetching, towel, reset
    for r in rallies:
        _solve_rally(r, rng)
    return rallies


def _contact_point(shot: Shot, target_x: float) -> None:
    facing = 1.0 if shot.player == "near" else -1.0
    _, racket = swing(shot.stroke, shot.player)
    ci = int(round(CONTACT * 60))
    local = racket[ci]
    offset = to_world(local[None], (0, 0), facing)[0]
    end_y = -HALF_L - 22 if shot.player == "near" else HALF_L + 22
    hit = np.array([target_x, end_y, offset[2]])
    shot.hit = hit
    shot.base = hit[:2] - offset[:2]


def _arc(t0, t1, p0, p1, apex):
    return (t0, t1, np.asarray(p0, float), np.asarray(p1, float), apex)


def _solve_rally(r: Rally, rng) -> None:
    for s in r.shots:
        _contact_point(s, float(rng.uniform(-35, 35)))
    path, ev = r.ball_path, r.events
    first = r.shots[0]
    # serve toss: ball rises from the palm ~0.75 s before contact
    palm = first.hit + np.array([-10.0, 6.0, -18.0])
    path.append(("toss", first.contact - 0.8, first.contact, palm, first.hit))
    surf = TABLE_Z + BALL_R
    for i, s in enumerate(r.shots):
        ev.append((s.contact, "bat"))
        nxt = r.shots[i + 1] if i + 1 < len(r.shots) else None
        side = 1.0 if s.player == "near" else -1.0        # direction the ball travels in y
        if i == 0:   # serve: bounce on own side, then the far side
            b1 = np.array([s.hit[0] * 0.6, -70.0 * side, surf])     # server's own half
            b2 = np.array([float(rng.uniform(-40, 40)), 75.0 * side, surf])
            tb1, tb2 = s.contact + 0.2, s.contact + 0.58
            path += [_arc(s.contact, tb1, s.hit, b1, 6), _arc(tb1, tb2, b1, b2, 26)]
            ev += [(tb1, "table"), (tb2, "table")]
            last_p, last_t = b2, tb2
        else:
            if nxt is None and r.ending == "net":
                hit_net = np.array([s.hit[0] * 0.5, -2.0 * side, TABLE_Z + 9])
                tn = s.contact + 0.32
                drop = np.array([hit_net[0], -18.0 * side, surf])
                path += [_arc(s.contact, tn, s.hit, hit_net, 12), _arc(tn, tn + 0.2, hit_net, drop, 2)]
                ev += [(tn, "net")]
                # dying bounces on the hitter's own side
                p, tt, hgt, dt = drop, tn + 0.2, 9.0, 0.26
                for _ in range(5):
                    q = p + np.array([0, -7.0 * side, 0])
                    path.append(_arc(tt, tt + dt, p, q, hgt))
                    ev.append((tt, "table"))
                    p, tt, hgt, dt = q, tt + dt, hgt * 0.55, dt * 0.78
                r.events.append((tt, "end"))
                continue
            b = np.array([float(rng.uniform(-55, 55)), float(rng.uniform(55, 105)) * side, surf])
            tb = s.contact + float(rng.uniform(0.34, 0.42))
            path.append(_arc(s.contact, tb, s.hit, b, 24 if "loop" in s.stroke else 16))
            ev.append((tb, "table"))
            last_p, last_t = b, tb
        if nxt is not None:
            path.append(_arc(last_t, nxt.contact, last_p, nxt.hit, 20))
        else:   # winner: ball flies past the opponent onto the floor
            floor = np.array([last_p[0] * 1.6, (HALF_L + 230) * side, BALL_R])
            tf = last_t + 0.55
            path.append(_arc(last_t, tf, last_p, floor, 40))
            ev.append((tf, "floor"))
            p, tt, hgt = floor, tf, 22.0
            for _ in range(3):
                q = p + np.array([0, 45.0 * side, 0])
                path.append(_arc(tt, tt + 0.3, p, q, hgt))
                p, tt, hgt = q, tt + 0.3, hgt * 0.5
            path.append(_arc(tt, tt + 1.2, p, p + np.array([0, 60.0 * side, 0]), 0))
            ev.append((tf, "end"))


def ball_at(r: Rally, t: float):
    for seg in r.ball_path:
        if seg[0] == "toss":
            _, t0, t1, p0, p1 = seg
            if t0 <= t <= t1:
                u = (t - t0) / (t1 - t0)
                if u < 0.12:                       # resting in the palm
                    return p0
                u = (u - 0.12) / 0.88
                apex = p1 + np.array([0, 0, 34.0])
                if u < 0.55:
                    v = u / 0.55
                    return p0 + (apex - p0) * (1 - (1 - v) ** 2)
                v = (u - 0.55) / 0.45
                return apex + (p1 - apex) * v * v
            continue
        t0, t1, p0, p1, apex = seg
        if t0 <= t <= t1:
            u = (t - t0) / max(t1 - t0, 1e-6)
            return p0 + (p1 - p0) * u + np.array([0, 0, apex * 4 * u * (1 - u)])
    return None


def player_pose(rallies: list[Rally], who: str, t: float) -> tuple[np.ndarray, np.ndarray]:
    """(world joints cm (33,3), local joints m (33,3)) for one player at time t."""
    facing = 1.0 if who == "near" else -1.0
    shots = [s for r in rallies for s in r.shots if s.player == who]
    default_y = -HALF_L - 75 if who == "near" else HALF_L + 75
    # base position: glide between the positions needed for each shot
    keys = [(s.contact, s.base) for s in shots]
    base = np.array([0.0, default_y])
    if keys:
        times = [k[0] for k in keys]
        j = int(np.searchsorted(times, t))
        if j == 0:
            base = keys[0][1].copy()
        elif j >= len(keys):
            base = keys[-1][1].copy()
        else:
            (ta, pa), (tb, pb) = keys[j - 1], keys[j]
            if tb - ta > 3.0:   # between rallies: step back and walk to the ready spot
                home = np.array([0.0, default_y + 40 * facing * -1])
                u1, u2 = _smooth((t - ta - 0.6) / 1.5), _smooth((t - (tb - 2.0)) / 1.5)
                base = pa + (home - pa) * u1 + (pb - home) * u2
            else:
                base = pa + (pb - pa) * _smooth((t - ta - 0.25) / max(tb - ta - 0.55, 0.1))
    local = ready_pose()[who].copy()
    for s in shots:
        seq, _ = swing(s.stroke, who)
        u = t - (s.contact - CONTACT)
        if -0.25 <= u <= 1.2 + 0.35:
            fi = int(np.clip(round(u * 60), 0, len(seq.times) - 1))
            pose = seq.world[fi].astype(float)
            if u < 0:
                w = _smooth((u + 0.25) / 0.25)
            elif u > 1.2:
                w = 1 - _smooth((u - 1.2) / 0.35)
            else:
                w = 1.0
            local = local + (pose - local) * w
            break
    # small idle bounce on the toes
    local = local + np.array([0, 0.012 * np.sin(t * 5.3 + (0 if who == "near" else 1.7)), 0])
    return to_world(local, base, facing), local


# ------------------------------------------------------------------ audio
def _audio(rallies: list[Rally], duration: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    t = np.arange(n) / SR
    # room tone: low rumble + air
    x = 0.006 * rng.standard_normal(n).astype(np.float32)
    x += (0.01 * np.sin(2 * np.pi * 60 * t)).astype(np.float32)

    def add(at, sig):
        i = int(at * SR)
        if 0 <= i < n:
            x[i:i + sig.size] += sig[: n - i]

    def click(freq, dur, amp, decay):
        tt = np.arange(int(SR * dur)) / SR
        s = amp * np.sin(2 * np.pi * freq * tt) * np.exp(-tt / decay)
        s += 0.4 * amp * np.sin(2 * np.pi * freq * 1.31 * tt) * np.exp(-tt / (decay * 0.7))
        s += 0.15 * amp * rng.standard_normal(tt.size) * np.exp(-tt / 0.0015)
        return s.astype(np.float32)

    for r in rallies:
        for at, kind in r.events:
            if kind == "bat":
                add(at, click(3000, 0.02, 0.55, 0.004) + np.pad(click(900, 0.02, 0.15, 0.006), (0, 0)))
            elif kind == "table":
                add(at, click(3500, 0.015, 0.45, 0.003))
            elif kind == "net":
                add(at, click(600, 0.05, 0.25, 0.01))
            elif kind == "floor":
                add(at, click(1200, 0.03, 0.15, 0.006))
        # footsteps and a towel shuffle during the dead time after each rally
        end = max(at for at, _ in r.events)
        for k in range(6):
            tt = np.arange(int(SR * 0.08)) / SR
            add(end + 1.0 + k * 0.55 + rng.uniform(0, 0.1),
                (0.2 * np.sin(2 * np.pi * 85 * tt) * np.exp(-tt / 0.02)).astype(np.float32))
    return np.clip(x, -1, 1)


# ------------------------------------------------------------------ render
def render_sample_match(out_path: str | Path, seed: int = 7, progress=None) -> dict:
    """Render the sample match to out_path (mp4). Returns timing metadata and the
    near player's pose track (for demo stroke analysis)."""
    import cv2

    rallies = plan_match(seed)
    duration = max(max(t for t, _ in r.events) for r in rallies) + 3.0
    cam = Camera()
    bg = _background(cam)
    table_img, table_mask, table_shadow = _table_layer(cam)
    shadow_f = (table_shadow.astype(np.float32) / 255)[..., None]
    bg = (bg * (1 - 0.5 * shadow_f)).astype(np.uint8)
    tm = (table_mask.astype(np.float32) / 255)[..., None]
    vignette = np.ones((H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    vignette = 1 - 0.28 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    vignette = np.clip(vignette, 0.6, 1)[..., None]
    rng = np.random.default_rng(seed)

    out_path = Path(out_path)
    tmp = Path(tempfile.mkdtemp())
    raw = tmp / "video.avi"
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    n_frames = int(duration * FPS)
    near_times, near_world, near_image = [], [], []
    trail: list = []
    for fi in range(n_frames):
        t = fi / FPS
        frame = bg.copy()
        wf, _ = player_pose(rallies, "far", t)
        draw_player(frame, cam, wf, FAR_STYLE)
        frame = (frame * (1 - tm) + table_img * tm).astype(np.uint8)

        ball = None
        for r in rallies:
            ball = ball_at(r, t)
            if ball is not None:
                break
        if ball is not None:
            trail.append(ball)
            trail = trail[-4:]
            if ball[2] >= TABLE_Z and abs(ball[0]) <= HALF_W and abs(ball[1]) <= HALF_L:
                sh = cam.project([ball[0], ball[1], TABLE_Z + 0.1])
                rad = cam.px(BALL_R * 1.4, ball)
                cv2.ellipse(frame, (int(sh[0]), int(sh[1])), (max(1, int(rad)), max(1, int(rad * 0.45))), 0, 0, 360,
                            (80, 40, 10), -1, cv2.LINE_AA)
            for k, p in enumerate(trail[:-1]):      # motion blur
                q = cam.project(p)
                cv2.circle(frame, (int(q[0]), int(q[1])), max(1, int(cam.px(BALL_R, p))),
                           (200, 225, 245), -1, cv2.LINE_AA)
            q = cam.project(ball)
            cv2.circle(frame, (int(q[0] * 16), int(q[1] * 16)), max(40, int(cam.px(BALL_R * 1.3, ball) * 16)),
                       (245, 250, 255), -1, cv2.LINE_AA, shift=4)
        else:
            trail = []

        wn, ln = player_pose(rallies, "near", t)
        draw_player(frame, cam, wn, NEAR_STYLE, back=True)
        uv = cam.project(wn)
        near_times.append(t)
        near_world.append(ln)
        near_image.append(np.column_stack([uv[:, 0] / W, uv[:, 1] / H, np.ones(33)]))

        f = frame.astype(np.float32) * vignette + rng.normal(0, 2.2, frame.shape)
        vw.write(np.clip(f, 0, 255).astype(np.uint8))
        if progress and fi % 60 == 0:
            progress(fi / n_frames)
    vw.release()

    wav = tmp / "audio.wav"
    pcm = (_audio(rallies, duration, seed) * 32767).astype(np.int16)
    with wave.open(str(wav), "wb") as wf_:
        wf_.setnchannels(1)
        wf_.setsampwidth(2)
        wf_.setframerate(SR)
        wf_.writeframes(pcm.tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-i", str(wav), "-c:v", "libx264",
                    "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-g", "30", "-c:a", "aac",
                    "-b:a", "96k", "-shortest", "-movflags", "+faststart", str(out_path)], check=True)
    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()
    return {
        "duration": duration,
        "rallies": [{"shots": [(s.player, s.stroke, s.contact) for s in r.shots], "ending": r.ending,
                     "events": r.events} for r in rallies],
        "near_pose": {"times": np.asarray(near_times), "world": np.asarray(near_world, np.float32),
                      "image": np.asarray(near_image, np.float32)},
    }
