"""Annotated video output.

Draws with cv2 and projects the mesh by hand: pyrender cannot load on the GPU
boxes (no libEGL, no sudo), and a 2D skeleton plus a metrics HUD is what actually
reads as coaching feedback anyway.
"""

import cv2
import numpy as np

from . import mhr

GREEN, RED, CYAN, YELLOW, WHITE = ((80, 220, 80), (60, 60, 235), (235, 200, 60),
                                   (60, 220, 235), (245, 245, 245))


def project(pts3d, focal, cam_t, cx, cy):
    """Project camera-space points with the model's own predicted pinhole camera."""
    p = np.asarray(pts3d, dtype=np.float64) + np.asarray(cam_t, dtype=np.float64).reshape(1, 3)
    z = np.clip(p[:, 2], 1e-6, None)
    f = float(np.asarray(focal).reshape(-1)[0])
    return np.stack([p[:, 0] / z * f + cx, p[:, 1] / z * f + cy], axis=1)


def draw_skeleton(img, k2, colour=GREEN, joint_colour=RED):
    for a, b in mhr.LIMBS:
        if a < len(k2) and b < len(k2):
            pa, pb = np.int32(k2[a]), np.int32(k2[b])
            cv2.line(img, tuple(pa), tuple(pb), colour, 2, cv2.LINE_AA)
    for i in mhr.NAMES:
        if i < len(k2):
            cv2.circle(img, tuple(np.int32(k2[i])), 3, joint_colour, -1, cv2.LINE_AA)
    return img


def draw_board_frame(img, bf, focal, cam_t, cx, cy, length=0.5):
    """Draw the fitted board axes so the frame can be sanity-checked visually."""
    o = bf.origin
    ends = np.stack([o, o + bf.long * length, o + bf.lat * length * 0.6,
                     o + bf.normal * length * 0.6])
    p = project(ends, focal, cam_t, cx, cy)
    for i, col in ((1, CYAN), (2, YELLOW), (3, WHITE)):
        cv2.arrowedLine(img, tuple(np.int32(p[0])), tuple(np.int32(p[i])), col, 2,
                        cv2.LINE_AA, tipLength=0.2)
    return img


def draw_hud(img, values, turn=None, x=18, y=30, line=26):
    """Metrics panel. Only the numbers a rider can act on."""
    rows = [
        ("inclination", "{:.0f} deg", values.get("inclination")),
        ("angulation", "{:.0f} deg", values.get("angulation")),
        ("knee flex L/R", None, None),
        ("fore/aft", "{:+.2f}", values.get("fore_aft")),
        ("toe/heel", "{:+.2f}", values.get("toe_heel")),
        ("hip-shoulder", "{:+.0f} deg", values.get("hip_shoulder_sep")),
    ]
    pad = 8
    w = 320
    h = line * (len(rows) + (2 if turn else 1)) + pad
    panel = img[y - 22:y - 22 + h, x - pad:x - pad + w]
    if panel.size:
        panel[:] = (panel * 0.35).astype(np.uint8)

    cy_ = y
    if turn:
        cv2.putText(img, f"turn {turn['index']}  {turn['edge'].upper()}SIDE "
                         f"{turn['duration_s']:.1f}s", (x, cy_),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    CYAN if turn["edge"] == "toe" else YELLOW, 2, cv2.LINE_AA)
        cy_ += line
    for name, fmt, val in rows:
        if name == "knee flex L/R":
            l, r = values.get("knee_flex_l"), values.get("knee_flex_r")
            txt = ("knee flex L/R  {:.0f} / {:.0f} deg".format(l, r)
                   if l is not None and r is not None and np.isfinite(l) and np.isfinite(r)
                   else "knee flex L/R  --")
        elif val is None or not np.isfinite(val):
            txt = f"{name}  --"
        else:
            txt = f"{name}  " + fmt.format(val)
        cv2.putText(img, txt, (x, cy_), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        cy_ += line
    return img


def draw_edge_trace(img, series, i, height=70, colour_pos=CYAN, colour_neg=YELLOW):
    """A scrolling toe/heel trace, so edge changes are visible as they happen."""
    h, w = img.shape[:2]
    x0, y0, ww = w - 340, h - height - 30, 320
    s = np.asarray(series, dtype=float)
    if not len(s):
        return img
    lo, hi = np.nanmin(s), np.nanmax(s)
    rng = max(1e-6, hi - lo)
    panel = img[y0:y0 + height, x0:x0 + ww]
    if panel.size:
        panel[:] = (panel * 0.35).astype(np.uint8)
    mid = y0 + int(height * (hi / rng)) if lo < 0 < hi else y0 + height // 2
    cv2.line(img, (x0, mid), (x0 + ww, mid), (140, 140, 140), 1)
    n = len(s)
    for j in range(1, n):
        xa = x0 + int(ww * (j - 1) / max(1, n - 1))
        xb = x0 + int(ww * j / max(1, n - 1))
        ya = y0 + int(height * (hi - s[j - 1]) / rng)
        yb = y0 + int(height * (hi - s[j]) / rng)
        col = colour_pos if s[j] > 0 else colour_neg
        cv2.line(img, (xa, np.clip(ya, y0, y0 + height)), (xb, np.clip(yb, y0, y0 + height)),
                 col, 2, cv2.LINE_AA)
    xi = x0 + int(ww * i / max(1, n - 1))
    cv2.line(img, (xi, y0), (xi, y0 + height), (255, 255, 255), 1)
    cv2.putText(img, "toe", (x0 + 4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour_pos, 1)
    cv2.putText(img, "heel", (x0 + 4, y0 + height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                colour_neg, 1)
    return img


def write_video(path, frames, fps):
    if not frames:
        raise ValueError("no frames to write")
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    return path
