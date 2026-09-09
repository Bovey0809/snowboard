"""Annotated video output.

Draws with cv2 and projects the mesh by hand: pyrender cannot load on the GPU
boxes (no libEGL, no sudo), and a 2D skeleton plus a metrics HUD is what actually
reads as coaching feedback anyway.

Text is the one thing cv2 cannot do here. Its Hershey fonts are vector strokes
with no glyph beyond ASCII, so a Chinese overlay is not a matter of passing a
different string — `putText` simply draws nothing for 内倾. So every label goes
through PIL against a real TrueType face instead.

Doing that per call would mean a full-frame BGR->RGB->PIL->back round trip for
each of the ~13 strings on a frame, which at 1920x1080 is most of the render
budget. Instead the draw functions *queue* their text and `render_text` applies
the whole frame's worth in one pass. Shapes stay on cv2 and are drawn first, so
the queue only ever holds text that belongs on top.
"""

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None

from . import config as CFG
from . import mhr

GREEN, RED, CYAN, YELLOW, WHITE = ((80, 220, 80), (60, 60, 235), (235, 200, 60),
                                   (60, 220, 235), (245, 245, 245))
AMBER, GREY, ORANGE = (60, 180, 245), (140, 140, 140), (40, 150, 250)
BAND, TRACK = (55, 105, 55), (72, 72, 72)

FONT = cv2.FONT_HERSHEY_SIMPLEX
STATUS_COLOUR = {"good": GREEN, "warn": AMBER, "bad": RED, "na": GREY}

_FONTS = {}


def font_ok():
    """Whether real text rendering is available. False means ASCII-only."""
    return Image is not None and CFG.FONT is not None


def _pil_font(px):
    px = max(6, int(round(px)))
    if px not in _FONTS:
        _FONTS[px] = ImageFont.truetype(CFG.FONT, px)
    return _FONTS[px]


_MEASURE = None


def text_size(text, px):
    """(width, height) of `text` at `px`, measured the way it will be drawn.

    Measured through `textbbox` with the same anchor `render_text` uses, not
    `font.getbbox` — the two disagree by the ascender offset, which is what
    makes a caption's background box sit slightly off its text.
    """
    global _MEASURE
    if not font_ok():
        scale = px / 22.0
        (w, h), _ = cv2.getTextSize(text, FONT, scale, 2)
        return w, h
    if _MEASURE is None:
        _MEASURE = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    l, t, r, b = _MEASURE.textbbox((0, 0), text, font=_pil_font(px), anchor="lt")
    return r - l, b - t


def queue_text(items, text, xy, px, colour, stroke=0, stroke_colour=(20, 20, 20)):
    """Add one string to a frame's text queue. `xy` is the top-left corner."""
    items.append({"text": text, "xy": (int(xy[0]), int(xy[1])), "px": px,
                  "colour": colour, "stroke": stroke,
                  "stroke_colour": stroke_colour})
    return items


def render_text(img, items):
    """Draw a whole frame's queued text in one pass. Returns the image."""
    if not items:
        return img
    if not font_ok():
        # No usable face: fall back to Hershey, which will mangle any non-ASCII.
        # `analyze_run` refuses --lang zh in this state rather than shipping a
        # video full of empty boxes.
        for it in items:
            cv2.putText(img, it["text"], (it["xy"][0], it["xy"][1] + int(it["px"])),
                        FONT, it["px"] / 22.0, it["colour"],
                        2 if it["stroke"] else 1, cv2.LINE_AA)
        return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for it in items:
        f = _pil_font(it["px"])
        # PIL wants RGB; every colour in this module is BGR.
        fill = tuple(int(c) for c in reversed(it["colour"]))
        kw = {}
        if it["stroke"]:
            kw = {"stroke_width": int(it["stroke"]),
                  "stroke_fill": tuple(int(c) for c in reversed(it["stroke_colour"]))}
        # "lt" anchors the top of the ascender, so rows line up on their box
        # rather than drifting with each string's tallest glyph.
        d.text(it["xy"], it["text"], font=f, fill=fill, anchor="lt", **kw)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


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


def _frac(v, lo, hi):
    """Where `v` sits in [lo, hi] as a 0-1 fraction, clamped."""
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0)) if hi > lo else 0.0


def _gauge_bar(img, x0, ytop, w, h, row):
    """One target-band bar: track, green target zone, marker at the live value.

    The target zone is the whole point. A bare "angulation 39 deg" is unreadable
    to a learner, but a marker inside or outside a green band needs no
    interpretation at all.
    """
    lo, hi = row["range"]
    cv2.rectangle(img, (x0, ytop), (x0 + w, ytop + h), TRACK, -1)
    if row["enabled"]:
        blo, bhi = row["band"]
        a, b = x0 + int(w * _frac(blo, lo, hi)), x0 + int(w * _frac(bhi, lo, hi))
        cv2.rectangle(img, (a, ytop), (max(a + 1, b), ytop + h), BAND, -1)
    # Bipolar metrics get a centre tick, so "which side" is legible as well as
    # "how far": fore/aft toward the tail and toward the nose are opposite faults.
    if row["bipolar"] and lo < 0 < hi:
        z = x0 + int(w * _frac(0.0, lo, hi))
        cv2.line(img, (z, ytop), (z, ytop + h), GREY, 1)
    if row["enabled"] and np.isfinite(row["value"]):
        m = x0 + int(w * _frac(row["value"], lo, hi))
        cv2.line(img, (m, ytop - 3), (m, ytop + h + 3),
                 STATUS_COLOUR[row["status"]], 2, cv2.LINE_AA)


def draw_gauges(img, rows, turn=None, level=None, lang="zh", text=None,
                x=20, y=20, line=30, label_w=132, bar_w=150,
                head_px=22, label_px=19, value_px=19, level_px=15):
    """Metrics panel as target-band gauges rather than bare numbers.

    `rows` comes from `coach.gauge_rows`, so the green zone drawn here and the
    thresholds `report.py` writes about are the same table by construction.

    Queues its text into `text` when given, for `render_text` to draw in one
    pass; otherwise renders its own and returns the new image.
    """
    from . import coach

    own = text is None
    items = [] if own else text
    pad = 10
    w = label_w + bar_w + 100
    head = (1 if turn else 0) + (1 if level else 0)
    h = line * (len(rows) + head) + 2 * pad
    panel = img[max(0, y - pad):y - pad + h, max(0, x - pad):x - pad + w]
    if panel.size:
        panel[:] = (panel * 0.35).astype(np.uint8)

    def centred(px):
        """Top edge that centres a `px` glyph box in one row."""
        return (line - px) // 2

    cy_ = y
    if turn:
        edge = coach.say(turn["edge"], lang)
        label = (f"{coach.say('turn', lang)} {turn['index']}  {edge} "
                 f"{turn['duration_s']:.1f}s" if lang == "en" else
                 f"第{turn['index']}个{coach.say('turn', lang)}  {edge} "
                 f"{turn['duration_s']:.1f}s")
        queue_text(items, label, (x, cy_ + centred(head_px)), head_px,
                   CYAN if turn["edge"] == "toe" else YELLOW, stroke=1)
        cy_ += line
    if level:
        # The bands are level-dependent guesses, so the video says which set it
        # is judging against instead of leaving that in a report nobody reads.
        sep = ": " if lang == "en" else "："
        queue_text(items, f"{coach.say('targets', lang)}{sep}{coach.say(level, lang)}",
                   (x, cy_ + centred(level_px)), level_px, GREY)
        cy_ += line

    for row in rows:
        queue_text(items, row["label"], (x, cy_ + centred(label_px)), label_px, WHITE)
        _gauge_bar(img, x + label_w, cy_ + (line - 12) // 2, bar_w, 12, row)
        queue_text(items, row["text"], (x + label_w + bar_w + 12,
                                        cy_ + centred(value_px)), value_px,
                   STATUS_COLOUR[row["status"]])
        cy_ += line
    return render_text(img, items) if own else img


def _overlaps(a, b):
    return not (a[0] >= b[0] + b[2] or a[0] + a[2] <= b[0] or
                a[1] >= b[1] + b[3] or a[1] + a[3] <= b[1])


def draw_cue(img, cue, box=None, avoid=None, text=None, min_px=24, max_px=60):
    """The single live coaching cue, anchored to the rider.

    Anchoring keeps the rider's eye on the body instead of a corner, but it has
    to survive a small subject: the broadcast clip tracks a rider only 267 px
    tall, so the caption is scaled from box height with a legibility floor, moved
    below the rider when there is no room above, and clamped inside the frame.
    """
    if not cue:
        return img
    own = text is None
    items = [] if own else text
    h, w = img.shape[:2]
    s = cue["text"]
    bh = float(box[3] - box[1]) if box is not None else h * 0.4
    px = int(np.clip(bh / 9.0, min_px, max_px))
    tw, tht = text_size(s, px)
    pad = int(0.42 * px)
    bw_, bh_ = tw + 2 * pad, tht + 2 * pad

    cx = int((box[0] + box[2]) / 2) if box is not None else w // 2
    x0 = int(np.clip(cx - bw_ // 2, 6, max(6, w - bw_ - 6)))
    if box is not None:
        y0 = int(box[1]) - bh_ - int(0.5 * px)
        if y0 < 6 or (avoid and _overlaps((x0, y0, bw_, bh_), avoid)):
            y0 = int(box[3]) + int(0.5 * px)
    else:
        y0 = h - bh_ - 130
    y0 = int(np.clip(y0, 6, max(6, h - bh_ - 6)))

    cv2.rectangle(img, (x0, y0), (x0 + bw_, y0 + bh_), ORANGE, -1)
    cv2.rectangle(img, (x0, y0), (x0 + bw_, y0 + bh_), (20, 20, 20), 2)
    queue_text(items, s, (x0 + pad, y0 + pad), px, WHITE, stroke=max(1, px // 16))
    return render_text(img, items) if own else img


def draw_edge_trace(img, series, i, height=70, colour_pos=CYAN, colour_neg=YELLOW,
                    lang="zh", text=None):
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
    from . import coach

    own = text is None
    items = [] if own else text
    queue_text(items, coach.say("toe_short", lang), (x0 + 5, y0 + 4), 15, colour_pos)
    queue_text(items, coach.say("heel_short", lang), (x0 + 5, y0 + height - 19), 15,
               colour_neg)
    return render_text(img, items) if own else img


def write_video(path, frames, fps):
    if not frames:
        raise ValueError("no frames to write")
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    return path
