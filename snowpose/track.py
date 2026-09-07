"""Subject selection across frames.

Snowboard footage routinely has more than one rider in shot, and their boxes
cross. Picking the largest detection per frame silently hops between riders, so
detections are linked into tracks first and the analysis runs on one track.
A greedy IoU tracker is enough — riders move fast but smoothly, and we only need
identity within a single clip.
"""

from dataclasses import dataclass, field

import numpy as np


def iou(a, b):
    """IoU of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class Track:
    tid: int
    frames: list = field(default_factory=list)   # frame indices
    boxes: list = field(default_factory=list)    # xyxy per frame
    confs: list = field(default_factory=list)

    @property
    def length(self):
        return len(self.frames)

    @property
    def mean_height(self):
        return float(np.mean([b[3] - b[1] for b in self.boxes])) if self.boxes else 0.0

    def gap_to(self, frame_idx):
        return frame_idx - self.frames[-1]


def link(detections, iou_thresh=0.2, max_gap=8):
    """Link per-frame detections into tracks.

    Args:
        detections: list over frames; each entry a list of dicts with "xyxy" and "conf".
        iou_thresh: minimum IoU to continue a track.
        max_gap: frames a track may go unmatched before it is closed (occlusion,
            missed detection, motion blur).

    Returns:
        list[Track]
    """
    active, done, next_id = [], [], 0

    for fi, dets in enumerate(detections):
        boxes = [np.asarray(d["xyxy"], dtype=float) for d in dets]
        confs = [float(d.get("conf", 1.0)) for d in dets]

        # Greedy matching, best IoU first, one detection per track.
        pairs = sorted(
            ((iou(t.boxes[-1], b), ti, bi)
             for ti, t in enumerate(active) for bi, b in enumerate(boxes)),
            reverse=True,
        )
        used_t, used_b = set(), set()
        for score, ti, bi in pairs:
            if score < iou_thresh or ti in used_t or bi in used_b:
                continue
            t = active[ti]
            t.frames.append(fi)
            t.boxes.append(boxes[bi])
            t.confs.append(confs[bi])
            used_t.add(ti)
            used_b.add(bi)

        for bi, b in enumerate(boxes):
            if bi in used_b:
                continue
            t = Track(next_id)
            next_id += 1
            t.frames.append(fi)
            t.boxes.append(b)
            t.confs.append(confs[bi])
            active.append(t)

        still, closed = [], []
        for t in active:
            (still if t.gap_to(fi) <= max_gap else closed).append(t)
        active, _ = still, done.extend(closed)

    done.extend(active)
    return done


def primary(tracks, min_length=8):
    """The track most likely to be the rider being analysed.

    Longest track wins; among comparable lengths prefer the larger subject, since
    a bigger box means more pixels on the body and a better mesh.
    """
    usable = [t for t in tracks if t.length >= min_length] or list(tracks)
    if not usable:
        return None
    best_len = max(t.length for t in usable)
    contenders = [t for t in usable if t.length >= 0.8 * best_len]
    return max(contenders, key=lambda t: t.mean_height)


def interpolate(track, n_frames):
    """Fill a track's gaps so every frame in its span has a box.

    Returns dict frame_idx -> xyxy, spanning the track's first to last frame.
    """
    out, fr, bx = {}, track.frames, track.boxes
    for i in range(len(fr) - 1):
        f0, f1 = fr[i], fr[i + 1]
        b0, b1 = np.asarray(bx[i]), np.asarray(bx[i + 1])
        out[f0] = b0
        for f in range(f0 + 1, f1):
            w = (f - f0) / (f1 - f0)
            out[f] = b0 * (1 - w) + b1 * w
    out[fr[-1]] = np.asarray(bx[-1])
    return {f: b for f, b in out.items() if 0 <= f < n_frames}
