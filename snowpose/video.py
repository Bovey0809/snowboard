"""Video decode helpers.

cv2 only — the GPU boxes have no ffmpeg binary and no sudo to install one.
"""

import cv2
import numpy as np


def probe(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    info["duration_s"] = info["frames"] / info["fps"] if info["fps"] else 0.0
    return info


def read_segment(path, start_s=0.0, end_s=None, target_fps=None):
    """Decode [start_s, end_s) as BGR frames, optionally subsampled.

    Metrics are smoothed derivatives, so ~12 fps carries the signal at a third of
    the inference cost of 25 fps. Returns (frames_bgr, sample_fps, source_indices).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = 1 if not target_fps else max(1, int(round(src_fps / target_fps)))

    first = int(round(start_s * src_fps))
    last = int(round(end_s * src_fps)) if end_s is not None else None
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    frames, idxs, i = [], [], first
    while True:
        if last is not None and i >= last:
            break
        ok = cap.grab()
        if not ok:
            break
        if (i - first) % step == 0:
            ok, f = cap.retrieve()
            if ok:
                frames.append(f)
                idxs.append(i)
        i += 1
    cap.release()
    return frames, src_fps / step, idxs


def shot_cuts(frames, threshold=0.45):
    """Frame indices where the shot changes.

    Broadcast footage cuts constantly, and a cut looks like a teleporting rider to
    any tracker. Compared on coarse grayscale histograms, which is cheap and cares
    about content rather than motion.
    """
    hists = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        h = cv2.calcHist([g], [0], None, [32], [0, 256]).flatten()
        s = h.sum()
        hists.append(h / s if s else h)
    cuts = []
    for i in range(1, len(hists)):
        d = 0.5 * np.abs(hists[i] - hists[i - 1]).sum()
        if d > threshold:
            cuts.append(i)
    return cuts


def split_shots(n_frames, cuts, min_len=8):
    """Turn cut positions into [start, end) shot spans."""
    bounds = [0] + list(cuts) + [n_frames]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= min_len]
