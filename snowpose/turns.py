"""Turn segmentation and per-turn scoring.

The rider's mass crosses the board from one edge to the other at every edge
change, so `toe_heel` (centre of mass across the board, + toward the toe edge)
changes sign once per turn. Segmenting on its zero crossings needs no world
frame and no board detection.
"""

import numpy as np

from . import mhr


def orient_long_axis(kpts, stance="auto"):
    """Decide which end of the board's long axis is the nose.

    The axis is built from `ankle_L - ankle_R`, so its sign is arbitrary; every
    fore/aft number is meaningless until it is pinned. This is a per-run
    decision, never per frame.

    Args:
        kpts: (T, 70, 3) joint sequence.
        stance: "regular" (left foot forward), "goofy" (right foot forward), or
            "auto" to infer it from where the rider is looking — riders look
            toward the nose of the board, so the head leads the hips along it.

    Returns:
        (sign, how) where multiplying the axis `ankle_L - ankle_R` by `sign`
        makes it point toward the nose, and `how` records the basis of the call.
    """
    if stance == "regular":
        return 1.0, "stance=regular (left foot forward)"
    if stance == "goofy":
        return -1.0, "stance=goofy (right foot forward)"

    K = np.asarray(kpts, dtype=float)
    votes = []
    for k in K:
        axis = k[mhr.L_ANKLE] - k[mhr.R_ANKLE]
        n = np.linalg.norm(axis)
        if n < 1e-6:
            continue
        axis = axis / n
        hip_mid = (k[mhr.L_HIP] + k[mhr.R_HIP]) / 2.0
        votes.append(float(np.dot(k[mhr.NOSE] - hip_mid, axis)))
    if not votes:
        return 1.0, "auto failed, defaulted to regular"
    lead = float(np.median(votes))
    sign = 1.0 if lead >= 0 else -1.0
    conf = abs(lead) / (np.std(votes) + 1e-6)
    return sign, (f"auto: head leads hips by {lead:+.3f}m along the axis "
                  f"(confidence {conf:.1f}); inferred "
                  f"{'regular' if sign > 0 else 'goofy'}")


def segment(toe_heel, fps, min_duration=0.35, min_amplitude=0.04):
    """Split a run into turns at edge changes.

    Args:
        toe_heel: smoothed per-frame COM across the board, + toward the toe edge.
        fps: frames per second of the series.
        min_duration: turns shorter than this are noise, not turns.
        min_amplitude: the signal must reach this far from zero inside a turn for
            the edge to count as actually engaged. Hysteresis on amplitude is what
            stops a rider coasting flat-based from registering dozens of turns.

    Returns:
        list of dicts with start/end frame, edge, duration and peak commitment.
    """
    x = np.asarray(toe_heel, dtype=float)
    n = len(x)
    if n < 3:
        return []

    # Candidate boundaries: sign changes of the (already smoothed) signal.
    sign = np.sign(x)
    sign[sign == 0] = 1
    crossings = [0] + [i for i in range(1, n) if sign[i] != sign[i - 1]] + [n]

    turns = []
    for a, b in zip(crossings[:-1], crossings[1:]):
        seg = x[a:b]
        if not len(seg):
            continue
        peak = float(np.max(np.abs(seg)))
        dur = (b - a) / fps
        if dur < min_duration or peak < min_amplitude:
            continue
        edge = "toe" if np.median(seg) > 0 else "heel"
        turns.append({
            "start": int(a), "end": int(b), "duration_s": round(dur, 3),
            "edge": edge, "peak_commitment": round(peak, 4),
            "peak_frame": int(a + int(np.argmax(np.abs(seg)))),
        })
    return turns


def score_turns(turns, metrics, fps):
    """Attach per-turn summaries of the frame metrics.

    `metrics` is a dict of name -> per-frame array, all the same length.
    """
    out = []
    for t in turns:
        a, b = t["start"], t["end"]
        row = dict(t)
        for key, arr in metrics.items():
            seg = np.asarray(arr, dtype=float)[a:b]
            seg = seg[np.isfinite(seg)]
            if not len(seg):
                continue
            row[f"{key}_mean"] = round(float(np.mean(seg)), 3)
            row[f"{key}_peak"] = round(float(np.max(np.abs(seg))), 3)
        out.append(row)
    return out


def symmetry(scored):
    """Compare toeside against heelside turns.

    Riders are almost always stronger on one edge; the gap is the single most
    actionable thing a run tells you.
    """
    toe = [t for t in scored if t["edge"] == "toe"]
    heel = [t for t in scored if t["edge"] == "heel"]
    if not toe or not heel:
        return {"comparable": False, "n_toe": len(toe), "n_heel": len(heel)}

    def m(rows, key):
        vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else np.nan

    keys = ["peak_commitment", "inclination_peak", "angulation_peak",
            "knee_flex_l_mean", "knee_flex_r_mean", "duration_s"]
    out = {"comparable": True, "n_toe": len(toe), "n_heel": len(heel)}
    for k in keys:
        tv, hv = m(toe, k), m(heel, k)
        out[k] = {"toe": round(tv, 3), "heel": round(hv, 3),
                  "delta": round(tv - hv, 3)}
    return out


def consistency(scored, key="peak_commitment"):
    """How repeatable the turns are. Low spread is the mark of a good rider."""
    vals = np.array([t[key] for t in scored if key in t], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        return {"n": int(len(vals))}
    return {"n": int(len(vals)), "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
            "cv": round(float(vals.std() / abs(vals.mean())), 3) if vals.mean() else None}
