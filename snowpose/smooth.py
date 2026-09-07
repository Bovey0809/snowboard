"""Temporal smoothing.

SAM 3D Body runs per frame with no temporal model, so the recovered joints jitter.
Measured on real snowboard-cross footage (36 tracked frames, 25 fps): the board
axes rotate a median 5.5 deg between adjacent frames with a 64 deg outlier, and
knee flexion moves 3.5-4.9 deg. Turns last 1-2 s (25-50 frames), so a short
window removes most of the noise without touching the signal.

Order matters: reject outliers, then smooth the joints, then derive metrics.
Smoothing the metrics instead would let one bad frame corrupt a whole window
through the non-linear angle computations.
"""

import numpy as np


def hampel(x, window=7, n_sigma=3.0):
    """Replace spikes with the local median.

    Returns (cleaned, mask_of_replaced). Uses the median absolute deviation, so a
    single wild frame cannot drag the threshold out with it.
    """
    x = np.asarray(x, dtype=float).copy()
    n = len(x)
    if n < 3:
        return x, np.zeros(n, dtype=bool)
    half = max(1, window // 2)
    bad = np.zeros(n, dtype=bool)
    med = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = x[lo:hi]
        seg = seg[np.isfinite(seg)]
        if len(seg) == 0:
            med[i] = x[i]
            continue
        med[i] = np.median(seg)
        mad = np.median(np.abs(seg - med[i]))
        sigma = 1.4826 * mad
        if sigma > 0 and np.isfinite(x[i]) and abs(x[i] - med[i]) > n_sigma * sigma:
            bad[i] = True
    x[bad] = med[bad]
    x[~np.isfinite(x)] = med[~np.isfinite(x)]
    return x, bad


def savgol(x, window=9, poly=2):
    """Savitzky-Golay smoothing that preserves local curvature.

    Falls back to a shorter window on short series. Implemented with a local
    polynomial fit so scipy is not required.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return x.copy()
    w = min(window, n if n % 2 else n - 1)
    if w < poly + 2:
        w = min(n if n % 2 else n - 1, poly + 3)
    if w < 3:
        return x.copy()
    half = w // 2
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        t = np.arange(lo, hi) - i
        seg = x[lo:hi]
        ok = np.isfinite(seg)
        if ok.sum() <= poly:
            out[i] = np.nanmean(seg) if ok.any() else np.nan
            continue
        deg = min(poly, ok.sum() - 1)
        c = np.polyfit(t[ok], seg[ok], deg)
        out[i] = np.polyval(c, 0.0)
    return out


def smooth_series(x, window=9, poly=2, hampel_window=7, n_sigma=3.0):
    """Despike then smooth a 1-D series."""
    clean, bad = hampel(x, hampel_window, n_sigma)
    return savgol(clean, window, poly), bad


def smooth_keypoints(kpts, window=9, poly=2, n_sigma=3.0):
    """Smooth a (T, J, 3) joint sequence independently per joint and axis.

    Returns (smoothed, n_spikes_replaced).
    """
    K = np.asarray(kpts, dtype=float)
    T, J, C = K.shape
    out = np.empty_like(K)
    spikes = 0
    for j in range(J):
        for c in range(C):
            s, bad = smooth_series(K[:, j, c], window, poly, n_sigma=n_sigma)
            out[:, j, c] = s
            spikes += int(bad.sum())
    return out, spikes


def fill_gaps(values, frames, all_frames):
    """Linearly interpolate a per-frame series onto a contiguous frame range."""
    v = np.asarray(values, dtype=float)
    return np.interp(np.asarray(all_frames, dtype=float),
                     np.asarray(frames, dtype=float), v)
