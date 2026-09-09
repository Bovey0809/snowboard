"""Level-aware coaching layer: target bands, fault thresholds and live cues.

`report.py` turns measurements into written findings. This module is the layer
underneath both it and the overlay renderer, and it exists for one reason: a
number is not advice. "angulation 39 deg" tells a learner nothing, because they
have no idea what 39 should be. Advice is the *gap* between the value and a
target, so the target has to be a first-class thing the renderer can draw.

Two audiences need different targets. A World Universiade snowboard-cross racer
measured +0.106 fore/aft on the validation clip and the recreational thresholds
called it a fault — but an aggressive forward stance is correct for racing (see
docs/design.md). So the tables are keyed by declared level rather than pretending
one set fits both.

**The bands are informed guesses, not validated norms.** They encode widely
taught technique and are kept here, in one place, so they can be recalibrated
against footage of riders whose level is known. The geometry that feeds them is
verified to about a degree; these numbers are not. Treat the measurements as
sound and the verdicts as provisional.
"""

import numpy as np

LEVELS = ("learner", "racer")

# Per-level display bands. `band` is the target the rider is being coached
# toward; `range` is only the gauge's drawn extent, chosen so a realistic value
# never pins to an end. Both are in the metric's own units.
_BANDS = {
    "learner": {
        "inclination": {"band": (5.0, 25.0), "range": (0.0, 60.0)},
        "angulation": {"band": (15.0, 45.0), "range": (0.0, 70.0)},
        "knee_flex": {"band": (35.0, 75.0), "range": (0.0, 100.0)},
        # Centred, because a learner's job is to stay stacked over the middle.
        "fore_aft": {"band": (-0.07, 0.07), "range": (-0.25, 0.25)},
        "hip_shoulder_sep": {"band": (-15.0, 15.0), "range": (-45.0, 45.0)},
    },
    "racer": {
        "inclination": {"band": (15.0, 50.0), "range": (0.0, 70.0)},
        "angulation": {"band": (25.0, 65.0), "range": (0.0, 80.0)},
        "knee_flex": {"band": (45.0, 90.0), "range": (0.0, 110.0)},
        # Deliberately asymmetric and nose-biased: this is the entry that stops
        # the racer false positive, and the reason bands beat a symmetric |x|
        # threshold. Sitting back is still a fault; driving the nose is not.
        "fore_aft": {"band": (-0.04, 0.16), "range": (-0.25, 0.30)},
        "hip_shoulder_sep": {"band": (-20.0, 20.0), "range": (-45.0, 45.0)},
    },
}

LANGS = ("zh", "en")

# Value formats are language-independent; only the words change. Degrees print
# as "°" rather than "deg" in both, since the symbol needs no translating and
# buys back panel width that Chinese glyphs want.
_FORMAT = {
    "inclination": ("{:.0f}", "°"),
    "angulation": ("{:.0f}", "°"),
    "knee_flex": ("{:.0f}", "°"),
    "fore_aft": ("{:+.2f}", ""),
    "hip_shoulder_sep": ("{:+.0f}", "°"),
}

# Cues are shouted from the side of the piste, so they are corrections, not
# descriptions: what to do, not what is wrong. The Chinese is written to the
# same brief — four to six characters, imperative, no hedging.
_TEXT = {
    "en": {
        "inclination": "inclination", "angulation": "angulation",
        "knee_flex": "knee flex", "fore_aft": "fore/aft",
        "hip_shoulder_sep": "hip-shoulder",
        "fore_aft_back": "GET FORWARD", "fore_aft_nose": "EASE OFF THE NOSE",
        "stiff_legs": "BEND YOUR KNEES", "too_low": "STAND TALLER",
        "knee_asymmetry": "EVEN UP YOUR LEGS",
        "excess_lean": "ANGULATE, DON'T LEAN",
        "flat_board": "COMMIT TO THE EDGE",
        "counter_rotation": "QUIET SHOULDERS",
        "learner": "learner", "racer": "racer",
        "toe": "TOE", "heel": "HEEL", "toe_short": "toe", "heel_short": "heel",
        "targets": "targets", "turn": "turn", "unknown": "unknown",
    },
    "zh": {
        "inclination": "内倾", "angulation": "屈体",
        "knee_flex": "屈膝", "fore_aft": "前后重心",
        "hip_shoulder_sep": "肩胯夹角",
        "fore_aft_back": "重心前移", "fore_aft_nose": "重心后移",
        "stiff_legs": "屈膝下沉", "too_low": "身体立起",
        "knee_asymmetry": "双腿均衡发力",
        "excess_lean": "屈髋立刃",
        "flat_board": "果断立刃",
        "counter_rotation": "肩别抢转",
        "learner": "初学", "racer": "竞技",
        "toe": "前刃", "heel": "后刃", "toe_short": "前刃", "heel_short": "后刃",
        "targets": "标准", "turn": "弯", "unknown": "未知",
    },
}


def say(key, lang="zh"):
    """One user-facing string. Falls back to English, then to the key itself."""
    return _TEXT.get(lang, _TEXT["en"]).get(key) or _TEXT["en"].get(key, key)

# Fault thresholds that vary with level. Everything derivable from a band is
# derived in `thresholds()` rather than repeated here, so the drawn target and
# the written finding can never drift apart.
_FAULTS = {
    "learner": {
        "knee_asymmetry": 10.0,       # deg between left and right knee flexion
        "low_edge_commitment": 0.06,  # peak |toe_heel| below this = riding flat
        "excess_lean": 0.55,          # inclination / (inclination+angulation)
        "edge_gap": 0.30,             # relative toe-vs-heel difference
        "inconsistency": 0.35,        # coefficient of variation of commitment
    },
    "racer": {
        "knee_asymmetry": 12.0,
        "low_edge_commitment": 0.10,  # a racer riding at 0.06 is not carving
        "excess_lean": 0.62,          # racers legitimately incline much harder
        "edge_gap": 0.22,             # and are held to a tighter symmetry
        "inconsistency": 0.25,
    },
}

# Level-independent: these describe what the clip *is*, not how well it is
# ridden, so a racer and a learner get the same test.
_COMMON = {
    "one_edge_fraction": 0.9,        # share of frames on a single edge
    "traverse_seconds": 5.0,         # an "edge" held longer than this is a traverse
    "traverse_fraction": 0.5,        # share of run spent traversing
    "sustained_seconds": 1.0,        # a fault held this long counts
    "implausible_commitment": 0.45,  # lateral offset no rider reaches while riding
}

# How long a cue must be the top candidate before it is shown, and how long it
# lingers once cleared. Without these the caption strobes: at 12 fps a value
# sitting on a band edge flips state several times a second and the overlay
# becomes unwatchable, which is a rendering fault masquerading as coaching.
CUE_HOLD_S = 0.3
CUE_LINGER_S = 1.0

# How far outside its band a value must sit before the caption speaks, in units
# of half the band width. This is the same boundary `_status` calls "bad", so
# the rule the viewer learns is simply: **the caption names the red bar.**
#
# It exists because merely leaving the band is not rare. Cueing on that fired on
# 82 of 82 frames of the race clip and 317 of 352 of the mixkit clip — a caption
# on every frame is wallpaper, and the whole point of picking one fault was to
# say less, not to say something different constantly. The gauges still show
# every drift continuously; the caption is reserved for what is clearly wrong.
CUE_EXCESS = 1.0

# ...but a mild fault *held* is not mild. On the clip that ends in a crash, hip
# and shoulder separation peaked at 25.1 deg and held above 20 deg for 1.6 s —
# never reaching the "bad" line at 30 deg, so the caption said nothing at all
# through the whole approach to the fall, on the one clip where a live cue most
# earns its keep. `report.py` already makes this argument for its written
# findings (`sustained_excursion`); without the same second path in, the overlay
# is quietest exactly when it should be loudest.
#
# So a value only into the amber zone speaks too, once it has stayed there for
# CUE_SUSTAIN_S. 0.33 is set by that clip: the wind-up clears 0.33 for 1.6 s but
# clears 0.5 for only 0.8 s, so a higher line misses it entirely.
CUE_SUSTAIN_EXCESS = 0.33
CUE_SUSTAIN_S = 1.0

# Say it, let them try it, say it again. A cue that has been up for
# CUE_MAX_SPELL_S goes quiet for CUE_REST_S before it may return.
#
# This is what makes the sustained path above affordable. Sensitivity and
# nagging are separate problems, and one threshold cannot control both: dropping
# the sustain line far enough to catch the developing wind-up took the median
# duty cycle over the six sample clips from 23% to 45%, worst case 68%. Capping
# the spell instead keeps the sensitivity and takes the worst case back to 49%,
# because a fault held for twenty seconds does not need twenty seconds of
# caption — the gauge is showing it the whole time regardless.
CUE_MAX_SPELL_S = 3.0
CUE_REST_S = 2.0

# Nothing flashes up for less than this. Within one caption spell the top-ranked
# fault can change, and the ranking is not stable enough to be worth showing at
# that rate: it produced 0.3-0.4 s fragments that flick to another message and
# back. A cue too brief to read is worse than silence, so short segments are
# dropped rather than padded — padding would just displace the cue after them.
CUE_MIN_SHOW_S = 1.0

# The causal hierarchy of snowboard teaching: balance, then pressure, then edge,
# then rotation. A rider who is in the back seat will also be counter-rotating,
# and coaching the rotation first is wasted breath. Lower tier wins, always —
# this is what turns six simultaneous findings into one thing to work on.
TIER_BALANCE, TIER_PRESSURE, TIER_EDGE, TIER_ROTATION = 1, 2, 3, 4


def bands(level="learner"):
    """Target bands and gauge ranges for a declared level."""
    if level not in _BANDS:
        raise ValueError(f"unknown level {level!r}; expected one of {LEVELS}")
    return _BANDS[level]


def thresholds(level="learner"):
    """Fault thresholds for `report.py`, with the band-derived ones filled in.

    `fore_aft_bias` and `stiff_legs` are not stored: they *are* the band edges,
    and storing them separately is how a drawn green zone ends up disagreeing
    with the sentence printed underneath it.
    """
    if level not in _FAULTS:
        raise ValueError(f"unknown level {level!r}; expected one of {LEVELS}")
    B = bands(level)
    fa_lo, fa_hi = B["fore_aft"]["band"]
    out = dict(_COMMON)
    out.update(_FAULTS[level])
    out["fore_aft_band"] = (fa_lo, fa_hi)
    # The sustained-excursion check works on |value|, so it needs a scalar. The
    # wider edge is the honest choice: a racer held at +0.16 is inside their
    # band and must not be flagged for staying there.
    out["fore_aft_bias"] = max(abs(fa_lo), abs(fa_hi))
    out["stiff_legs"] = B["knee_flex"]["band"][0]
    out["counter_rotation"] = B["hip_shoulder_sep"]["band"][1]
    return out


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def gauge_value(values, key):
    """The single number a gauge shows, from a frame's metric dict.

    Knee flexion is the only gauge that is not a metric in its own right: both
    legs are measured, but the rider is coached on how flexed they are overall,
    with the left/right gap handled as its own fault.
    """
    if key == "knee_flex":
        l, r = _f(values.get("knee_flex_l")), _f(values.get("knee_flex_r"))
        return np.nanmean([l, r]) if np.isfinite([l, r]).any() else np.nan
    return _f(values.get(key))


def _status(v, lo, hi):
    """Where a value sits relative to its band: good, warn or bad.

    The warn zone is half a band-width outside the band, so a rider drifting out
    sees amber before red rather than a binary that snaps at the edge.
    """
    if not np.isfinite(v):
        return "na"
    if lo <= v <= hi:
        return "good"
    margin = 0.5 * (hi - lo)
    return "warn" if (lo - margin) <= v <= (hi + margin) else "bad"


def gauge_rows(values, level="learner", stance_ok=True, lang="zh"):
    """The rows a gauge panel draws for one frame.

    Args:
        values: per-frame metric dict (the same keys `geometry.frame_metrics`
            produces).
        level: declared rider level; picks the band table.
        stance_ok: whether the board's nose direction is trustworthy. When it is
            not, fore/aft is returned disabled rather than mirrored — see
            `cue_stream` for why that matters more in a drawing than in prose.
        lang: language for the row labels and the "unknown" marker.

    Returns:
        list of dicts: label, value, text, band, range, status, enabled.
    """
    B = bands(level)
    rows = []
    for key in ("inclination", "angulation", "knee_flex", "fore_aft",
                "hip_shoulder_sep"):
        fmt, unit = _FORMAT[key]
        label = say(key, lang)
        v = gauge_value(values, key)
        enabled = stance_ok or key != "fore_aft"
        if not enabled:
            text, status = say("unknown", lang), "na"
        elif np.isfinite(v):
            text = fmt.format(v) + unit
            status = _status(v, *B[key]["band"])
        else:
            text, status = "--", "na"
        rows.append({"key": key, "label": label, "value": v, "text": text,
                     "band": B[key]["band"], "range": B[key]["range"],
                     "status": status, "enabled": enabled,
                     "bipolar": key in ("fore_aft", "hip_shoulder_sep")})
    return rows


def _candidates(v, T, B, stance_ok):
    """Every cue that applies to one frame, as (tier, excess, id).

    `excess` is how far past the band the value sits, normalised by half the
    band width, so faults measured in degrees and faults measured in body
    heights can be ranked against each other inside a tier.
    """
    out = []

    def over(value, lo, hi):
        if not np.isfinite(value):
            return None
        half = max(1e-6, 0.5 * (hi - lo))
        if value < lo:
            return -(lo - value) / half
        if value > hi:
            return (value - hi) / half
        return None

    # Tier 1 — balance. Suppressed outright when the nose direction is a guess:
    # "GET FORWARD" with the board reversed coaches the wrong leg, and a caption
    # burned into the video is far more persuasive than a footnote in a report.
    if stance_ok:
        fa = _f(v.get("fore_aft"))
        e = over(fa, *B["fore_aft"]["band"])
        if e is not None:
            out.append((TIER_BALANCE, abs(e), "fore_aft_back")
                       if e < 0 else
                       (TIER_BALANCE, abs(e), "fore_aft_nose"))

    # Tier 2 — pressure. Legs that are straight or unevenly loaded have nothing
    # left to give, so nothing above this tier can be fixed until they do.
    kf = gauge_value(v, "knee_flex")
    e = over(kf, *B["knee_flex"]["band"])
    if e is not None:
        out.append((TIER_PRESSURE, abs(e), "stiff_legs")
                   if e < 0 else
                   (TIER_PRESSURE, abs(e), "too_low"))
    kl, kr = _f(v.get("knee_flex_l")), _f(v.get("knee_flex_r"))
    if np.isfinite(kl) and np.isfinite(kr):
        gap = abs(kl - kr)
        if gap > T["knee_asymmetry"]:
            out.append((TIER_PRESSURE, gap / T["knee_asymmetry"] - 1.0,
                        "knee_asymmetry"))

    # Tier 3 — edge. Pure lean and no edge at all are different faults with
    # opposite fixes, so they are separate cues rather than one "edge" warning.
    incl, ang = _f(v.get("inclination")), _f(v.get("angulation"))
    if np.isfinite(incl) and np.isfinite(ang) and (incl + ang) > 1.0:
        ratio = incl / (incl + ang)
        if ratio > T["excess_lean"]:
            # Scaled so 1.0 lands at a ratio of about 0.70 for a learner, which
            # is lean with almost no angulation left to steer with.
            out.append((TIER_EDGE, (ratio - T["excess_lean"]) / 0.15,
                        "excess_lean"))
        else:
            # Not on an edge at all: both halves of the tipping short of target.
            # Averaging the two shortfalls keeps this on the same scale as every
            # other cue, so it can be ranked against them.
            si = over(incl, *B["inclination"]["band"])
            sa = over(ang, *B["angulation"]["band"])
            if si is not None and sa is not None and si < 0 and sa < 0:
                out.append((TIER_EDGE, (abs(si) + abs(sa)) / 2.0,
                            "flat_board"))

    # Tier 4 — rotation.
    sep = _f(v.get("hip_shoulder_sep"))
    e = over(sep, *B["hip_shoulder_sep"]["band"])
    if e is not None:
        out.append((TIER_ROTATION, abs(e), "counter_rotation"))

    return out


def _debounce(ids, fps, hold_s=CUE_HOLD_S, linger_s=CUE_LINGER_S):
    """Keep a cue only once it has held, then let it linger after it clears.

    Scans left to right, so a newly qualified cue overwrites the tail of the
    previous one's linger — the rider always sees the current problem, never a
    stale caption sitting on top of a fresh fault.
    """
    n = len(ids)
    out = [None] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ids[j + 1] == ids[i]:
            j += 1
        if ids[i] is not None and (j - i + 1) / fps >= hold_s:
            for t in range(i, min(n, j + 1 + int(round(linger_s * fps)))):
                out[t] = ids[i]
        i = j + 1
    return out


def _rest(ids, fps, max_spell_s=None, rest_s=None):
    """Break a long unbroken caption into spells separated by silence.

    This deliberately works on the *caption channel*, not on each cue id: a run
    of frames counts as one spell even if the top fault changes partway through.
    Resting per id looks equivalent and is not — when two faults alternate, each
    change restarts its own spell and nothing ever rests. That is how the race
    clip reached a caption on 98% of frames while every individual cue was
    inside its cap.
    """
    max_spell = max(1, int(round((max_spell_s or CUE_MAX_SPELL_S) * fps)))
    rest = max(1, int(round((rest_s or CUE_REST_S) * fps)))
    out = list(ids)
    n = len(out)
    i = 0
    while i < n:
        if out[i] is None:
            i += 1
            continue
        j = i
        while j + 1 < n and ids[j + 1] is not None:
            j += 1
        k = i
        while k <= j:
            k += max_spell                       # speak
            for f_i in range(k, min(j + 1, k + rest)):
                out[f_i] = None                  # then rest
            k += rest
        i = j + 1
    return out


def _drop_short(ids, fps, min_show_s=None):
    """Remove any cue segment too brief to read."""
    need = max(1, int(round((min_show_s or CUE_MIN_SHOW_S) * fps)))
    out = list(ids)
    n = len(out)
    i = 0
    while i < n:
        if ids[i] is None:
            i += 1
            continue
        j = i
        while j + 1 < n and ids[j + 1] == ids[i]:
            j += 1
        if (j - i + 1) < need:
            for f_i in range(i, j + 1):
                out[f_i] = None
        i = j + 1
    return out


def cue_stream(metrics, fps, level="learner", stance_ok=True, lang="zh"):
    """One coaching cue per frame, or None.

    At most one, always: six findings on a static report is an audit a rider can
    read at their own pace, but six captions flashing over a video is noise. The
    tier order decides which fault gets the frame.

    Args:
        metrics: dict of name -> per-frame array, all the same length (already
            smoothed).
        fps: frame rate of those series, for the hold and linger windows.
        level: declared rider level.
        stance_ok: False when the stance inference was too weak to trust, which
            disables every cue that depends on the sign of the board's long axis.
        lang: language the cue text is rendered in. The cue *id* is unchanged by
            it, so debouncing, resting and tallies stay language-independent.

    Returns:
        list, one entry per frame: None, or {"id", "text", "tier", "excess"}.
    """
    keys = list(metrics)
    if not keys:
        return []
    n = len(np.asarray(metrics[keys[0]]))
    T, B = thresholds(level), bands(level)

    # Pass 1: every candidate on every frame, ungated, keyed by cue id.
    per_frame = []
    for i in range(n):
        v = {k: np.asarray(metrics[k], dtype=float)[i] for k in keys}
        # A centre of mass this far off the board's centreline is not riding —
        # it is a fall, a sit-down or a stop that survived the clip trim. Coach
        # nothing over those frames rather than shouting at someone lying down.
        th = _f(v.get("toe_heel"))
        if np.isfinite(th) and abs(th) > T["implausible_commitment"]:
            per_frame.append({})
            continue
        per_frame.append({c[2]: c for c in _candidates(v, T, B, stance_ok)})

    # Pass 2: which ids may speak on which frames. Either the value is already
    # past the "bad" line, or it has sat in amber long enough to count as a
    # fault that is developing rather than a wobble.
    eligible = [set(cid for cid, c in f.items() if c[1] >= CUE_EXCESS)
                for f in per_frame]
    hold = max(1, int(round(CUE_SUSTAIN_S * fps)))
    for cid in {cid for f in per_frame for cid in f}:
        amber = [cid in f and f[cid][1] >= CUE_SUSTAIN_EXCESS for f in per_frame]
        i = 0
        while i < n:
            if not amber[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and amber[j + 1]:
                j += 1
            # Marks the whole window, not just the tail past the hold. The
            # hold decides *whether* this counts as a developing fault; once it
            # does, the interesting stretch is the whole wind-up, and this is
            # review footage, not live coaching — there is no reason to withhold
            # the caption over the very frames that show the fault growing.
            # On the crash clip that is the difference between a 0.7 s cue,
            # truncated by the end of the video, and the full 1.6 s.
            for f_i in range(i, j + 1):
                eligible[f_i].add(cid)
            i = j + 1

    # Rank by tier *after* gating, never before: a mild balance drift must not
    # shadow a genuinely bad rotation fault two tiers down.
    picks = [min((per_frame[i][cid] for cid in eligible[i]),
                 key=lambda c: (c[0], -c[1])) if eligible[i] else None
             for i in range(n)]

    kept = _drop_short(_rest(_debounce([p[2] if p else None for p in picks], fps),
                             fps), fps)
    by_id = {}
    for p in picks:
        if p:
            by_id.setdefault(p[2], p)
    out = []
    for i, cue_id in enumerate(kept):
        if cue_id is None:
            out.append(None)
            continue
        # Prefer this frame's own reading of the cue; fall back to any frame's,
        # so a cue still has text through its linger tail.
        p = picks[i] if picks[i] and picks[i][2] == cue_id else by_id[cue_id]
        out.append({"id": cue_id, "text": say(cue_id, lang), "tier": p[0],
                    "excess": round(float(p[1]), 3)})
    return out
