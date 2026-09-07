"""Turn measurements into ranked, actionable feedback.

The thresholds below are coaching heuristics, not validated norms. They encode
widely-taught snowboard technique (stay stacked over the board, drive the edge
with angulation rather than pure lean, keep the shoulders quiet, load both legs)
and they are deliberately kept in one place so they can be recalibrated once
there is footage of a known-good rider to compare against.
"""

import numpy as np

# name -> (threshold, severity weight). Tuned to flag only clear faults.
THRESHOLDS = {
    "fore_aft_bias": 0.07,        # |mean COM along board| / body scale
    "knee_asymmetry": 10.0,       # deg between left and right knee flexion
    "low_edge_commitment": 0.06,  # peak |toe_heel| below this = riding flat
    "excess_lean": 0.55,          # inclination / (inclination+angulation) above this
    "counter_rotation": 15.0,     # deg hip-to-shoulder separation
    "stiff_legs": 35.0,           # deg mean knee flexion below this = riding tall
    "edge_gap": 0.30,             # relative toe-vs-heel difference
    "inconsistency": 0.35,        # coefficient of variation of turn commitment
    "one_edge_fraction": 0.9,     # share of frames on a single edge = not turning
}


def _f(x):
    a = np.asarray(x, dtype=float)
    return a[np.isfinite(a)]


def _mean(x, default=np.nan):
    a = _f(x)
    return float(a.mean()) if len(a) else default


def analyse(metrics, scored_turns, sym, cons, notes=None):
    """Produce ranked findings.

    Args:
        metrics: dict name -> per-frame array (already smoothed).
        scored_turns: output of turns.score_turns.
        sym: output of turns.symmetry.
        cons: output of turns.consistency.

    Returns:
        dict with "findings" (ranked list) and "summary" numbers.
    """
    T = THRESHOLDS
    findings = []

    fa = _mean(metrics.get("fore_aft", []))
    incl = _mean(metrics.get("inclination", []))
    ang = _mean(metrics.get("angulation", []))
    kl = _mean(metrics.get("knee_flex_l", []))
    kr = _mean(metrics.get("knee_flex_r", []))
    sep = _f(metrics.get("hip_shoulder_sep", []))
    sep_mag = float(np.mean(np.abs(sep))) if len(sep) else np.nan

    if np.isfinite(fa) and abs(fa) > T["fore_aft_bias"]:
        where = "toward the nose" if fa > 0 else "toward the tail"
        drill = ("Ride a few runs deliberately driving your front knee over your "
                 "front toes; on a groomer, try straight-lining with your weight "
                 "clearly forward so you learn what centred feels like.")
        if fa < 0:
            drill = ("Classic back-seat riding. On an easy pitch, ride with your "
                     "hands on your front knee for a run — it forces your mass "
                     "forward and you will feel the nose start to bite.")
        findings.append({
            "id": "fore_aft_bias",
            "severity": round(min(1.0, abs(fa) / (2 * T["fore_aft_bias"])), 2),
            "title": f"Mass sits {where} rather than centred",
            "detail": (f"Centre of mass averages {fa:+.3f} of body height along the "
                       f"board (centred would be near 0.00). "
                       + ("Sitting back makes the nose wash out and the board hard "
                          "to steer." if fa < 0 else
                          "Too far forward loads the nose and makes the tail break away.")),
            "drill": drill,
        })

    if np.isfinite(kl) and np.isfinite(kr) and abs(kl - kr) > T["knee_asymmetry"]:
        stiffer = "front" if kl < kr else "rear"
        findings.append({
            "id": "knee_asymmetry",
            "severity": round(min(1.0, abs(kl - kr) / (2 * T["knee_asymmetry"])), 2),
            "title": f"Uneven leg flexion — {stiffer} leg is doing less",
            "detail": (f"Knee flexion averages {kl:.0f} deg left and {kr:.0f} deg "
                       f"right, a {abs(kl-kr):.0f} deg gap. One leg absorbing most "
                       f"of the terrain costs you both edge hold and control."),
            "drill": ("Ride a pitch you find easy and consciously match the bend in "
                      "both knees through the whole turn. Slow, exaggerated "
                      "flex-and-extend on a green run builds the pattern."),
        })

    if np.isfinite(incl) and np.isfinite(ang) and (incl + ang) > 1:
        lean_ratio = incl / (incl + ang)
        if lean_ratio > T["excess_lean"]:
            findings.append({
                "id": "excess_lean",
                "severity": round(min(1.0, (lean_ratio - T["excess_lean"]) / 0.3), 2),
                "title": "Leaning the whole body instead of angulating",
                "detail": (f"Inclination averages {incl:.0f} deg against only "
                           f"{ang:.0f} deg of angulation ({lean_ratio:.0%} of the "
                           f"total is pure lean). Leaning tips you in but gives you "
                           f"nothing left to adjust with, so the edge either holds "
                           f"or lets go all at once."),
                "drill": ("Work on bending at the waist and knees to tip the board "
                          "while keeping your head and shoulders more upright — "
                          "'knees in, shoulders level'. Garlands across the fall "
                          "line are the standard drill."),
            })

    if np.isfinite(sep_mag) and sep_mag > T["counter_rotation"]:
        findings.append({
            "id": "counter_rotation",
            "severity": round(min(1.0, sep_mag / (2 * T["counter_rotation"])), 2),
            "title": "Upper body twisting against the board",
            "detail": (f"Hips and shoulders sit {sep_mag:.0f} deg apart on average. "
                       f"Steering with the shoulders instead of the feet makes turns "
                       f"late and skiddy."),
            "drill": ("Ride with your lead hand pointed down the board and keep it "
                      "there through the turn; initiate with your ankles and knees. "
                      "Holding a pole across your hips for a run makes the twist "
                      "obvious."),
        })

    kmean = np.nanmean([kl, kr])
    if np.isfinite(kmean) and kmean < T["stiff_legs"]:
        findings.append({
            "id": "stiff_legs",
            "severity": round(min(1.0, (T["stiff_legs"] - kmean) / T["stiff_legs"]), 2),
            "title": "Riding too tall — legs barely flexed",
            "detail": (f"Mean knee flexion is only {kmean:.0f} deg. Straight legs "
                       f"leave no travel to absorb terrain or to build pressure."),
            "drill": ("Practise a low, athletic stance: flex ankles, knees and hips "
                      "together and hold it for a whole run so it stops feeling odd."),
        })

    # Riding one edge the whole way down is a descent, not linked turns, and it is
    # the single most useful thing to say about such a run. Checked before the
    # turn-quality findings, which are meaningless without edge changes.
    th = _f(metrics.get("toe_heel", []))
    if len(th) > 10:
        toe_share = float((th > 0).mean())
        one_edge = max(toe_share, 1.0 - toe_share)
        if one_edge >= T["one_edge_fraction"]:
            edge = "toe" if toe_share > 0.5 else "heel"
            other = "heel" if edge == "toe" else "toe"
            findings.append({
                "id": "single_edge",
                "severity": 1.0,
                "title": f"The whole run is on the {edge} edge — no edge changes",
                "detail": (f"{one_edge:.0%} of frames have the mass on the {edge} "
                           f"side of the board. That is a {edge}side descent — "
                           f"sideslipping or skidding down the fall line — rather "
                           f"than linked turns, so nothing below about turn quality "
                           f"means much."),
                "drill": (f"This is the standard plateau. Work on committing to "
                          f"{other}side: {other}side traverses across the slope, "
                          f"then {other}side garlands, then link one {edge}side to "
                          f"one {other}side turn at a time on gentle pitch."),
            })

    if scored_turns:
        peaks = _f([t.get("peak_commitment", np.nan) for t in scored_turns])
        if len(peaks) and peaks.mean() < T["low_edge_commitment"]:
            findings.append({
                "id": "low_edge_commitment",
                "severity": round(min(1.0, 1 - peaks.mean() / T["low_edge_commitment"]), 2),
                "title": "Turns are skidded rather than carved",
                "detail": (f"Peak cross-board mass transfer averages only "
                           f"{peaks.mean():.3f} of body height. The board is being "
                           f"steered flat instead of set on an edge."),
                "drill": ("Ride slow, deliberate traverses on one edge only until "
                          "the board tracks a clean line, then link them."),
            })

    if sym.get("comparable"):
        tv = sym["peak_commitment"]["toe"]
        hv = sym["peak_commitment"]["heel"]
        denom = max(abs(tv), abs(hv), 1e-6)
        gap = abs(tv - hv) / denom
        if gap > T["edge_gap"]:
            weak = "toeside" if abs(tv) < abs(hv) else "heelside"
            findings.append({
                "id": "edge_gap",
                "severity": round(min(1.0, gap / (2 * T["edge_gap"])), 2),
                "title": f"{weak.capitalize()} is your weaker edge",
                "detail": (f"Edge commitment reaches {tv:.3f} toeside against "
                           f"{hv:.3f} heelside, a {gap:.0%} difference across "
                           f"{sym['n_toe']} toeside and {sym['n_heel']} heelside turns."),
                "drill": (f"Spend a run doing {weak} traverses and {weak}-only "
                          f"garlands. Almost everyone is lopsided here; deliberate "
                          f"single-edge mileage is the fix."),
            })

    cv = cons.get("cv")
    if cv is not None and cv > T["inconsistency"]:
        findings.append({
            "id": "inconsistency",
            "severity": round(min(1.0, cv / (2 * T["inconsistency"])), 2),
            "title": "Turn-to-turn consistency is low",
            "detail": (f"Edge commitment varies {cv:.0%} between turns across "
                       f"{cons.get('n')} turns. Consistency, not peak angle, is what "
                       f"separates riders at this level."),
            "drill": ("Pick a pitch and count a rhythm out loud, making every turn "
                      "the same size and duration. Rhythm before amplitude."),
        })

    findings.sort(key=lambda f: -f["severity"])
    return {
        "findings": findings,
        "summary": {
            "frames": int(len(_f(metrics.get("inclination", [])))),
            "turns": len(scored_turns),
            "mean_inclination_deg": None if not np.isfinite(incl) else round(incl, 1),
            "mean_angulation_deg": None if not np.isfinite(ang) else round(ang, 1),
            "mean_knee_flex_deg": None if not np.isfinite(kmean) else round(float(kmean), 1),
            "mean_fore_aft": None if not np.isfinite(fa) else round(fa, 3),
            "mean_abs_hip_shoulder_sep_deg": None if not np.isfinite(sep_mag) else round(sep_mag, 1),
        },
        "notes": notes or [],
    }


def to_markdown(result, title="Run analysis"):
    s = result["summary"]
    out = [f"# {title}", ""]
    out.append(f"{s['turns']} turns over {s['frames']} analysed frames.")
    out.append("")
    out.append("| measure | value |")
    out.append("|---|---|")
    for k, label in [("mean_inclination_deg", "Mean inclination"),
                     ("mean_angulation_deg", "Mean angulation"),
                     ("mean_knee_flex_deg", "Mean knee flexion"),
                     ("mean_fore_aft", "Mean fore/aft balance"),
                     ("mean_abs_hip_shoulder_sep_deg", "Mean hip-shoulder separation")]:
        v = s.get(k)
        out.append(f"| {label} | {'--' if v is None else v} |")
    out.append("")

    if not result["findings"]:
        out.append("No faults crossed the flagging thresholds on this run.")
    else:
        out.append("## What to work on")
        out.append("")
        for i, f in enumerate(result["findings"], 1):
            out.append(f"### {i}. {f['title']}")
            out.append(f"*severity {f['severity']:.2f}*")
            out.append("")
            out.append(f["detail"])
            out.append("")
            out.append(f"**Drill.** {f['drill']}")
            out.append("")

    if result.get("notes"):
        out.append("## Caveats")
        out.append("")
        for n in result["notes"]:
            out.append(f"- {n}")
        out.append("")
    return "\n".join(out)
