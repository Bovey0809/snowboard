"""Turn measurements into ranked, actionable feedback.

The thresholds are coaching heuristics, not validated norms. They encode
widely-taught snowboard technique (stay stacked over the board, drive the edge
with angulation rather than pure lean, keep the shoulders quiet, load both legs).

They now live in `coach.py`, keyed by declared rider level, because this module
is no longer their only consumer: the overlay draws the same bands as target
zones on its gauges. Two copies of a threshold is two things to recalibrate and
one chance for a green zone to contradict the sentence printed under it.
"""

import numpy as np

from . import coach


def _f(x):
    a = np.asarray(x, dtype=float)
    return a[np.isfinite(a)]


def sustained_excursion(values, threshold, fps, min_seconds=1.0):
    """The longest stretch where |value| stays at or above `threshold`.

    A run mean hides a fault that develops. Measured on a clip that ended in a
    crash, hip-shoulder separation grew from 10 deg to 24 deg over the final 1.5 s
    and held there — but the run mean was 13 deg, under the 15 deg threshold, so
    the report said "no faults" 0.2 s before the rider hit the ground.

    Returns (duration_s, start_idx, end_idx, peak) for the longest qualifying
    stretch, or None.
    """
    v = np.abs(np.asarray(values, dtype=float))
    over = np.isfinite(v) & (v >= threshold)
    best = None
    i = 0
    n = len(over)
    while i < n:
        if not over[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and over[j + 1]:
            j += 1
        dur = (j - i + 1) / fps
        if dur >= min_seconds and (best is None or dur > best[0]):
            best = (dur, i, j + 1, float(np.nanmax(v[i:j + 1])))
        i = j + 1
    return best


def _mean(x, default=np.nan):
    a = _f(x)
    return float(a.mean()) if len(a) else default


def analyse(metrics, scored_turns, sym, cons, notes=None, fps=None, times=None,
            level="learner", stance_ok=True):
    """Produce ranked findings.

    Args:
        metrics: dict name -> per-frame array (already smoothed).
        scored_turns: output of turns.score_turns.
        sym: output of turns.symmetry.
        cons: output of turns.consistency.
        fps: frames per second, needed to judge how long a fault was held.
        times: optional per-frame timestamps, so findings can be located in the clip.
        level: declared rider level, which picks the band table in `coach`.
        stance_ok: False when the stance inference was too weak to trust. Every
            fore/aft finding depends on the sign of the board's long axis, so
            they are suppressed rather than caveated — telling a rider to get
            forward when the nose is the tail coaches the wrong leg.

    Returns:
        dict with "findings" (ranked list) and "summary" numbers.
    """
    T = coach.thresholds(level)
    findings = []

    fa = _mean(metrics.get("fore_aft", []))
    incl = _mean(metrics.get("inclination", []))
    ang = _mean(metrics.get("angulation", []))
    kl = _mean(metrics.get("knee_flex_l", []))
    kr = _mean(metrics.get("knee_flex_r", []))
    sep = _f(metrics.get("hip_shoulder_sep", []))
    sep_mag = float(np.mean(np.abs(sep))) if len(sep) else np.nan

    # The target is a band, not a magnitude, and for a racer it is deliberately
    # asymmetric: an aggressive forward stance is correct technique. A symmetric
    # |fa| > threshold test is what flagged a Universiade snowboard-cross racer
    # at +0.106 for riding the way their discipline requires.
    fa_lo, fa_hi = T["fore_aft_band"]
    fa_flagged = stance_ok and np.isfinite(fa) and not (fa_lo <= fa <= fa_hi)
    if fa_flagged:
        back = fa < fa_lo
        where = "toward the tail" if back else "toward the nose"
        half = max(1e-6, 0.5 * (fa_hi - fa_lo))
        excess = (fa_lo - fa) if back else (fa - fa_hi)
        drill = ("Ride a few runs deliberately driving your front knee over your "
                 "front toes; on a groomer, try straight-lining with your weight "
                 "clearly forward so you learn what centred feels like.")
        if back:
            drill = ("Classic back-seat riding. On an easy pitch, ride with your "
                     "hands on your front knee for a run — it forces your mass "
                     "forward and you will feel the nose start to bite.")
        findings.append({
            "id": "fore_aft_bias",
            "severity": round(min(1.0, excess / half), 2),
            "title": f"Mass sits {where} rather than centred",
            "detail": (f"Centre of mass averages {fa:+.3f} of body height along the "
                       f"board, outside the {fa_lo:+.2f} to {fa_hi:+.2f} target for "
                       f"a {level}. "
                       + ("Sitting back makes the nose wash out and the board hard "
                          "to steer." if back else
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

    # A "turn" whose mass sits half a body-height off the board centreline is not a
    # turn. Across every clip measured, real turns peaked at 0.04-0.30; the 10 s a
    # rider spent lying in the snow after a crash scored 0.76 and was reported as an
    # 11 s heelside turn, quadrupling the run's apparent best. This guard rejects
    # such segments without claiming to know why they are not riding — see
    # docs/design.md for why detecting a fall properly is harder than it looks.
    bogus = [t for t in scored_turns
             if t.get("peak_commitment", 0) > T["implausible_commitment"]]
    if bogus:
        worst = max(bogus, key=lambda t: t["peak_commitment"])
        findings.append({
            "id": "not_riding",
            "severity": 1.0,
            "title": "Part of this clip is not riding",
            "detail": (f"{len(bogus)} segment(s) show the centre of mass more than "
                       f"{T['implausible_commitment']:.2f} body-heights off the board "
                       f"centreline, peaking at {worst['peak_commitment']:.2f} over "
                       f"{worst['duration_s']:.0f}s. No rider holds that while riding — "
                       f"it means a fall, a sit-down or a stop got included. Trim the "
                       f"clip to the riding and re-run; every number here is skewed "
                       f"until you do."),
            "drill": ("Not a technique fault — re-run with --start/--end bracketing "
                      "the riding only."),
        })

    # A run mean hides a fault that develops mid-run, which is exactly the shape of
    # a fault that ends in a crash. These checks fire on a fault *held* for a while
    # even when the average is clean.
    if fps:
        def when(i, j):
            if times is not None and len(times) > max(i, j - 1):
                return f" between {float(times[i]):.1f}s and {float(times[j-1]):.1f}s"
            return ""

        sep_series = metrics.get("hip_shoulder_sep", [])
        if len(_f(sep_series)) > 3 and not (np.isfinite(sep_mag) and sep_mag > T["counter_rotation"]):
            hit = sustained_excursion(sep_series, T["counter_rotation"], fps,
                                      T["sustained_seconds"])
            if hit:
                dur, i, j, peak = hit
                findings.append({
                    "id": "counter_rotation_developing",
                    "severity": round(min(1.0, peak / (2 * T["counter_rotation"])), 2),
                    "title": "Upper body winds up against the board during the run",
                    "detail": (f"Hips and shoulders stay more than "
                               f"{T['counter_rotation']:.0f} deg apart for {dur:.1f}s"
                               f"{when(i, j)}, peaking at {peak:.0f} deg, even though the "
                               f"run averages {sep_mag:.0f} deg. A fault that builds like "
                               f"this is how a caught edge starts: once the shoulders lead, "
                               f"the board follows late and then bites."),
                    "drill": ("Watch that stretch of the overlay and note what your lead "
                              "hand does. Keep it pointed down the board and turn from "
                              "the ankles and knees; if the wind-up returns, stop the run "
                              "and reset rather than riding through it."),
                })

        fa_series = metrics.get("fore_aft", [])
        if stance_ok and len(_f(fa_series)) > 3 and not fa_flagged:
            hit = sustained_excursion(fa_series, T["fore_aft_bias"], fps,
                                      T["sustained_seconds"])
            if hit:
                dur, i, j, peak = hit
                findings.append({
                    "id": "fore_aft_developing",
                    "severity": round(min(1.0, peak / (2 * T["fore_aft_bias"])), 2),
                    "title": "Balance drifts off centre for part of the run",
                    "detail": (f"Mass sits more than {T['fore_aft_bias']:.2f} body-heights "
                               f"off centre along the board for {dur:.1f}s{when(i, j)}, "
                               f"peaking at {peak:.2f}, while the run averages "
                               f"{fa:+.3f}. The average looks fine because the drift "
                               f"cancels out."),
                    "drill": ("Re-watch that stretch. Drifting fore/aft mid-run usually "
                              "means the terrain or speed changed and your stance did not "
                              "move with it."),
                })

    # A turn lasts 1-3 s. An "edge" held for 20 s is a traverse across the slope,
    # and counting it as one turn flatters the run badly: it inflates the turn
    # count and makes duration-based statistics meaningless. Reported before the
    # turn-quality findings for the same reason as single_edge.
    if scored_turns:
        durs = np.array([t.get("duration_s", np.nan) for t in scored_turns], dtype=float)
        durs = durs[np.isfinite(durs)]
        if len(durs):
            long_mask = durs > T["traverse_seconds"]
            total = float(durs.sum())
            trav = float(durs[long_mask].sum())
            share = trav / total if total > 0 else 0.0
            if long_mask.any() and share >= T["traverse_fraction"]:
                real = int((~long_mask).sum())
                findings.append({
                    "id": "traversing",
                    "severity": round(min(1.0, share), 2),
                    "title": "Long traverses, not linked turns",
                    "detail": (f"{int(long_mask.sum())} of {len(durs)} segments last "
                               f"longer than {T['traverse_seconds']:.0f}s — "
                               f"{trav:.0f}s of {total:.0f}s is spent holding one edge "
                               f"across the slope, against only {total-trav:.0f}s of "
                               f"actual turning across {real} genuine turn(s). A turn "
                               f"lasts 1-3s; anything longer is a traverse, so treat "
                               f"the turn count below as segments, not turns."),
                    "drill": ("Shorten the traverses deliberately: pick a rhythm and "
                              "change edge every 3 seconds, then every 2, even if the "
                              "turns are scruffy at first. Linking is a separate skill "
                              "from holding an edge, and it only comes from reps."),
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
        "level": level,
        "summary": {
            "level": level,
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
    level = result.get("level", "learner")
    B = coach.bands(level)
    out = [f"# {title}", ""]
    out.append(f"{s['turns']} turns over {s['frames']} analysed frames, judged "
               f"against **{level}** targets.")
    out.append("")
    # The target column is the same band the overlay draws as a green zone, so
    # the written report and the video always agree about what "good" was.
    out.append("| measure | value | target |")
    out.append("|---|---|---|")
    for k, label, bk in [
            ("mean_inclination_deg", "Mean inclination", "inclination"),
            ("mean_angulation_deg", "Mean angulation", "angulation"),
            ("mean_knee_flex_deg", "Mean knee flexion", "knee_flex"),
            ("mean_fore_aft", "Mean fore/aft balance", "fore_aft"),
            ("mean_abs_hip_shoulder_sep_deg", "Mean hip-shoulder separation",
             "hip_shoulder_sep")]:
        v = s.get(k)
        lo, hi = B[bk]["band"]
        fmt = "{:+.2f}" if bk == "fore_aft" else "{:.0f}"
        tgt = (f"|{fmt.format(hi)}| or less" if bk == "hip_shoulder_sep"
               else f"{fmt.format(lo)} to {fmt.format(hi)}")
        out.append(f"| {label} | {'--' if v is None else v} | {tgt} |")
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
