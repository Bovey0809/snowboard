"""Regenerate report.md from an existing run's metrics.csv and turns.json.

Inference is the expensive part (about a second a frame), so re-running a whole
clip just to change how findings are worded or thresholded is wasteful. This
recomputes the turn scoring and the report from artifacts already on disk.

Optionally re-signs fore/aft for a known stance, which is the one metric whose
sign depends on a per-run inference that can be wrong.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snowpose import report as R, turns as TU  # noqa: E402

SKIP = {"frame", "source_frame", "t_s"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="directory holding metrics.csv and turns.json")
    ap.add_argument("--stance", default=None, choices=["regular", "goofy"],
                    help="re-sign fore/aft for a known stance")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    rows = list(csv.DictReader(open(run / "metrics.csv")))
    if not rows:
        sys.exit("metrics.csv is empty")
    meta = {}
    tj = run / "turns.json"
    if tj.exists():
        meta = json.loads(tj.read_text())
    fps = float(meta.get("fps") or 12.0)

    metrics = {k: np.array([float(r[k]) for r in rows], dtype=float)
               for k in rows[0] if k not in SKIP}
    frames = [int(r["frame"]) for r in rows]

    notes = []
    orientation = meta.get("orientation", "")
    if args.stance:
        inferred_goofy = "goofy" in orientation
        want_goofy = args.stance == "goofy"
        if inferred_goofy != want_goofy:
            metrics["fore_aft"] = -metrics["fore_aft"]
            for k in ("hip_vs_board", "shoulder_vs_board"):
                if k in metrics:
                    metrics[k] = np.where(metrics[k] > 0, metrics[k] - 180.0,
                                          metrics[k] + 180.0)
            notes.append(f"Fore/aft re-signed for a stated {args.stance} stance; the "
                         f"run had inferred the opposite ({orientation}).")
        else:
            notes.append(f"Stated stance ({args.stance}) agrees with the run's "
                         f"inference, so signs are unchanged.")
    elif orientation:
        notes.append(f"Board nose direction resolved by {orientation}.")

    seg = TU.segment(metrics["toe_heel"], fps, frames=frames)
    scored = TU.score_turns(seg, metrics, fps)
    sym, cons = TU.symmetry(scored), TU.consistency(scored)
    if len(scored) < 6:
        notes.append(f"Only {len(scored)} segments; symmetry and consistency should "
                     f"not be trusted on so few.")
    notes.append("Metrics are in a frame fitted to the rider's feet, so they are "
                 "independent of camera angle but say nothing about the board's "
                 "angle to the actual slope.")

    times = [float(r["t_s"]) for r in rows] if "t_s" in rows[0] else None
    res = R.analyse(metrics, scored, sym, cons, notes=notes, fps=fps, times=times)
    md = R.to_markdown(res, title=args.title or run.name)
    (run / "report.md").write_text(md)
    (run / "turns.json").write_text(json.dumps(
        {"turns": scored, "symmetry": sym, "consistency": cons,
         "orientation": orientation, "fps": fps,
         "stance_applied": args.stance}, indent=1))
    print(md)


if __name__ == "__main__":
    main()
