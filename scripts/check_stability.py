"""Measure how stable the board frame is on real footage.

Every metric rests on a frame fitted to the rider's feet, and the feet are the
joints most at risk from boots, bindings, spray and self-occlusion. This reports
frame-to-frame jitter, left/right leg swaps, and toe-vector collapse, so the
smoothing that follows is sized from measurement rather than guessed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snowpose import geometry as G, mhr, track as T  # noqa: E402
from snowpose.model import BodyModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="/data/rick/sam3d/smoke_frames")
    ap.add_argument("--dets", default="/data/rick/sam3d/smoke_dets.json")
    ap.add_argument("--out", default="/data/rick/sam3d/stability")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    paths = sorted(Path(args.frames).glob("*.jpg"))
    raw = json.loads(Path(args.dets).read_text())
    per_frame = [raw.get(p.name, []) for p in paths]

    tracks = T.link(per_frame)
    print(f"frames={len(paths)}  tracks={len(tracks)}  "
          f"lengths={sorted((t.length for t in tracks), reverse=True)[:6]}")
    tr = T.primary(tracks)
    if tr is None:
        sys.exit("no usable track")
    boxes = T.interpolate(tr, len(paths))
    print(f"primary track {tr.tid}: {tr.length} dets, mean box height "
          f"{tr.mean_height:.0f}px, analysing {len(boxes)} frames\n")

    model = BodyModel()

    rows, frames_done = [], []
    for fi in sorted(boxes):
        o = model.infer(paths[fi], boxes[fi])
        if o is None:
            continue
        k = np.asarray(o["pred_keypoints_3d"], dtype=np.float64)
        res = G.frame_metrics(k, o.get("pred_vertices"))
        if res is None:
            continue
        m, bf = res
        m["frame"] = fi
        m["ankle_axis"] = (k[mhr.L_ANKLE] - k[mhr.R_ANKLE]).tolist()
        m["toe_len_l"] = float(np.linalg.norm(k[mhr.L_TOE_BIG] - k[mhr.L_HEEL]))
        m["toe_len_r"] = float(np.linalg.norm(k[mhr.R_TOE_BIG] - k[mhr.R_HEEL]))
        m["basis"] = bf.basis.tolist()
        rows.append(m)
        frames_done.append(fi)

    if len(rows) < 2:
        sys.exit(f"only {len(rows)} frames produced metrics")

    def series(key):
        return np.array([r[key] for r in rows], dtype=float)

    def ang(u, v):
        u, v = np.asarray(u), np.asarray(v)
        c = np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)
        return float(np.degrees(np.arccos(c)))

    print(f"=== board-frame stability over {len(rows)} frames ===")
    names = ["long", "lat", "normal"]
    for ax in range(3):
        d = [ang(rows[i]["basis"][ax], rows[i + 1]["basis"][ax]) for i in range(len(rows) - 1)]
        d = np.array(d)
        print(f"  {names[ax]:<7} frame-to-frame rotation: median {np.median(d):5.2f}deg  "
              f"p90 {np.percentile(d, 90):6.2f}  max {d.max():6.2f}")

    # A left/right swap flips the ankle axis ~180deg between adjacent frames.
    aa = [np.array(r["ankle_axis"]) for r in rows]
    flips = [i for i in range(len(aa) - 1) if ang(aa[i], aa[i + 1]) > 90]
    print(f"\n  left/right leg swaps: {len(flips)}/{len(aa)-1} adjacent pairs"
          + (f" at frames {[rows[i]['frame'] for i in flips][:10]}" if flips else " (none)"))

    tl = np.concatenate([series("toe_len_l"), series("toe_len_r")])
    print(f"  heel->toe length: min {tl.min():.3f}m median {np.median(tl):.3f}m "
          f"max {tl.max():.3f}m  ({(tl < 0.05).sum()} of {len(tl)} below 5cm)")
    pl = series("planarity")
    print(f"  deck planarity (0=perfect): median {np.median(pl):.3f} p90 {np.percentile(pl,90):.3f}")
    sw = series("stance_width")
    print(f"  stance width: median {np.median(sw):.3f}m  std {sw.std():.3f}m")

    print(f"\n=== metric jitter (median |frame-to-frame delta|) ===")
    for key in ["inclination", "angulation", "knee_flex_l", "knee_flex_r",
                "ankle_flex_l", "ankle_flex_r", "hip_shoulder_sep",
                "fore_aft", "toe_heel"]:
        s = series(key)
        d = np.abs(np.diff(s))
        unit = "" if key in ("fore_aft", "toe_heel") else "deg"
        print(f"  {key:<18} range [{np.nanmin(s):+7.2f},{np.nanmax(s):+7.2f}]  "
              f"jitter {np.nanmedian(d):6.3f}{unit}  p90 {np.nanpercentile(d,90):6.3f}")

    np.savez_compressed(out / "stability.npz",
                        **{k: series(k) for k in rows[0] if k not in ("basis", "ankle_axis")})
    (out / "stability.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k not in ("basis",)} for r in rows], indent=1))
    print(f"\n[ok] wrote {out}/stability.json")


if __name__ == "__main__":
    main()
