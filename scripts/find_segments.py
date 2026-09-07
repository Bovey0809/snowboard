"""Locate segments of a video worth analysing.

Broadcast footage is mostly interviews, graphics and wide shots. Turn analysis
needs a continuous stretch with one clearly-resolved rider, so this scans the
whole file cheaply and reports the best contiguous windows.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snowpose import config as CFG  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default=CFG.DETECTOR)
    ap.add_argument("--sample-fps", type=float, default=1.0)
    ap.add_argument("--min-height", type=float, default=150.0,
                    help="px; below this the mesh gets unreliable")
    ap.add_argument("--min-seconds", type=float, default=6.0)
    ap.add_argument("--out", default="segments.json")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / args.sample_fps)))
    print(f"{Path(args.video).name}: {total} frames @ {fps:.2f}fps "
          f"({total/fps/60:.1f} min), sampling every {step}")

    model = YOLO(args.weights)
    samples = []
    fi = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if fi % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                r = model.predict(frame, classes=[0], conf=0.4, imgsz=960, verbose=False,
                                  device=args.device, save=False,
                                  project="./out/yolo", name="seg",
                                  exist_ok=True)[0]
                hs = []
                if r.boxes is not None and len(r.boxes):
                    b = r.boxes.xyxy.cpu().numpy()
                    hs = sorted((b[:, 3] - b[:, 1]).tolist(), reverse=True)
                samples.append({"frame": fi, "t": fi / fps, "n": len(hs),
                                "h1": hs[0] if hs else 0.0,
                                "h2": hs[1] if len(hs) > 1 else 0.0})
            if len(samples) % 120 == 0 and samples:
                print(f"  ...{samples[-1]['t']/60:.1f} min", flush=True)
        fi += 1
    cap.release()

    good = [s["h1"] >= args.min_height for s in samples]
    runs, i = [], 0
    while i < len(good):
        if not good[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(good) and good[j + 1]:
            j += 1
        dur = samples[j]["t"] - samples[i]["t"]
        if dur >= args.min_seconds:
            win = samples[i:j + 1]
            runs.append({
                "start_s": round(samples[i]["t"], 2),
                "end_s": round(samples[j]["t"], 2),
                "duration_s": round(dur, 2),
                "median_h": round(float(np.median([s["h1"] for s in win])), 1),
                "max_h": round(max(s["h1"] for s in win), 1),
                "mean_n": round(float(np.mean([s["n"] for s in win])), 2),
                # a lone rider scores best: h2 much smaller than h1
                "solo_ratio": round(float(np.mean(
                    [(s["h2"] / s["h1"]) if s["h1"] else 1.0 for s in win])), 3),
            })
        i = j + 1

    runs.sort(key=lambda r: (-r["duration_s"], -r["median_h"]))
    Path(args.out).write_text(json.dumps(
        {"video": args.video, "fps": fps, "segments": runs}, indent=1))
    print(f"\n{len(runs)} segments >= {args.min_seconds}s with a rider >= {args.min_height}px:\n")
    print(f"{'start':>8} {'end':>8} {'dur':>6} {'med_h':>6} {'max_h':>6} {'riders':>7} {'solo':>6}")
    for r in runs[:20]:
        print(f"{r['start_s']:8.1f} {r['end_s']:8.1f} {r['duration_s']:6.1f} "
              f"{r['median_h']:6.1f} {r['max_h']:6.1f} {r['mean_n']:7.2f} {r['solo_ratio']:6.3f}")


if __name__ == "__main__":
    main()
