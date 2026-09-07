"""End-to-end: a video segment in, coaching feedback out.

    frames -> YOLO boxes -> track -> SAM 3D Body -> smooth -> board-frame metrics
           -> turn segmentation -> overlay video + CSV + markdown report
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snowpose import geometry as G, mhr, overlay as O, report as R  # noqa: E402
from snowpose import smooth as S, track as T, turns as TU, video as V  # noqa: E402

METRIC_KEYS = ["inclination", "angulation", "lower_leg_lean", "knee_flex_l",
               "knee_flex_r", "ankle_flex_l", "ankle_flex_r", "hip_shoulder_sep",
               "hip_vs_board", "shoulder_vs_board", "fore_aft", "toe_heel",
               "com_height", "planarity", "stance_width", "body_scale"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--fps", type=float, default=12.0, help="analysis frame rate")
    ap.add_argument("--stance", default="auto", choices=["auto", "regular", "goofy"])
    ap.add_argument("--weights", default="/home/rick/assets/yolo26x.pt")
    ap.add_argument("--out", default="/data/rick/sam3d/run_out")
    ap.add_argument("--device", default="0")
    ap.add_argument("--min-box-height", type=float, default=120.0)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    notes = []

    info = V.probe(args.video)
    print(f"source: {info['width']}x{info['height']} {info['fps']:.2f}fps "
          f"{info['duration_s']/60:.1f}min")
    frames, fps, src_idx = V.read_segment(args.video, args.start, args.end, args.fps)
    if not frames:
        sys.exit("no frames decoded")
    print(f"decoded {len(frames)} frames at {fps:.2f}fps "
          f"({args.start:.1f}s -> {args.end if args.end else info['duration_s']:.1f}s)")

    # Broadcast footage cuts constantly and a cut looks like a teleporting rider,
    # so tracking is confined to within a shot.
    cuts = V.shot_cuts(frames)
    shots = V.split_shots(len(frames), cuts, min_len=max(8, int(fps)))
    print(f"{len(cuts)} shot cuts -> {len(shots)} shots >= {max(8,int(fps))} frames")

    from ultralytics import YOLO
    det = YOLO(args.weights)

    dets = []
    for f in frames:
        r = det.predict(f, classes=[0], conf=0.4, imgsz=1280, verbose=False,
                        device=args.device, save=False,
                        project=str(out / "yolo"), name="det", exist_ok=True)[0]
        row = []
        if r.boxes is not None and len(r.boxes):
            for xyxy, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                if xyxy[3] - xyxy[1] >= args.min_box_height:
                    row.append({"xyxy": xyxy.tolist(), "conf": float(c)})
        dets.append(row)
    print(f"detections: {sum(len(d) for d in dets)} boxes, "
          f"{sum(1 for d in dets if d)}/{len(dets)} frames occupied")

    # Best (shot, track) pair: a long track on a big subject.
    best = None
    for a, b in shots:
        tracks = T.link(dets[a:b])
        tr = T.primary(tracks, min_length=max(6, int(fps * 0.5)))
        if tr is None:
            continue
        score = tr.length * tr.mean_height
        if best is None or score > best[0]:
            best = (score, a, b, tr)
    if best is None:
        sys.exit("no usable track in any shot")
    _, a, b, tr = best
    boxes = {a + f: bx for f, bx in T.interpolate(tr, b - a).items()}
    print(f"chosen shot [{a},{b}) ({(b-a)/fps:.1f}s), track {tr.tid}: "
          f"{tr.length} dets, mean height {tr.mean_height:.0f}px, "
          f"analysing {len(boxes)} frames")
    if (b - a) / fps < 3.0:
        notes.append(f"Longest usable shot was only {(b-a)/fps:.1f}s — too short for "
                     f"reliable turn statistics.")

    from snowpose.model import BodyModel
    model = BodyModel()

    order = sorted(boxes)
    kpts, verts, used = [], [], []
    for fi in order:
        rgb = cv2.cvtColor(frames[fi], cv2.COLOR_BGR2RGB)
        o = model.infer(rgb, boxes[fi])
        if o is None:
            continue
        kpts.append(np.asarray(o["pred_keypoints_3d"], dtype=np.float64))
        verts.append(np.asarray(o["pred_vertices"], dtype=np.float64))
        used.append((fi, o))
    if len(kpts) < 5:
        sys.exit(f"only {len(kpts)} frames inferred")
    K = np.stack(kpts)
    print(f"inferred {len(K)} frames")

    Ks, spikes = S.smooth_keypoints(K, window=max(5, int(round(fps * 0.6)) | 1), poly=2)
    print(f"smoothed joints (window {max(5,int(round(fps*0.6))|1)}), "
          f"{spikes} spike samples replaced")

    sign, how = TU.orient_long_axis(Ks, args.stance)
    print(f"board orientation: {how}")
    notes.append(f"Board nose direction resolved by {how}.")

    series = {k: [] for k in METRIC_KEYS}
    frames_kept, bfs = [], []
    for i in range(len(Ks)):
        res = G.frame_metrics(Ks[i], verts[i], flip_long=(sign < 0))
        if res is None:
            continue
        m, bf = res
        for k in METRIC_KEYS:
            series[k].append(m.get(k, np.nan))
        frames_kept.append(used[i][0])
        bfs.append((bf, used[i][1]))
    metrics = {k: np.asarray(v, dtype=float) for k, v in series.items()}
    print(f"metrics on {len(frames_kept)} frames")

    # Light second pass: the metrics are non-linear in the joints, so despike again.
    for k in metrics:
        metrics[k], _ = S.smooth_series(metrics[k],
                                        window=max(5, int(round(fps * 0.4)) | 1), poly=2)

    # frames_kept guards against inference dropping frames mid-track: without it
    # the series compacts and every turn duration is understated.
    seg = TU.segment(metrics["toe_heel"], fps, frames=frames_kept)
    scored = TU.score_turns(seg, metrics, fps)
    sym = TU.symmetry(scored)
    cons = TU.consistency(scored)
    print(f"turns: {len(scored)} "
          f"({sym.get('n_toe',0)} toeside / {sym.get('n_heel',0)} heelside)")
    for t in scored:
        print(f"  t={frames_kept[t['start']]/fps+args.start:6.2f}s {t['edge']:>4}side "
              f"{t['duration_s']:.2f}s commit {t['peak_commitment']:.3f}")

    if len(scored) < 4:
        notes.append(f"Only {len(scored)} turns were segmented; symmetry and "
                     f"consistency numbers need more turns to mean much.")
    notes.append("Metrics are expressed in a frame fitted to the rider's feet, so "
                 "they are independent of camera angle but say nothing about the "
                 "board's angle to the actual slope.")

    res = R.analyse(metrics, scored, sym, cons, notes=notes)
    md = R.to_markdown(res, title=f"{Path(args.video).stem} "
                                  f"[{args.start:.0f}s-{(args.end or info['duration_s']):.0f}s]")
    (out / "report.md").write_text(md)

    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "source_frame", "t_s"] + METRIC_KEYS)
        for i, fi in enumerate(frames_kept):
            w.writerow([i, src_idx[fi], round(args.start + fi / fps, 4)]
                       + [round(float(metrics[k][i]), 5) for k in METRIC_KEYS])

    (out / "turns.json").write_text(json.dumps(
        {"turns": scored, "symmetry": sym, "consistency": cons,
         "orientation": how, "fps": fps}, indent=1))

    if not args.no_video:
        turn_of = {}
        for idx, t in enumerate(scored, 1):
            for i in range(t["start"], t["end"]):
                turn_of[i] = dict(t, index=idx)
        canvas = []
        for i, fi in enumerate(frames_kept):
            img = frames[fi].copy()
            bf, o = bfs[i]
            h, w = img.shape[:2]
            k2 = np.asarray(o["pred_keypoints_2d"]).reshape(-1, 2)
            vp = O.project(np.asarray(o["pred_vertices"]), o["focal_length"],
                           o["pred_cam_t"], w / 2.0, h / 2.0)
            for x, y in vp[::16]:
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(img, (int(x), int(y)), 1, (0, 170, 255), -1)
            O.draw_skeleton(img, k2)
            O.draw_board_frame(img, bf, o["focal_length"], o["pred_cam_t"], w / 2.0, h / 2.0)
            O.draw_hud(img, {k: metrics[k][i] for k in METRIC_KEYS}, turn=turn_of.get(i))
            O.draw_edge_trace(img, metrics["toe_heel"], i)
            canvas.append(img)
        O.write_video(out / "overlay.mp4", canvas, fps)
        print(f"wrote {out}/overlay.mp4")

    print(f"\n[ok] {out}/report.md, metrics.csv, turns.json")
    print("\n" + "=" * 70)
    print(md)


if __name__ == "__main__":
    main()
