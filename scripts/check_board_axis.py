"""Test the foot-derived board axis against a detected snowboard.

Every metric in this project rests on inferring the board's plane from the
rider's feet. This checks that inference against independent evidence: a COCO
segmentation mask of the snowboard gives a real 2D long axis, which is compared
against the foot-derived long axis projected through the model's own camera.

Agreement near 0 deg means the feet really do recover the board's orientation.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snowpose import board as B, geometry as G, mhr, overlay as O  # noqa: E402
from snowpose import config as CFG  # noqa: E402
from snowpose import smooth as S, track as T, turns as TU, video as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--det-weights", default=CFG.DETECTOR)
    ap.add_argument("--seg-weights", default=CFG.SEGMENTOR)
    ap.add_argument("--out", default="./out/board_check")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames, fps, _ = V.read_segment(args.video, args.start, args.end, args.fps)
    print(f"decoded {len(frames)} frames at {fps:.2f}fps")
    cuts = V.shot_cuts(frames)
    shots = V.split_shots(len(frames), cuts, min_len=max(8, int(fps)))

    from ultralytics import YOLO
    det = YOLO(args.det_weights)
    seg = YOLO(args.seg_weights)

    dets = []
    for f in frames:
        r = det.predict(f, classes=[0], conf=0.4, imgsz=1280, verbose=False,
                        device=args.device, save=False, project=str(out / "yolo"),
                        name="d", exist_ok=True)[0]
        row = []
        if r.boxes is not None and len(r.boxes):
            for xyxy, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                if xyxy[3] - xyxy[1] >= 120:
                    row.append({"xyxy": xyxy.tolist(), "conf": float(c)})
        dets.append(row)

    best = None
    for a, b in shots:
        tr = T.primary(T.link(dets[a:b]), min_length=max(6, int(fps * 0.5)))
        if tr is None:
            continue
        sc = tr.length * tr.mean_height
        if best is None or sc > best[0]:
            best = (sc, a, b, tr)
    if best is None:
        sys.exit("no usable track")
    _, a, b, tr = best
    boxes = {a + f: bx for f, bx in T.interpolate(tr, b - a).items()}
    print(f"shot [{a},{b}), track {tr.tid}: {len(boxes)} frames "
          f"at {tr.mean_height:.0f}px")

    from snowpose.model import BodyModel
    model = BodyModel()

    order = sorted(boxes)
    kpts, outs, idxs = [], [], []
    for fi in order:
        o = model.infer(cv2.cvtColor(frames[fi], cv2.COLOR_BGR2RGB), boxes[fi])
        if o is None:
            continue
        kpts.append(np.asarray(o["pred_keypoints_3d"], dtype=np.float64))
        outs.append(o)
        idxs.append(fi)
    K, _ = S.smooth_keypoints(np.stack(kpts), window=max(5, int(round(fps * 0.6)) | 1))
    sign, how = TU.orient_long_axis(K, "auto")
    print(f"orientation: {how}")

    rows, agreements, vis = [], [], []
    for i, fi in enumerate(idxs):
        res = G.frame_metrics(K[i], outs[i].get("pred_vertices"), flip_long=(sign < 0))
        if res is None:
            continue
        m, bf = res
        img = frames[fi]
        h, w = img.shape[:2]
        o = outs[i]

        # Project the foot-derived long axis into the image.
        p = O.project(np.stack([bf.origin, bf.origin + bf.long * 0.5]),
                      o["focal_length"], o["pred_cam_t"], w / 2.0, h / 2.0)
        long2d = p[1] - p[0]

        # The rider's feet in 2D, for associating the right board.
        k2 = np.asarray(o["pred_keypoints_2d"]).reshape(-1, 2)
        feet2d = k2[[mhr.L_ANKLE, mhr.R_ANKLE, mhr.L_TOE_BIG, mhr.R_TOE_BIG,
                     mhr.L_HEEL, mhr.R_HEEL]].mean(axis=0)

        boards = B.detect(seg, img, device=args.device, project=out / "yolo")
        bd, dist = B.nearest_to(boards, feet2d, max_dist=max(80.0, 0.6 * (
            boxes[fi][3] - boxes[fi][1])))
        row = {"frame": int(fi), "n_boards": len(boards),
               "assoc_dist_px": None if bd is None else round(dist, 1)}
        if bd is not None and bd["axis"] is not None:
            ag = B.axis_agreement(bd["axis"], long2d)
            row.update(agreement_deg=round(ag, 2),
                       elongation=round(bd["elongation"], 2),
                       board_conf=round(bd["conf"], 3))
            if np.isfinite(ag):
                agreements.append(ag)
            if len(vis) < 6:
                im = img.copy()
                O.draw_skeleton(im, k2)
                c = np.int32(bd["centroid"])
                ax = np.asarray(bd["axis"]) * 120
                cv2.line(im, tuple(c - np.int32(ax)), tuple(c + np.int32(ax)),
                         (0, 0, 255), 3, cv2.LINE_AA)
                pa, pb = np.int32(p[0]), np.int32(p[1])
                cv2.arrowedLine(im, tuple(pa), tuple(pb), (255, 220, 0), 3, cv2.LINE_AA)
                cv2.putText(im, f"agreement {ag:.1f} deg", (40, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
                vis.append(im)
        rows.append(row)

    ag = np.array(agreements, dtype=float)
    print(f"\n=== foot-derived axis vs detected board mask ===")
    print(f"  frames analysed        : {len(rows)}")
    print(f"  frames with a board    : {sum(1 for r in rows if r['assoc_dist_px'] is not None)}")
    print(f"  frames with a mask axis: {len(ag)}")
    if len(ag):
        print(f"  agreement (0=perfect, 90=perpendicular):")
        print(f"    median {np.median(ag):5.1f} deg   p25 {np.percentile(ag,25):5.1f}   "
              f"p75 {np.percentile(ag,75):5.1f}   max {ag.max():5.1f}")
        print(f"    within 15 deg: {(ag<15).mean():.0%}   within 30 deg: {(ag<30).mean():.0%}")
        print(f"  random-chance median would be ~45 deg")
    else:
        print("  no snowboard masks were associated — cannot validate this way")

    (out / "board_check.json").write_text(json.dumps(
        {"rows": rows, "median_agreement_deg": None if not len(ag) else float(np.median(ag)),
         "n_with_axis": int(len(ag)), "orientation": how}, indent=1))
    for i, im in enumerate(vis):
        cv2.imwrite(str(out / f"board_{i}.jpg"), im)
    print(f"\n[ok] {out}/board_check.json + {len(vis)} example frames")


if __name__ == "__main__":
    main()
