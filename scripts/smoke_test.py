"""Validate SAM 3D Body on real snowboard frames before building metrics on top.

Prints the shape/range of every output field so coordinate conventions (units,
axis order, root-relative vs camera-space) come from data rather than
assumption, and draws a cv2 overlay. Deliberately avoids pyrender/OpenGL: the
Shenzhen boxes have no libEGL and no sudo, so the mesh is projected with the
model's own predicted camera and drawn as points.
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
import torch

# Joint indices from sam_3d_body/metadata/mhr70.py
JOINTS = {
    0: "nose", 5: "l_shoulder", 6: "r_shoulder", 7: "l_elbow", 8: "r_elbow",
    9: "l_hip", 10: "r_hip", 11: "l_knee", 12: "r_knee", 13: "l_ankle",
    14: "r_ankle", 15: "l_toe_big", 17: "l_heel", 18: "r_toe_big",
    20: "r_heel", 41: "r_wrist", 62: "l_wrist", 69: "neck",
}
LIMBS = [(13, 11), (11, 9), (14, 12), (12, 10), (9, 10), (9, 69), (10, 69),
         (69, 0), (69, 5), (69, 6), (5, 7), (7, 62), (6, 8), (8, 41),
         (13, 15), (13, 17), (14, 18), (14, 20)]


def describe(name, a, indent="    "):
    a = np.asarray(a)
    if a.dtype == object or a.size == 0:
        print(f"{indent}{name}: {a.shape} {a.dtype}")
        return
    f = a.reshape(-1).astype(np.float64)
    print(f"{indent}{name}: shape={str(a.shape):<16} dtype={str(a.dtype):<9} "
          f"min={f.min():+.4f} max={f.max():+.4f} mean={f.mean():+.4f}")


def project(pts3d, focal, cam_t, cx, cy):
    """Project camera-space points with the model's predicted pinhole camera."""
    p = np.asarray(pts3d, dtype=np.float64) + np.asarray(cam_t, dtype=np.float64).reshape(1, 3)
    z = np.clip(p[:, 2], 1e-6, None)
    f = float(np.asarray(focal).reshape(-1)[0])
    return np.stack([p[:, 0] / z * f + cx, p[:, 1] / z * f + cy], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="./frames")
    ap.add_argument("--dets", default="./dets.json")
    ap.add_argument("--ckpt", default=CFG.CKPT)
    ap.add_argument("--mhr", default=CFG.MHR_PATH)
    ap.add_argument("--out", default="./out/smoke")
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

    model, cfg = load_sam_3d_body(args.ckpt, device=torch.device("cuda"), mhr_path=args.mhr)
    est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    print(f"[ok] model loaded; mesh faces={est.faces.shape}")

    dets = json.loads(Path(args.dets).read_text())
    frames = sorted(Path(args.frames).glob("*.jpg"))[: args.limit]
    saved, ok = {}, 0

    for i, fp in enumerate(frames):
        d = dets.get(fp.name, [])
        if not d:
            print(f"[skip] {fp.name}: no detection")
            continue
        box = np.array(d[0]["xyxy"], dtype=np.float32).reshape(1, 4)
        outs = est.process_one_image(str(fp), bboxes=box)
        if not outs:
            print(f"[skip] {fp.name}: estimator returned nothing")
            continue
        o, ok = outs[0], ok + 1
        img = cv2.imread(str(fp))
        h, w = img.shape[:2]

        if i == 0:
            print(f"\n=== output fields ({fp.name}) ===")
            for k, v in o.items():
                if v is None:
                    print(f"    {k}: None")
                elif isinstance(v, np.ndarray):
                    describe(k, v)
                else:
                    print(f"    {k}: {type(v).__name__}={v}")

            k3 = np.asarray(o["pred_keypoints_3d"])
            print(f"\n=== coordinate conventions ===")
            print(f"    n_joints={len(k3)}  (expect 70 for MHR70)")
            for idx, nm in JOINTS.items():
                if idx < len(k3):
                    print(f"    {nm:<11} idx{idx:<3} {np.round(k3[idx], 4)}")
            ank = (k3[13] + k3[14]) / 2
            dv = k3[0] - ank
            axis = int(np.argmax(np.abs(dv)))
            print(f"\n    nose - ankle_mid = {np.round(dv, 4)}")
            print(f"    -> vertical axis = index {axis}, sign {np.sign(dv[axis]):+.0f} "
                  f"({'head is -' if dv[axis] < 0 else 'head is +'}{'XYZ'[axis]}"
                  f" => {'Y-down camera convention' if axis == 1 and dv[1] < 0 else 'check'})")
            span = np.linalg.norm(k3[0] - ank)
            print(f"    nose-to-ankle distance = {span:.4f} "
                  f"(~1.4-1.6 if metres, ~140-160 if cm/mm)")
            # stance width tells us whether the feet are resolved sensibly
            print(f"    ankle separation = {np.linalg.norm(k3[13] - k3[14]):.4f}")
            print(f"    l_heel->l_toe    = {np.round(k3[15] - k3[17], 4)}")
            print(f"    r_heel->r_toe    = {np.round(k3[18] - k3[20], 4)}")

        # 2D overlay: model's own 2D keypoints + projected mesh, no OpenGL needed
        k2 = np.asarray(o["pred_keypoints_2d"]).reshape(-1, 2)
        verts = np.asarray(o["pred_vertices"])
        vp = project(verts, o["focal_length"], o["pred_cam_t"], w / 2.0, h / 2.0)
        for x, y in vp[::12]:
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (int(x), int(y)), 1, (0, 180, 255), -1)
        for a, b in LIMBS:
            if a < len(k2) and b < len(k2):
                cv2.line(img, tuple(np.int32(k2[a])), tuple(np.int32(k2[b])), (0, 255, 0), 2)
        for idx in JOINTS:
            if idx < len(k2):
                cv2.circle(img, tuple(np.int32(k2[idx])), 3, (0, 0, 255), -1)
        x1, y1, x2, y2 = box[0].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.imwrite(str(out_dir / f"{fp.stem}_overlay.jpg"), img)

        saved[fp.name] = {k: np.asarray(v) for k, v in o.items()
                          if isinstance(v, np.ndarray) and k != "mask"}

    if saved:
        np.savez_compressed(out_dir / "raw.npz",
                            **{f"{k}__{kk}": vv for k, v in saved.items() for kk, vv in v.items()})
        print(f"\n[ok] {ok}/{len(frames)} frames inferred -> {out_dir}")
    else:
        print("\n[FAIL] no frames produced output")


if __name__ == "__main__":
    main()
