# Snowboard pose analysis on SAM 3D Body — design

## What the model gives us

`SAM3DBodyEstimator.process_one_image(img, bboxes=...)` returns one dict per person.
Verified on real snowboard-cross frames (2026-09-07, ultra11):

| field | shape | notes |
|---|---|---|
| `pred_keypoints_3d` | (70, 3) | MHR70 joints, **metres**, camera-space, root-relative |
| `pred_keypoints_2d` | (70, 2) | image pixels — free overlay, no projection needed |
| `pred_vertices` | (18439, 3) | mesh; `faces` is (36874, 3) |
| `pred_cam_t` | (3,) | root translation; z was ~8.9 m on the test shot |
| `focal_length` | scalar | predicted pinhole focal, px |
| `pred_global_rots` | (127, 3, 3) | per-joint global rotation matrices |
| `global_rot` | (3,) | body orientation in camera frame |
| `body_pose_params` / `shape_params` / `scale_params` | (133,) / (45,) / (28,) | MHR parameters |

Measured sanity on a carving rider: nose→ankle 1.18 m, ankle separation 0.38 m.

## The two constraints that shape everything

**1. There is no board.** SAM 3D Body is body-only. But MHR70 includes
`heel`, `big_toe_tip` and `small_toe_tip` for both feet (indices 15-20), and both
feet are strapped to the board. So the board plane is recoverable from the feet:

- board long axis  `L = ankle_L - ankle_R`
- toe direction    `T = mean(toe - heel)` over both feet
- board normal     `N = L × T`, then re-orthogonalise

That makes YOLO board detection a *refinement* (it pins the board's true edges and
catches boot/binding offsets), not a prerequisite.

**2. World-up is unknown.** The model works per-frame in camera space, and the
camera on real footage is tilted, moving, or both. The "vertical axis" test on a
crouched rider is inconclusive by construction — the body's long axis projected
mostly onto camera X in our test shot.

So **every MVP metric is expressed in the board frame `(L, T, N)`**, which is
rotation-invariant and needs no gravity estimate. Absolute edge angle relative to
the slope is the one thing that genuinely needs world-up, and it is deferred.

The coaching-relevant substitute is available body-relatively: **inclination** =
angle between the body's stacking axis (`hip_mid → neck`) and the board normal.
During a carve the rider's apparent vertical (gravity + centripetal) is what they
stack against, so this is arguably the more useful number than absolute edge angle.

## Metrics (board frame, no world-up needed)

- **Fore/aft balance** — COM projected on `L`, relative to ankle midpoint. The
  classic beginner fault (sitting back) shows up here.
- **Toe/heel balance** — COM projected on `T`. Signs the edge being ridden.
- **Knee / ankle flexion** — joint angles from `pred_keypoints_3d`; absorption and
  "stack". Independent of camera entirely.
- **Inclination vs angulation** — whole-body lean vs hip/knee bending. The
  distinction every carving coach teaches.
- **Upper/lower separation** — hip axis vs shoulder axis twist; counter-rotation.
- **Stance symmetry** — left vs right leg flexion difference.

## Turn segmentation

Toe/heel balance (COM on `T`) changes sign at each edge change, so turns segment
from its zero crossings without any world frame. Per-turn scoring then compares
toeside against heelside, and turn-to-turn consistency.

## Pipeline

    footage → frames (cv2) → YOLO person boxes → SAM 3D Body per frame
            → subject tracking → temporal smoothing → board frame → metrics
            → turn segmentation → overlay video + metrics CSV + coaching report

## Known risks

- Rider height in frame was a median 176 px on the test footage. It works, but
  quality degrades on small subjects; chairlift-distance footage will be weak.
- Per-frame inference means jitter. Smoothing is required before differentiating
  anything.
- Two riders in shot: subject selection must be tracked, not per-frame "tallest".
