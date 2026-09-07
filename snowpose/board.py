"""Board detection, used to check the foot-derived board frame against pixels.

The whole metric design infers the board's plane from the rider's feet, because
SAM 3D Body has no notion of a board. That inference is load-bearing, so it is
worth testing against something independent. COCO includes `snowboard` (class 31),
and a segmentation mask gives a genuine 2D long axis — a bounding box does not,
since a box has no orientation.

This is a validator first. Pinning the true edges and correcting for boot and
binding offsets would need the mask fused into the 3D fit, which is not done here.
"""

import numpy as np

SNOWBOARD_CLASS = 31


def detect(model, frame, conf=0.25, imgsz=1280, device="0", project=None):
    """Return snowboard instances as dicts with box, mask centroid and 2D axis."""
    kw = dict(classes=[SNOWBOARD_CLASS], conf=conf, imgsz=imgsz, verbose=False,
              device=device, save=False)
    if project:
        kw.update(project=str(project), name="board", exist_ok=True)
    r = model.predict(frame, **kw)[0]
    out = []
    if r.boxes is None or not len(r.boxes):
        return out
    masks = r.masks.data.cpu().numpy() if r.masks is not None else None
    h, w = frame.shape[:2]
    for i, (xyxy, c) in enumerate(zip(r.boxes.xyxy.cpu().numpy(),
                                      r.boxes.conf.cpu().numpy())):
        item = {"xyxy": xyxy.tolist(), "conf": float(c), "axis": None,
                "centroid": [float((xyxy[0] + xyxy[2]) / 2),
                             float((xyxy[1] + xyxy[3]) / 2)],
                "elongation": None}
        if masks is not None and i < len(masks):
            m = masks[i]
            ys, xs = np.nonzero(m > 0.5)
            if len(xs) >= 12:
                # Masks come back at the model's letterboxed resolution.
                sx, sy = w / m.shape[1], h / m.shape[0]
                pts = np.stack([xs * sx, ys * sy], axis=1).astype(np.float64)
                item["centroid"] = pts.mean(axis=0).tolist()
                centred = pts - pts.mean(axis=0)
                _, s, vt = np.linalg.svd(centred, full_matrices=False)
                item["axis"] = vt[0].tolist()          # principal direction, 2D
                item["elongation"] = float(s[0] / s[1]) if s[1] > 1e-9 else np.inf
        out.append(item)
    return out


def nearest_to(boards, point, max_dist):
    """The board whose centroid is closest to `point` (the rider's feet in 2D)."""
    best, bd = None, np.inf
    p = np.asarray(point, dtype=float)
    for b in boards:
        d = float(np.linalg.norm(np.asarray(b["centroid"]) - p))
        if d < bd:
            best, bd = b, d
    if best is None or bd > max_dist:
        return None, bd
    return best, bd


def axis_agreement(board_axis_2d, long_axis_2d):
    """Acute angle in degrees between two undirected 2D axes.

    Both axes are sign-ambiguous — the board's principal component and the
    foot-derived long axis each point either way — so the comparison folds to
    [0, 90]. 0 means perfect agreement, 90 means perpendicular.
    """
    a = np.asarray(board_axis_2d, dtype=float)
    b = np.asarray(long_axis_2d, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return np.nan
    c = abs(float(np.dot(a / na, b / nb)))
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))
