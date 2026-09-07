"""Board-frame geometry from MHR70 joints.

Everything here is expressed in a frame built from the rider's own feet, so no
gravity or world-up estimate is needed — see docs/design.md. Inputs are
`pred_keypoints_3d` arrays, shape (70, 3), in metres, camera-space.
"""

import numpy as np

from . import mhr


def _unit(v, eps=1e-9):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > eps else np.zeros_like(v)


def _angle_between(u, v):
    """Unsigned angle in degrees between two vectors."""
    u, v = _unit(u), _unit(v)
    if not u.any() or not v.any():
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def _signed_angle(u, v, axis):
    """Signed angle in degrees from u to v, measured about `axis` (right-handed)."""
    n = _unit(axis)
    up = _unit(u - np.dot(u, n) * n)
    vp = _unit(v - np.dot(v, n) * n)
    if not up.any() or not vp.any():
        return np.nan
    s = np.dot(np.cross(up, vp), n)
    c = np.dot(up, vp)
    return float(np.degrees(np.arctan2(s, c)))


def foot_points(k):
    """The four points that lie on the board deck: both heels and both big toes."""
    return np.stack([k[mhr.L_HEEL], k[mhr.L_TOE_BIG],
                     k[mhr.R_HEEL], k[mhr.R_TOE_BIG]])


class BoardFrame:
    """Orthonormal frame (long, lat, normal) fitted to the rider's feet.

    - `normal` is perpendicular to the board deck, signed to point toward the body.
    - `long` runs along the board's length (tip-to-tail); its sign is arbitrary here
      and must be resolved once per run from travel direction, not per frame.
    - `lat` = normal x long, so +lat is one edge and -lat the other. Which one is
      "toe" follows from the feet and is set in `toe_sign`.
    """

    __slots__ = ("origin", "long", "lat", "normal", "planarity", "stance_width",
                 "toe_sign", "ok")

    def __init__(self, k):
        pts = foot_points(k)
        self.origin = pts.mean(axis=0)

        # Plane through the four deck points. Smallest singular vector is the normal;
        # the ratio of singular values says how planar the four points actually are.
        centred = pts - self.origin
        _, s, vt = np.linalg.svd(centred, full_matrices=False)
        normal = vt[2]
        self.planarity = float(s[2] / s[0]) if s[0] > 1e-9 else np.inf

        # The body sits above the deck, so the normal points toward the hips.
        hip_mid = (k[mhr.L_HIP] + k[mhr.R_HIP]) / 2.0
        if np.dot(normal, hip_mid - self.origin) < 0:
            normal = -normal
        self.normal = _unit(normal)

        # Board long axis: the ankle-to-ankle line, projected into the deck plane.
        ankle_axis = k[mhr.L_ANKLE] - k[mhr.R_ANKLE]
        self.stance_width = float(np.linalg.norm(ankle_axis))
        long = ankle_axis - np.dot(ankle_axis, self.normal) * self.normal
        self.long = _unit(long)
        self.lat = _unit(np.cross(self.normal, self.long))

        # Which lateral direction the toes point. Feet are strapped across the
        # board, so heel->toe has a large component along one edge.
        toe_dir = ((k[mhr.L_TOE_BIG] - k[mhr.L_HEEL]) +
                   (k[mhr.R_TOE_BIG] - k[mhr.R_HEEL])) / 2.0
        self.toe_sign = 1.0 if np.dot(toe_dir, self.lat) >= 0 else -1.0

        self.ok = bool(self.long.any() and self.normal.any() and self.stance_width > 1e-3)

    @property
    def basis(self):
        return np.stack([self.long, self.lat, self.normal])

    def to_board(self, p):
        """Express a point (or (N,3) array) in board coordinates about the feet."""
        d = np.atleast_2d(np.asarray(p, dtype=np.float64)) - self.origin
        out = d @ self.basis.T
        return out[0] if np.ndim(p) == 1 else out

    def flip_long(self):
        """Reverse the board's long axis, keeping the frame right-handed."""
        self.long = -self.long
        self.lat = _unit(np.cross(self.normal, self.long))
        self.toe_sign = -self.toe_sign


def com(k, vertices=None):
    """Centre of mass.

    Mesh centroid when vertices are available (uniform-density proxy over 18k
    points, far steadier than a joint mean); otherwise a trunk-weighted joint
    estimate, since most body mass sits in the pelvis and torso.
    """
    if vertices is not None and len(vertices):
        return np.asarray(vertices, dtype=np.float64).mean(axis=0)
    w = {mhr.L_HIP: 0.15, mhr.R_HIP: 0.15, mhr.NECK: 0.15, mhr.NOSE: 0.07,
         mhr.L_SHOULDER: 0.05, mhr.R_SHOULDER: 0.05,
         mhr.L_KNEE: 0.06, mhr.R_KNEE: 0.06, mhr.L_ANKLE: 0.03, mhr.R_ANKLE: 0.03,
         mhr.L_ELBOW: 0.025, mhr.R_ELBOW: 0.025, mhr.L_WRIST: 0.015, mhr.R_WRIST: 0.015}
    tot = sum(w.values())
    return sum(k[i] * wt for i, wt in w.items()) / tot


def body_scale(k):
    """A per-frame length scale for normalising displacements: ankle to neck."""
    ankle_mid = (k[mhr.L_ANKLE] + k[mhr.R_ANKLE]) / 2.0
    return float(np.linalg.norm(k[mhr.NECK] - ankle_mid)) or 1.0


def joint_flexion(k, prox, mid, dist):
    """Flexion at `mid` in degrees. 0 = fully extended, larger = more bent."""
    return 180.0 - _angle_between(k[prox] - k[mid], k[dist] - k[mid])


def leg_angles(k):
    """Knee and ankle flexion for both legs."""
    return {
        "knee_flex_l": joint_flexion(k, mhr.L_HIP, mhr.L_KNEE, mhr.L_ANKLE),
        "knee_flex_r": joint_flexion(k, mhr.R_HIP, mhr.R_KNEE, mhr.R_ANKLE),
        "ankle_flex_l": joint_flexion(k, mhr.L_KNEE, mhr.L_ANKLE, mhr.L_TOE_BIG),
        "ankle_flex_r": joint_flexion(k, mhr.R_KNEE, mhr.R_ANKLE, mhr.R_TOE_BIG),
    }


def stack_angles(k, bf):
    """Inclination and angulation, both relative to the board — no world-up needed.

    - `inclination`: angle between the torso axis (hip_mid -> neck) and the board
      normal. 0 means stacked square over the deck; it grows as the rider leans.
    - `angulation`: the bend between lower body (ankle_mid -> hip_mid) and torso.
      0 means a straight line from feet to neck; it grows as the rider breaks at
      the hip to hold an edge without leaning the whole body.
    """
    hip_mid = (k[mhr.L_HIP] + k[mhr.R_HIP]) / 2.0
    ankle_mid = (k[mhr.L_ANKLE] + k[mhr.R_ANKLE]) / 2.0
    torso = k[mhr.NECK] - hip_mid
    lower = hip_mid - ankle_mid
    return {
        "inclination": _angle_between(torso, bf.normal),
        "angulation": _angle_between(lower, torso),
        "lower_leg_lean": _angle_between(lower, bf.normal),
    }


def separation(k, bf):
    """Upper/lower body twist about the board normal, in degrees.

    Positive means the shoulders are rotated ahead of the hips in the frame's
    right-handed sense; the sign becomes meaningful once the long axis is oriented
    to travel direction. Magnitude is what matters for counter-rotation.
    """
    hip_axis = k[mhr.L_HIP] - k[mhr.R_HIP]
    sh_axis = k[mhr.L_SHOULDER] - k[mhr.R_SHOULDER]
    return {
        "hip_shoulder_sep": _signed_angle(hip_axis, sh_axis, bf.normal),
        "hip_vs_board": _signed_angle(bf.long, hip_axis, bf.normal),
        "shoulder_vs_board": _signed_angle(bf.long, sh_axis, bf.normal),
    }


def balance(k, bf, vertices=None):
    """Where the mass sits over the board, normalised by body scale.

    - `fore_aft`: along the board. Sign is only meaningful after the long axis is
      oriented to travel direction; then + is toward the nose.
    - `toe_heel`: across the board, signed so + is toward the toe edge. This is the
      quantity whose sign flips at every edge change, so turns segment from it.
    """
    c = com(k, vertices)
    b = bf.to_board(c)
    s = body_scale(k)
    return {
        "fore_aft": float(b[0] / s),
        "toe_heel": float(b[1] * bf.toe_sign / s),
        "com_height": float(b[2] / s),
    }


def frame_metrics(k, vertices=None, flip_long=False):
    """All per-frame metrics for one rider, or None if the feet are unusable.

    `flip_long` reverses the board's long axis so + points toward the nose. It is
    a per-run decision (see turns.orient_long_axis), never per frame, and must be
    applied before the metrics are read or every fore/aft sign is arbitrary.
    """
    k = np.asarray(k, dtype=np.float64)
    bf = BoardFrame(k)
    if not bf.ok:
        return None
    if flip_long:
        bf.flip_long()
    m = {"planarity": bf.planarity, "stance_width": bf.stance_width,
         "body_scale": body_scale(k)}
    m.update(leg_angles(k))
    m.update(stack_angles(k, bf))
    m.update(separation(k, bf))
    m.update(balance(k, bf, vertices))
    return m, bf
