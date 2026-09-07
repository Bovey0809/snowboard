"""MHR70 skeleton constants.

Index -> name taken verbatim from the SAM 3D Body source
(`sam_3d_body/metadata/mhr70.py`). Only the joints this project actually uses
are named here; the hand joints (21-62, minus the wrists) are not.
"""

NOSE = 0
L_EYE, R_EYE, L_EAR, R_EAR = 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_HIP, R_HIP = 9, 10
L_KNEE, R_KNEE = 11, 12
L_ANKLE, R_ANKLE = 13, 14
L_TOE_BIG, L_TOE_SMALL, L_HEEL = 15, 16, 17
R_TOE_BIG, R_TOE_SMALL, R_HEEL = 18, 19, 20
R_WRIST = 41
L_WRIST = 62
L_ACROMION, R_ACROMION = 67, 68
NECK = 69

N_JOINTS = 70

NAMES = {
    NOSE: "nose", L_EYE: "left_eye", R_EYE: "right_eye",
    L_EAR: "left_ear", R_EAR: "right_ear",
    L_SHOULDER: "left_shoulder", R_SHOULDER: "right_shoulder",
    L_ELBOW: "left_elbow", R_ELBOW: "right_elbow",
    L_HIP: "left_hip", R_HIP: "right_hip",
    L_KNEE: "left_knee", R_KNEE: "right_knee",
    L_ANKLE: "left_ankle", R_ANKLE: "right_ankle",
    L_TOE_BIG: "left_big_toe_tip", L_TOE_SMALL: "left_small_toe_tip", L_HEEL: "left_heel",
    R_TOE_BIG: "right_big_toe_tip", R_TOE_SMALL: "right_small_toe_tip", R_HEEL: "right_heel",
    L_WRIST: "left_wrist", R_WRIST: "right_wrist",
    L_ACROMION: "left_acromion", R_ACROMION: "right_acromion",
    NECK: "neck",
}

# Drawing order for 2D overlays.
LIMBS = [
    (L_ANKLE, L_KNEE), (L_KNEE, L_HIP), (R_ANKLE, R_KNEE), (R_KNEE, R_HIP),
    (L_HIP, R_HIP), (L_HIP, NECK), (R_HIP, NECK), (NECK, NOSE),
    (NECK, L_SHOULDER), (NECK, R_SHOULDER), (L_SHOULDER, L_ELBOW),
    (L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_ANKLE, L_HEEL), (L_HEEL, L_TOE_BIG), (L_ANKLE, L_TOE_BIG),
    (R_ANKLE, R_HEEL), (R_HEEL, R_TOE_BIG), (R_ANKLE, R_TOE_BIG),
]

# The joints that carry the snowboard-relevant signal. The feet matter most:
# both are strapped to the board, so heel/toe/ankle together fix the board plane.
FOOT = {
    "left": {"ankle": L_ANKLE, "heel": L_HEEL, "toe": L_TOE_BIG, "toe_small": L_TOE_SMALL},
    "right": {"ankle": R_ANKLE, "heel": R_HEEL, "toe": R_TOE_BIG, "toe_small": R_TOE_SMALL},
}
