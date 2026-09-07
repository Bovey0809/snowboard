"""Where the weights live.

Defaults follow the layout SAM 3D Body's own INSTALL.md produces, so a fresh
checkout works with no configuration. Override per machine with environment
variables rather than editing code:

    SAM3D_CKPT        path to model.ckpt
    SAM3D_MHR_PATH    path to assets/mhr_model.pt
    SNOWPOSE_DETECTOR person detector weights (any Ultralytics model)
    SNOWPOSE_SEGMENTOR segmentation weights, for the board cross-check
"""

import os

CKPT = os.environ.get("SAM3D_CKPT", "./checkpoints/sam-3d-body-dinov3/model.ckpt")
MHR_PATH = os.environ.get(
    "SAM3D_MHR_PATH", "./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
)
# Ultralytics fetches these on first use if they are not present locally.
DETECTOR = os.environ.get("SNOWPOSE_DETECTOR", "yolo11x.pt")
SEGMENTOR = os.environ.get("SNOWPOSE_SEGMENTOR", "yolo11x-seg.pt")
