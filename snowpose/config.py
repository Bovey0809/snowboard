"""Where the weights live.

Defaults follow the layout SAM 3D Body's own INSTALL.md produces, so a fresh
checkout works with no configuration. Override per machine with environment
variables rather than editing code:

    SAM3D_CKPT        path to model.ckpt
    SAM3D_MHR_PATH    path to assets/mhr_model.pt
    SNOWPOSE_DETECTOR person detector weights (any Ultralytics model)
    SNOWPOSE_SEGMENTOR segmentation weights, for the board cross-check
    SNOWPOSE_FONT     TrueType font for overlay text (needed for Chinese)
"""

import os

CKPT = os.environ.get("SAM3D_CKPT", "./checkpoints/sam-3d-body-dinov3/model.ckpt")
MHR_PATH = os.environ.get(
    "SAM3D_MHR_PATH", "./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
)
# Ultralytics fetches these on first use if they are not present locally.
DETECTOR = os.environ.get("SNOWPOSE_DETECTOR", "yolo11x.pt")
SEGMENTOR = os.environ.get("SNOWPOSE_SEGMENTOR", "yolo11x-seg.pt")

# Overlay text. OpenCV's Hershey fonts are ASCII-only — they cannot draw a
# single Chinese glyph — so any non-Latin overlay needs a real TrueType face
# rendered through PIL. Nothing is bundled: fonts are large and separately
# licensed, so this resolves one already on the machine, or takes an explicit
# path. Without one the overlay falls back to Hershey and English.
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "C:/Windows/Fonts/msyh.ttc",
)


def _resolve_font():
    p = os.environ.get("SNOWPOSE_FONT")
    if p:
        return p if os.path.exists(p) else None
    return next((c for c in _FONT_CANDIDATES if os.path.exists(c)), None)


FONT = _resolve_font()
