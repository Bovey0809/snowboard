"""Thin wrapper around SAM3DBodyEstimator for batch/video use.

Keeps the model loaded across frames and takes boxes from a detector we control,
because the repo's own detectors need weights from hosts the GPU boxes cannot
reach (see docs/design.md).
"""

from pathlib import Path

import numpy as np

DEFAULT_CKPT = "/data/rick/sam3d/ckpt/model.ckpt"
DEFAULT_MHR = "/data/rick/sam3d/ckpt/assets/mhr_model.pt"


class BodyModel:
    def __init__(self, ckpt=DEFAULT_CKPT, mhr_path=DEFAULT_MHR, device="cuda"):
        import torch
        from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

        model, cfg = load_sam_3d_body(str(ckpt), device=torch.device(device),
                                      mhr_path=str(mhr_path))
        self.est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
        self.faces = self.est.faces

    def infer(self, image, box):
        """Run one person. `image` is a path or an RGB array; `box` is xyxy.

        The estimator only *warns* when handed an array and assumes it is RGB, so
        callers decoding with cv2 must convert. Passing a path is always safe.
        """
        if isinstance(image, Path):
            image = str(image)
        b = np.asarray(box, dtype=np.float32).reshape(1, 4)
        outs = self.est.process_one_image(image, bboxes=b)
        return outs[0] if outs else None
