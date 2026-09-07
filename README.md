# snowboard

3D pose analysis of snowboard riding, built on Meta's
[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body), to turn footage of a
run into specific, actionable technique feedback.

## Status

Setup validated end to end on **ultra11** (Shenzhen, 8× RTX 6000D): the model runs on
real snowboard-cross frames and the recovered mesh lands correctly on a rider in a
deep carve, wearing helmet, goggles and bulky kit. See `docs/design.md` for the
output schema, the coordinate-frame findings, and the metric design.

## Layout

    snowpose/       library code
      mhr.py        MHR70 joint indices (from the model's own metadata)
    scripts/
      smoke_test.py validates the install and prints output conventions
    docs/design.md  what the model gives us and how the metrics are derived
    footage/        clips (gitignored)

## Environment

Remote, because the model needs a GPU and the checkpoints are large:

    ssh ultra11
    cd /data/rick/sam3d/sam-3d-body
    CUDA_VISIBLE_DEVICES=0 YOLO_CONFIG_DIR=/data/rick/ultra_cfg \
      /data/rick/sam3d/venv/bin/python scripts/smoke_test.py

Two gotchas are documented in `docs/design.md` and worth reading before touching the
install: `pyrender` cannot load (no libEGL, no sudo — project the mesh with numpy
instead), and the DINOv3 backbone fetches code via `torch.hub` on every load, which
needs a pre-populated cache on a box that cannot reach GitHub reliably.
