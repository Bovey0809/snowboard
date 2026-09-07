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

    snowpose/
      mhr.py        MHR70 joint indices (from the model's own metadata)
      model.py      SAM 3D Body wrapper, kept loaded across frames
      video.py      cv2 decode, shot-cut detection
      track.py      greedy IoU tracking + primary-subject selection
      geometry.py   board frame from the feet; all per-frame metrics
      smooth.py     Hampel despike + Savitzky-Golay
      turns.py      stance orientation, turn segmentation, symmetry
      report.py     ranked findings and drills
      overlay.py    skeleton, board axes, metrics HUD, edge trace
    scripts/
      analyze_run.py           video segment -> overlay + CSV + report
      find_segments.py         locate segments worth analysing in a long video
      check_stability.py       measure board-frame jitter on real footage
      smoke_test.py            validate the install, print output conventions
      patch_dinov3_offline.py  stop the backbone load hanging without egress
    docs/design.md  what the model gives us, how the metrics are derived, results
    footage/        clips (gitignored)

## Usage

    python scripts/find_segments.py --video run.mp4          # where is the riding?
    python scripts/analyze_run.py --video run.mp4 --start 120 --end 150 \
        --stance regular --out out/

Writes `overlay.mp4`, `metrics.csv`, `turns.json` and `report.md`. Pass
`--stance regular|goofy` when you know it; `auto` infers it from where the rider
looks and prints its confidence.

## Environment

Remote, because the model needs a GPU and the checkpoints are large:

    ssh ultra11
    cd /data/rick/sam3d/sam-3d-body
    CUDA_VISIBLE_DEVICES=0 YOLO_CONFIG_DIR=/data/rick/ultra_cfg \
      /data/rick/sam3d/venv/bin/python scripts/smoke_test.py

Read `docs/design.md` before trusting the report: the geometry is verified
against synthetic ground truth and real footage, but the coaching thresholds are
uncalibrated heuristics that currently flag an elite racer's deliberate forward
stance as a fault.

Two gotchas are documented in `docs/design.md` and worth reading before touching the
install: `pyrender` cannot load (no libEGL, no sudo — project the mesh with numpy
instead), and the DINOv3 backbone fetches code via `torch.hub` on every load, which
needs a pre-populated cache on a box that cannot reach GitHub reliably.
