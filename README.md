# snowboard

3D pose analysis of snowboard riding, built on Meta's
[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body), to turn footage of a
run into specific, actionable technique feedback.

![annotated overlay](docs/examples/overlay_frame.jpg)

*One frame of `overlay.mp4`: projected mesh and skeleton on the tracked rider, the
board axes fitted to the feet, the gauge panel, one coaching cue anchored to the
rider, and the scrolling toe/heel trace whose sign changes mark the turns. Two
other riders are in shot; the tracker holds the one it locked on. Every gauge is
inside its target band here — this is a snowboard-cross racer judged against
`--level racer` — so the only thing being said is about uneven leg loading.
Shown with `--lang en`; the default overlay language is Chinese.*

## What it measures

<img src="docs/examples/hud.jpg" width="420" alt="gauge panel with target bands">

Every number is expressed in a frame fitted to the rider's own feet, so it is
independent of camera angle — no world-up or gravity estimate needed. Inclination
is lean relative to the board; angulation is the bend at the hip that lets you hold
an edge without leaning; fore/aft is where the mass sits along the board.

Each is drawn against its target band, green, with the live value as a marker:
amber outside the band, red well outside. Here the rider is short of racer
inclination, angulation and knee flexion at this instant, and `fore/aft +0.16`
sits right at the top edge of the racer band — a value the learner bands would
call a fault.

<img src="docs/examples/edge_trace.jpg" width="420" alt="toe/heel edge trace">

The centre of mass crosses the board at every edge change, so turns segment from
the sign changes of this trace — cyan toeside, yellow heelside.

## The coaching is drawn, not just written

A number is not advice: "angulation 39°" tells a learner nothing, because they
have no idea what 39 should be. Advice is the *gap* between the value and a
target, so every metric is drawn as a bar with its target band as a green zone
and the live value as a marker — inside or outside, no interpretation needed.

Targets depend on who is riding. A World Universiade racer measured +0.106
fore/aft, which is a fault for a learner and correct technique for a racer, so
the bands are keyed by `--level` and the overlay states which set it is judging
against.

Over the top of that, **one** cue at a time, anchored to the rider. Which one is
decided by the causal order snowboarding is actually taught in — balance, then
pressure, then edge, then rotation — because a rider in the back seat will also
be counter-rotating, and coaching the rotation first is wasted breath. The cue
speaks only when a value reaches the same threshold the gauge paints red, so the
rule is simply: the caption names the red bar. Cueing on any band exit fired on
82 of 82 frames of one clip, and a caption on every frame is wallpaper.

A milder fault that is *held* speaks too, because that is the shape of the fault
that ends in a crash. On the clip where the rider goes down, shoulder separation
grows to 25° and holds — never reaching the "bad" line, so a red-bar-only rule
left the overlay silent through the entire approach to the fall.

Three rules bound how much it talks: a sustained fault is captioned across its
whole window, nothing shows for under a second, and the caption speaks for at
most 3s before resting 2s. That last one applies to the caption as a whole rather
than per message — resting per message looks equivalent and is not, because two
faults that alternate each restart their own spell and nothing ever rests.

The overlay is in Chinese by default (`--lang zh|en`). That is not a string swap:
OpenCV's Hershey fonts have no glyph beyond ASCII and draw *nothing* for 内倾, so
all text renders through PIL against a real TrueType face. Point `SNOWPOSE_FONT`
at a `.ttf`/`.ttc`, or install one (`fonts-wqy-zenhei`); without a usable font the
run warns and falls back to English rather than producing a video of empty boxes.
The written `report.md` stays English.

When the stance inference is too weak to trust, fore/aft advice is withdrawn
rather than caveated — the gauge reads `unknown`. With the nose and tail swapped,
"GET FORWARD" coaches the wrong leg, and a caption burned into a video is far
more persuasive than a footnote.

## The board comes from the feet

<img src="docs/examples/board_axis_check.jpg" width="560" alt="board axis validation">

SAM 3D Body has no notion of a snowboard, but MHR70 gives heels and toe tips and
both feet are strapped to the deck. Red is a detected snowboard mask's principal
axis; cyan is the axis recovered from the feet alone.

They agree to a **median 1.3°** — p25 0.5°, p75 2.5°, 98% within 15° — measured
across **771 frames** of a clip where the rider was large enough for the segmenter
to find the board reliably. On the broadcast frame shown above, where the rider is
only 267 px tall and just 15 frames produced a mask, agreement was 3.2°. Chance
would be 45° either way.

## Status

Validated end to end on an 8x RTX 6000D box: the model runs on real
snowboard-cross frames and the recovered mesh lands correctly on a rider in a deep
carve, wearing helmet, goggles and bulky kit. See `docs/design.md` for the output
schema, the coordinate-frame findings, and the metric design.

## Layout

    snowpose/
      mhr.py        MHR70 joint indices (from the model's own metadata)
      model.py      SAM 3D Body wrapper, kept loaded across frames
      video.py      cv2 decode, shot-cut detection
      track.py      greedy IoU tracking + primary-subject selection
      geometry.py   board frame from the feet; all per-frame metrics
      smooth.py     Hampel despike + Savitzky-Golay
      turns.py      stance orientation, turn segmentation, symmetry
      coach.py      per-level target bands, fault thresholds, live cue selection
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
        --stance regular --level learner --out out/

Writes `overlay.mp4`, `metrics.csv`, `turns.json` and `report.md`. Pass
`--stance regular|goofy` when you know it; `auto` infers it from where the rider
looks and prints its confidence. `--level learner|racer` picks which target bands
the gauges draw and the report judges against — the same numbers are correct
technique for one and a fault for the other. `--lang zh|en` sets the overlay
language (default Chinese).

`scripts/rebuild_report.py --run out/ --level racer` re-judges an analysed run
against the other level without re-running inference.

## Setup

Needs a CUDA GPU. SAM 3D Body itself is a separate install:

1. Clone [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) and follow
   its `INSTALL.md` (python 3.11, torch, detectron2 at the pinned commit).
2. **Request access** to the checkpoints on Hugging Face — they are gated and
   approved manually by Meta: [`facebook/sam-3d-body-dinov3`](https://huggingface.co/facebook/sam-3d-body-dinov3).
   Then `hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3`.
3. Install this package's requirements (`numpy`, `opencv-python`, `ultralytics`) into
   the same environment, and make `sam_3d_body` importable (`PYTHONPATH=/path/to/sam-3d-body`).

Paths come from the environment, with defaults matching the layout above:

    SAM3D_CKPT          checkpoints/sam-3d-body-dinov3/model.ckpt
    SAM3D_MHR_PATH      checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
    SNOWPOSE_DETECTOR   person detector weights   (default yolo11x.pt)
    SNOWPOSE_SEGMENTOR  segmentation weights      (default yolo11x-seg.pt)
    SNOWPOSE_FONT       TrueType font for overlay text, required for --lang zh

Then verify the install and print the model's output conventions:

    python scripts/smoke_test.py --frames ./frames --dets ./dets.json

Read `docs/design.md` before trusting the report: the geometry is verified
against synthetic ground truth and real footage, but the coaching thresholds are
uncalibrated heuristics that currently flag an elite racer's deliberate forward
stance as a fault.

Two gotchas are documented in `docs/design.md` and worth reading before touching the
install: `pyrender` cannot load (no libEGL, no sudo — project the mesh with numpy
instead), and the DINOv3 backbone fetches code via `torch.hub` on every load, which
needs a pre-populated cache on a box that cannot reach GitHub reliably.
