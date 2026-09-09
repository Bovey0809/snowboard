# Snowboard pose analysis on SAM 3D Body — design

## What the model gives us

`SAM3DBodyEstimator.process_one_image(img, bboxes=...)` returns one dict per person.
Verified on real snowboard-cross frames (2026-09-07):

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

## Validation results (2026-09-07)

**Board frame holds up.** Measured on 36 tracked frames of snowboard-cross
footage at 25 fps:

- **zero** left/right leg swaps across 35 adjacent frame pairs — the model's leg
  labelling is temporally consistent, so no temporal L/R disambiguation is needed
- heel→toe length 0.201–0.207 m (MHR has a fixed foot), so the toe direction never
  collapses and `N` is always well-conditioned
- deck planarity median 0.052, p90 0.109 — the four foot points really are close to
  a plane
- stance width median 0.413 m, std 0.090 m

**But it is noisy.** Board axes rotate a median 5.5°/frame (p90 13°, max 64°), and
knee flexion moves 3.5–4.9°/frame against signal ranges of 20–35°. Hence
`smooth.py`: Hampel despiking then Savitzky-Golay, which cuts noise 3.5× on a
turn-shaped signal while preserving amplitude. Order is joints → smooth → metrics,
never metrics → smooth, because the angle computations are non-linear.

**End-to-end run** on a 20 s race segment (`--start 1184 --end 1204`): 3 shots
detected, an 11 s shot chosen, one rider tracked at 267 px mean height through
82 frames while two other riders were in shot, stance auto-inferred as regular,
and **4 turns segmented (2 toeside, 2 heelside)** with durations 0.88–2.96 s.
Mean inclination 33.8°, angulation 53.2°, knee flexion 75.2°.

## The thresholds are not calibrated, but they are now level-aware

On that segment the report flagged "mass too far forward" (fore/aft +0.106) as a
top finding. The subject is a World Universiade snowboard-cross racer, and an
aggressive forward stance is *correct* technique for racing — so this was a false
positive produced by thresholds meant for recreational riding.

The fix was not a better number, it was a better shape. A single symmetric
`|fore_aft| > 0.07` test cannot express "sitting back is a fault but driving the
nose is not", so the targets became **bands**, keyed by declared rider level
(`--level learner|racer`, `coach.py`). The racer band is deliberately asymmetric,
`-0.04 to +0.16`, and the same +0.106 now passes:

| clip | mean fore/aft | learner target | racer target | flagged? |
|---|---|---|---|---|
| race (Universiade) | +0.106 | -0.07 to +0.07 | -0.04 to +0.16 | learner only |
| mixkit_sportsman | +0.073 | -0.07 to +0.07 | -0.04 to +0.16 | learner only |

That is the honest state of the coaching layer: the geometry is verified to about
a degree, the bands are informed guesses. Both level tables live in one module so
they can be recalibrated against footage of riders whose level is known. Until
then, treat the *measurements* as sound and the *verdicts* as provisional — which
is why the overlay prints `targets: learner` on every frame rather than letting
the viewer assume the green zone is authoritative.

## Advice is the gap, so the target has to be drawable

"angulation 39 deg" is not advice. A learner has no idea what 39 should be, so
the number carries no instruction — the *gap between the value and a target* is
what carries it. That is the whole reason `coach.py` sits underneath both the
report and the renderer: a target that only exists inside an `if` statement in
`report.py` cannot be drawn, and a threshold copied into the renderer is a second
thing to recalibrate and a chance for the green zone to contradict the sentence
printed under it. `report.to_markdown` prints a `target` column populated from the
same `coach.bands()` the gauges draw.

Two visual channels, deliberately different in rate:

- **Gauges** show every metric continuously against its band. Always on, no
  interpretation needed — marker inside the green zone or not.
- **One cue**, anchored to the rider, speaks only when a value is *clearly* out.

The split matters because a caption is not a gauge. Cueing on any band exit fired
on **82 of 82** frames of the race clip and **317 of 352** of mixkit_sportsman: a
caption on every frame is wallpaper, and the point of the causal hierarchy
(balance → pressure → edge → rotation, one cue wins) was to say *less*, not to say
something different constantly. Requiring the value to reach the same "bad"
boundary the gauge paints red drops that to 26-80 frames, and gives the viewer
one rule to learn: **the caption names the red bar.**

### But the crash clip then went silent

That gate alone made `vert_ride` produce **zero** cues — the clip whose written
headline is a shoulder wind-up growing from 10° to 25° in the final 1.5 s before
the rider goes down. Separation peaked at 25.1° and held above 20° for 1.6 s,
never reaching the 30° "bad" line, so the overlay was quietest exactly where it
should have been loudest. `report.py` already argues this case for its written
findings (`sustained_excursion`); the cue needed the same second path in.

So a value only into *amber* also speaks, once it has held there for a second.
The line is 0.33 of a half-band because that clip sets it: the wind-up clears
0.33 for 1.6 s but clears 0.5 for only 0.8 s.

That change works, and it costs duty cycle, so three rules bound how much the
caption talks. Sensitivity and nagging are separate problems and one threshold
cannot serve both.

1. **The sustained window is marked whole**, not just the part past the hold.
   The hold decides *whether* a slow drift counts; once it does, the interesting
   stretch is the entire wind-up. This is review footage, not live coaching —
   withholding the caption over the very frames that show the fault growing had
   no upside, and on the crash clip it was the difference between a 0.7 s cue
   truncated by the end of the video and the full 1.4 s.
2. **Nothing shows for under a second.** Within one spell the top-ranked fault
   can change, and the ranking is not stable enough to show at that rate — it
   produced 0.3-0.4 s fragments that flicked to another message and back. Those
   are dropped rather than padded, since padding only displaces the next cue.
3. **At most 3 s, then 2 s of silence.** Applied to the caption *channel*, not
   per cue id: resting per id looks equivalent and is not, because when two
   faults alternate each change restarts its own spell and nothing ever rests.
   That bug alone put the race clip at a caption on 98% of frames while every
   individual cue was inside its cap.

| clip | level | duty | cues | shortest |
|---|---|---|---|---|
| vert_ride | learner | 39% | 2 | 1.33s |
| rick_indoor | learner | 40% | 13 | 1.00s |
| mixkit_sportsman | learner | 44% | 4 | 2.27s |
| mixkit_downhill | learner | 45% | 4 | 1.20s |
| vert_full | learner | 46% | 4 | 1.33s |
| race | racer | 66% | 3 | 1.04s |

Every cue is now 1.0-3.0 s, median 1.8 s. `vert_ride` is the shape to want: quiet
through the clean riding, then `重心后移` and `肩别抢转` across the 2.7 s that end
in the fall.

## The overlay speaks Chinese, and that is not a string swap

OpenCV's `putText` uses Hershey fonts — vector strokes with no glyph outside
ASCII. It does not error on `内倾`; it draws nothing. So the overlay renders all
text through PIL against a real TrueType face, resolved by `config.FONT` from
`SNOWPOSE_FONT` or from the usual system paths, with `--lang zh|en` choosing the
strings. No font is bundled: they are large and separately licensed, and without
one `analyze_run.py` says so and falls back to English rather than burning 20
minutes of inference into a video of empty boxes.

The catch is cost. A PIL round trip is per *image*, not per string, so calling it
for each of the ~13 labels on a frame would convert 1920x1080 thirteen times over.
The draw functions therefore queue their text and `render_text` applies the frame's
whole queue in one pass; shapes stay on cv2 and are drawn first, so the queue only
ever holds what belongs on top.

The Chinese is written to the same brief as the English — imperative corrections,
four to six characters, what to do rather than what is wrong: 重心前移, 屈膝下沉,
双腿均衡发力, 屈髋立刃, 果断立刃, 肩别抢转.

The other two guards are about not lying persuasively:

- A cue must hold 0.3 s before it appears and lingers 0.5 s after it clears. At
  12 fps a value resting on a band edge flips state several times a second, and
  a strobing caption is a rendering fault wearing coaching clothes.
- **A weak stance inference withdraws fore/aft advice rather than caveating it.**
  Two of the three sample clips inferred stance below the confidence floor (0.2
  and 1.6 against a floor of 1.0). Every fore/aft sign depends on which end of
  the board is the nose, so with it wrong "GET FORWARD" coaches the wrong leg —
  and a caption burned into the video is far more persuasive than a footnote at
  the bottom of a report. Below the floor the gauge reads `unknown`, draws no
  band and no marker, and the finding is suppressed.

## A turn-segmentation trap, found and fixed

A raw zero-crossing split over-segments. A rider holding one long edge produces
brief opposite-sign wobbles as the smoothed centre of mass crosses the board's
centreline, and every wobble cut a turn in two. On this segment a single 2.96 s
heelside turn came apart into three "turns" separated by blips of 0.08–0.24 s with
peaks as low as 0.003 — which both inflated the turn count and made the
"inconsistency" finding partly an artifact of its own fragmentation.

Discarding sub-threshold segments is not enough; they have to be *absorbed* and
the same-edge neighbours either side merged. Three consecutive same-edge turns in
the output is the signature of getting this wrong, since zero-crossings alone can
never produce one.

## Runs on three clips

Screened with `find_segments.py`, then analysed end to end. All three came from
freely-licensed sources; the two stock clips are not redistributed here.

| clip | segment | rider px | frames | turns | leading finding |
|---|---|---|---|---|---|
| Universiade snowboard cross (CC BY 3.0) | 11 s of a 20 s window | 267 | 82 | 4 (2 toe / 2 heel) | mass toward the nose |
| Mixkit "sportsman down the hill" | 23.5 s, one shot | 463 | 352 | 3, 92% heelside | whole run on the heel edge |
| Mixkit "snowboarding down the hill" | 16 s tracked of 29.5 s | 295 | 240 | 1, 100% heelside | whole run on the heel edge |
| follow-cam, indoor slope (private) | 65.7 s, one shot | 608 | 982 | 6 segments, 3 real turns | long traverses, not linked turns |
| vertical phone clip, ends in a fall (private) | 6.9 s of riding | 336 | 104 | 3 real turns | upper body winds up before the crash |

The two stock clips are continuous single shots, so the tracker holds one rider
for the whole clip — 352 unbroken frames on the first. Both riders descend
entirely on the heel edge, which the pipeline now reports as the headline rather
than dressing it up as turn statistics.

The screening step earned its place. Of eight candidate clips, `find_segments.py`
plus a frame check rejected six: three were selfie-stick or helmet POV (the rider
is out of frame or fisheye-distorted), one was a promo edit of portraits and
walking, one was rail jibbing rather than turns, and two were drone shots with the
subject a few pixels tall. **A high median subject height is a warning sign, not a
recommendation** — near-frame-height means a selfie or a talking head. The usable
band on 720p footage was roughly 250-500 px.

## A run mean hides the fault that causes the crash

A 17 s vertical phone clip contained 6.9 s of riding and then a fall. On the riding
portion the report said **"No faults crossed the flagging thresholds"** — 0.2 s
before the rider hit the snow. That is the worst possible failure for a coaching
tool: silence in front of the one event that mattered.

The data had it all along. Hip-shoulder separation grew from ~10° to ~24° and
*stayed* there for the final 1.7 s. But the run mean was 13.4°, under the 15°
threshold, because the clean earlier seconds diluted it. Faults that end in a crash
are precisely the ones that develop, so a mean-based test is structurally blind to
them.

`sustained_excursion()` now checks whether a fault is *held* above threshold for at
least a second, independent of the mean, and reports the window:

> Hips and shoulders stay more than 15 deg apart for 1.7s **between 5.2s and 6.9s**,
> peaking at 25 deg, even though the run averages 13 deg.

The window closes exactly at the fall. Verified against the overlay frames first —
the mesh fits the body well through that stretch and the shoulder twist is visible
in the image, so this is signal, not tracking noise. The race segment, which has no
sustained excursion, correctly gains no such finding.

## Two dead ends in detecting "not riding"

The same clip spent 10 s lying in the snow, which the pipeline reported as an
**11 s heelside turn with commitment 0.759** — quadruple any real turn — and which
doubled the run's mean inclination. Non-riding frames do not merely add noise; they
produce confident, plausible, wrong numbers.

Two obvious detectors were measured and both are dead:

1. **Board-frame metrics cannot see it.** Riding inclination spanned 3-23°, on the
   ground 3-71° — overlapping. This is not a tuning problem but a direct consequence
   of the design: the board stays strapped to the feet, so the frame rotates with
   the body, and the frame is deliberately invariant to exactly the rotation that
   separates upright from fallen. The property that makes the metrics
   camera-independent also makes them blind here.
2. **Bounding-box aspect ratio cannot either.** On this clip it looked perfect —
   riding w/h 0.37-0.64 against 1.01-1.45 on the ground, cleanly separable. Then
   the same measure on the race clip returned **0.95-2.49, with 100% of frames
   above any workable threshold**: a snowboard-cross racer in a deep tuck is wider
   than they are tall. A detector tuned on an upright beginner would label every
   frame of a racing run a fall.

So `report.py` ships a *guard*, not a classifier: a segment whose mass sits more
than 0.45 body-heights off the board centreline is flagged "part of this clip is not
riding" and the user is told to trim and re-run. Real turns measured 0.04-0.30
across every clip, so the threshold has wide clearance. It makes no claim about
*why* those frames are not riding.

Detecting a fall properly needs a temporal signal — sustained loss of translation
combined with an orientation change — or a small dedicated classifier. Both are
future work; the guard stops the silent corruption in the meantime.

## A turn is 1-3 seconds; anything longer is a traverse

The 66 s follow-cam clip segmented into "6 turns" — but three of them lasted 23.7 s,
20.2 s and 13.8 s. Holding an edge for 20 s is a traverse across the slope, and
counting it as a turn flatters the run badly: it inflates the turn count and makes
every duration statistic meaningless. On that clip **58 s of 62 s was traverse and
only 4 s was actual turning**, across 3 genuine turns.

`report.py` now says so first (`traversing`), and labels the count as *segments*
rather than turns. The race segment, whose turns run 0.9-3.0 s, correctly does not
trigger it — which is the check that the threshold discriminates rather than just
firing on everything.

Because inference dominates the cost, `scripts/rebuild_report.py` regenerates a
report from an existing run's `metrics.csv` and `turns.json`, and can re-sign
fore/aft for a stance you know, so rewording or rethresholding findings never means
re-running a clip.

## A near-miss: the offset that was not an offset

Two Mixkit clips both measured a `toe_heel` median of **-0.0695 and -0.0693**.
Two unrelated riders landing on the same number to four decimal places looks
exactly like a geometric bias — plausibly the body's mass projecting near the
ankles, which sit heel-ward of the heel/toe midpoint used as the frame origin.
Subtracting a per-run baseline duly "fixed" the segmentation, turning one clip's
single long turn into four.

It was wrong. Two checks killed it:

1. A third clip, of racers, sat at median **-0.006 with 46% of samples positive**
   and 14 zero crossings. A constant offset would have pushed that clip negative
   too. The metric is well centred.
2. Frame-by-frame inspection of the Mixkit clip showed the rider on the **heel
   edge for all 24 seconds** — a heelside descent, never once switching to toe.

So the "offset" was two riders genuinely riding one edge, and the baseline
correction was inventing turns from the neutral middle of a heelside descent,
including a 10.7 s phantom "toeside". Absolute zero was right all along.

The lesson kept in the code: **riding one edge the whole way down is real, common,
and the single most useful thing to say about such a run** — so it is now a
first-class finding (`single_edge`) rather than something a normalisation hides.
`rolling_baseline` survives as opt-in, documented with why it is off.

## Still open

- **Absolute edge angle** needs world-up. Candidate: fit the slope plane to foot
  contact points across a run, which is self-contained but needs camera motion
  compensation.
- **Fusing the board mask into the 3D fit**, to pin true edges and correct for
  boot and binding offsets. The cross-check above validates the feet-only frame;
  fusion would tighten it.
- **Threshold calibration** against footage of riders whose level is known.
- Rider height in frame drives quality; chairlift-distance footage will be weak.

## The foot-derived board frame checks out against pixels (validated)

The design's load-bearing claim is that the rider's feet recover the board's
orientation. That is now tested against independent evidence rather than assumed.
COCO includes `snowboard` (class 31), and a *segmentation mask* yields a genuine
2D long axis — a bounding box does not, since a box carries no orientation.

`scripts/check_board_axis.py` compares the mask's principal axis against the
foot-derived long axis projected through the model's own predicted camera. Both
axes are sign-ambiguous, so agreement folds to [0°, 90°] where 0 is perfect and
random chance sits near 45°.

Measured on two clips:

| clip | rider px | frames with a mask | median | p25 / p75 | within 15° |
|---|---|---|---|---|---|
| follow-cam, indoor slope | 608 | **771 of 982** | **1.3°** | 0.5° / 2.5° | 98% |
| broadcast race segment | 267 | 15 of 82 | 3.2° | 1.9° / 5.6° | 87% |

Chance would be ~45°. So the feet recover the board's plane to about a degree when
the rider is big enough in frame for the segmenter to find the board at all — and
mask yield is what scales with subject size, not accuracy: 79% of frames at 608 px
against 18% at 267 px, while the median only moved from 1.3° to 3.2°.

This makes board detection a *refinement* rather than a dependency, as designed:
fusing the mask into the 3D fit to pin true edges and correct boot/binding offsets
remains future work.

## Example output

See the README — the annotated frame, metrics panel, edge trace and the board-axis
validation view are all shown there.
