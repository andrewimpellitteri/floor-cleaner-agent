# Operator pace, measured from video

The last unmeasured variable in the control loop, and the one that converts
every ranking in ANSWER.md into a clock: job time is `T = A / (W_eff * v)`,
linear in `v`, and jet dwell time is `t_patch / v`, inverse in it.

## Result

Purpose-shot fixed-camera clip `20260919_165046.mp4` (2m16s, 1280x720 @ 30 fps),
with a 10 ft tape laid on the floor as the metric reference. 102 s tracked,
23 sustained runs of >1.2 s continuous travel. **The distribution is bimodal**,
which is what makes it useful -- it separates the two speeds the model needs:

| mode | runs | median | what the model assumes | verdict |
|---|---|---|---|---|
| **working** (wand down, pushing) | 11 | **0.37 m/s** | `push_speed` 0.45 | ~1.2x too fast |
| **transit** (repositioning) | 12 | **0.95 m/s** | `walk_speed` 1.00 | **confirmed** |

Spot-checked by eye: frames drawn from the slow-mode runs show him bent with the
wand down over wet floor; the fastest runs show an unmistakable walking stride.

`walk_speed` had never been checked against anything and comes out right.
`push_speed` is high but by far less than the first clip suggested.

## Correction to the earlier estimate

The first pass on `20260919_082524.mp4` reported **0.240 m/s** and "1.9x too
fast". That is superseded. Two reasons, both mine:

1. **It blended working with standing still.** That clip had him working largely
   in place; 50% of frames were stationary, and a median taken over "moving"
   frames at a 0.15 m/s threshold still folds in a lot of shuffling. Measuring
   sustained runs instead of thresholded instants raises the working figure to
   0.37.
2. **It assumed constant depth.** Fair for that clip, wrong in general. In the
   new one his foot row runs 344-456 px and his apparent height 134-252 px -- a
   1.9x change in scale across the clip. Andrew flagged this himself ("depth is
   weird").

## Method

Ground-plane model. Camera at height `hc`, no roll, horizon at image row `yh`,
focal length `f`, principal point at centre. A floor point at image `(x, y)` is
at `X = (x-cx)*hc/(y-yh)` laterally and `Z = f*hc/(y-yh)` in depth.

* **Horizon** from the operator himself: for an upright figure on a plane,
  `h_px = k*(y_foot - yh)`. Regressing height on foot row over 1147 detections
  gives `yh = 210` px (r = 0.916).
* **Scale** from the 10 ft tape, at the two placements it occupied. One placement
  cannot separate `hc` from `f`; two at different positions and orientations
  can. They agree on `hc = 1.62 m` to 8%.
* **Speed** from displacement over sustained runs, NOT frame-to-frame gradients.
  Depth is badly conditioned here -- `dZ/dy ~ 0.032 m/px`, so 3 px of foot-row
  wobble manufactures ~1 m/s at 10 fps, and the naive gradient gave a nonsense
  IQR topping 1.07 m/s. Lateral is 5x better at `dX/dx ~ 0.007 m/px`. Over a
  2 s run the real displacement dwarfs the jitter in both axes.

Uncertainty ~±10-15%: `f` is inferred rather than measured (the two tape
placements agree best near 1100 px, a normal phone lens), and the two placements
differ by 8% on `hc`.

## Three wrong turns, recorded because each produced plausible numbers

1. **Grayscale segmentation caught only his hoodie.** A light band at his waist
   splits the silhouette, so the largest connected component was the torso and
   "foot row" was often his waist. Luminance contrast on the shorts is ample
   (108 against a 168-181 floor) -- the fix was morphological (close with a tall
   kernel before labelling), not a threshold change. Caught by drawing the boxes
   and looking, not by the numbers, which looked fine.
2. **Crude tape endpoints.** Taking the mean row of the few leftmost pixels of a
   2 px line returned y = 431 one second and 466 the next for the same physical
   placement. A 15 px error in row is a large error in depth, and it made the two
   placements algebraically inconsistent with *any* focal length. Fixed by
   tracing the ridge column-by-column and line-fitting over hundreds of columns.
3. **Frame-to-frame gradients** — see above.

## What this still does not establish

* One operator, one session. The 11 working runs spread 0.23-0.58 m/s; that is
  real technique variation and the sample does not characterise it.
* The implied height from the calibration is 1.59 m, which is short. Most likely
  the detected foot row sits slightly above his true feet (grey shoes against
  white epoxy), which would bias distances up a few percent.

## Consequence for the model

`push_speed` 0.45 -> ~0.37 m/s does NOT change any cleaning conclusion. Cutting
is saturated (Da ~ 42, `math_analysis.pdf`), so a slower pass removes the same
grit, and every comparison in ANSWER.md is between strategies at equal speed.

It changes the CLOCK by ~20% on the pushing phase, and confirms the transit
constant outright. That is a much smaller correction than the 1.9x the first
clip implied, and it is the direction that matters: the 15-minute routine pass
and the ~30-minute deep clean both stand.
