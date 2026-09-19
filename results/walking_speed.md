# Operator pace, measured from video

Andrew asked for this directly. It was the last unmeasured variable in the
control loop, and it is the one that converts every ranking in ANSWER.md into a
clock: job time is `T = A / (W_eff * v)`, linear in `v`, and jet dwell time is
`t_patch / v`, inverse in it.

## Result

Source: `20260919_082524.mp4`, the fixed-camera stretch t = 19.5-31.0 s
(10.6 s, 159 usable frames at 15 fps).

| quantity | measured |
|---|---|
| in motion | 50% of the time |
| pace while moving | **0.240 m/s** (IQR 0.211-0.292) |
| averaged over motion + pauses | 0.166 m/s |
| sustained push, best run | 0.357 m/s over 0.71 m / 2.0 s |
| sustained push, next two | 0.234, 0.218 m/s |

**The model's `push_speed = 0.45 m/s` is about 1.9x too fast** against the
median moving pace.

## Method, and why it is trustworthy where it is

The camera is static (phone propped on the floor), so a per-pixel median over
the window is the empty bay and the largest moving blob is the operator. He
crosses the frame at constant depth -- apparent area varies only 34% across the
window -- so pixels map to metres by a single constant rather than a
perspective model.

SCALE, from two references that agree to under 1%:

* **His body.** Read off a ruler overlay at full resolution rather than trusted
  to a threshold: feet at x = 120, head at x = 465, so 345 px at 720 wide = 259
  px at the 540 analysis width. He is bent at the hips while pushing, so
  standing height is ~259 / 0.937 = 276 px (legs straight, torso at ~30 deg).
  1.78 m / 276 px = **0.645 cm/px**.
* **The wall flag.** 95 x 72 px at 720 wide = 71 px long at 540. A 12x18 in
  flag (0.457 m) gives **0.643 cm/px**. No other standard flag size is within a
  factor of three of being consistent, so the data picks the size rather than
  the size being assumed.

Adopted 0.644 cm/px. Residual scale uncertainty is his true height, +-4.5%.

## Two wrong turns, recorded so they are not repeated

1. **Grayscale segmentation caught only his legs.** His shirt is tan and the
   wall is neutral, so in luminance the torso vanished and the bounding box was
   his lower body. Using that as "height" put the scale out by ~35%. Fixed with
   an R-B chroma channel, and then verified by eye on annotated frames rather
   than trusted -- the first two attempts both looked plausible in the numbers
   and were both wrong.
2. **The first fit averaged walking with standing still.** He works in place for
   half the window. A single regression over the whole segment returned 0.22 m/s,
   which is neither his working pace nor his transit pace but a blend of the two
   weighted by how much he happened to pause. Motion is now separated at a
   0.15 m/s threshold and reported as pace-while-moving plus a duty cycle.

## What this does NOT establish

* **One clip, 10.6 s, one operator.** Three sustained pushes, spread
  0.218-0.357 m/s. That spread is real technique variation, not noise, and the
  sample is far too small to characterise it.
* **No transit speed.** `walk_speed = 1.0 m/s` (repositioning between passes)
  remains completely unchecked -- he is working throughout this clip.
* **The 50% duty cycle is not interpretable yet.** It could be stubborn spots,
  hose management, or filming. It matters: at face value it would nearly double
  the job estimate again, on top of the 1.9x.

A purpose-shot replacement is specified in the "Wash Bay Pace Test" field card
(two markers a measured distance apart, so the measurement reduces to
timestamps and needs no camera calibration at all).

## Consequence for the model

`push_speed` 0.45 -> ~0.25 m/s does NOT change any cleaning conclusion. Cutting
is saturated (Da ~ 42, see math_analysis), so a slower pass removes the same
grit; the ranking of strategies is unchanged, and every comparison in ANSWER.md
is between strategies at the same speed.

What it changes is the CLOCK, and only the clock -- the job is ~1.9x longer per
unit area swept than the model assumes. Do not adjust the constant until the
purpose-shot clips land: correcting a 15-minute episode length on 10.6 seconds
of evidence would be trading a known error for an unknown one.
