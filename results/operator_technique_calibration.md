# Operator technique, measured from Andrew — and what it implies

2026-09-19. Ground truth from Andrew, in his words:

  standoff: "it varies and the agent should be able to vary the distance?
             But like 12\" average?"
  impact:   "And it hits like 12-18\" in front?"

He is right that it varies, and the model already agrees with him on that
point: `standoff` is an explicit per-step action, range 2.0-43.3 in, with a
slew limit of 47 in/s. His 12 in average sits comfortably inside that.

## The geometry this pins down

The jet lands `standoff * tan(tilt)` ahead of the operator, so a 12 in standoff
with a 12-18 in lead fixes the working tilt:

    lead 12 in  ->  tilt 45 deg off vertical
    lead 18 in  ->  tilt 56 deg off vertical

This is the first direct measurement of the wand angle, which is the control
variable the whole model turns on. It was previously unmeasured.

## What it does NOT show (a claim I nearly made and had to retract)

ORIENTATION.md illustrates the lead distance with "a 72 deg tilt and 30 cm
standoff ... most of a metre". At 12 in standoff, 72 deg delivers 0.2 kPa --
0.1x the base yield stress, i.e. it cannot cut at all. I was about to report
that the model is calibrated into a non-cutting regime.

That would have been WRONG, and checking `baselines.py` before claiming it is
what caught it. The baselines do not use 72 deg:

    PushSweep / far_to_near : standoff 0.40 m, tilt 0.55 rad = 31.5 deg
    BlastThenSweep blast    : standoff 0.40 m, tilt 0.22 rad = 12.6 deg
    BlastThenSweep sweep    : standoff 0.35 m, tilt 1.25 rad = 71.6 deg

72 deg appears only as the SWEEP phase, where cutting is deliberately traded
away for push. The ORIENTATION figure is an illustration, not the operating
point. No miscalibration.

## The real comparison, and the finding

Peak normal (cutting) pressure from `jet_peak_pressure`, against
`yield_mean = 2500 Pa` and a worn-lane centre of `2500 + 3000 = 5500 Pa`:

  technique                      standoff  tilt   lead    p_normal   x base  x worn
  far_to_near baseline            15.7 in  31.5d   9.7 in  10.56 kPa   4.23    1.92
  Andrew, 12 in lead              12.0 in  45.0d  12.0 in   8.70 kPa   3.48    1.58
  Andrew, 18 in lead              12.0 in  56.0d  17.8 in   3.28 kPa   1.31    0.60

TWO THINGS FALL OUT.

1. The simulation's baseline is slightly MORE aggressive than Andrew's actual
   technique -- 0.82x his cutting pressure at the 12 in lead, 0.31x at the
   18 in lead. So the sim is OPTIMISTIC about cutting relative to real
   practice. The real job should be harder than the sim says, not easier.
   That matters for every time-to-clean number produced so far.

2. ANDREW'S OWN STATED RANGE STRADDLES THE WORN-LANE CUTTING THRESHOLD.
   At a 12 in lead the jet delivers 1.58x the worn-lane yield stress and cuts.
   At an 18 in lead it delivers 0.60x and CANNOT CUT WORN LANES AT ALL.
   Between those two the cutting pressure changes by 2.6x (8.70 -> 3.28 kPa)
   for what is, to the operator, a small change of wrist angle.

## The practical recommendation this supports

If stubborn patches are being left in the traffic lanes -- the worn strips
where pieces get dragged -- the fix is to STAND THE WAND UP so the jet lands
closer to your feet, about 12 in rather than 18 in ahead. Not to get closer to
the floor, and not to go slower.

This is consistent with the independently measured residual-location result
(`results/residual_where.txt`): in 1 of 4 seeds, 86.8% of the leftover dirt sat
in the worn lanes. Grit in worn lanes is exactly the grit that a laid-over wand
cannot lift.

CAVEAT, and it is a real one: the worn-lane threshold depends on
`worn_lane_boost = 3000 Pa`, which is a guess, and one that the current photos
arguably do not support (see `results/vision_audit_crosscheck.md` section 6 --
two model families, asked blind, see no worn-through lanes). A sweep of that
parameter is running. If the boost is smaller than assumed, the 18 in lead
stops being a cliff and becomes merely less efficient. The DIRECTION of the
recommendation (stand it up) survives either way; its URGENCY does not.

## Still open

The one number still missing is the walking speed, which together with the
swath sets coverage. Everything else in the operator's control loop is now
either measured or bounded.
