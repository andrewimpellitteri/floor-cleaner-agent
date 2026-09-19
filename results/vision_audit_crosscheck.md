# Cross-model audit of the shop media (2026-09-19)

Motivation: the in-house Haiku subagent pass produced confident, precise, WRONG
readings three times on this drop -- a 3-inch pond depth that was a perspective
artifact, "no wand in frame" on a frame containing a full side-on lance, and
"no tape in frame" on a frame containing a tape. One vision reader cannot be
trusted for anything metric.

So every claim below was re-tested against `gpt-5.5` (OpenAI) at high reasoning
effort, BLIND: the prompt never contained my reading, the Haiku reading, or
Andrew's. Critical questions were asked twice in independent calls, on the
principle that two reads disagreeing is itself the finding.

Tooling added: `scripts/media/vision_audit.py` (stills, OpenAI) and
`scripts/media/video_audit.py` (native video, OpenRouter).

## 1. Pond depth -- CORRECTION CONFIRMED, and tightened

Two independent blind reads of 20260919_082256, neither told what to expect:

  Read 1: "I do not see a waterline crossing the blade at any inch graduation.
           The apparent horizontal/irregular line near the 2-3 inch area looks
           like the far edge of a puddle on the floor behind the tape, not a
           water surface intersecting the tape blade."
  Read 2: "The horizontal/irregular line visible behind the tape appears to be
           the far edge of a puddle on the floor, not a water surface crossing
           the vertical blade."

Both independently identified the exact artifact, and both put the depth at a
thin film, "perhaps under about 1/8 inch". Both confirm the hook is visible and
not submerged.

So three independent sources now agree the 3-3.5 inch reading was wrong: my own
direct read, Andrew ("def not 3 in ... it barely over the metal thing that
stops the measure"), and two blind GPT-5.5 reads.

NUMBER FOR THE MODEL: open-floor film is UNDER ~3 mm (1/8 in), tighter than the
2-7 mm bracket I derived from the tape-hook geometry alone. `initial_film_mean
= 2.5e-3` is consistent with this. NOTE this is OPEN FLOOR, and says nothing
about `trough_retain_depth`, which governs the trough and remains unmeasured.

## 2. Wand geometry -- my camera diagnosis confirmed, my nozzle read CORRECTED

Two blind reads of the 082524 t=9.0 s frame:
  - Camera is "nearly within or looking along the wand's plane of motion, not
    cleanly positioned to the side for a true profile view". Exactly the
    problem I identified.
  - Both: this camera position is NOT adequate to recover true tilt. Apparent
    angle 55-60 deg above floor (~30-35 deg off vertical), and both volunteered
    unprompted that the apparent angle cannot be trusted as the 3D angle.
  - CORRECTS ME: I wrote that the nozzle "sits essentially at the floor". Both
    reads put it 1-4 inches above. A later magnified crop supports "held
    slightly above, soft shadow beneath the tip, no definite hard contact".
  - No spray in any frame.

My own projected figure of ~48 deg off the tape line and their 30-35 deg off
vertical are both projections of the same unmeasurable quantity, and the fact
that two careful estimates disagree by ~15 deg -- exactly the width of the
model's regime boundary -- is the proof that this frame cannot settle it.

## 3. What the wand video is actually FOR

An ordered 9-frame sequence (6.0-12.0 s) given to gpt-5.5 as a time series:
  "setting up / demonstrating / measuring, not actually washing"
  "No visible spray is supported in any frame"
  nozzle lowered to the tape at 8.67-9.67 s, then withdrawn as he walks off
  "most likely ... demonstrating or measuring the nozzle's distance/stand-off
   from the floor and/or angle relative to the floor, using the tape measure as
   a reference"

So the clip is a deliberate geometry-documentation shot. Andrew set it up
correctly and it records the right quantity.

## 4. Why it still cannot be cashed in -- a hard resolution limit

Magnified crops (8x upscale of the native 1280x720 frame) put to gpt-5.5:
  "No legible numbers or inch/foot graduation marks. The tape blade is visible
   only as a blurred diagonal line/strip."
It also establishes the layout: the blade passes UNDERNEATH the nozzle and
continues to a hook beyond it. So the tape is a ruler laid across the floor and
the nozzle marks a POSITION ALONG IT -- and that position's reading is exactly
the lead distance `standoff*tan(tilt)`.

The number is in the scene and is destroyed by the encode. 720p is not enough
to resolve tape graduations at that working distance. No processing recovers it.

RESOLUTION: Andrew reads the number off the tape himself, or reshoots that one
frame as a close-up STILL of the tape at the nozzle position. Either is seconds
of work and turns the existing footage into a tilt measurement.

## 5. Wet pattern (issue #9) -- weak support for the line source

Blind 8-frame walkthrough sequence of 083021:
  - wet regions are SHEETS WITH SCALLOPED EDGES, not circular patches
  - wet areas sit directly beneath hanging awnings in several frames
  - best fit "line/multiple-drip source plus shallow runoff, but not proven";
    explicitly cannot rule out tap water that then spread by slope
  - roughly half the visible floor is wet
  - no trough confirmed visible in this clip

This is genuine but WEAK support for `ambient_layout="bar"` over `two_tap`. The
honest statement is that the geometry is consistent with the bar and
inconsistent with neat point sources, not that the bar is proven.

## 6. The floor condition question -- now independently supported

Unprompted, on the walkthrough frames:
  "I do not see distinct, consistent traffic lanes where the epoxy visibly
   differs from surrounding areas. Worn-through traffic lanes are not supported
   by these frames."

That is a second model family, asked blind, reaching the same conclusion I did
from the tape close-ups. ORIENTATION.md records the floor as two years overdue
for recoating and worn through in the traffic lanes. The current photos do not
show that. This governs the spatially varying yield stress in `geometry.py`, so
it needs Andrew to settle: has the bay been recoated?

## Method note

The blind two-read protocol earned its keep. It confirmed one correction,
overturned one of my own readings (nozzle height), and produced two
independently-derived estimates that disagree by exactly the amount that proves
the measurement is not there. A single confident reader would have given a
number in every one of those cases.
