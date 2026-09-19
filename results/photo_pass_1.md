# Photo pass 1 — four existing `room_pics/`, one vision agent each

2026-09-19. Four Haiku subagents, one photo each, identical brief, each told to
answer VISIBLE / PARTLY VISIBLE / NOT VISIBLE per item and to write NOT VISIBLE
rather than guess a number.

## What this pass established

**1. The wet pattern is a line source under the bar, not a radial patch from a tap.**
Three of four photos show a distinct darker/wet stripe on the floor directly
below where awnings hang, and all three describe the wetness as linear and
directional rather than radial.

- 20221227: "A distinct wet stripe/drip pattern runs along the floor directly
  below where the hanging awnings terminate"; "does NOT suggest a point-source
  tap origin; wetter areas correlate with where awnings are resting/draining"
- 20260416: "consistent with a line source (awning drip line or the bar
  draining) rather than a point tap source. No obvious radial wetness pattern"
- 20260513: "suggests line-source wetting (possibly drip from awning bar above)"

This supports `ambient_layout="bar"` as the dominant wetting mechanism during
the wash, over `two_tap`. It matches Andrew's own account (the floor is washed
while the awnings drain on the bar) and it is the first *independent* support
for that geometry.

CAVEAT ON THIS ITEM: the brief told the agents both that there is a bar and
that there are two taps, so they were primed with both options. The robust part
is the physical observation (a wet stripe under the bar, in three separate
photos years apart); the line-vs-point *interpretation* is partly my framing
and should not be counted as strongly as the observation.

**2. Dirt is fine grit, and it collects in the worn lanes.**
All four: fine grit/dust, explicitly not coarse debris. Three of four put the
heaviest concentration in the worn lanes. Consistent with the 20 g/m² fine
loading and with the model coupling high yield stress to the worn lanes --
grit that is harder to shift is grit that accumulates.

**3. Worn lanes follow the awning footprints.**
20221227: worn zones "appear to follow the footprint of the awnings". Supports
the worn-lane structure in `geometry.py` being placed by traffic, not arbitrary.

## What this pass did NOT establish

**No metric quantity is recoverable from these photos.** Every agent that
offered a width in feet also stated it had no usable scale reference; one
contradicted itself inside a single sentence ("difficult to quantify without
scale reference, but visible lanes are 2-4 feet wide"). Those figures are
confabulated and are NOT recorded here as measurements. Asked for a number
without a scale, a vision model will produce one anyway.

- Issue #10 (trough pond depth): STILL OPEN. The trough is at best partly
  visible in two photos and no standing water was identified in any of them.
  These photos cannot answer it.
- Issue #11 (single-pass swath): STILL OPEN. Needs the tape-measure video.
- Source locations: no tap and no corner pressure washer is visible in any
  photo (one photo shows a person holding a wand). Cannot confirm the source
  geometry.

## Method finding, for the next pass

Single-photo vision is reliable for presence/absence and qualitative structure
and unreliable for anything metric. So the split should be:

- ask vision for STRUCTURE (is it wet, where is the boundary, which lane, is
  there ponding) -- it is good at this
- get METRICS from a frame that contains a tape measure IN THE PLANE BEING
  MEASURED, and have the model read the tape rather than estimate the object

The "NOT VISIBLE" instruction worked for structural items and failed for metric
ones; for metric items the instruction needs to be a hard refusal plus a named
scale object ("report the width ONLY as a count of tape graduations").

---

# Photo pass 2 — 16 photos + 4 videos from the shop, 2026-09-19

Four Haiku agents over the 16 stills in groups of four, plus my own direct
reads of the tape close-ups. Videos extracted to frames at 1 fps (125 frames).

## The pond depth (issue #10) — corrected

An agent reported "3 to 3.5 inches" of standing water against a vertical tape
in 20260919_082256, confidence PROBABLE. THIS IS WRONG and was not used.

How it was caught: the same agent described 20260919_082310 as a vertical tape
with a waterline, and that photo plainly shows the tape lying FLAT on the floor
held in a hand. A demonstrable misread of a sibling image is grounds to verify
the headline number directly, which I did.

What 082256 actually shows: the tape is vertical with its hook resting ON the
floor and NOT submerged. The bright reflective band the agent read as a
waterline at the 3-4 inch graduation is the far surface of the puddle receding
away from a near-grazing camera, which projects to that height in image space.
20260919_082305 is unambiguous -- hook on essentially dry floor, small thin-film
puddle alongside.

Andrew, independently, on the same photo: "def not 3 in ? you are seeing wrong
it barely over the metal thing that stops the measure ? I could barely see any
reading ?"

Where that leaves the number. "Barely over the hook" is bounded by the hook's
own geometry: the flat foot plate is ~1/16 in (1.6 mm), the bent-up lip is
~1/4 in (6.4 mm). So the open-floor film is somewhere in 2-7 mm. The model's
`trough_retain_depth = 5.0e-3` sits inside that band -- NOT refuted, NOT
confirmed closer than a factor of ~3.

STILL OPEN: every close-up in this set is OPEN FLOOR. The pond Andrew described
is at the TROUGH base, and no photo in this set is of the trough. Issue #10 is
not yet answered by measurement.

## Floor condition — a possible change to a ground-truth assumption

ORIENTATION.md records the floor as two years overdue for recoating and "worn
through in the traffic lanes". The agents split hard on this: group A read the
wide shots as "does not appear freshly recoated ... established but moderately
worn"; group C read its four as "pristine", "topcoat appears intact", "no
visible wear-through".

My own reads of 082256 and 082305: a glossy, intact, light-grey floor with
sharp specular reflections and scattered fine grit. That is not a worn-through
surface.

This needs Andrew to resolve, because it changes the yield-stress map: has the
bay been recoated since the worn-lane description was written, or are these
photos of a section that was never in the traffic lanes? Recorded as a QUESTION,
not as a finding.

## Standing water is a thin film, widely distributed

Group A on the wide shots, consistently: mirror-like reflections of the ceiling
lights spread broadly across the floor, "not pooled in one area", "broad sheen
rather than dark edges". My own reads agree. This is the `initial_film_mean =
2.5e-3` regime, and it is what the shallow-water solver assumes.

## Dirt is fine grit, and the loading looks LIGHT

All eight agent-groups across both passes: fine grit specks, explicitly not
coarse or chunky. But the close-ups show a floor that is mostly clean with
isolated speck clusters, not a uniformly loaded one. Issue #2's open question
(is the ~30-min deep clean a normal floor or a filthy one?) is NOT resolved by
these photos, and if anything they suggest the 20 g/m2 uniform loading is
heavier and more uniform than what the floor actually looks like between jobs.

## Method: what worked and what did not

WORKED: forcing VISIBLE / PARTLY VISIBLE / NOT VISIBLE per item; grouping four
photos per agent so it could cross-reference; normalising EXIF rotation before
handing images over.

FAILED: the hard rule against unscaled measurements. Both passes produced
confident feet-and-inches numbers with no scale in frame, and pass 2 produced a
precise wrong number WITH a scale in frame by misreading perspective. The rule
reduced the rate; it did not stop it.

RULE GOING FORWARD: a measurement from a vision agent is a HYPOTHESIS until a
human or a second reader verifies it against the original image. Never let one
into the model directly. The cheap check that caught this one was looking for
an internal contradiction across the agent's own sibling images.

## Video pipeline

`scripts/media/video_frames.py` (probe / extract / montage / auto). Two traps
found and fixed against these clips:
  - ffmpeg drops display-matrix auto-rotation as soon as -vf is supplied, so
    portrait phone clips extract on their side. `_transpose_for()` reapplies it
    from the probed rotation (these clips: rotation=-90 -> transpose=2).
  - the ffmpeg `xstack` montage is brittle; contact sheets are tiled with PIL.
Frames are named by timestamp (t0007.00s.jpg) so a filename IS a time
coordinate -- which is what makes walking speed recoverable from two frames.

## Videos — CORRECTED after looking properly

An earlier version of this file said the videos contain no tape measure and no
side-on wand, based on an agent triage of contact sheets. THAT WAS WRONG, and
it was wrong because I delegated the looking after already having caught the
same agent class misreading images twice. Andrew pushed back ("Please try to
look harder? I took a lot video"). He was right. Looking at every sheet myself
and then at native-resolution frames:

### 20260919_082524 (41 s) — THE MEASUREMENT CLIP
t0008-t0010 is a continuous side-on sequence with the ENTIRE lance visible from
grip to nozzle, AND the 6 ft tape laid flat on the floor running from its hook
out to directly beneath the nozzle tip. This is the issue #11 setup and it is
set up correctly. The triage reported "WAND SIDE-ON: none" and "TAPE MEASURE:
none" for this exact sheet.

Note the clip's phone was ROTATED MID-RECORDING: t0001-t0011 need a further 90
CCW while t0018+ are already upright. One display-matrix flag cannot describe
both halves, which is why a global transpose kept failing. Per-segment rotation
is needed, not a per-file one.

### 20260919_082607 (30 s) — chest-mounted POV of a pass
Top-down operator view: shoes, lance, nozzle and floor all in frame throughout,
with the tape appearing around t0016-t0017. Good for where the nozzle sits
relative to the floor during an actual walking pass.

### 20260919_082903 (34 s) — layout tour
Drums and buckets along the wall, awnings hanging on the bar (t0003-t0004),
hoses, and floor wet patches from t0012 on. Best clip for bay layout.

### 20260919_083021 (20 s) — THE WET-PATTERN CLIP
The best evidence in the whole drop for issue #9. Large standing-water regions
with sharp scalloped wet/dry boundaries (t0000-t0013), and at t0009-t0011 dark
awnings hanging directly above wet floor. That is the line-source wetting seen
directly, not inferred: awnings above, wet floor beneath.

## What is still blocked, and why it is NOT the footage's fault

The tilt angle cannot be recovered from 082524 even though the geometry is all
in frame. The camera is wide-angle, close to the subject, and lies nearly IN
the plane of the wand's motion, so the lance-to-floor angle in the image is a
projection of the true 3D angle, not the angle itself. Measured in projection
the lance sits about 48 deg off the tape line; the true value could differ by
20 deg or more, and the model's two regimes are separated by 15 deg. So this
frame cannot settle it. Recording that as a measurement would repeat exactly
the error that produced the bogus 3-inch pond reading.

THE ONE QUESTION THAT UNLOCKS THE CLIP: what was the tape laid out to measure,
and what did it read? If the hook was at the operator's feet and the blade ran
to where the jet lands, then that reading IS the lead distance
`standoff*tan(tilt)` -- and with standoff it gives tilt directly, with no
image geometry needed at all. The footage plus one number from Andrew is
enough; the footage alone is not.

