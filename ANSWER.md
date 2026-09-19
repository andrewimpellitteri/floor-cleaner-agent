# How to wash the bay floor — what the simulation actually supports

This is the deliverable ORIENTATION.md asks for: a defensible answer to the
practical question, not a trained model. Everything below is measured, and each
claim says how strongly.

**Read the confidence labels.** They are not decoration — some of these are
solid across every test thrown at them, and some are single-seed hints.

---

## The headline

**Coverage beats technique.** Across every experiment run, what separates a good
result from a bad one is what fraction of the floor the jet actually passes
over — not how hard it hits, not how much water is running, and not how
cleverly the operator picks where to go next.

The attribution measurement is blunt about it: of the grit still stuck at the
end of a run, **coverage gaps outweigh genuinely-uncuttable grit by 3–4 : 1 by
mass**. In one seed, 98% of the leftover grit was simply never walked over.

So the single most valuable habit is **overlap your passes enough that no
stripes are left between them.**

---

## 1. Work in full passes from the wall to the trough — CONFIDENT

| strategy | coverage | fraction clean |
|---|---|---|
| full passes, wall → trough | **0.935** | **0.681** |
| short bands, trough outward | 0.716 | 0.461 |
| random | 0.831 | 0.401 |
| always go to the dirtiest spot | 0.041 | 0.061 |

*8 gpm, 30 min, 4 seeds.*

The long march from the wall wins by **+22 points of clean** over working
outward in bands near the trough. The reason is visible in the coverage column
and nowhere else: banding leaves 28% of the floor untouched.

## 2. Do not chase the dirty patches — CONFIDENT, AND COUNTERINTUITIVE

"Always go to the worst spot" is the most natural-sounding tactic and it is the
worst one tested, by a wide margin — 6% clean against 68%.

The interesting part: measured per unit of work it is the **most efficient**
strategy of the four (eta 0.858, the highest). It cuts well. It just never goes
anywhere, so coverage collapses to 0.041. Efficiency per pass is not the thing
that matters; area swept is.

## 3. Do not blast everything then sweep everything — CONFIDENT

| | cut % | delivered % | grit left stuck | clean |
|---|---|---|---|---|
| one continuous pass | **92.3 ± 2.0** | 79.8 ± 3.9 | **0.088 kg** | **72.2 ± 7.0** |
| blast phase, then sweep phase | 71.5 ± 3.5 | 60.1 ± 3.2 | 0.322 kg | 53.6 ± 4.9 |

*6 seeds, 30 min.* Splitting the job into a cutting pass and a pushing pass
leaves **3.7× more grit stuck** and ends ~19 points dirtier. Cut and push in the
same motion.

This ranking survives a 4× sweep of the worn-lane hardness assumption
(far_to_near wins at every level, +0.16 to +0.20, 1.9–2.4 sd), so it does not
depend on how worn the floor actually is.

## 4. Keep the wand low — CONFIDENT

There is a hard cliff at about **0.80 m (31 in)** of standoff: the effective
swath goes to **0.0 cm** and cutting to **zero**. Above that height the jet
simply stops cutting, because the pressure has fallen below the grit's yield
stress.

This was verified to be water-invariant (15.6 / 15.9 / 16.0% across a wide range
of ambient flow, ±2.0), which is what you expect from a pressure threshold
rather than a flow effect.

## 5. Stand the wand up, especially in the traffic lanes — LIKELY

From Andrew's own measurements (12 in standoff, jet landing 12–18 in ahead,
i.e. 45–56° off vertical):

| lead distance | tilt | cutting pressure | vs worn-lane yield |
|---|---|---|---|
| 12 in | 45° | 8.70 kPa | **1.58× — cuts** |
| 18 in | 56° | 3.28 kPa | **0.60× — does not cut** |

A small wrist movement changes cutting pressure by **2.6×** and crosses the
threshold at which worn lanes stop responding at all.

Independent support, from a prior experiment recorded in `baselines.py`: an
earlier default of 54° "pushed well but delivered only 1.1x adhesion, so it
swept the loose layer to the trough and left the adhered grit untouched — the
floor stopped getting cleaner after about two minutes." 54° is inside Andrew's
stated range.

Labelled LIKELY rather than CONFIDENT because the worn-lane threshold depends on
`worn_lane_boost`, which is an estimate the current photos do not clearly
support. A direct outcome test is running.

## 6. More water does not clean better — CONFIDENT

Going from 8 to 20 gpm leaves the **cutting identical** (92.3% both) and the
stuck grit identical (0.088 kg both). What extra water buys is **delivery** —
moving already-loosened grit to the trough (79.8% → 86.0%).

Andrew's own observation matches: "even if you run continuous water through the
faucets without power washing it doesn't really clean it from a point source."
Water carries; the jet cuts. And since you cannot flood the bay from a tap, this
is not an available lever anyway.

Source layout (taps vs the drip line under the bar) moves outcomes **within
noise**, with the sign flipping between water levels. Do not worry about it.

---

## What this does NOT establish

**The absolute times are not validated.** The simulation never fully finishes:
the best strategy reaches 72% of cells under threshold in 30 minutes and leaves
a worst-cell residual 47× over the limit. Andrew reports ~30 minutes for a deep
clean where "the floor looks good". Those two statements are not yet reconciled,
and the most likely explanation is that the simulator's completion test —
*every single cell* under 2 g/m² — is a stricter standard than "looks good"
(issue #2). **Treat the rankings as the result and the clock as unvalidated.**

**The learned policy is not part of this answer.** Reinforcement learning was
attempted and does not currently work, for a well-understood reason: the reward
has no gradient (see `results/shaping_gradient_ablation.txt`). Run to completion
the trained policy cleans 3.1% against the hand-coded strategy's 72.2%. Every
recommendation above comes from physics experiments and scripted strategies.

**Still unmeasured:** walking speed, which together with swath sets coverage —
and coverage is the thing that matters most. This is the highest-value number
still outstanding.

---

## If you only remember three things

1. **Overlap your passes.** Most of what gets left behind was never touched.
2. **Full passes wall-to-trough, not short bands, and never chase dirty spots.**
3. **Keep the wand low and stood up** — under ~2.5 ft, landing ~1 ft ahead of
   you, not 1.5 ft.
