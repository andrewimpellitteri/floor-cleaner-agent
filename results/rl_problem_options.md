# Where an interesting RL problem could live in this simulation

Andrew asked what we are doing wrong with the RL, and whether the variation is
just small. The answer turned out to be neither: the learner works, and the
variation is large, but the task as modelled has no structure a learner can
exploit. This is the survey of what would change that, with what is measured
separating the options rather than what sounds good.

## What a learnable problem looks like here, measured

The `reach` diagnostic and the cleaning task run on the same trainer, network,
optimiser and batch size, and differ by a factor of fifteen in the only number
that matters to a policy gradient:

| | `adv_std` | random vs optimum |
|---|---|---|
| `reach` (walk to a target) | **1.57** | random −116 against a +33 ceiling |
| cleaning | 0.08 – 0.14, *collapsing* | random 0.235 against 0.333 |

That second column is the real tell. In the cleaning task a uniform-random
policy already scores **70% of the best known strategy**. There is no gradient
to climb because almost everything works.

## Why the current task has no structure

Three properties, each measured, and each one removes a reason to think:

1. **Cutting is saturated.** Da ≈ 42 (`math_analysis.pdf`), and the measurement
   agrees: `cut%` is *identical* at 92.3% for 8 gpm and 20 gpm
   (`results/phase_breakdown.txt`). Technique collapses to a constant.
2. **Coverage is order-independent.** Sweeping the floor gives near enough the
   same result in any order, so there is nothing to sequence. Five hand-built
   lane spacings over a 2.6× range all land at 0.28–0.35
   (`results/macro_ceiling.txt`).
3. **The agent is told the answer.** `_channels` hands it a live `grip` map —
   the yield field, i.e. exactly where the worn lanes are — for the whole floor,
   every step. Finding those is the only genuinely hard thing in the job.

## The water claim is untested, not refuted

`math_analysis.pdf` argues that because a lone wand sustains a bed shear near
the 0.1 Pa deposit threshold, "operators should help each other superlinearly".
`config.py` flags the same thing and is explicit that it is "a PREDICTION, not a
fudge ... worth testing directly rather than assuming". The two disagree on the
numbers (0.61 mm / 0.09 Pa against 0.85 mm / 0.125 Pa), but both put a lone wand
within about 25% of the threshold.

The only sweep we have is `far_to_near` clean fraction against supply:

| supply | 8 gpm | 20 gpm | 45 gpm |
|---|---|---|---|
| clean  | 71.5% | 85.3% | 91.3% |
| step   | -- | +13.8 | +6.0 |

That is concave -- but all three points sit ABOVE the predicted threshold, so it
does not test the claim. A regime change at 4 gpm would not appear anywhere in
this table. I first read this as refuting the superlinearity; it does not. It is
measured in the wrong place.

This is the cheapest open question left, and it gates two of the options below:
sweep 2/4/6/8 gpm and look for convexity as bed shear crosses 0.1 Pa. If it is
there, low water is the interesting regime AND the shared-water prediction
holds, which revives the two-operator option. If it is not, both weaken together.

## Where the loss actually is

`far_to_near`, 8 gpm, 6 seeds, 30 min (`results/phase_breakdown.txt`):

- **7.7%** of the grit is never cut at all (0.088 kg stuck)
- **12.5%** is cut but never reaches the trough (0.142 kg left loose)

The second is nearly twice the first, and it is a **transport** failure, not a
cutting one. Transport is genuinely path-dependent in a way coverage is not:
where you push slurry determines whether it lands somewhere with enough film to
carry it onward, and later passes can re-strand what earlier ones moved. That
is the one place in this model where ordering provably matters.

## Options, ranked

### 1. Partial observability — BUILT, untrained (`--blind`, issue #8)

The global map becomes a remembered view; unvisited floor reads blank,
including the yield field. The agent must *discover* the worn lanes.

- **Structure**: exploration/exploitation, and memory of coverage.
- **Evidence for a prize**: one floor in four has a worn lane holding 87% of the
  leftover grit, 44.3% survival in lanes against 1.8% elsewhere; the other three
  have 7–14% (`results/residual_where.txt`). A fixed sweep cannot condition on
  which floor it drew.
- **Fair benchmark**: `far_to_near` needs no perception at all, so a blind
  operator can execute it. The comparison stays honest.
- **Cost**: done. One flag, ~$1 to train.
- **Risk**: strictly harder than the current task, and harder is not the same as
  learnable. The worn-lane prize is large on one floor in four and small on the
  rest.

### 2. Scarce water, where transport binds — CHEAPEST TO TRY

Run at 4 gpm instead of 8–20. Cutting is saturated at every supply, so lowering
water does not make cutting matter; it makes **delivery** the binding constraint,
and delivery is the path-dependent part.

- **Structure**: routing slurry to the trough, where earlier passes change what
  later ones can move. Genuinely sequential.
- **Evidence for a prize**: 12.5% of the grit is already cut-but-stranded at
  8 gpm. At 4 gpm the model puts bed shear below the deposit threshold entirely.
- **Cost**: a config change. Essentially free.
- **Open question, and the thing to measure first**: does strategy variance
  actually *widen* at 4 gpm? At 8 and 20 gpm the spread between good strategies
  is about the same, so the regime change is predicted, not observed.

### 3. Non-uniform effort allocation

Spending strokes where the grit is rather than sweeping uniformly. Measured
separately in `results/` — every schedule in the ceiling test allocated
uniformly, which was a gap in that test rather than a finding.

### 4. Two operators sharing water — GATED ON THE 4 gpm SWEEP

The most interesting option if the threshold prediction holds, and a weak one if
it does not, which is exactly why the sweep above comes first.

- **If the prediction holds**: water is a genuinely shared resource with a
  nonlinearity, so where the second operator stands changes what the first can
  achieve. That is real coordination, it is the one thing single-agent scripts
  structurally cannot do, and it answers a question Andrew actually has
  ("split up or work together?" -- he describes 1-4 people in the bay).
- **If it does not**: two operators are close to additive, each cleaning their
  half, and the coordination collapses back into dividing coverage.

- **Cost**: the largest of these. `p_normal` is a scalar peak pressure paired
  with an `intensity` field, so two jets at different poses cannot be expressed
  without changing `physics_substep` to take a pressure field. Plus a second
  operator in the state, action dim 5 → 10, and the observation.

## What would close this out

The honest summary is that the deliverable never needed RL, and the sim as
built is a coverage problem with a known-good scripted answer. Everything above
is about making a *different* problem that is interesting to train on. If that
is the goal, option 2 is nearly free and should be measured before option 1 is
trained, because if transport does not bind harder at 4 gpm then the most
promising source of sequential structure in this model is not there either.
