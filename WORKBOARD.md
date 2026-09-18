# Workboard — floor_clean

Optimising how a wash-bay floor gets pressure-washed, well enough to settle the
argument about technique with evidence instead of opinion.

**Read `ORIENTATION.md` first.** It has the domain facts, the physics, and the
invariants you must not break. This file is only the task list.

Status key: `TODO` · `WIP` · `DONE` · `BLOCKED`

---

## Done

| | Task | Where |
|---|---|---|
| DONE | Physical config calibrated to the real bay | `floorclean/config.py` |
| DONE | Bay geometry, worn lanes, flat spots, standing water | `floorclean/geometry.py` |
| DONE | Fan-tip impingement model (standoff/tilt trade-off) | `floorclean/jet.py` |
| DONE | Local-inertial shallow water + 3-phase sediment | `floorclean/physics.py` |
| DONE | Batched JAX environment, potential-based shaping | `floorclean/env.py` |
| DONE | Actor-critic network (no BatchNorm — see ORIENTATION) | `floorclean/networks.py` |
| DONE | PPO with correct truncation bootstrapping | `floorclean/ppo.py` |
| DONE | Scripted baseline strategies | `floorclean/baselines.py` |
| DONE | Physical calibration report | `scripts/calibrate.py` |
| DONE | 8 physics tests (conservation, transport, grid independence) | `tests/test_physics.py` |

---

## P0 — blocking everything else

### T1 · Calibrate against the 15-minute ground truth `TODO`
**The single most important open task.** Everything downstream is only as
trustworthy as this.

Andrew (shop floor, awning cleaning) reports **~15 minutes for the floor, one
person, not crazy dirty**. The simulated section is 4.2 m × 14 m = 59 m².

- Run `PushSweep(name="far_to_near")` to completion (not the 5-min training
  window) and measure wall-clock simulated time to get every cell under
  `dirt.clean_threshold`.
- If it lands far from ~15 min, tune **in this order**, and say in the commit
  message which knob moved and why:
  1. `dirt.entrainment_rate` — how fast the jet cuts. Check the effect on the
     "single pass" table in `scripts/calibrate.py`; one slow pass should strip
     a ~15 cm swath to bare floor, and that currently holds.
  2. `dirt.settling_velocity` — sets transport length `u*h/v_settle`. Andrew
     confirmed push distance **depends on standing water**, which is exactly
     this relationship, so do not break it by making transport water-independent.
  3. `dirt.deposit_entrainment_rate` — how readily a laid-over wand sweeps.
- Do **not** tune by changing the reward. The reward is a measurement
  instrument here, not a knob.

**Done when:** a competent baseline finishes the section in 12–20 simulated
minutes, and `scripts/calibrate.py` still reports a physically sensible
single-pass table.

#### Findings so far (2026-09-18) — read before touching any constant

Transport is **fixed**; cutting is now the bottleneck, and the cause is a real
physical trade-off rather than a bug. Breakdown by phase over 15 simulated
minutes of `far_to_near`:

| t (min) | bound | deposited | suspended | drained |
|---|---|---|---|---|
| 0 | 7.24 | 1.66 | 0.00 | 0.00 |
| 15 | 7.09 | 0.27 | 0.01 | 1.55 |

The loose layer is swept to the trough and drained almost completely — the
sweeping half of the job works. But **adhered grit barely moves: 7.24 → 7.09 kg**.

The reason is visible in the pressure table in `scripts/calibrate.py`. The
baseline holds a 54° tilt at a 40 cm standoff, which delivers **1.1× adhesion** —
enough tangential momentum to push slurry, nowhere near enough normal stress to
cut, and never enough for the 5.5 kPa worn lanes. Impingement falls as
`cos²(tilt)/L²`, so a single compromise angle cannot do both jobs. Meanwhile the
"single pass strips a 17 cm swath" figure in the calibration table is measured at
tilt 0.5 (29°), which is far more upright than any baseline actually uses — so
that table was not describing the baselines' behaviour.

**Do not respond by raising `entrainment_rate` again.** It has already been
raised 5× (5e-4 → 2.5e-3) and barely changed the outcome, precisely because the
binding constraint is pressure-below-threshold, not rate. Raising it further
would paper over the trade-off that is the whole subject of the study.

What this actually means:

1. The physics is asserting that **cutting and pushing must be separated in
   time** — which is exactly the hypothesis `BlastThenSweep` encodes. That is a
   result, not an obstacle, and it is the first genuinely interesting output of
   the model.
2. `BlastThenSweep` is currently **broken** and cannot confirm it. Its sweep
   phase targets `end_y` (the trough) while its transition out of that phase
   waits for `at_start`, so once it reaches the trough it sweeps there forever.
   Fix the state machine before drawing any conclusion about the comparison.
3. Its blast pass also uses a 16 cm standoff, where the fan is only ~7 cm wide,
   against an 18 cm lane spacing — so it cuts under 40% of the floor it walks
   over. Lane spacing must match the swath at the standoff actually used.
4. Note the non-obvious result already in the calibration table: standing
   **back** to 50 cm gives a *wider* swath (27 cm vs 16 cm at 10 cm) while still
   cutting 100%, because the fan spreads faster than the pressure falls. The
   optimum standoff is very likely not "as close as possible", which is worth
   the study on its own.

### T2 · Rendering `DONE` → `floorclean/render.py`
Needed *before* trusting any training run, not after.

A bug in the baseline sweep geometry — walking along the fan's long axis, so
each pass cleared a 3 cm strip instead of 20 cm — was invisible in the reward
curve and would have been obvious in one frame of video. Assume there are more
like it.

- Render: residual grit (all three phases), film depth, the wand position with
  its impact patch drawn where it actually lands (**not** at the tip — at high
  tilt the jet strikes most of a metre ahead), and the trough.
- Export mp4 via `imageio`; a still-frame grid mode for quick checks.
- Must work headless (`matplotlib.use("Agg")`) — it will run on RunPod.

**Done when:** you can watch a baseline episode and see slurry being walked to
the trough.

---

## P1 — the deliverable

### T3 · Training driver `TODO` → `scripts/train.py`
`floorclean/ppo.py` already exposes `init_runner` and `make_chunk`. This is the
outer loop only.

- Loop chunks, print metrics, checkpoint with `orbax`, resume from checkpoint.
- Log to CSV (no wandb dependency — keep it runnable offline).
- `tyro` CLI for config overrides.

### T4 · Benchmark — **this is the actual answer** `TODO` → `scripts/benchmark.py`
The table that settles the argument. Depends on T1 and T3.

Run each strategy in `baselines.default_suite()` plus the trained policy, from
identical seeds, **to completion** rather than for a fixed window. Report per
strategy:

- time to get the whole section under threshold (mean ± spread over seeds)
- water used (gallons)
- time to 90% clean vs 100% clean — the long tail is where methods differ
- fraction of grit pushed *past* the trough onto the far slope (the overshoot
  failure Andrew described)

**Done when:** there is a table with confidence intervals that a skeptical
coworker could not wave away as a fluke of one run.

### T5 · Environment tests `DONE` → `tests/test_env.py`
Mirror what `tests/test_physics.py` does for the physics.

- Shaping telescopes: over an episode, summed shaping reward equals
  `REWARD_SCALE * (Φ_final − Φ_initial)` to float tolerance. If this fails the
  agent is being paid for something other than progress.
- No free reward: an agent parked far from the trough with the wand at max
  standoff earns approximately `−TIME_COST * dt` per step and nothing more.
- Truncation ≠ termination: at `max_steps`, `truncated=True` and
  `terminated=False` unless the floor is genuinely clean.
- `reset` is deterministic given a key; `vmap` and `jit` both work.
- Overshoot is possible: a hard push near the trough can land grit on the far
  side. (Guards the reason the full bay is modelled instead of half.)

### T6 · Policy analysis — turn the policy into advice `TODO` → `scripts/analyse.py`
A trained network nobody can read will not convince anyone. The output of this
project should be a technique somebody can follow.

Extract from rollouts:
- standoff and tilt as a function of *what the policy is doing* — cutting
  adhered grit vs sweeping loose slurry. Does it separate the two jobs the way
  `BlastThenSweep` assumes, or find a compromise angle?
- where it chooses to work relative to the trough, over the course of a job
- how far it pushes before re-attacking
- **the headline**: three or four sentences of plain instruction, each backed
  by a number from the benchmark.

---

## P2 — the extensions Andrew asked about

### T7 · Two operators `TODO`
> *"the dirt gets pushed without an opposing force so usually we do two agents"*

The full bay is already modelled (both sides of the trough) precisely so this
drops in. Plan:
- `n_agents` in `ObsConfig`/`EnvState`; stack the wand state; action becomes
  `(n_agents, 5)`.
- Shared policy, per-agent egocentric observation, plus the other operator's
  position in the vector observation.
- Benchmark one operator vs two, **normalised by labour-minutes, not wall-clock**
  — two people finishing in 60% of the time is a *loss* on labour.

Answers the real question: is the second person buying opposing force at the
trough, or just parallelism?

### T8 · Bucket dumps `TODO`
> *"we dump buckets from the 55 gal to 5 gal to help move the dirt"*

Deliberately out of scope for now (Andrew: *"just do power wash for now"*), but
the physics already supports it — a bucket is a large, brief `water_source` with
no momentum. Likely a large effect, since transport scales with `u*h`. Worth an
action-space flag once T4 gives a baseline to compare against.

### T9 · Retire the old stack `TODO`
`cleaning_room.py`, `fluid_sim.py`, `feature_extraction.py`, `main.py` are the
previous Stable-Baselines3 + numba implementation, superseded entirely. They do
not import and `numba` is broken against the installed numpy. Delete once T4
has produced results, not before — keep them reachable in git history for
comparison.

### T10 · RunPod training `TODO` → `scripts/runpod_launch.py`
Adapt the launcher from `../4o_clone/scripts/runpod_launch.py` (GraphQL deploy,
`terminateAfter` cost guard, S3 log mirroring — all proven).

- Key is at `~/.runpod/config.toml` and `~/Documents/dev/4o_clone/.env`;
  verified working, balance was $124 on 2026-09-18.
- Needs `jax[cuda12]` — see the `cuda` extra in `pyproject.toml`.
- A 4090 is plenty; the whole training loop is one compiled JAX program, so the
  pod needs no babysitting.
- **Set `terminateAfter`.** Do not leave a pod running.

### T11 · README rewrite `TODO`
The current `README.md` describes the old implementation and is wrong in every
particular.

---

## P0 — verified solver/baseline issues (run-checked 2026-09-18, CPU)

Do not start T1 calibration until B1–B4 are resolved — each one moves the
numbers T1 is supposed to anchor. Reproduction commands are single-step probes
in `floorclean/physics.py` / `jet.py` / `baselines.py` (full `calibrate.py`
does not finish on CPU; it timed out at 120 s).

### B1 · Entrainment uses peak-minus-yield instead of local-minus-yield `TODO` → `floorclean/physics.py:219`
Code computes `max(p_peak − yield, 0) · coverage`; correct is
`max(p_peak·exp(−r²) − yield, 0)`. Measured at 30 cm / 0.5 rad:
8 cells entrain vs 4 physically justified, integrated weight 0.20 vs 0.07
(~3× high). The `single_pass_removal` swath in `scripts/calibrate.py` —
the exact table T1 calibrates against — absorbs this error.

### B2 · Reward shaping discounts with γ=1, trainer uses γ=0.997 `TODO` → `floorclean/env.py:355`
`F = Φ′ − Φ` in env vs `γ=0.997` in `ppo.py`. Measured `Φ₀=−30.5 kg-eq`:
per-step bias `(1−γ)·Φ·SCALE ≈ −1.1` vs `TIME_COST·dt = −0.2` (5.5×),
cumulative ≈ −1649 over an episode. Pass `γ` into the env or the Ng
guarantee (which `tests/test_env.py::test_shaping_telescopes` pins with
γ=1) does not cover what PPO actually optimises.

### B3 · `BlastThenSweep` sweep pass never runs `TODO` → `floorclean/baselines.py:224-249`
Blast arrives at the trough → `next_phase=2.0`; sweeping starts already at
`reached_trough` → `finished` is immediately true → lane advances after one
0.2 s step. A 60-step physics run never saw phase 2.Needs a reposition
between blast and sweep. Benchmarking this strategy as-is measures a slow
blast pass, not two-pass technique.

### B4 · 2D CFL clip allows 1.8·h outflow per substep (latent) `TODO` → `floorclean/physics.py:144-148`
Per-face `|q| ≤ cfl·dx/dt·h_upwind` with `cfl=0.45` permits 4×0.45=1.8·h
drain; the `maximum(h,0)` / `maximum(susp−dt·div,0)` clamps then create
mass. Proven with adversarial 4-way 5000 Pa `tau`: `Σh` 0.0020→0.0027 in
one substep; sediment 0.10→0.18 (+80%). Realistic jets conserve to ~1e-7
(200 sustained substeps, sealed) because Manning friction and the `0.5·dx`
kernel clamp keep `|q|` below the clip — so fix is a guard (`/2` in 2D or
per-cell outflow rescale + clip-hit counter test), not a behaviour change.

### B5 · Unvalidated 10× `settling_velocity` change `TODO` → `floorclean/config.py:235`
`settling_velocity` is now `3.0e-4`; ORIENTATION and the committed physics
say ~3 mm/s. Transport length `u·h/v` is 10× longer, which masks B1/B3 and
pre-empts T1's prescribed knob order. Either revert or validate via
`scripts/run_to_completion.py` + the calibrate single-pass table per the T1
protocol — do not train through it.

### B6 · `test_overshoot_possible` fails on a test bug `TODO` → `tests/test_env.py:173`
`blob` masks with scalar `state.tip_y` (operator start, 13.85) instead of
`env.floor.y`, so the deposited blob is empty (measured dep 0.0000 before
and after 60 pushes; far side bit-identical 8.273950→8.273950 kg) and the
assert fails. One-line fix: `jnp.abs(env.floor.y − (fc.trough_y + 0.6))`.
Other 4 env tests pass (139 s CPU total).

### Audit of concurrent work (no action taken here)
- `scripts/train.py` (T3, new): CSV key order matches `make_chunk` summary;
  suspected bug — `PyTreeCheckpointer.save` to the same path every chunk
  without `force=True`/step subdirs will raise on the second save.
- Old-stack edits (`cleaning_room.py`, `fluid_sim.py`, `main.py` modified,
  `fluid.py` deleted, all uncommitted) are pre-existing T9-retire material;
  untouched.

---

## Invariants — do not break these

1. **Mass is conserved exactly.** `tests/test_physics.py` checks this to 1e-4
   relative. The reward is a potential-based shaping term over remaining grit,
   which is only sound because grit cannot appear or vanish. The previous
   implementation used a non-conservative advection scheme and paid the agent
   for numerical error. If a conservation test fails, the physics is wrong —
   do not loosen the tolerance.
2. **Truncation is not termination.** Almost every episode ends on the time
   limit. Conflating them teaches the agent the world ends at five minutes.
3. **No BatchNorm in the policy**, and no running-statistics normalisation over
   the observation maps. Channels are scaled by fixed physical constants so a
   pixel value means the same thing all run.
4. **Jet pressure is computed from the physical footprint**, then rasterised.
   Keeps results independent of `dx`; there is a test.
5. **Do not add reward terms to fix behaviour.** If the policy does something
   daft, the physics or the observation is wrong. The only legitimate
   objectives are time, water, and grit delivered to the trough.
