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

#### Update — cutting is solved, TRANSPORT is the bottleneck

After fixing the subgrid entrainment gate, the compromise wand angle, the
`BlastThenSweep` state machine and the side-switching rule, `far_to_near` over
30 simulated minutes now gives:

| phase | start | end |
|---|---|---|
| adhered (`bound`) | ~2.40 | **0.33** |
| loose (`deposited`) | ~0.50 | **1.54** |
| delivered to trough | 0 | 1.06 |

So the jet now cuts 86% of the adhered grit off the epoxy, and most of it then
**sits on the floor as loose slurry instead of reaching the trough**. Time to
clean is still `inf` within 30 min, and the gap is entirely transport.

This is worth taking seriously rather than tuning away, because it is exactly
what Andrew described unprompted: *"the dirt gets pushed without an opposing
force so usually we do two agents."* The model reproducing that failure mode
independently is evidence it has the mechanism right.

Next steps, in order:
1. Measure, do not guess: use `push_delivery` in `scripts/calibrate.py` to get
   delivered-fraction vs standoff/tilt/speed on a WET floor (it currently probes
   a near-dry one, which is not how the bay is worked).
2. The likely physical lever is water depth, since transport length is
   `u*h/v_settle`. A sweeping pass that walks with its own bow wave should carry
   much further than one that outruns it — check whether `push_speed` is simply
   above the wave speed, in which case the operator is walking away from their
   own water.
3. Only then consider `deposit_entrainment_rate` or `settling_velocity`.
4. `near_to_far` currently regresses badly (bound 2.29 vs 0.33): with the new
   `cross = wrapped & finished` rule it switches sides after finishing only its
   first band, so it never completes its outward progression. Its side-switch
   needs to wait until all bands are done, not all lanes in one band.

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

#### 2026-09-18, second probe — superseded (kept for the mechanism table)
First 40-min `far_to_near` run, before the side-switch fix — numbers below
are stale, mechanism stands. Effective excess G = P − Y − Y·ln(P/Y) (kPa):

| wand | P | normal (2.5k) | lane (5.5k) | lane+ (8.5k) |
|---|---|---|---|---|
| 0.16 m / 0.22 rad (old blast) | 106 | 94.6 | 84.7 | 76.5 |
| 0.40 m / 0.22 rad (new blast) | 18 | 10.9 | 6.3 | 3.3 |
| 0.40 m / 0.55 rad (compromise) | 11 | 4.5 | 1.5 | 0.2 |

The compromise angle delivers 0.2 kPa against the worst lanes — ~380× less
than the close blast, functionally zero (30+ passes per lane). And the 0.40 m
blast from point 3 above sits in the middle: width-matched but 23× weaker in
the lanes than 0.16 m. So blast standoff is now a genuine two-sided trade —
fan width vs lane-threshold punch — that only a timed comparison settles, and
`BlastThenSweep` likely needs per-pass lane spacing (narrow blast lanes, wide
sweep lanes) instead of one shared `lane_width`. Do not resolve this by
lowering `worn_lane_boost`: the lanes are domain ground truth (ORIENTATION).
Related: push delivery over 3 m is 10% dry / 40% on a 2.5 mm working film /
69% at 6 mm — Andrew's "depends on standing water" reproduced exactly, so
transport needs no knob; the sim starting wet is load-bearing, keep it.

#### 2026-09-18, third probe (current code: fixed sides, load 0.10)
Same command, seed 0: **still NOT CLEAN after 40 min** — worst 0.48 (24×
threshold), 43.6% cells clean, 5.38 → 2.94 kg on floor, 2.44 drained, mass
closure +0.000 kg. The ping-pong is gone and most adhered grit cuts, but the
worst cells do not move: at G_lane+ = 0.2 kPa the compromise angle is on a
knife-edge in the worn lanes, so completion is seed-fragile (the commit's
lighter seed finished cutting; seed 0 does not). Raising `entrainment_rate`
cannot fix G ≈ 0 — only pressure (closer/lower tilt) or dwell (re-attack)
moves those cells.
Also fixed here: `near_to_far` crossed sides after its FIRST band
(`cross = wrapped & finished` reset the band index), so it never progressed
outward — verified by trace (bands stuck at 0), now crosses only after the
last band (`side_covered`). `BlastThenSweep` defaults to full-height passes
and is unaffected; the mean-metric `_side_is_done` helper is now dead code
(left in place — do not use it).

#### Resolution coarsening for the training runs (verified 2026-09-18)

The "dx 7→10 cm for ~2.9× speedup" proposal is invalid as stated: 4.2/0.10 =
42 cells along x and `CleaningEnv` asserts `nx % pool == 0` (pool=4), so the
env refuses to construct (checked directly, `AssertionError`). Nearest valid
choices are **dx=8.75 cm (48×160, ~1.56× fewer cells)** or **dx≈11.67 cm
(36×120, ~2.78× fewer cells)**. Whichever is picked, extend the
grid-independence test to it and keep the planned ranking-invariance check
(same benchmark order at both resolutions) as the convergence evidence.

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

### T3a · wandb — the one metric pair that matters `DONE`

Not duplicating the implementation. Handing over the only non-obvious part, for
whoever is wiring it up:

**Plot `drained_kg` and `fraction_removed` on the same axes, and watch for them
to DIVERGE.** That divergence is this project's characteristic failure and it is
invisible in total reward.

The policy can learn to CUT and then walk away. Breaking adhered grit loose pays
immediately through the shaping term; carrying the slurry twenty feet to the
trough pays much later and is much harder. A policy that does the first and
neglects the second scores well on reward and on `fraction_removed` while
leaving the floor covered in loose slurry — which is precisely the outcome the
scripted baselines already fall into (see T1: adhered grit 2.40 → 0.33 kg, but
1.54 kg left lying loose and only 1.06 kg actually drained).

So: `fraction_removed` climbing while `drained_kg` flattens means stop the run.
Third metric worth a panel is `worst_residual` — it is the thoroughness measure
and the one that decides whether the floor is ever actually finished, since the
mean goes quiet long before the worn lanes do.

Two practical notes: `WANDB_API_KEY` is in `~/Documents/dev/4o_clone/.env`, and
whatever wraps it should degrade to CSV rather than raise — a failed `wandb.init`
must not kill a GPU run that has already been paid for.

Wired 2026-09-18: `ppo.py` summary + `train.py` CSV now carry `drained_kg`,
`adhered_kg`, `deposited_kg`, `suspended_kg` per update (same dict feeds W&B,
so pinning them to one dashboard panel is one click — that is the divergence
panel). Eval stills are wrapped so render failures degrade to curves-only,
and `wandblog` never raises by construction. Verified offline end-to-end
(curves + stills + final mp4); full suite 22 passed.

### T3 · Training driver `DONE` → `scripts/train.py`
`floorclean/ppo.py` already exposes `init_runner` and `make_chunk`. This is the
outer loop only.

- Loop chunks, print metrics, checkpoint with `orbax`, resume from checkpoint.
- Log to CSV (no wandb dependency — keep it runnable offline).
- `tyro` CLI for config overrides.

Implemented and CLI-verified; the chunk has not yet been executed at full size
(512 envs needs the GPU box). Resume path restores params/opt-state/env/obs
from a single pytree checkpoint.
Smoke-verified on CPU 2026-09-18 (4 envs × 8 steps: 3 updates, CSV +
checkpoint + `--resume` all OK, including the repeated-save `force` path).
Independently roundtripped the same day: save → fresh init → restore gives
bit-identical params/opt-state/env-step/rng, and training continues from the
restored state.

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

#### Training run 1 (`main`, interim at update 1500/4576 — 2026-09-18)

Sources: S3 `training.csv` + eval stills (W&B run `1lzvcwhj` mirrors the same
rows). Pod healthy ($0.74/hr), GPU backend, ~7.5k steps/s → ~5.1 h total;
the 5 h guard likely fires at ~85–90%, resume tail is built and verified.

**Literature double-check (all claims re-verified this session):** matched-γ
shaping per Ng 1999 (code + unit test + startup assert); Φ(terminal)=0
automatic per Grzes 2017; timeout bootstrap per Pardo 2018 in `ppo.py`;
Wiewiora offset → EV watch; Devlin–Kudenko dynamic form covers future
constant moves; value-clip removed per SB3 default + Andrychowicz 2020;
λ stays 0.95 (dense shaping does the temporal decomposition; longer
fragments would buy correlated-gradient variance per the ICLR batch-structure
analysis); N=512/T=64 already the variance-safe shape.

| metric | first 100 upd | last 100 upd | reading |
|---|---|---|---|
| explained_variance | 0.97 | 1.00 | critic learned the offset; no value problem |
| value_loss | 3.1 | 0.27 | ÷10, consistent with EV |
| standoff / tilt means | 0.54 m / 0.65 rad | 0.32 m / 0.22 rad | cutting posture emerging |
| fraction_removed | 0.156 | 0.118 | **down** while reward flat (0.31→0.35) |
| drained_kg | 0.084 | 0.060 | **down** — see below |
| adhered left | 0.49 | 0.60 | **up** — cutting deep but narrow? |
| worst_residual | 0.103 | 0.069 | better where it works |
| entropy / KL | 4.60→4.26 / ~0.006 | — | still exploring, stable updates |
| finished_clean | 0 throughout | — | no env has ever finished (expected) |

Eval stills (update ~1400): 14 cm standoff, 0° tilt, operator effectively
stationary in a corner — mid and end frames 75 sim-s apart are pixel-identical
with grit unchanged. Consistent with the table: the policy cuts where it
stands (immediate shaping) and never pushes (delayed, discounted payoff),
harvesting the discount drizzle for staying dirty. **Not stop-criteria:**
33% in, entropy healthy, totals still order correctly (parked ≈ +297 vs
clean-fast ≈ +434), and this may be a transient cutting-first phase. If it
persists past ~50%, escalate per plan (λ ladder first — T stays 64).

**Two corrections to earlier analysis in this file and chat:**
- Eval floors differ per eval (`PRNGKey(10_000 + eval_idx)`), so `BEST_UPDATE`
  0.475 → 0.31 kg partly reflects floor lottery. Stills stay valid behavior
  samples; final selection needs fixed-floor completions (this T4). Best-eval
  tracking should move to a fixed key set.
- The S3 log mirror looked dead mid-run while training was at update 100+:
  `print()` block-buffered through `tee`, and `JAX_LOG_COMPILES=1` buried the
  mirror in ~1600 lines of spam. Fixed (`PYTHONUNBUFFERED=1`, compiles
  silenced — `cce756b`); visibility-only, training unaffected.

### T5 · Environment tests `DONE` → `tests/test_env.py`
Mirror what `tests/test_physics.py` does for the physics.

- Shaping telescopes: over an episode, summed shaping reward equals
  `REWARD_SCALE * (Φ_final − Φ_initial)` to float tolerance. If this fails the
  agent is being paid for something other than progress.
- No free reward: a parked wand earns the time cost plus only the known
  discount drizzle `SCALE·(γ−1)·Φ₀` per step (positive, since Φ < 0) and
  nothing more. (`test_no_free_reward` asserts exactly this; the old
  "≈ −TIME_COST·dt" wording predates the matched-γ design.)
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

### T10 · RunPod training `DONE` → `scripts/runpod_launch.py`
Adapt the launcher from `../4o_clone/scripts/runpod_launch.py` (GraphQL deploy,
`terminateAfter` cost guard, S3 log mirroring — all proven).

- Key is at `~/.runpod/config.toml` and `~/Documents/dev/4o_clone/.env`;
  verified working, balance was $124 on 2026-09-18.
- Needs `jax[cuda12]` — see the `cuda` extra in `pyproject.toml`.
- A 4090 is plenty; the whole training loop is one compiled JAX program, so the
  pod needs no babysitting.
- **Set `terminateAfter`.** Do not leave a pod running.

Implemented 2026-09-18 (`scripts/runpod_launch.py` + `scripts/jobs/train.sh`):
GraphQL deploy, mandatory terminateAfter (`--hours 0` is refused), wait-healthy
startup check, `--status/--logs/--kill`, `--dry-run` with machine-readable
stdout. Pod runs a CUDA 12.8 image, installs `.[cuda]` via uv, refuses to train
unless JAX sees a CUDA device, streams checkpoints/CSV/logs to S3, and
self-terminates; `--ref` defaults to the current branch and stale refs are
refused. Verified with `--dry-run` and `--status` (auth works, nothing
billing). Not yet launched: push the branch first — the pod clones origin,
and `jax-rewrite` is still local-only.

S3 costs (kept low by design, 2026-09-18): a checkpoint is ~220 MB
(512-env state dominates); periodic 10-min syncs stream rolling + snapshots
as transient traffic (PUTs measured in cents per run). Steady state per run
is rolling + best-eval + last snapshots + media + CSV ≈ 0.7 GB ≈ $0.02/mo —
`train.sh` prunes superseded snapshots locally and mirrors the prune with
`sync --delete`, so seeds and ablations cannot accumulate unboundedly. The
rolling `state` (resume point) is never pruned. `WANDB_API_KEY` is forwarded
like the other secrets (dry-run redacted).

#### GPU execution notes (from docs + roofline, 2026-09-18)
Throughput model for the full config (512 envs, 60×200 grid, 16 substeps):
~1 GB HBM traffic per physics substep (all envs) → ~15 GB per control step →
~1 TB per PPO update → ~2 s/update rollout on a 4090, ~5–7 h total training.
The loop is memory-bandwidth bound stencil code; the network update is a
minor share. Consequences:
- Reset-path micro-opts (box-blur, FFT filters, quantile) are NOT worth it:
  ~512 resets/chunk vs ~10M substep passes ≈ 0.005% of traffic. Killed with
  numbers; do not revisit.
- Untried, zero code risk, try in this order on the pod: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`,
  `JAX_LOG_COMPILES=1`, persistent `jax_compilation_cache_dir` in `train.py`
  (recompiling the chunk every pod start costs minutes),
  `--xla_gpu_enable_while_loop_double_buffering=true`,
  `--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL`, then O1. Skip Triton
  GEMM / NCCL / PGLE (no matmuls, no collectives, single GPU).
- Structural speedups all trade science for speed and need the T4 validation
  protocol, not blind application: 16→8 substeps halves the CFL speed limit
  to 0.63 m/s (clips bow waves); coarser grid per the resolution note;
  bf16 observations (pointless — update phase is already the minor share).
- Bug-hunt side: no host-numpy inside any jitted path (checked `ppo/env/
  physics/jet/networks/rollout/geometry`); `ppo.py`'s unused numpy import
  removed. First-ever executions all pass: PPO chunk smoke, `train.py`
  end-to-end + `--resume`, `run_episode(fresh=True)`, headless still render.

### T11 · README rewrite `TODO`
The current `README.md` describes the old implementation and is wrong in every
particular.

---

## P0 — verified solver/baseline issues (run-checked 2026-09-18, CPU)

All resolved. T1 calibration may proceed -- but re-run `scripts/calibrate.py`
first: B1 narrows the effective swath and B5 restores the transport length,
so the single-pass table has moved. (Full `calibrate.py` does not finish on
CPU; it timed out at 120 s. Reproduction probes were single-step.)

### B1 · DONE — entrainment gated by local pressure → `floorclean/physics.py`, `floorclean/jet.py`
Code computes `max(p_peak − yield, 0) · coverage`; correct is
`max(p_peak·exp(−r²) − yield, 0)`. Measured at 30 cm / 0.5 rad:
8 cells entrain vs 4 physically justified, integrated weight 0.20 vs 0.07
(~3× high). The `single_pass_removal` swath in `scripts/calibrate.py` —
the exact table T1 calibrates against — absorbs this error.
FIXED (final form, concurrent agent — verified here): for a Gaussian peak P
over threshold Y, INT max(0, P·e − Y) dA over the above-threshold region is
exactly A_patch·(P − Y − Y·ln(P/Y)), so the code spreads effective excess
G = P − Y − Y·ln(P/Y) over cells with `coverage`. Total removal is exact and
dx-independent; G falls smoothly to zero as P→Y and tends to P for P≫Y. The
`intensity` field added to `JetImpact` in the first pass is kept as a
diagnostic (true-footprint profile for render). Pinned by
`test_entrainment_integrates_patch_excess` (pins the integral to 5e-3 plus a
below-threshold zero). Note found while testing: Σcoverage·cell undercounts
A_patch by ~1.6% on this grid (thin direction under-resolved) — absorbed in
T1, do not chase it here.

### B2 · DONE — shaping uses the trainer's discount → `floorclean/env.py`
`F = Φ′ − Φ` in env vs `γ=0.997` in `ppo.py`. Measured `Φ₀=−30.5 kg-eq`:
free reward `(γ−1)·Φ·SCALE ≈ +1.08/step` vs `TIME_COST·dt = −0.2` (5×),
cumulative ≈ +1620 over an episode. Pass `γ` into the env or the Ng
guarantee (which `tests/test_env.py::test_shaping_telescopes` pins with
γ=1) does not cover what PPO actually optimises.
FIXED: `CleaningEnv(discount=0.997)`, `F = γΦ′ − Φ` in `step`. Measured
worse than first estimated -- with Φ < 0 the mismatch was a FREE reward of
+1.08/step (5× the time cost), not a penalty. `test_no_free_reward` now
asserts parked shaping equals the known discount drizzle within tolerance;
`test_env_discount_matches_trainer_gamma` pins env/PPO agreement so a
future gamma change cannot silently re-break it.

### B3 · DONE — four-phase `BlastThenSweep` → `floorclean/baselines.py`
Blast arrives at the trough → `next_phase=2.0`; sweeping starts already at
`reached_trough` → `finished` is immediately true → lane advances after one
0.2 s step. A 60-step physics run never saw phase 2.Needs a reposition
between blast and sweep. Benchmarking this strategy as-is measures a slow
blast pass, not two-pass technique.
FIXED (concurrent agent, phase machine verified here by mock-state probe):
reposition/blast/return/sweep with side crossing; 0→1→2→3→0 with the lane
advancing only after the sweep reaches the trough.

### B4 · DONE — 2D CFL clip halved + regression tests → `floorclean/physics.py`
Per-face `|q| ≤ cfl·dx/dt·h_upwind` with `cfl=0.45` permits 4×0.45=1.8·h
drain; the `maximum(h,0)` / `maximum(susp−dt·div,0)` clamps then create
mass. Proven with adversarial 4-way 5000 Pa `tau`: `Σh` 0.0020→0.0027 in
one substep; sediment 0.10→0.18 (+80%). Realistic jets conserve to ~1e-7
(200 sustained substeps, sealed) because Manning friction and the `0.5·dx`
kernel clamp keep `|q|` below the clip — so fix is a guard (`/2` in 2D or
per-cell outflow rescale + clip-hit counter test), not a behaviour change.
FIXED: `u_max = cfl·dx/dt/2`, config comment corrected to ~1.3 m/s.
`test_four_way_outflow_creates_no_water/sediment` fail pre-fix, pass post-fix.

### B5 · DONE — `settling_velocity` restored to 0.003 → `floorclean/config.py`
`settling_velocity` is now `3.0e-4`; ORIENTATION and the committed physics
say ~3 mm/s. Transport length `u·h/v` is 10× longer, which masks B1/B3 and
pre-empts T1's prescribed knob order. Either revert or validate via
`scripts/run_to_completion.py` + the calibrate single-pass table per the T1
protocol — do not train through it.
FIXED: reverted to 0.003 (ORIENTATION ground truth: a few mm/s, ~0.7 m
transport in a 2 mm film). At 3e-4 the stranded-slurry mechanic behind the
FarToNear/NearToFar debate nearly vanishes. T1 owns any re-tune, with
benchmark evidence.

### B6 · `test_overshoot_possible` fails on a test bug `DONE` → `tests/test_env.py:173`
`blob` masks with scalar `state.tip_y` (operator start, 13.85) instead of
`env.floor.y`, so the deposited blob is empty (measured dep 0.0000 before
and after 60 pushes; far side bit-identical 8.273950→8.273950 kg) and the
assert fails. FIXED: blob now keyed on `env.floor.y`, and the impact point is
aimed onto the blob (tip at trough+1.4 so the 72° jet lands ~0.9 m ahead, ON
the blob, then walks it across). Overshoot measured at ~2–4e-4 kg per hard
push; test asserts > 2e-4. Full suite: 14 passed (physics + env).

### Audit of concurrent work (no action taken here)
- `scripts/train.py` (T3, new): CSV key order matches `make_chunk` summary;
  suspected bug — `PyTreeCheckpointer.save` to the same path every chunk
  without `force=True`/step subdirs will raise on the second save.
  RESOLVED: `force=True` added to both saves.
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
