# Orientation — floor_clean

Read this before touching anything. `WORKBOARD.md` has the task list.

## What this is

Andrew works at an awning cleaning shop. Awnings are laid out on a big bay floor
and washed; the floor itself then has to be cleaned. He wants to settle an
argument with his coworkers about the **optimal way to clean the floor** —
backed by physics rather than by who is most confident.

So the deliverable is not "a trained model". It is **a defensible answer to a
practical question**, which needs three things: a simulation faithful enough to
believe, the methods people actually use implemented inside it, and a
comparison a skeptic cannot dismiss.

## Domain facts (from Andrew — these are ground truth, not assumptions)

- **The floor, not the awnings.** Awnings are cleaned separately. The floor is
  washed **clear** — pieces are moved out first, so no obstacles.
- **Geometry.** A big open bay, sloped in two directions into a **central
  trough**. Roughly **20–25 ft from the wall to the trough on each side**.
- **Surface.** Grit-broadcast epoxy, the type used in aircraft hangars.
  Normally recoated yearly; **two years overdue**, so it is worn through in the
  traffic lanes where pieces get dragged. Grit ground into worn epoxy is much
  harder to shift — hence the spatially varying yield stress and the worn-lane
  structure in `geometry.py`.
- **Machine.** ~1200 psi soft-wash unit, **water only, no chemicals**, ~4 gpm.
- **Technique.** **Standoff — how high the wand is held — is the variable that
  matters most**, which is why it is an explicit action.
- **The floor is worked wet.** Hoses run for rinse; the bay rarely dries out.
  There are also **random flat spots that trap standing water**. Push distance
  **depends on standing water** — Andrew confirmed this directly, and the model
  reproduces it because transport length is `u*h/v_settle`.
- **Ground truth: two different numbers for two different jobs.**
  **~15 minutes** for a routine pass, one person, floor not unusually dirty —
  "the floor looks good". **~30 minutes** for a *full deep clean*, one person
  (Andrew, 2026-09-18: *"for a full deep good clean with one person probably
  closer to 30"*).
  Which one applies depends on the criterion being scored. The simulator's
  completion test is **every cell** under `dirt.clean_threshold`, which is a
  deep-clean standard — so it should be compared against the **30-minute**
  figure, not the 15. Scoring a deep-clean criterion against a routine-clean
  time is what T1 did for months. See issue #2.
  **Still open:** whether that 30 minutes is a normally dirty floor done
  thoroughly (in which case the model is close) or a genuinely filthy one (in
  which case the 20 g/m² loading is too light for the time it is being asked to
  match).
- **Water comes from point sources, and water alone does not clean.** Andrew:
  *"even if you run continuous water through the faucets without power washing
  it doesn't really clean it from a point source."* There are **two taps at the
  midpoints of the room** and **four power washers, one in each corner** (so the
  bay is equipped for up to four simultaneous operators). The floor is washed
  **while still wet**, right after the awnings come up and while they drain on
  the bar — a line source. You cannot flood the bay from a tap, so raising the
  ambient flow is not an available intervention. See issue #1.
- **The trough ponds.** It does not drain freely: *"trough has a small pond near
  base due to warping and wear of room and painting and fiberglass."* See
  issue #3.
- **Known failure mode:** *"the dirt gets pushed without an opposing force"* —
  slurry driven past the trough and up the far slope. This is why the **full**
  bay is modelled, not half of it: cutting the domain at the trough would put a
  wall there and make overshoot impossible by construction.
- Also done in practice but **out of scope for now** (Andrew: *"just do power
  wash for now"*): dumping 5-gal buckets to help move dirt; working in pairs.

Photos of the real bay are in `room_pics/`.

## The physics, and why it is the way it is

Three modules, each solving one thing.

### `jet.py` — what the wand does to the floor
The tip sits at height `standoff`, tilted `tilt` off vertical, pointed along
`azimuth`. The beam travels `standoff/cos(tilt)` and lands **`standoff*tan(tilt)`
ahead of the operator** — at a 72° tilt and 30 cm standoff that is most of a
metre, which is far too much to ignore when aiming.

Two things happen at the impact patch, and they trade off against each other:

- **Normal** momentum cuts adhered grit. It falls off as `cos²(tilt)/L²`.
  Getting close and standing the wand up is how you break grit loose.
- **Tangential** momentum pushes the water film, scaling as `sin(tilt)`. Laying
  the wand over is how you move slurry.

You cannot have both at once. **That tension is the entire problem**, and
whether the optimum interleaves the two or compromises between them is one of
the things worth finding out.

Pressure is computed from the **true physical footprint** (a 25° fan at 30 cm is
about 13 cm × 1.5 cm — thinner than one grid cell) and only then rasterised, so
results do not depend on `dx`. There is a test.

### `physics.py` — water and dirt
**Water:** the local inertial form of the shallow water equations (Bates,
Horritt & Fewtrell 2010), staggered grid, semi-implicit friction.

Retaining **inertia** is the point, not a refinement. An earlier version used
the diffusive-wave approximation, which drops acceleration and solves directly
for a friction-balanced velocity. Water leaving the jet then decelerates
*instantly*, transport length collapses from over a metre to a few centimetres,
and a pushed slurry can never travel more than a few inches no matter the
technique. The whole question is how far a push carries, so that approximation
could not answer it.

**Dirt:** three phases, Hairsine–Rose.

| phase | what it is | what moves it |
|---|---|---|
| `bound` | adhered to the epoxy | jet impingement only — **kilopascals** |
| `deposited` | knocked loose, lying on the floor | tangential stress — **pascals** |
| `suspended` | carried by the film | advects with the water |

Splitting `bound` from `deposited` is essential. With one threshold, loosened
slurry would be as hard to move as virgin adhered grit and the jet would have to
physically strike every square inch — which is not how the job is done. The
split reproduces the real two-part technique: get close and stand the wand up to
break grit loose, then lay it over to sweep it in.

Suspended grit settles back into `deposited`, **never** into `bound` — blasted
grit does not re-bond. So a stalled push costs transport, not the cutting work
already done.

### `env.py` — the RL problem
- **Reward** is strict potential-based shaping, `F = γΦ(s') − Φ(s)` with `Φ` =
  negative outstanding transport work (grit mass weighted by phase and by
  distance to the trough). Ng et al. (1999) guarantee this leaves the optimal
  policy unchanged, so the agent gets dense feedback with no way to farm it.
  Everything else in the reward is a real cost: elapsed time, plus a finish
  bonus. **That guarantee depends on mass being conserved exactly.**
- **Episodes are the full 15-minute job** (`sim.max_steps = 4500` at
  `control_dt = 0.2 s`), not a window. They *were* five-minute windows when the
  floor carried 40 lb of grit and took 20–40 minutes; at the corrected 20 g/m²
  loading the window became actively wrong, because thoroughness means getting
  every cell under threshold and that needs more time than the window had — so
  no policy could ever finish and the finish bonus was unreachable. See the
  reasoning in `config.py` above `max_steps`.
  `reset` still starts the floor at a uniformly random point through the job, so
  the policy sees fresh floors, half-done floors, and floors down to the last
  stubborn worn-lane patches. A policy that only knew the opening move would
  score badly on most of that distribution. The benchmark then runs **to
  completion**, which is the number that answers the question.

## Stack

Rewritten from Stable-Baselines3 + numba to **end-to-end JAX**: the environment
is pure array maths, so environments, rollouts, GAE and optimiser updates all
run inside one compiled `lax.scan` on the GPU. The old setup stepped one
environment at a time in Python and was given 20,000 timesteps — about twenty
episodes, nowhere near enough to learn anything. (The old `numba` stack also no
longer imports: it refuses the installed numpy.)

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest tests/ -q      # physics must pass
.venv/bin/python scripts/calibrate.py     # read this before believing anything
```

GPU: `uv pip install -e ".[cuda]"`.

## Conventions

- Arrays are indexed `field[i, j]` ↔ `x = i*dx`, `y = j*dx`. **`x` runs along
  the trough, `y` across it.** Renderers transpose for display; no other module
  is allowed to be clever about axes. The old implementation had x/y swapped
  inconsistently between the physics and the renderer, which is the kind of bug
  that costs a day.
- All physical quantities are **SI** and correspond to something measurable.
  Where a number is a guess, the comment says so and names the calibration knob.

## Invariants

See the end of `WORKBOARD.md`. The short version: mass is conserved exactly,
truncation is not termination, no BatchNorm or observation normalisation, and
**do not add reward terms to fix behaviour** — if the policy does something
daft, the physics or the observation is wrong.
