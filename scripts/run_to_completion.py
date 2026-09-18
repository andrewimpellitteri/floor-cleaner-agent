#!/usr/bin/env python3
"""Run a scripted baseline to COMPLETION and report simulated minutes to clean.

This is the T1 measurement: a competent strategy swept until every cell is
under `dirt.clean_threshold`, timed against Andrew's ~15-minute ground truth.
Training episodes are five-minute windows; this is the whole job.

    python scripts/run_to_completion.py --strategy far_to_near --seed 0
    python scripts/run_to_completion.py --strategy blast_then_sweep --record out.mp4

The run starts from a FRESH floor (no `_apply_progress` randomisation) with a
fixed seed, the operator at the wall, and steps until `worst residual <
clean_threshold` or the time cap. A timeline is printed every 30 simulated
seconds so the shape of the job -- fast first coverage, long stubborn tail --
is visible rather than collapsed into one number.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from floorclean.baselines import BlastThenSweep, PushSweep
from floorclean.config import Config, GPM
from floorclean.env import CleaningEnv, EnvState
from floorclean.geometry import episode_elevation, initial_dirt, initial_water
from floorclean.physics import initial_fields

# Column order of the per-step diagnostic vector packed inside the scan.
DIAG_NAMES = (
    "worst_residual",
    "remaining_kg",
    "drained_kg",
    "adhered_kg",
    "deposited_kg",
    "suspended_kg",
    "fraction_clean",
    "water_m3",
)
# Index of `worst_residual`; completion is judged on it.
WORST = DIAG_NAMES.index("worst_residual")

MAX_FRAMES = 400  # cap recorded states so memory stays bounded


def make_strategy(name: str):
    if name == "far_to_near":
        return PushSweep(name="far_to_near", near_to_far=False)
    if name == "near_to_far":
        return PushSweep(name="near_to_far", near_to_far=True)
    if name == "blast_then_sweep":
        return BlastThenSweep()
    raise SystemExit(f"unknown strategy {name!r}")


def fresh_state(env: CleaningEnv, key: jax.Array) -> EnvState:
    """A fresh, uniformly dirty floor and an operator at the wall.

    Bypasses `reset`'s random job-progress: a completion run must start the
    whole job, not a random slice of one.
    """
    cfg = env.cfg
    k_dirt, k_z, k_water, k_next = jax.random.split(key, 4)

    z = episode_elevation(k_z, cfg, env.floor)
    bound, yield_stress = initial_dirt(k_dirt, cfg, env.floor)
    fields = initial_fields(cfg.floor.nx, cfg.floor.ny)._replace(
        bound=bound, h=initial_water(k_water, cfg, z)
    )
    return EnvState(
        fields=fields,
        yield_stress=yield_stress,
        z=z,
        tip_x=jnp.array(0.15),
        tip_y=jnp.array(cfg.floor.length_y - 0.15),
        standoff=jnp.array(0.4),
        tilt=jnp.array(0.6),
        azimuth=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
        potential=env._potential(fields),
        initial_mass=jnp.sum(bound) * cfg.floor.cell_area,
        water_used=jnp.array(0.0),
        key=k_next,
    )


def run(env: CleaningEnv, policy, state: EnvState, total_steps: int, stride: int):
    """JIT a single scan over the whole run; frames recorded every `stride`."""

    def sim_step(state_carry):
        state, carry = state_carry
        carry, action = policy.act(env, state, carry)
        new_state, _, _, _, _, info = env.step(state, action)
        diag = jnp.stack([info[k] for k in DIAG_NAMES])
        return (new_state, carry), diag

    frames_total = total_steps // stride
    remainder = total_steps - frames_total * stride

    def outer(state_carry, _):
        (state, carry), diags = jax.lax.scan(
            lambda c, _: sim_step(c), state_carry, None, length=stride
        )
        return (state, carry), (state, diags)

    init = (state, policy.init(env, state))
    (state, carry), (frames, diags) = jax.lax.scan(
        outer, init, None, length=max(frames_total, 1)
    )
    if not frames_total:
        frames, diags = None, None

    tail_diags = None
    if remainder:
        (_, _), tail_diags = jax.lax.scan(
            lambda c, _: sim_step(c), (state, carry), None, length=remainder
        )
    return frames, diags, tail_diags


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default="far_to_near")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=90.0)
    ap.add_argument("--record", default=None, help="write an mp4 of the job here")
    args = ap.parse_args()

    cfg = Config()
    env = CleaningEnv(cfg)
    policy = make_strategy(args.strategy)

    control_dt = cfg.sim.control_dt
    total_steps = int(args.max_minutes * 60 / control_dt)
    stride = max(1, total_steps // MAX_FRAMES)
    frames_total = total_steps // stride

    print(f"strategy          {args.strategy}")
    print(f"seed              {args.seed}")
    print(f"cap               {args.max_minutes:.0f} sim min "
          f"({total_steps} steps, recording 1 in {stride})")

    state = fresh_state(env, jax.random.PRNGKey(args.seed))
    initial_kg = float(state.initial_mass)

    t0 = time.time()
    frames, diags, tail = run(env, policy, state, total_steps, stride)
    # First call includes compilation.
    print(f"(jitted run took {time.time() - t0:.1f} s wall for "
          f"{total_steps * control_dt / 60:.0f} sim min)")

    diags = np.asarray(diags).reshape(frames_total * stride, len(DIAG_NAMES))
    if tail is not None:
        diags = np.concatenate([diags, np.asarray(tail)], axis=0)

    steps = diags.shape[0]
    t = np.arange(1, steps + 1) * control_dt
    clean_idx = int(np.argmax(diags[:, WORST] < cfg.dirt.clean_threshold)) + 1
    finished = diags[clean_idx - 1, WORST] < cfg.dirt.clean_threshold

    print()
    print(f"{'min':>6} {'worst kg/m2':>12} {'left kg':>9} {'clean %':>8} "
          f"{'susp kg':>8} {'dep kg':>7}")
    for k in range(0, steps, 150):  # every 30 simulated seconds
        d = diags[k]
        print(f"{t[k] / 60:6.1f} {d[WORST]:12.4f} {d[1]:9.2f} {d[6] * 100:7.1f}% "
              f"{d[5]:8.3f} {d[4]:7.2f}")
    d = diags[-1]
    print(f"{t[-1] / 60:6.1f} {d[WORST]:12.4f} {d[1]:9.2f} {d[6] * 100:7.1f}% "
          f"{d[5]:8.3f} {d[4]:7.2f}")

    print()
    if finished:
        minutes = t[clean_idx - 1] / 60
        water_gal = diags[clean_idx - 1, 7] * 264.172
        print(f"CLEAN at {minutes:.1f} simulated minutes "
              f"({clean_idx} steps), {water_gal:.0f} gal water")
    else:
        minutes = float("nan")
        print(f"NOT CLEAN after {args.max_minutes:.0f} sim min "
              f"(worst {diags[-1, WORST]:.4f} vs threshold "
              f"{cfg.dirt.clean_threshold})")

    last = diags[clean_idx - 1] if finished else diags[-1]
    on_floor = last[1] + 0.0  # remaining
    print(f"mass check      started {initial_kg:6.2f} kg, "
          f"drained {last[2]:6.2f} kg, still on floor {on_floor:6.2f} kg "
          f"(closure {initial_kg - last[2] - on_floor:+.3f} kg)")

    if args.record and frames is not None:
        from floorclean.render import render_episode, save_still

        out = pathlib.Path(args.record)
        out.parent.mkdir(parents=True, exist_ok=True)
        n = int(np.ceil(clean_idx / stride)) if finished else frames_total
        rollout = [jax.tree.map(lambda x: np.asarray(x[i]), frames)
                   for i in range(n)]
        still_dir = out.parent / (out.stem + "_stills")
        still_dir.mkdir(parents=True, exist_ok=True)
        for frac in (0.0, 0.25, 0.5, 0.75, 0.999):
            k = min(n - 1, int(frac * n))
            save_still(env, rollout[k], str(still_dir / f"t{k * stride * control_dt:05.0f}s.png"))
        try:
            render_episode(env, rollout, str(out), fps=12)
            print(f"wrote {out} ({n} frames) and stills in {still_dir}")
        except Exception as e:  # ffmpeg missing etc -- stills are the fallback
            print(f"mp4 failed ({e}); stills written to {still_dir}")


if __name__ == "__main__":
    main()
