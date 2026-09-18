"""Running policies: to completion, for benchmarking and for rendering.

Training uses five-minute windows, but the question being answered is "how long
does the floor take", so the benchmark has to run past the training horizon
until the floor is genuinely clean. These helpers do that for both scripted
baselines and trained networks through one interface.

Two practical constraints shape the design:

* Storing every state for a twenty-minute episode would be gigabytes, so states
  are recorded every `record_every` steps and the run proceeds in compiled
  chunks between recordings.
* `lax.scan` cannot stop early, so a run continues to a fixed cap and the
  completion time is recovered afterwards as the first step at which the floor
  came clean. Spraying an already-clean floor costs nothing but simulated time.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from .env import CleaningEnv, EnvState
from .networks import ActorCritic


class Policy(Protocol):
    name: str

    def init(self, env: CleaningEnv, state: EnvState) -> Any: ...
    def act(self, env: CleaningEnv, state: EnvState, carry: Any): ...


class NeuralPolicy:
    """Wraps trained network parameters in the scripted-baseline interface."""

    def __init__(self, params, action_dim: int, name: str = "learned",
                 deterministic: bool = True):
        self.params = params
        self.name = name
        self.deterministic = deterministic
        self._net = ActorCritic(action_dim=action_dim)

    def init(self, env: CleaningEnv, state: EnvState):
        return jnp.zeros(())  # no carry; key comes from the env state

    def act(self, env: CleaningEnv, state: EnvState, carry):
        obs = env._observe(state)
        batched = jax.tree.map(lambda x: x[None], obs)
        mean, log_std, _ = self._net.apply(self.params, batched)
        action = mean[0]
        if not self.deterministic:
            key = jax.random.fold_in(state.key, state.step)
            action = action + jnp.exp(log_std[0]) * jax.random.normal(key, action.shape)
        return carry, jnp.clip(action, -1.0, 1.0)


class EpisodeResult(NamedTuple):
    states: list  # recorded EnvStates, for rendering
    seconds_to_clean: float  # inf if never finished inside the cap
    seconds_to_90: float
    final_remaining_kg: float
    initial_mass_kg: float
    water_litres: float
    overshoot_kg: float  # grit driven past the trough onto the far slope
    remaining_trace: np.ndarray
    clean_fraction_trace: np.ndarray


def _far_side_mass(env: CleaningEnv, state: EnvState, start_side: float):
    """Grit currently sitting on the far side of the trough, kg.

    Only meaningful as a DIFFERENCE against the same quantity at reset. The far
    half of the bay starts dirty like everything else, so the absolute number is
    dominated by grit that was always there and says nothing about technique.
    What matters is the increase: slurry driven past the trough and up the far
    slope, where it now has the full run back. That is the overshoot failure
    Andrew described, and it is why the full bay is modelled rather than half.
    """
    fc = env.cfg.floor
    residual = state.fields.bound + state.fields.deposited + state.fields.suspended
    far = jnp.where(start_side > 0, env.floor.y < fc.trough_y, env.floor.y > fc.trough_y)
    return jnp.sum(jnp.where(far, residual, 0.0)) * fc.cell_area


def run_episode(
    env: CleaningEnv,
    policy: Policy,
    key: jax.Array,
    max_seconds: float = 2400.0,
    record_every: int = 25,
    fresh: bool = False,
) -> EpisodeResult:
    """Run one episode to completion (or to `max_seconds`) and record the trace.

    `fresh=True` starts a uniformly dirty floor (`env.fresh_state`) -- required
    for benchmark/completion times. The default starts at a random point
    through the job, matching training windows.
    """
    cfg = env.cfg
    total_steps = int(max_seconds / cfg.sim.control_dt)
    n_chunks = total_steps // record_every

    if fresh:
        state = env.fresh_state(key)
    else:
        state, _ = env.reset(key)
    carry = policy.init(env, state)
    start_side = jnp.where(state.tip_y >= cfg.floor.trough_y, 1.0, -1.0)
    initial_mass = float(state.initial_mass)
    far_side_at_reset = float(_far_side_mass(env, state, start_side))

    def one_step(cs, _):
        c, s = cs
        c, action = policy.act(env, s, c)
        s, _obs, reward, _term, _trunc, info = env.step(s, action)
        return (c, s), (info["remaining_kg"], info["fraction_clean"],
                        info["worst_residual"], info["water_m3"])

    @jax.jit
    def chunk(cs):
        cs, trace = jax.lax.scan(one_step, cs, None, length=record_every)
        return cs, trace

    states, remaining, cleanfrac = [state], [], []
    worst_trace, water_trace = [], []
    cs = (carry, state)
    for _ in range(n_chunks):
        cs, (rem, cf, worst, water) = chunk(cs)
        states.append(cs[1])
        remaining.append(np.asarray(rem))
        cleanfrac.append(np.asarray(cf))
        worst_trace.append(np.asarray(worst))
        water_trace.append(np.asarray(water))

    remaining = np.concatenate(remaining)
    cleanfrac = np.concatenate(cleanfrac)
    worst = np.concatenate(worst_trace)
    water = np.concatenate(water_trace)

    dt = cfg.sim.control_dt
    clean_idx = np.flatnonzero(worst < cfg.dirt.clean_threshold)
    t_clean = float(clean_idx[0] + 1) * dt if clean_idx.size else float("inf")

    target90 = 0.10 * initial_mass
    idx90 = np.flatnonzero(remaining <= target90)
    t_90 = float(idx90[0] + 1) * dt if idx90.size else float("inf")

    finished_at = clean_idx[0] if clean_idx.size else len(remaining) - 1
    return EpisodeResult(
        states=states,
        seconds_to_clean=t_clean,
        seconds_to_90=t_90,
        final_remaining_kg=float(remaining[finished_at]),
        initial_mass_kg=initial_mass,
        water_litres=float(water[finished_at]) * 1000.0,
        overshoot_kg=float(_far_side_mass(env, cs[1], start_side)) - far_side_at_reset,
        remaining_trace=remaining,
        clean_fraction_trace=cleanfrac,
    )


def batched_window_return(env: CleaningEnv, policy: Policy, key: jax.Array,
                          num_envs: int = 64, steps: int | None = None):
    """Mean return and cleaning progress over a batch of training-length windows.

    Cheap enough to call during training as an evaluation signal, and it is the
    same quantity the trainer optimises, so it is comparable across policies.
    """
    steps = steps or env.cfg.sim.max_steps
    keys = jax.random.split(key, num_envs)
    state, _ = jax.vmap(env.reset)(keys)
    carry = jax.vmap(policy.init, in_axes=(None, 0))(env, state)

    def step(cs, _):
        c, s = cs
        c, a = jax.vmap(policy.act, in_axes=(None, 0, 0))(env, s, c)
        s, _o, r, _te, _tr, info = jax.vmap(env.step)(s, a)
        return (c, s), (r, info["fraction_removed"], info["drained_kg"])

    (_, final), (rewards, removed, drained) = jax.lax.scan(
        step, (carry, state), None, length=steps
    )
    return {
        "return": float(jnp.mean(jnp.sum(rewards, axis=0))),
        "fraction_removed": float(jnp.mean(removed[-1])),
        "drained_kg": float(jnp.mean(drained[-1])),
    }
