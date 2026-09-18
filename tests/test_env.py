"""Environment-level invariants, mirroring what test_physics.py does for the
physics (see WORKBOARD.md T5).

The reward is only a faithful measurement instrument if these hold:

1. Shaping telescopes: summed (reward + time cost) over any stretch equals
   REWARD_SCALE * (Phi_final - Phi_initial). If this fails, the agent is being
   paid for something other than progress.
2. No free reward: a parked wand earns (approximately) -TIME_COST*dt and
   nothing more.
3. Truncation is not termination.
4. reset is deterministic given a key; vmap and jit both work.
5. Overshoot is possible: a hard push near the trough can drive grit onto the
   far slope. This is why the full bay is modelled instead of half.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from floorclean.config import Config, SimConfig
from floorclean.env import (
    ALPHA_DISTANCE,
    COST_BOUND,
    COST_DEPOSITED,
    COST_SUSPENDED,
    FINISH_BONUS,
    REWARD_SCALE,
    TIME_COST,
    CleaningEnv,
    EnvState,
)
from floorclean.physics import residual_map


def fresh_state(env: CleaningEnv, key: jax.Array) -> EnvState:
    """Fresh uniformly-dirty floor, operator at the wall (no random progress)."""
    return env.fresh_state(key)


def rollout(env: CleaningEnv, state: EnvState, actions):
    """Step with a fixed action sequence; return (final_state, summed reward)."""
    total = 0.0
    for a in actions:
        state, _, r, term, trunc, _ = env.step(state, jnp.array(a))
        total += float(r)
    return state, total


def test_shaping_telescopes():
    """sum(reward) + time costs == REWARD_SCALE * (Phi_T - Phi_0), exactly.

    This is the property that makes dense shaping equivalent to the true
    objective (Ng et al. 1999). It holds by construction only while the
    potential is computed from the same conserved fields the reward is
    computed from -- a future edit that breaks mass conservation, or that
    recomputes Phi with different constants, shows up here.

    Undiscounted env (discount=1.0) so the sum telescopes exactly; the
    production discount is pinned to the trainer's gamma by
    test_env_discount_matches_trainer_gamma below.
    """
    env = CleaningEnv(discount=1.0)
    state = fresh_state(env, jax.random.PRNGKey(3))
    phi0 = float(state.potential)

    # A plausible working action: move, hold the wand mid-height, moderate tilt.
    actions = [(0.4, -0.6, 0.0, 0.1, 0.3)] * 40
    final, total_r = rollout(env, state, actions)

    time_cost = 40 * TIME_COST * env.cfg.sim.control_dt
    phiT = float(final.potential)
    assert total_r + time_cost == pytest.approx(REWARD_SCALE * (phiT - phi0), rel=1e-4, abs=1e-6)


def test_env_discount_matches_trainer_gamma():
    """The shaping discount must equal the PPO discount (WORKBOARD B2).

    F = gamma*Phi(s') - Phi(s) is policy-invariant only for the gamma the
    optimiser discounts with. A silent mismatch biases every step by
    (1-gamma)*Phi, which measured ~5x TIME_COST*dt.
    """
    from floorclean.ppo import PPOConfig
    assert CleaningEnv().discount == pytest.approx(PPOConfig().gamma)


def test_no_free_reward():
    """Parked at max standoff: only the known discount drizzle plus time cost.

    The wand is held as high as the arm allows, where impingement pressure is
    under the adhesion of nearly all grit. Whatever tiny loosening remains at
    the weakest cells must stay within 1% of the episode's shaping scale -- a
    policy that could farm reward standing still would invalidate the whole
    objective.
    """
    env = CleaningEnv()
    state = fresh_state(env, jax.random.PRNGKey(4))
    phi0 = float(state.potential)

    n = 20
    # action: [x, y, azimuth rate, standoff target, tilt target]
    parked = [(0.0, 0.0, 0.0, 1.0, 0.0)] * n
    final, total_r = rollout(env, state, parked)

    # With discount gamma < 1, a parked wand earns the known discount drizzle
    # SCALE*(gamma-1)*Phi0 per step (Phi < 0, so this is positive) -- part of
    # the Ng-invariant transform, not farmable progress. Anything beyond it is.
    drizzle = REWARD_SCALE * (env.discount - 1.0) * phi0
    time_cost = n * TIME_COST * env.cfg.sim.control_dt
    per_step = (total_r + time_cost) / n  # mean shaping only
    scale = abs(REWARD_SCALE * phi0)
    assert abs(per_step - drizzle) < 0.01 * scale / n
    # And the time cost really is being charged on top of the drizzle.
    assert abs((total_r - n * drizzle) + time_cost) < 0.01 * scale / n


def test_truncation_is_not_termination():
    """At max_steps: truncated=True, terminated=False on a dirty floor."""
    cfg = Config(sim=SimConfig(max_steps=6))
    env = CleaningEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(5))

    term = trunc = None
    for _ in range(6):
        state, _, _, term, trunc, _ = env.step(state, jnp.zeros(5))
        assert state.step > 0
    assert bool(trunc) and not bool(term)

    # And the floor is nowhere near clean -- six steps cannot finish a bay.
    worst = float(jnp.max(residual_map(state.fields)))
    assert worst > cfg.dirt.clean_threshold


def test_reset_deterministic_vmap_jit():
    env = CleaningEnv()
    key = jax.random.PRNGKey(6)

    s1, o1 = env.reset(key)
    s2, o2 = env.reset(key)
    same = jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda a, b: jnp.allclose(a, b), s1, s2)
    )
    assert bool(same)

    # vmap over resets: distinct keys give distinct floors.
    states, _ = jax.vmap(env.reset)(jax.random.split(key, 4))
    assert states.fields.bound.shape == (4, env.cfg.floor.nx, env.cfg.floor.ny)

    # jit steps fine.
    stepped = jax.jit(env.step)(s1, jnp.zeros(5))
    assert stepped[0].step == 1


def test_overshoot_possible():
    """A hard laid-over push at the trough drives grit onto the far slope.

    Guards the reason the FULL bay is modelled: cut the domain at the trough
    and this failure mode becomes impossible by construction, no matter what
    the policy does.
    """
    env = CleaningEnv()
    state = fresh_state(env, jax.random.PRNGKey(7))
    fc = env.cfg.floor

    # A loose blob 0.5 m from the trough, and the operator standing back so
    # the IMPACT (not the tip -- at 72 deg it lands ~0.9 m ahead) starts on
    # the blob and is walked hard across the basin.
    blob = (jnp.abs(env.floor.y - (fc.trough_y + 0.5)) < 0.2) & (
        jnp.abs(env.floor.x - fc.length_x / 2) < 0.6
    )
    fields = state.fields._replace(deposited=jnp.where(blob, 0.3, 0.0))
    state = state._replace(
        fields=fields,
        tip_y=jnp.array(fc.trough_y + 1.4),
        standoff=jnp.array(0.30),
        tilt=jnp.array(1.25),
        azimuth=jnp.array(-jnp.pi / 2),
        potential=env._potential(fields, state.initial_mass),
    )

    def far_side(s):
        far = env.floor.y < fc.trough_y
        res = s.fields.bound + s.fields.deposited + s.fields.suspended
        return float(jnp.sum(jnp.where(far, res, 0.0)) * fc.cell_area)

    before = far_side(state)
    state, _ = rollout(env, state, [(0.0, -1.0, 0.0, 0.4, 0.9)] * 100)

    # The bar is deliberately low: this guards EXISTENCE of the failure mode
    # (a few tenths of a gram crossing), not its magnitude.
    assert far_side(state) - before > 2e-4, (
        f"no overshoot: far-side grit {before:.6f} -> {far_side(state):.6f} kg"
    )


def test_shaping_weights_are_ordered():
    """The phase costs must order bound > deposited > suspended: each phase
    transition must pay, or the agent is not rewarded for making progress
    through bound -> deposited -> suspended -> drained."""
    assert COST_BOUND > COST_DEPOSITED > COST_SUSPENDED > 0.0
    assert ALPHA_DISTANCE > 0.0
    assert FINISH_BONUS > 0.0
