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


def test_wiewiora_offset_cancels_drizzle():
    """V(s) = f(s) - SCALE*Phi(s) removes shaping from TD residuals (issue #4).

    With r = r_gen + SCALE*(gamma*Phi' - Phi) and V = f - SCALE*Phi, the TD
    residual delta = r + gamma*V' - V reduces to r_gen + gamma*f' - f: every
    shaping term cancels identically (given env.discount == trainer gamma).
    This is the analytic justification for the Option C critic baseline -- not
    merely "the critic's job gets easier". Pins SCALE (not SCALE/(1-gamma),
    not gamma*SCALE) and the no-gamma-factor offset from Ng.
    """
    from floorclean.ppo import PPOConfig

    env = CleaningEnv()
    assert env.discount == PPOConfig().gamma
    gamma = env.discount

    # Pure algebra half: arbitrary potentials, zero residual.
    for phi, phi_next, r_gen in [(-2.72, -2.70, -0.2), (-1.3, -1.25, -0.2),
                                 (-0.05, -0.02, -0.2), (-2.72, -2.72, -0.2)]:
        r = r_gen + REWARD_SCALE * (gamma * phi_next - phi)
        v, v_next = -REWARD_SCALE * phi, -REWARD_SCALE * phi_next  # f = 0
        delta = r + gamma * v_next - v
        assert delta == pytest.approx(r_gen), (phi, phi_next)

    # Env-grounded half: one real step, f = 0, delta must equal time cost
    # (plus finish bonus only if the floor actually came clean).
    state = fresh_state(env, jax.random.PRNGKey(11))
    phi0 = float(state.potential)
    dt = env.cfg.sim.control_dt
    state2, _, r, term, _trunc, _info = env.step(state, jnp.zeros(5))
    phi1 = float(state2.potential)
    v0, v1 = -REWARD_SCALE * phi0, -REWARD_SCALE * phi1
    r_gen = float(r) - REWARD_SCALE * (gamma * phi1 - phi0)
    delta = float(r) + gamma * float(jnp.where(term, 0.0, v1)) - v0
    expected = -TIME_COST * dt + (FINISH_BONUS if bool(term) else 0.0)
    assert r_gen == pytest.approx(expected, rel=1e-4, abs=1e-6)
    assert delta == pytest.approx(expected, rel=1e-4, abs=1e-6)


def _zero_value_head(params):
    """Force the critic to output exactly 0, so V(s) == -REWARD_SCALE*Phi(s).

    Makes the analytic offset the ONLY thing in the value, which is what lets
    the tests below read it straight out of the summary. The value head is an
    unnamed `nn.Dense(1)` (networks.py), so it is the only leaf in the tree
    whose last axis is 1 -- kernel (hidden, 1) and bias (1,). Everything else
    is a conv kernel, a LayerNorm vector, log_std (5,), or an actor Dense.
    """
    import jax.tree_util as jtu

    def fix(leaf):
        if leaf.ndim >= 1 and leaf.shape[-1] == 1:
            return jnp.zeros_like(leaf)
        return leaf

    zeroed = jtu.tree_map(fix, params)
    n = sum(1 for leaf in jtu.tree_leaves(params)
            if leaf.ndim >= 1 and leaf.shape[-1] == 1)
    assert n == 2, f"expected value kernel+bias, matched {n} leaves"
    return zeroed


def _tiny_chunk(max_steps=6, num_envs=4, num_steps=8):
    """A chunk small enough to run on CPU, short enough to truncate inside it."""
    from floorclean.ppo import PPOConfig, init_runner, make_chunk

    cfg = Config(sim=SimConfig(max_steps=max_steps))
    env = CleaningEnv(cfg)
    ppo = PPOConfig(num_envs=num_envs, num_steps=num_steps, num_minibatches=1,
                    update_epochs=1, updates_per_chunk=1)
    runner = init_runner(env, ppo, jax.random.PRNGKey(0))
    return env, ppo, runner, make_chunk(env, ppo)


def test_chunk_applies_the_wiewiora_offset_with_the_pre_step_potential():
    """The offset must be wired to Phi(s_t), with the right sign and scale.

    This tests `make_chunk`, not the algebra -- the algebra is pinned above.
    With the value head zeroed, f == 0 identically, so the summary must satisfy
    value_mean == -REWARD_SCALE * phi_mean exactly. That single identity catches
    a missing offset, a flipped sign, a wrong scale, and (via phi_mean) the
    off-by-one where Phi(s_{t+1}) is paired with V(s_t).
    """
    env, ppo, runner, chunk = _tiny_chunk()
    zeroed = _zero_value_head(runner.train_state.params)
    runner = runner._replace(
        train_state=runner.train_state.replace(params=zeroed))

    _runner, summary = chunk(runner)
    value_mean = float(jnp.ravel(summary["value_mean"])[0])
    phi_mean = float(jnp.ravel(summary["phi_mean"])[0])

    assert value_mean == pytest.approx(-REWARD_SCALE * phi_mean, rel=1e-5), (
        f"V != -SCALE*Phi: value_mean {value_mean:.4f} vs "
        f"{-REWARD_SCALE * phi_mean:.4f} -- the offset is not wired to the "
        f"pre-step potential")
    # Sanity: a dirty floor has Phi < 0, so the offset is a large POSITIVE value.
    assert phi_mean < 0.0
    assert value_mean > 0.0


def test_chunk_phi_is_the_pre_step_potential_not_the_post_step_one():
    """`traj.phi` must be Phi(s_t): read before env_step overwrites env_state.

    A one-line ordering slip there pairs V(s_t) with Phi(s_{t+1}) and quietly
    stops the cancellation working. Caught by running a single-step rollout from
    a known state and comparing against the potential before and after.
    """
    env, ppo, runner, chunk = _tiny_chunk(num_steps=1, num_envs=2)
    phi_before = float(jnp.mean(runner.env_state.potential))

    _runner, summary = chunk(runner)
    phi_mean = float(jnp.ravel(summary["phi_mean"])[0])

    assert phi_mean == pytest.approx(phi_before, rel=1e-6), (
        f"phi_mean {phi_mean:.6f} != pre-step {phi_before:.6f}; traj.phi is "
        f"being read after env_step")


def test_chunk_advantages_are_not_drizzle_scaled():
    """The learning signal must not carry the shaping drift (issue #4).

    The drizzle is REWARD_SCALE*(gamma-1)*Phi ~ +0.54/step on a fresh floor. If
    the offset were absent or mis-signed, the critic would start ~540 off and
    the raw advantages would inherit that scale. With the cancellation working
    they sit near the genuine per-step reward instead.

    Deliberately a loose bound: this is a smoke test for an order-of-magnitude
    failure, not a precision claim. It is the difference between "the drizzle
    cancelled" and "it did not".
    """
    env, ppo, runner, chunk = _tiny_chunk()
    zeroed = _zero_value_head(runner.train_state.params)
    runner = runner._replace(
        train_state=runner.train_state.replace(params=zeroed))

    _runner, summary = chunk(runner)
    adv_std = float(jnp.ravel(summary["adv_std_global"])[0])
    phi_mean = float(jnp.ravel(summary["phi_mean"])[0])
    offset = abs(REWARD_SCALE * phi_mean)  # ~540 on a dirty floor

    assert jnp.isfinite(adv_std)
    assert adv_std < 0.1 * offset, (
        f"adv_std {adv_std:.3f} is within an order of magnitude of the "
        f"offset {offset:.1f} -- the shaping term is still in the advantage")


def test_ambient_source_conserves_total_flow():
    """Every layout delivers the SAME total water -- geometry is not a volume knob.

    The point of `ambient_layout` is to change WHERE the rinse lands without
    changing how much of it there is (issue #1). If the layouts did not
    integrate to the same Q, any comparison between them would be confounded by
    flow rate, which is the one thing the A/B is supposed to hold fixed.
    """
    import dataclasses as _dc

    from floorclean.geometry import ambient_source, build_floor

    base = Config()
    for layout in ("uniform", "two_tap", "bar", "real"):
        cfg = _dc.replace(base, floor=_dc.replace(base.floor, ambient_layout=layout))
        src = ambient_source(cfg, build_floor(cfg))
        q = float(jnp.sum(src) * cfg.floor.cell_area)
        assert q == pytest.approx(cfg.floor.ambient_inflow, rel=2e-2), layout
        assert float(jnp.min(src)) >= 0.0, layout


def test_ambient_layouts_differ_in_shape():
    """...and that they are genuinely different distributions, not the same field.

    Guards against a layout silently falling back to uniform: the concentrated
    ones must be markedly more unequal. Measured jet-free at equal Q, the real
    bay's layouts reach a HIGHER mobile fraction than uniform rain despite a
    LOWER median depth -- they trade a dead majority for a live channel.
    """
    import dataclasses as _dc

    from floorclean.geometry import ambient_source, build_floor

    base = Config()

    def spread(layout):
        cfg = _dc.replace(base, floor=_dc.replace(base.floor, ambient_layout=layout))
        src = np.asarray(ambient_source(cfg, build_floor(cfg)))
        return src.std() / max(src.mean(), 1e-12)

    uniform = spread("uniform")
    assert uniform < 1e-6, "uniform layout is not uniform"
    for layout in ("two_tap", "bar", "real"):
        assert spread(layout) > 0.5, layout


def test_env_uses_the_configured_ambient_layout():
    """The env must actually consume the field, not rebuild a uniform one."""
    import dataclasses as _dc

    from floorclean.geometry import ambient_source, build_floor

    cfg = Config()
    env = CleaningEnv(cfg)
    expected = ambient_source(cfg, build_floor(cfg))
    assert jnp.allclose(env.ambient, expected)
    # and the default is the real bay, not the physically unavailable rain
    assert cfg.floor.ambient_layout == "real"


def test_trough_retains_a_pond_but_still_drains():
    """The trough is a sink, not a perfect one (issue #3).

    Andrew: "trough has a small pond near base due to warping and wear". Water
    above the retained depth must still leave -- otherwise the bay floods -- but
    the puddle itself must stay. Setting `trough_retain_depth = 0` has to
    restore the old perfect-sink behaviour exactly, which is what makes this a
    modelling choice rather than a behaviour change hidden in the physics.
    """
    import dataclasses as _dc

    def pond_after(retain, steps=900):
        cfg = _dc.replace(Config(),
                          floor=_dc.replace(Config().floor, trough_retain_depth=retain))
        env = CleaningEnv(cfg)
        state = env.fresh_state(jax.random.PRNGKey(0))
        act = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])

        def step(s, _):
            s, _o, _r, _t, _tr, info = env.step(s, act)
            return s, info["pond_m3"]

        _s, trace = jax.jit(lambda s: jax.lax.scan(step, s, None, length=steps))(state)
        return float(trace[-1])

    ponded = pond_after(5.0e-3)
    perfect_sink = pond_after(0.0)

    assert ponded > 5.0 * perfect_sink, (
        f"retained pond {ponded:.4g} m^3 is not meaningfully more than the "
        f"perfect sink's {perfect_sink:.4g}")
    # ...and it is a puddle, not a reservoir: bounded by depth x trough area.
    fc = Config().floor
    cap = 5.0e-3 * fc.trough_width * fc.length_x * 1.5
    assert ponded < cap, f"pond {ponded:.4g} m^3 exceeds the retainable {cap:.4g}"


def test_delivered_is_at_least_drained():
    """`drained` is what left the building; `delivered` also counts the pond.

    They were the same number while the trough was a perfect sink, and the
    workboard read `drained_kg` as "left the floor" throughout. With a pond they
    differ, and conflating them would overstate progress.
    """
    env = CleaningEnv(Config())
    state = env.fresh_state(jax.random.PRNGKey(0))
    act = jnp.array([0.0, 0.0, 0.0, -1.0, 1.0])

    def step(s, _):
        s, _o, _r, _t, _tr, info = env.step(s, act)
        return s, jnp.stack([info["drained_kg"], info["delivered_kg"]])

    _s, tr = jax.jit(lambda s: jax.lax.scan(step, s, None, length=400))(state)
    drained, delivered = np.asarray(tr).T
    assert np.all(delivered >= drained - 1e-9)
    assert delivered[-1] > 0.0


def test_floor_cleanliness_ignores_the_trough():
    """Grit sitting in the trough has left the FLOOR, which is the job.

    Once the trough can retain grit, counting its cells would make `done_clean`
    unreachable and would have the shaping term penalise delivery -- the exact
    opposite of the intended incentive.
    """
    env = CleaningEnv(Config())
    state = env.fresh_state(jax.random.PRNGKey(0))
    in_trough = (env.floor.trough >= 0.5)
    assert bool(jnp.any(in_trough)), "no trough cells to test with"

    # A spotless floor with a heavily loaded trough must read as clean.
    clean_floor = state.fields._replace(
        bound=jnp.zeros_like(state.fields.bound),
        deposited=jnp.where(in_trough, 10.0 * Config().dirt.clean_threshold, 0.0),
        suspended=jnp.zeros_like(state.fields.suspended),
    )
    s2 = state._replace(fields=clean_floor)
    _s, _o, _r, terminated, _tr, info = env.step(s2, jnp.zeros(5))
    assert float(info["fraction_clean"]) == pytest.approx(1.0)
    assert float(info["trough_grit_kg"]) > 0.0
    assert bool(terminated), "a clean floor must terminate even with a loaded trough"
