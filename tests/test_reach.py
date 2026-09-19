"""The `reach` diagnostic: can a policy gradient move the policy at all?

This mode is not part of the answer to Andrew's question. It exists to separate
two explanations for the RL failure that the cleaning task cannot distinguish:

  the learner is broken            -- the policy scored BELOW a uniform-random
                                      baseline (0.031 against 0.235)
  the problem is flat              -- nothing to learn, so the policy never moved

The evidence is ambiguous because both look the same from the outside. The
critic reaching EV 1.000 says the machinery, optimiser and data path all work;
a freshly initialised network emitting a near-constant action explains the
below-random score, because a constant action stands still and drains nothing.

So: a task with a large dense signal, a known optimum (walk straight there) and
a constant-action score of about zero. If PPO learns this, the machinery is fine
and the cleaning landscape really is flat. If it does not, something on the
policy side is broken and no amount of interesting problem structure will help.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from floorclean.env import TIME_COST, CleaningEnv

STRAIGHT = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])


def _toward(env, st, sign=1.0):
    """Action that walks the tip at the target (sign=-1 walks away)."""
    step = env.cfg.washer.walk_speed * env.cfg.sim.control_dt
    return jnp.stack([
        jnp.clip((st.target_x - st.tip_x) / step, -1, 1) * sign,
        jnp.clip((st.target_y - st.tip_y) / step, -1, 1) * sign,
        jnp.array(0.0), jnp.array(0.0), jnp.array(0.0),
    ])


def test_default_mode_is_untouched():
    """The diagnostic must not perturb the real objective."""
    env = CleaningEnv()
    assert env.reward_mode == "clean"
    assert env.obs_shapes["vector"] == (11,)


def test_reach_mode_shows_the_bearing():
    """Without the target in the observation the task is unsolvable, not hard."""
    env = CleaningEnv(reward_mode="reach")
    assert env.obs_shapes["vector"] == (13,)
    st = env.fresh_state(jax.random.PRNGKey(0))
    obs = env._observe(st)
    assert obs.vector.shape == (13,)
    fc = env.cfg.floor
    assert float(obs.vector[11]) == pytest.approx(
        (float(st.target_x) - float(st.tip_x)) / fc.length_x, rel=1e-5)
    assert float(obs.vector[12]) == pytest.approx(
        (float(st.target_y) - float(st.tip_y)) / fc.length_y, rel=1e-5)


def test_the_target_is_on_the_floor():
    env = CleaningEnv(reward_mode="reach")
    fc = env.cfg.floor
    for seed in range(8):
        st = env.fresh_state(jax.random.PRNGKey(seed))
        assert 0.0 < float(st.target_x) < fc.length_x
        assert 0.0 < float(st.target_y) < fc.length_y


def test_walking_toward_pays_and_walking_away_costs():
    """The sign of the signal is the whole diagnostic."""
    env = CleaningEnv(reward_mode="reach")
    st = env.fresh_state(jax.random.PRNGKey(1))
    _s, _o, r_to, *_ = env.step(st, _toward(env, st, +1.0))
    _s, _o, r_away, *_ = env.step(st, _toward(env, st, -1.0))
    assert float(r_to) > 0.0 > float(r_away)


def test_a_constant_action_scores_about_nothing():
    """What an untrained network emits. This is the FAIL line for the run."""
    env = CleaningEnv(reward_mode="reach")
    st = env.fresh_state(jax.random.PRNGKey(2))
    total = 0.0
    for _ in range(60):
        st, _o, r, _t, _tr, _i = env.step(st, STRAIGHT)
        total += float(r)
    # Standing still earns only the clock: no progress, no hits.
    assert total == pytest.approx(-TIME_COST * env.cfg.sim.control_dt * 60,
                                  rel=0.25, abs=0.5)


def test_progress_reward_telescopes():
    """Sum of progress = start distance - end distance, so circling pays zero.

    A reward that did not telescope could be farmed by oscillating toward and
    away from the target, and the diagnostic would measure exploitation of a
    bug rather than competence.
    """
    env = CleaningEnv(reward_mode="reach")
    st = env.fresh_state(jax.random.PRNGKey(3))
    d0 = float(jnp.hypot(st.target_x - st.tip_x, st.target_y - st.tip_y))
    dt = env.cfg.sim.control_dt
    # Accumulate only up to the first hit: that step pays a bonus and respawns
    # the target, so it is not part of a telescoping sum over one approach.
    total, d1, steps = 0.0, d0, 0
    for _ in range(60):
        st, _o, r, _t, _tr, i = env.step(st, _toward(env, st))
        if float(i["reach_hit"]) > 0.0:
            break
        total += float(r) + TIME_COST * dt
        d1 = float(jnp.hypot(st.target_x - st.tip_x, st.target_y - st.tip_y))
        steps += 1
    assert steps >= 5, "target was already within reach; nothing was tested"
    assert total == pytest.approx(d0 - d1, rel=1e-3, abs=1e-3)


def test_reaching_respawns_the_target_and_never_terminates():
    env = CleaningEnv(reward_mode="reach")
    st = env.fresh_state(jax.random.PRNGKey(4))
    first = (float(st.target_x), float(st.target_y))
    hits = 0.0
    for _ in range(400):
        st, _o, _r, term, _tr, i = env.step(st, _toward(env, st))
        hits += float(i["reach_hit"])
        assert not bool(term), "reach mode must run the full episode"
    assert hits >= 2.0, f"an oracle hit only {hits} targets in 80 s"
    assert (float(st.target_x), float(st.target_y)) != first


def test_reach_keeps_the_full_info_key_set():
    """A partial info dict fails deep inside the trainer's jitted scan."""
    clean = CleaningEnv()
    reach = CleaningEnv(reward_mode="reach")
    _s, _o, _r, _t, _tr, ic = clean.step(clean.fresh_state(jax.random.PRNGKey(5)),
                                         STRAIGHT)
    _s, _o, _r, _t, _tr, ir = reach.step(reach.fresh_state(jax.random.PRNGKey(5)),
                                         STRAIGHT)
    assert set(ic).issubset(set(ir))
    assert {"reach_hit", "reach_dist"} <= set(ir)
