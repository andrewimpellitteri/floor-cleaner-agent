"""Partial observability: the operator remembers, he does not clairvoy.

Issue #8. With `ObsConfig.memory` on, the global map is a REMEMBERED view --
what he saw when last near each cell, frozen since -- plus a `seen` channel so
"never been there" is distinguishable from "been there, it was clean". The
local crop stays live.

The load-bearing property is the last test here: perception must not touch the
physics or the reward. If enabling memory changed the dynamics, every number
measured under the omniscient view would become incomparable, and the whole
point is to compare them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from floorclean.env import CleaningEnv, ObsConfig

BLIND = ObsConfig(memory=True)
ACTION = jnp.array([1.0, -0.6, 0.2, 0.0, -0.1])


def test_memory_is_off_by_default():
    """Every result in results/ was measured omniscient; the default must stay."""
    assert ObsConfig().memory is False
    assert ObsConfig().map_channels == 4
    assert BLIND.map_channels == 5


def test_blind_env_reports_the_extra_channel():
    env = CleaningEnv(obs_cfg=BLIND)
    shapes = env.obs_shapes
    assert shapes["global_map"][-1] == 5
    assert shapes["local_map"][-1] == 5
    obs = env._observe(env.fresh_state(jax.random.PRNGKey(0)))
    assert obs.global_map.shape == shapes["global_map"]
    assert obs.local_map.shape == shapes["local_map"]


def test_unvisited_floor_reads_blank_and_is_most_of_the_bay():
    """At t=0 he has seen only where he stands."""
    env = CleaningEnv(obs_cfg=BLIND)
    st = env.fresh_state(jax.random.PRNGKey(0))
    frac = float(jnp.mean(st.seen))
    assert 0.0 < frac < 0.10, f"saw {frac:.1%} of the bay from a standing start"
    # Unseen cells carry nothing -- including their yield stress, which is the
    # channel that otherwise gives away where the worn lanes are.
    unseen = st.seen < 0.5
    assert float(jnp.max(jnp.abs(st.remembered[unseen]))) == 0.0

    # ...whereas the omniscient view hands over the whole yield field at t=0.
    omni_env = CleaningEnv()
    omni_obs = omni_env._observe(omni_env.fresh_state(jax.random.PRNGKey(0)))
    assert float(jnp.std(omni_obs.global_map[..., 3])) > 0.0
    blind_obs = env._observe(st)
    assert float(jnp.std(blind_obs.global_map[..., 3])) < \
        float(jnp.std(omni_obs.global_map[..., 3])), \
        "the blind global map reveals as much yield structure as the omniscient one"


def test_seen_only_ever_grows_and_grows_by_walking():
    env = CleaningEnv(obs_cfg=BLIND)
    st = env.fresh_state(jax.random.PRNGKey(1))
    prev = float(jnp.sum(st.seen))
    start = prev
    for _ in range(60):
        st, _obs, _r, _t, _tr, _i = env.step(st, ACTION)
        now = float(jnp.sum(st.seen))
        assert now >= prev - 1e-6, "coverage went backwards"
        prev = now
    assert prev > start, "walking 12 s revealed nothing"


def test_the_remembered_map_goes_stale():
    """A cell cleaned after he walked away still reads as he left it.

    This is what gives revisiting a purpose, and it is the difference between a
    remembered map and a live one masked by a coverage stencil.
    """
    env = CleaningEnv(obs_cfg=BLIND)
    st = env.fresh_state(jax.random.PRNGKey(2))
    for _ in range(10):
        st, *_ = env.step(st, ACTION)

    seen_now = st.seen > 0.5
    snapshot = np.asarray(st.remembered)

    # Wipe the floor clean everywhere by fiat, then walk AWAY from where he is.
    f = st.fields
    st = st._replace(fields=f._replace(bound=jnp.zeros_like(f.bound),
                                       deposited=jnp.zeros_like(f.deposited),
                                       suspended=jnp.zeros_like(f.suspended)))
    away = jnp.array([-1.0, 1.0, 0.0, 0.0, 0.0])
    for _ in range(40):
        st, *_ = env.step(st, away)

    after = np.asarray(st.remembered)
    still_stale = seen_now & (st.seen > 0.5)
    changed = np.abs(after - snapshot).max(axis=-1) > 1e-6
    # Somewhere he has walked away from must still carry the old, dirty reading.
    assert np.any(np.asarray(still_stale) & ~changed), \
        "every remembered cell updated -- the map is live, not remembered"


def test_reset_clears_the_memory():
    env = CleaningEnv(obs_cfg=BLIND)
    st = env.fresh_state(jax.random.PRNGKey(3))
    for _ in range(40):
        st, *_ = env.step(st, ACTION)
    assert float(jnp.sum(st.seen)) > 0
    fresh, _obs = env.reset(jax.random.PRNGKey(4))
    assert float(jnp.mean(fresh.seen)) < 0.10


def test_perception_does_not_touch_physics_or_reward():
    """THE invariant. Memory changes what is SEEN, nothing else.

    Same seed, same actions, memory on and off: identical rewards and identical
    grit. Without this, every omniscient number in results/ would stop being
    comparable to anything measured here.
    """
    omni = CleaningEnv()
    blind = CleaningEnv(obs_cfg=BLIND)
    a = omni.fresh_state(jax.random.PRNGKey(5))
    b = blind.fresh_state(jax.random.PRNGKey(5))

    for i in range(30):
        act = jnp.array([0.8, -0.5, 0.3, -0.2, 0.1]) * (1.0 if i % 2 else -0.7)
        a, _o, ra, ta, _, ia = omni.step(a, act)
        b, _o, rb, tb, _, ib = blind.step(b, act)
        assert float(ra) == pytest.approx(float(rb), rel=1e-6, abs=1e-6)
        assert bool(ta) == bool(tb)
        assert float(ia["remaining_kg"]) == pytest.approx(
            float(ib["remaining_kg"]), rel=1e-6)

    assert float(a.tip_x) == pytest.approx(float(b.tip_x), rel=1e-6)
    assert float(jnp.sum(a.fields.bound)) == pytest.approx(
        float(jnp.sum(b.fields.bound)), rel=1e-6)


def test_memory_off_costs_a_placeholder_not_a_grid():
    """Omniscient runs must not carry full-size memory arrays..."""
    fc = CleaningEnv().cfg.floor
    st = CleaningEnv().fresh_state(jax.random.PRNGKey(6))
    assert st.seen.size < fc.nx * fc.ny
    assert st.remembered.size < fc.nx * fc.ny


def test_no_state_array_is_zero_size():
    """...but they must not be ZERO-size either, in any configuration.

    The trainer checkpoints env_state, and orbax refuses zero-size arrays:
    "Cannot save arrays with zero size: ParamInfo: [name=env_state.seen]".
    A (0, 0) placeholder killed a live run at its first checkpoint, and would
    have killed every run, because memory-off is the default path. Nothing in
    the unit tests touched checkpointing, so only the GPU run caught it.
    """
    for env in (CleaningEnv(), CleaningEnv(obs_cfg=BLIND),
                CleaningEnv(reward_mode="reach")):
        st = env.fresh_state(jax.random.PRNGKey(7))
        for name, leaf in zip(st._fields, jax.tree.leaves(st)):
            assert jnp.asarray(leaf).size > 0, f"{name} is zero-size"
