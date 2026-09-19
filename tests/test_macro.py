"""Semi-MDP invariants for the macro-action wrapper.

Both things tested here failed silently when I got them wrong, which is why
they are tests rather than comments:

1. The discount must be gamma**K. A self-consistent WRONG pair (env and
   trainer agreeing on a bad value) passes train.py's existing guard, and the
   only symptom is that every value estimate is quietly mis-scaled.

2. The reward and the Wiewiora offset must be divided by the SAME K. Dividing
   the reward alone cannot change the optimal policy, but it breaks the
   offset's cancellation of the shaping -- and the offset is what makes the
   critic's job tractable. Measured symptom of getting the scale wrong in the
   other direction: value_loss 4459 against policy_loss 0.12.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from floorclean.env import FINISH_BONUS, REWARD_SCALE, CleaningEnv
from floorclean.macro import MacroEnv


def test_macro_discount_is_gamma_to_the_k():
    env = CleaningEnv()
    m = MacroEnv(env)
    assert m.discount == pytest.approx(env.discount ** m.macro_steps, rel=1e-12)
    # And it must be a real discount, not accidentally >= 1.
    assert 0.0 < m.discount < 1.0


def test_macro_reward_scale_matches_the_reward_division():
    """offset scale / control scale == 1/K, the same factor step() applies."""
    env = CleaningEnv()
    m = MacroEnv(env)
    assert m.reward_scale == pytest.approx(REWARD_SCALE / m.macro_steps, rel=1e-12)
    assert env.reward_scale == pytest.approx(REWARD_SCALE, rel=1e-12)


def test_macro_step_advances_exactly_k_control_steps():
    env = CleaningEnv()
    m = MacroEnv(env)
    st = m.fresh_state(jax.random.PRNGKey(0))
    before = int(st.step)
    ns, _obs, _r, _term, _trunc, _info = m.step(st, jnp.array([0.0, 1.0, -0.4, -0.2]))
    assert int(ns.step) - before == m.macro_steps


def test_macro_reward_is_the_scaled_discounted_inner_sum():
    """Reconstruct the macro reward from the control steps it wraps.

    This is the property that makes the semi-MDP bookkeeping correct: the macro
    reward is sum_i gamma^i r_i, divided by K. If either the discounting inside
    the window or the 1/K is dropped, this fails.
    """
    env = CleaningEnv()
    m = MacroEnv(env)
    action = jnp.array([0.2, 1.0, -0.3, -0.1])
    st = m.fresh_state(jax.random.PRNGKey(2))

    _ns, _o, macro_r, _t, _tr, _i = m.step(st, action)

    # Replay the same stroke one control step at a time, latch included.
    lane_x, side, standoff, tilt = m._decode(action)
    g, acc, es = env.discount, 0.0, st
    pushing = jnp.bool_(False)
    for i in range(m.macro_steps):
        pushing = pushing | m._at_start(es, lane_x, side)
        a = m._low_level(es, lane_x, side, standoff, tilt, pushing)
        es, _, r, _, _, _ = env.step(es, a)
        acc += (g ** i) * float(r)

    assert float(macro_r) == pytest.approx(acc / m.macro_steps, rel=2e-3, abs=1e-4)


def test_macro_env_delegates_attributes_the_eval_path_needs():
    """render/eval reach for env.floor et al; without delegation the eval
    silently degrades to curves-only and the fresh-floor benchmark is lost."""
    m = MacroEnv(CleaningEnv())
    for name in ("floor", "floor_mask", "n_floor", "obs_cfg"):
        assert hasattr(m, name), name
    # ...but a missing attribute must still raise rather than recurse forever.
    with pytest.raises(AttributeError):
        _ = m.definitely_not_a_real_attribute


def test_macro_step_completes_a_full_stroke():
    """One macro action must be one wall->trough push, from wherever it starts.

    Regression test for the stall: the phase used to be derived from position
    alone, as `at_lane & (tip_y <= wall_y - 0.12)`, which is true almost
    everywhere rather than only at the wall. A stroke begun at the trough aimed
    at the trough it was standing on and never moved (y stayed in [7.0, 8.0]);
    one that reached the wall stalled there. No macro action executed a push at
    all, and two training runs measured that rather than a policy.
    """
    env = CleaningEnv()
    m = MacroEnv(env)
    fc = env.cfg.floor
    action = jnp.array([0.0, 1.0, 0.0, 0.0])          # mid-lane, high side
    lane_x, side, standoff, tilt = m._decode(action)
    wall_y = fc.length_y

    # Start at the trough -- where every stroke after the first begins.
    st = m.fresh_state(jax.random.PRNGKey(0))._replace(
        tip_x=jnp.array(1.0), tip_y=jnp.array(fc.trough_y))

    es, pushing, visited_wall = st, jnp.bool_(False), False
    for _ in range(m.macro_steps):
        pushing = pushing | m._at_start(es, lane_x, side)
        es, _, _, _, _, _ = env.step(
            es, m._low_level(es, lane_x, side, standoff, tilt, pushing))
        visited_wall |= abs(float(es.tip_y) - wall_y) < 0.12

    assert visited_wall, "never repositioned out to the wall"
    assert float(es.tip_y) == pytest.approx(fc.trough_y, abs=0.15), (
        "did not finish the push at the trough")
    assert float(es.tip_x) == pytest.approx(float(lane_x), abs=0.15)


def test_macro_step_pays_the_finish_bonus_once():
    """A terminating window must not re-pay FINISH_BONUS on every sub-step.

    env.step pays FINISH_BONUS whenever the floor is clean -- it is a predicate,
    not a one-shot event -- and the low-level trainer hides that by resetting on
    `done`. An unmasked macro window paid it up to 128 times: measured 399.0 for
    a single macro step on a clean floor, against the 3.125 one bonus is worth
    after the 1/K scaling.
    """
    env = CleaningEnv()
    m = MacroEnv(env)
    st = m.fresh_state(jax.random.PRNGKey(0))
    f = st.fields
    cleaned = f._replace(bound=jnp.zeros_like(f.bound),
                         deposited=jnp.zeros_like(f.deposited),
                         suspended=jnp.zeros_like(f.suspended))
    # Recompute the cached potential too, or step 0 books a spurious shaping
    # jump from the dirty floor this state was derived from.
    clean = st._replace(
        fields=cleaned, potential=env._potential(cleaned, st.initial_mass))

    ns, _obs, r, terminated, _trunc, _info = m.step(clean, jnp.array([0.0, 1.0, 0.0, 0.0]))

    assert bool(terminated)
    # The window stops at the terminal step rather than running on for 128.
    assert int(ns.step) - int(clean.step) == 1
    g, K = env.discount, m.macro_steps
    repeated = FINISH_BONUS * (1.0 - g ** K) / (1.0 - g) / K   # ~395, the bug
    assert float(r) == pytest.approx(FINISH_BONUS / K, rel=0.05), (
        f"expected one bonus ({FINISH_BONUS / K:.3f}), got {float(r):.3f}; "
        f"repeated-bonus value is {repeated:.1f}")


def test_macro_obs_shapes_is_a_property_like_the_inner_env():
    """`CleaningEnv.obs_shapes` is a property; the wrapper must match.

    This was a plain method returning `self.env.obs_shapes()` -- calling the
    dict the property returns. Nothing reads obs_shapes today (networks build
    from a sample obs), so it was a trap rather than a live failure.
    """
    env = CleaningEnv()
    m = MacroEnv(env)
    assert m.obs_shapes == env.obs_shapes
    assert m.obs_shapes["vector"] == (11,)
