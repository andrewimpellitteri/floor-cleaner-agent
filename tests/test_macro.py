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

from floorclean.env import REWARD_SCALE, CleaningEnv
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

    # Replay the same stroke one control step at a time.
    lane_x, side, standoff, tilt = m._decode(action)
    g, acc, es = env.discount, 0.0, st
    for i in range(m.macro_steps):
        a = m._low_level(es, lane_x, side, standoff, tilt)
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
