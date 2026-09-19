"""A macro-action (semi-MDP) wrapper: decide strokes, not wrist angles.

WHY THIS EXISTS -- the measurement that forced it.

`results/action_leverage.txt`: from a fixed mid-job state, varying a single
low-level action and then behaving identically for 60 s moves the discounted
genuine return by a std of 0.006 against a mean of -111.89. That is 5.6e-5 of
the return, and 0.19% of the variation between different floors. PPO has to
recover the first number from inside the second, a ~500:1 noise-to-signal
ratio at the batch level.

That is a property of the FRAMING, not of the reward or the critic. Eight
training configurations spanning alpha, ent_coef and two different reward
designs all landed in a narrow band, and both honest diagnostics (EV, then
EV_f) saturated at 1.000 because the critic can predict the 99.8% of return
that is set by which floor was drawn and how far through the job the reset
landed.

The physics says the same thing from the other end: coverage gaps outweigh
genuinely-uncuttable grit 3-4:1 by mass, and in one seed 98% of the grit left
behind had simply never been walked over. The floor is cleaned by WHERE YOU
WALK, not by what the wand does in any given 0.2 s.

So this wrapper changes the decision, not the physics. One macro-action commits
to a whole stroke -- which lane, which side of the trough, how high, how laid
over -- and a scripted low-level controller executes it for MACRO_STEPS control
steps. An episode becomes ~35 decisions that each move a real share of the
outcome, instead of 4500 that each move 5.6e-5 of it.

SEMI-MDP BOOKKEEPING. A macro-step spans K control steps, so the reward is the
gamma-discounted sum WITHIN the window and the discount handed to the trainer
is gamma**K. Getting this wrong would silently rescale every value estimate.
The potential-based shaping guarantee survives unchanged: summing
gamma^i * (gamma*Phi_{i+1} - Phi_i) over the window telescopes to
gamma^K * Phi_K - Phi_0, which is exactly the shaping term for one macro-step.

WHAT IS DELIBERATELY NOT LEARNED. The low-level controller is scripted, and
that is the point -- it encodes the part the physics already settled (push
toward the trough, lift the wand while repositioning so the return trip does
not undo the pass). The agent is left with the part the physics says actually
decides the outcome: where to put the next stroke, and how to hold the wand
for it.
"""

from __future__ import annotations

import dataclasses
import jax
import jax.numpy as jnp

from .baselines import _aim, _standoff, _tilt, _walk
from .env import CleaningEnv, EnvState, Obs
from .physics import residual_map

# Control steps per macro-action. A stroke is a reposition to the wall end of
# the commanded lane, then ~7 m of push to the trough at 0.45 m/s (~78 steps).
# `_walk` clips each axis independently at walk_speed = 1 m/s, so repositioning
# costs max(dx, dy) seconds, not the diagonal.
#
# Every stroke after the first starts where the previous one ended -- at the
# trough -- so the reposition is max(4.2, 7) = 7 s = 35 steps and the whole
# stroke fits in 113 of the 128 steps. Measured: from (1.0, 7.0) one macro step
# now reaches y = 14.00 and returns to exactly y = 7.00.
#
# The ONE case 128 does not cover is a stroke commanded from the far side of the
# trough, which needs max(4.2, 14) + 78 = 148 steps; measured, it ends at
# y = 8.69, short of the trough. That can only happen on the first stroke of an
# episode, because nothing else leaves the operator across the trough from the
# lane it is about to work. Raising MACRO_STEPS would fix it, but macro serial
# cost is num_steps * MACRO_STEPS and that budget has already bitten once, so
# the first stroke of each episode is allowed to be partial instead.
MACRO_STEPS = 128


# NOTE: there is deliberately no MacroState wrapper. The macro env operates
# directly on EnvState, because ppo.init_runner staggers episode clocks with
# `env_state._replace(step=...)` and reads `env_state.potential` for the
# Wiewiora offset. A wrapper would have to forward both, and a _replace that
# silently created a wrapper-level `step` instead of setting the real one would
# leave every environment's clock at zero -- the exact lockstep-batch problem
# the stagger exists to prevent. Nothing needed a wrapper: the macro-step index
# is just env_state.step // macro_steps.


@dataclasses.dataclass(frozen=True)
class MacroEnv:
    """Wraps CleaningEnv so one agent decision is one stroke.

    Action is 4 values in [-1, 1]:
      0  lane_x   -- where along the bay to put this pass
      1  side     -- which half of the trough (sign)
      2  standoff -- wand height held during the push
      3  tilt     -- wand angle from vertical during the push
    """

    env: CleaningEnv
    macro_steps: int = MACRO_STEPS
    push_speed: float = 0.45  # you push slower than you walk

    # ---- plumbing the trainer needs -------------------------------------
    @property
    def action_dim(self) -> int:
        return 4

    @property
    def reward_scale(self) -> float:
        """REWARD_SCALE / K, matching the 1/K applied to the macro reward.

        A macro reward is the discounted SUM of K control-step rewards, so left
        unscaled its variance dwarfs the analytic offset and the value loss
        swamps the policy gradient -- measured on a smoke run: value_loss 4459
        against policy_loss 0.12, i.e. vf_coef*value was ~20000x the policy
        term. Dividing the reward by K restores a per-control-step-equivalent
        scale. Dividing rewards by a constant cannot change the optimal policy,
        but the offset MUST be divided by the same constant or it stops
        cancelling the shaping.
        """
        return self.env.reward_scale / self.macro_steps

    @property
    def discount(self) -> float:
        """gamma**K -- the semi-MDP discount for one macro-step."""
        return float(self.env.discount ** self.macro_steps)

    @property
    def cfg(self):
        return self.env.cfg

    def __getattr__(self, name):
        """Delegate anything not overridden here to the wrapped CleaningEnv.

        The render and eval paths reach for env.floor, env.floor_mask,
        env.n_floor and friends. Without this the first eval raises
        AttributeError inside train.py's try/except, which degrades the run to
        "curves-only" and silently drops the fresh-floor benchmark -- the one
        honest metric, and the whole point of the eval fix. Observed on the
        first macro launch: "eval renders failed (AttributeError: 'MacroEnv'
        object has no attribute 'floor')".

        Guarded against recursion: dataclass __init__ sets `env` via
        object.__setattr__, and any lookup before that must raise rather than
        re-enter __getattr__ looking for `env` again.
        """
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def obs_shapes(self):
        return self.env.obs_shapes()

    def _observe(self, state: EnvState) -> Obs:
        return self.env._observe(state)

    # ---- episode management ---------------------------------------------
    def reset(self, key: jax.Array):
        return self.env.reset(key)

    def fresh_state(self, key: jax.Array) -> EnvState:
        return self.env.fresh_state(key)

    # ---- the low-level controller ----------------------------------------
    def _decode(self, action: jnp.ndarray):
        """Macro action in [-1,1]^4 -> physical stroke parameters."""
        fc, wc = self.env.cfg.floor, self.env.cfg.washer
        a = jnp.clip(action, -1.0, 1.0)
        lane_x = (a[0] * 0.5 + 0.5) * fc.length_x
        side = jnp.where(a[1] >= 0.0, 1.0, -1.0)
        standoff = wc.standoff_min + (a[2] * 0.5 + 0.5) * (wc.standoff_max - wc.standoff_min)
        tilt = (a[3] * 0.5 + 0.5) * wc.tilt_max
        return lane_x, side, standoff, tilt

    def _at_start(self, env_state: EnvState, lane_x, side):
        """Is the operator standing at the wall end of `lane_x` on `side`?

        This is the ONLY place a push may begin, which is what makes a macro
        action mean the same stroke regardless of where the last one ended.
        """
        fc = self.env.cfg.floor
        wall_y = jnp.where(side > 0, fc.length_y, 0.0)
        return ((jnp.abs(env_state.tip_x - lane_x) < 0.12)
                & (jnp.abs(env_state.tip_y - wall_y) < 0.12))

    def _low_level(self, env_state: EnvState, lane_x, side, standoff, tilt, pushing):
        """One control action implementing the commanded stroke.

        Two phases, selected by the `pushing` latch that `step` carries through
        the scan:
          reposition -- wand lifted clear, walk to the wall end of `lane_x`
          push       -- walk down the slope to the trough at the commanded
                        standoff and tilt, jet aimed at the trough

        THE LATCH IS NOT OPTIONAL. The first version derived the phase from
        position alone, as `at_lane & (tip_y <= wall_y - 0.12)`. That predicate
        is true almost everywhere on the floor rather than only out at the wall,
        so both ends stalled: a stroke beginning at the trough decided it was
        already pushing, aimed at the trough it was standing on, and sat there
        for all 128 control steps; a stroke that did reach the wall flipped the
        predicate false, targeted the wall, and stalled there instead. Measured
        on the broken version -- from (1.0, 7.0) the tip never left
        y in [7.00, 8.00]; from (0.5, 13.9) it ended pinned at y = 14.00. No
        macro action ever executed a wall->trough push, so the flat 0.37 clean
        fraction of the first two macro runs was measuring a controller that
        could not move slurry, not a policy that could not learn.

        Position alone cannot work even in principle: at an interior point,
        outbound (repositioning) and inbound (pushing) look identical. Hence a
        latch, which is also what PushSweep carries for the same reason.
        """
        env, fc, wc = self.env, self.env.cfg.floor, self.env.cfg.washer
        wall_y = jnp.where(side > 0, fc.length_y, 0.0)

        target_y = jnp.where(pushing, fc.trough_y, wall_y)
        speed = jnp.where(pushing, self.push_speed, 1.0)
        ax, ay = _walk(env, env_state, lane_x, target_y, speed)

        # Always drive the slurry down the slope toward the trough.
        target_az = jnp.where(side > 0, -jnp.pi / 2, jnp.pi / 2)
        # Lift clear while repositioning so the walk back does not undo the
        # pass just made (the same reason PushSweep does it).
        so = jnp.where(pushing, standoff, wc.standoff_max)

        return jnp.stack([
            ax, ay,
            _aim(env, env_state, target_az),
            _standoff(env, so),
            _tilt(env, tilt),
        ])

    # ---- the macro step ---------------------------------------------------
    def step(self, state: EnvState, action: jnp.ndarray):
        """Execute one stroke. Returns the semi-MDP transition."""
        g = self.env.discount
        lane_x, side, standoff, tilt = self._decode(action)

        def inner(carry, i):
            es, acc, done, pushing = carry
            # Latch before acting, so a stroke that already begins in position
            # pushes on step 0 rather than wasting one step repositioning.
            pushing = pushing | self._at_start(es, lane_x, side)
            a = self._low_level(es, lane_x, side, standoff, tilt, pushing)
            nes, _, r, terminated, _trunc, info = self.env.step(es, a)

            # Mask everything after the FIRST termination. env.step pays
            # FINISH_BONUS on every step where the floor is clean -- it is not
            # a one-shot event -- so an unmasked window paid it up to 128 times:
            # measured 399.0 for one macro step on an already-clean floor,
            # against the 3.125 a single bonus is worth at this scale. Worse,
            # the same stroke scored differently depending on which sub-step it
            # happened to finish on. The low-level trainer never saw this
            # because ppo.make_chunk resets on `done`; inside a macro window
            # nothing did.
            live = ~done
            acc = acc + jnp.where(live, (g ** i) * r, 0.0)
            # Freeze the terminal state rather than go on spraying a clean
            # floor for the rest of the window.
            nes = jax.tree.map(lambda new, old: jnp.where(live, new, old), nes, es)
            return (nes, acc, done | terminated, pushing), info

        # The inner env already produces every diagnostic the trainer logs, so
        # carry the whole per-step info out and keep the LAST one rather than
        # rebuilding a partial dict here -- a hand-rolled subset silently drops
        # keys and fails deep inside a jitted scan.
        (env_state, macro_reward, terminated, _pushing), infos = jax.lax.scan(
            inner, (state, jnp.array(0.0), jnp.bool_(False), jnp.bool_(False)),
            jnp.arange(self.macro_steps))
        info = jax.tree.map(lambda x: x[-1], infos)

        truncated = env_state.step >= self.env.cfg.sim.max_steps

        # Stroke parameters, so a trained policy is readable as technique.
        info = dict(info)
        info.update({"lane_x": lane_x, "side": side,
                     "macro_standoff": standoff, "macro_tilt": tilt})
        # /K -- see `reward_scale`. Keeps value targets the same order as the
        # low-level MDP so vf_coef means the same thing in both.
        return (env_state, self._observe(env_state), macro_reward / self.macro_steps,
                terminated, truncated, info)
