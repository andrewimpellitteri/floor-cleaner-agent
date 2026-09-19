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

# Control steps per macro-action. A full pass is ~7 m of push at 0.45 m/s
# (~78 steps) plus up to ~4.2 m of repositioning in x and a walk back out to
# the wall. 128 steps (25.6 s) covers the worst case, so a macro-action
# reliably completes the stroke it commits to rather than being cut off
# mid-push -- which would make the same action mean different things depending
# on where the operator happened to be standing.
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

    def _low_level(self, env_state: EnvState, lane_x, side, standoff, tilt):
        """One control action implementing the commanded stroke.

        Two phases, decided from position rather than from a stored counter so
        the controller is stateless and the macro-step stays a pure scan:
          reposition -- wand lifted clear, walk to the wall end of `lane_x`
          push       -- walk down the slope to the trough at the commanded
                        standoff and tilt, jet aimed at the trough
        """
        env, fc, wc = self.env, self.env.cfg.floor, self.env.cfg.washer
        wall_y = jnp.where(side > 0, fc.length_y, 0.0)

        # In position to start a push? Must be at the right lane AND out at the
        # wall end of it, otherwise a stroke would start from wherever the last
        # one finished and the action would not mean what it says.
        at_lane = jnp.abs(env_state.tip_x - lane_x) < 0.12
        past_start = jnp.where(side > 0,
                               env_state.tip_y <= wall_y - 0.12,
                               env_state.tip_y >= wall_y + 0.12)
        pushing = at_lane & past_start

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
            es, acc, term_any = carry
            a = self._low_level(es, lane_x, side, standoff, tilt)
            nes, _, r, terminated, _trunc, info = self.env.step(es, a)
            acc = acc + (g ** i) * r
            return (nes, acc, term_any | terminated), info

        # The inner env already produces every diagnostic the trainer logs, so
        # carry the whole per-step info out and keep the LAST one rather than
        # rebuilding a partial dict here -- a hand-rolled subset silently drops
        # keys and fails deep inside a jitted scan.
        (env_state, macro_reward, terminated), infos = jax.lax.scan(
            inner, (state, jnp.array(0.0), jnp.bool_(False)),
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
