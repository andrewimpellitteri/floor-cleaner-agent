# Action-space redesign: decide strokes, not wrist angles

## Why

`results/action_leverage.txt` measured the thing that eight training
configurations had been failing against without naming it. From a fixed
mid-job state, varying ONE low-level action and then behaving identically for
60 s moves the discounted genuine return by:

    std 0.006254  on a return of -111.89        = 5.6e-5 relative
    against a between-floor std of 3.375        = 0.19% of the nuisance

PPO has to recover the first number from inside the second. That is a property
of the framing -- 4500 sequential decisions each worth 5.6e-5 of the outcome --
and no reward, critic or entropy setting touches it. It is also why both honest
diagnostics died: EV, then EV_f, both saturate at 1.000 because the critic can
predict the 99.8% of return fixed by which floor was drawn and where the reset
landed.

The physics said the same thing from the other end and had done for a while:
coverage gaps outweigh genuinely-uncuttable grit 3-4:1 by mass, and in one seed
98% of the grit left behind had never been walked over. The floor is cleaned by
WHERE YOU WALK.

## What changed

`floorclean/macro.py`. One agent decision commits to a whole stroke; a scripted
low-level controller executes it for K = 128 control steps (25.6 s), giving 35
decisions per episode instead of 4500.

Action, 4 values in [-1, 1]:

    lane_x    where along the bay to put this pass
    side      which half of the trough
    standoff  wand height held during the push
    tilt      wand angle from vertical during the push

The controller is deliberately scripted for the part the physics already
settled -- walk to the wall end of the chosen lane with the wand lifted clear,
then push down the slope to the trough at the commanded geometry, jet aimed at
the trough. What is left to the agent is exactly what the measurements say
decides the outcome.

## Does it work? (the measurement, not the intention)

Identical protocol on the macro action space:

    quantity                        low-level     macro      ratio
    std / |return|                   5.6e-5       9.7e-4      17x
    one decision / whole floor       0.0019       0.0324      17x

Why 17x is the relevant threshold rather than just "better": with ~32k samples
per update, averaging cuts noise by sqrt(N) ~ 181.

    low-level effective SNR = 181 * 0.0019 = 0.34   (below 1 -- not learnable)
    macro     effective SNR = 181 * 0.0324 = 5.9    (learnable)

So the redesign moves the problem across the line rather than merely improving
a number. It does NOT claim the policy will beat the scripted sweep; that is
what the training run is for.

## Two implementation traps, both measured rather than guessed

**Semi-MDP discounting.** A macro-step spans K control steps, so the reward is
the gamma-discounted sum WITHIN the window and the trainer's discount must be
gamma**K (0.9998^128 = 0.974722). `train.py --macro` overrides `ppo.gamma`
rather than asking for it, because a self-consistent WRONG pair would pass the
existing guard. The potential-based shaping guarantee survives untouched:
summing gamma^i (gamma*Phi_{i+1} - Phi_i) telescopes to gamma^K Phi_K - Phi_0.

**Reward scale vs the Wiewiora offset.** A macro reward is a sum of K control
rewards, so unscaled its variance dwarfs the analytic offset. Measured on a
smoke run: value_loss 4459 against policy_loss 0.12, i.e. with vf_coef = 0.5
the gradient was ~20000:1 value-dominated and the policy would barely move. Fix
is to divide the macro reward by K -- which cannot change the optimal policy --
but the offset V = f - alpha*scale*Phi MUST be divided by the same K or it
stops cancelling the shaping it was derived to cancel. `ppo.py` now reads
`env.reward_scale` instead of importing REWARD_SCALE, so the wrapper can set
both consistently (200.0 low-level, 1.5625 macro).

## No wrapper state

The macro env operates directly on `EnvState`. A `MacroState` wrapper was
written first and removed: `ppo.init_runner` staggers episode clocks with
`env_state._replace(step=...)` and reads `env_state.potential`, and a wrapper
whose `_replace` silently created a wrapper-level `step` would have left every
environment's real clock at zero -- reintroducing the lockstep-batch
correlation the stagger exists to prevent. The macro index is just
`env_state.step // macro_steps`, so nothing needed wrapping.
