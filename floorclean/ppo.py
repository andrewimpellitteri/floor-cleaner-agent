"""PPO, written end-to-end in JAX so the whole training loop lives on the GPU.

The environment is pure array maths, so there is no reason to pay for a Python
step loop or for copying observations across the PCI bus. Environments, rollout
collection, GAE and the optimiser updates all run inside one `lax.scan`, and the
whole thing is compiled once. On a 4090 this is worth several orders of
magnitude over the previous `DummyVecEnv` + Stable-Baselines3 setup, which
stepped ONE environment at a time in Python and was given 20,000 timesteps --
about twenty episodes, which is nowhere near enough to learn anything.

Two correctness points that the previous setup got wrong and that matter a lot
here:

1. TRUNCATION IS NOT TERMINATION. Hitting the five-minute limit does not mean
   the world ended, it means we stopped watching. The value function must
   bootstrap through it, or the agent learns that all value vanishes at 1500
   steps and front-loads its work. Since almost every episode here ends by
   truncation rather than by finishing the floor, getting this wrong would
   dominate the result. Handled by adding `gamma * V(s_final)` to the reward on
   truncated steps and only then treating the step as an episode boundary.

2. NO OBSERVATION NORMALISATION OVER THE MAPS. The old config wrapped a
   `VecNormalize` around a dict observation containing an image, which rescales
   every pixel by a running mean and variance and destroys the physical meaning
   of "how much grit is here". The observation channels are instead scaled by
   fixed, known physical constants in `env._channels`, so a given pixel value
   means the same thing for the whole run.
"""

from __future__ import annotations

import dataclasses
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from .env import REWARD_SCALE, CleaningEnv, Obs
from .networks import ActorCritic, entropy, log_prob


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    num_envs: int = 512
    num_steps: int = 64  # rollout length per environment per update
    updates_per_chunk: int = 20  # compiled together; checkpoint between chunks
    total_timesteps: int = 200_000_000

    lr: float = 3e-4
    anneal_lr: bool = True
    # 0.2 s per step, and the episode is sim.max_steps = 4500 steps (15 min).
    #
    # This was 0.999, justified as "a ~1000-step horizon, which is what a
    # 15-minute episode needs". That reasoning was wrong: 1/(1-gamma) = 1000
    # steps is 3.3 min, less than a quarter of the episode, so the agent could
    # not see its own terminal state. Measured consequence: gamma^4500 = 0.0111,
    # so FINISH_BONUS = 400 was worth 4.4 reward units at episode start against
    # ~200 of accumulated time cost over the same horizon -- a ratio of 0.022.
    # The finish bonus was invisible even before you ask whether it is
    # reachable, and it fired exactly zero times in 750M steps of training.
    #
    # 0.9998 gives a 5000-step (16.7 min) horizon, just longer than the
    # episode, and gamma^4500 = 0.407. The terminal event is now worth seeing.
    # MUST match CleaningEnv(discount=...) -- the shaping term uses it.
    gamma: float = 0.9998

    # How much of the potential to hand the critic analytically, in
    # V(s) = f(s) - alpha*REWARD_SCALE*Phi(s).
    #
    # This is a dial between two known-bad endpoints, and the reason it exists
    # is that they are the SAME quantity seen from two sides (issue #4):
    #
    #   alpha = 0  -- the critic must learn the whole -SCALE*Phi offset. It does,
    #                 to EV 0.99995, and the residual left for the policy is
    #                 4.6e-5 of return variance. Run 1: 150M steps, 0 episodes
    #                 ever clean, every metric worse than init.
    #   alpha = 1  -- exact cancellation. delta = r_gen + gamma*f' - f, and
    #                 r_gen is CONSTANT at -TIME_COST*dt (measured: std 0.0
    #                 over 3000 steps, one unique value) because the finish
    #                 bonus never fires. So the shaping is gone from the
    #                 learning problem entirely -- Wiewiora's equivalence run in
    #                 reverse -- and there is nothing to learn from. gate4:
    #                 adv_std collapses 16x, EV_f RISES to 0.79 as the policy
    #                 randomises into predictability.
    #
    # Intermediate alpha keeps some shaping in the TD residual while removing
    # most of the offset the critic would otherwise have to memorise.
    #
    # SUPERSEDED, and the history above is kept only because it is what forced
    # the real diagnosis. Every one of those failures was a symptom of an EMPTY
    # OBJECTIVE, not of a bad alpha: r_gen was constant because the reward had
    # no policy-dependent term at all, so the advantages were identically zero
    # and no alpha could help. A five-run sweep confirmed it -- alpha 0, 0.5 and
    # 1 and a 1000x range of ent_coef all converged to fraction_clean
    # 0.354-0.364, a spread of 0.010.
    #
    # env.DIRT_COST fixes the cause: r_gen now varies (std 0.021, 2928 unique
    # values over 3000 steps). With a genuine signal present, alpha = 1 is once
    # again the PRINCIPLED setting rather than a measured failure -- it cancels
    # the policy-invariant shaping out of the TD residual exactly, leaving
    # delta = r_gen + gamma*f' - f over a genuine, varying r_gen. The critic
    # then learns the real value function instead of memorising -SCALE*Phi.
    potential_baseline_alpha: float = 1.0
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.003
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 8

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // self.batch_size


class Transition(NamedTuple):
    done: jnp.ndarray  # episode boundary (terminated OR truncated)
    action: jnp.ndarray
    value: jnp.ndarray  # full V(s) = f(s) - SCALE*Phi(s), never the residual alone
    reward: jnp.ndarray  # already bootstrap-adjusted for truncation
    log_prob: jnp.ndarray
    obs: Obs
    phi: jnp.ndarray  # Phi(s_t) pre-step, (num_envs,) -- pairs with obs


class RunnerState(NamedTuple):
    train_state: TrainState
    env_state: Any
    obs: Obs
    rng: jax.Array


def _tree_where(mask, a, b):
    """Select between two pytrees per batch element."""
    return jax.tree.map(
        lambda x, y: jnp.where(mask.reshape((-1,) + (1,) * (x.ndim - 1)), x, y), a, b
    )


def init_runner(env: CleaningEnv, cfg: PPOConfig, rng: jax.Array) -> RunnerState:
    rng, k_net, k_reset, k_stagger = jax.random.split(rng, 4)

    network = ActorCritic(action_dim=env.action_dim)
    env_state, obs = jax.vmap(env.reset)(jax.random.split(k_reset, cfg.num_envs))

    # Stagger the episode clocks, once, at startup.
    #
    # Without this every environment starts at step 0 and truncates at
    # max_steps together, so the whole batch marches through the job in
    # lockstep: 4500 steps at 64 steps per update is ~70 updates, and the first
    # smoke run's metrics duly rose for 45 updates and fell for the next 25 as
    # all 512 envs reset at once. That is bad twice over -- it makes the
    # learning curves unreadable (episode phase dominates policy quality), and
    # it correlates every sample in the batch, which is exactly the variance
    # that having 512 environments is supposed to buy away.
    #
    # `reset` already randomises how far through the JOB a floor is; this
    # randomises where in the CLOCK each episode sits. Startup only: after the
    # first truncation they stay spread out on their own.
    start = jax.random.randint(k_stagger, (cfg.num_envs,), 0, env.cfg.sim.max_steps)
    env_state = env_state._replace(step=start.astype(jnp.int32))
    obs = jax.vmap(env._observe)(env_state)

    params = network.init(k_net, obs)

    if cfg.anneal_lr:
        schedule = optax.linear_schedule(cfg.lr, 0.0, cfg.num_updates * cfg.update_epochs
                                         * cfg.num_minibatches)
    else:
        schedule = cfg.lr

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(schedule, eps=1e-5),
    )
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
    return RunnerState(train_state=train_state, env_state=env_state, obs=obs, rng=rng)


def make_chunk(env: CleaningEnv, cfg: PPOConfig):
    """Build the jitted function that runs `cfg.updates_per_chunk` PPO updates."""
    # The Wiewiora offset V = f - REWARD_SCALE*Phi cancels the shaping term out
    # of the TD residual EXACTLY -- but only when the env shapes with the same
    # discount the optimiser uses. `train.py` checks this too; repeated here
    # because tests and scripts call make_chunk directly and bypass that path,
    # and a mismatch is silent: advantages simply stop being drizzle-free.
    if abs(env.discount - cfg.gamma) > 1e-12:
        raise ValueError(
            f"env.discount {env.discount} != cfg.gamma {cfg.gamma}; the shaping "
            f"term would not cancel in the TD residual (issue #4)")

    network = ActorCritic(action_dim=env.action_dim)
    env_reset = jax.vmap(env.reset)
    env_step = jax.vmap(env.step)

    def update(runner: RunnerState, _):
        # ---------------- rollout ----------------
        def env_step_fn(runner: RunnerState, _):
            train_state, env_state, last_obs, rng = runner
            rng, k_act, k_reset = jax.random.split(rng, 3)

            # Wiewiora 2003 (issue #4): V(s) = f(s) - SCALE*Phi(s), so the
            # shaping drizzle cancels analytically in delta (delta = r_gen +
            # gamma*f' - f). The network returns the residual f; the offset
            # uses the cached EnvState.potential (already paid for in env.step).
            # phi_pre MUST be read before env_step shadows env_state -- after
            # is off-by-one (pairs V(s_t) with Phi(s_{t+1})).
            phi_pre = env_state.potential
            mean, log_std, f_pre = network.apply(train_state.params, last_obs)
            off = cfg.potential_baseline_alpha * REWARD_SCALE
            value = f_pre - off * phi_pre
            action = mean + jnp.exp(log_std) * jax.random.normal(k_act, mean.shape)
            logp = log_prob(mean, log_std, action)

            env_state, obs, reward, terminated, truncated, info = env_step(env_state, action)
            # Phi(s') pre-reset: after env_step but before the reset-select
            # below. Using the post-reset potential here would bootstrap the
            # fresh floor at truncation -- silent corruption of Pardo's boundary.
            phi_next_pre_reset = env_state.potential

            # Bootstrap through the time limit. `obs` here is still the real
            # final observation -- the reset has not happened yet. The extra
            # forward is skipped unless some env actually truncated. At 512
            # envs and 4500-step episodes that is ~11% of rollout steps (see
            # the reset note below), and the physics dwarfs the network anyway.
            def with_bootstrap(_):
                _, _, f_final = network.apply(train_state.params, obs)
                final_value = f_final - off * phi_next_pre_reset
                return reward + cfg.gamma * final_value * truncated * (1.0 - terminated)

            def no_bootstrap(_):
                return reward

            reward = jax.lax.cond(
                jnp.any(truncated & ~terminated), with_bootstrap, no_bootstrap, None
            )

            done = jnp.logical_or(terminated, truncated)
            # Resets run every step in the naive version and cost ~10% of a
            # step (FFTs, blur, quantile over the whole batch) while being
            # discarded whenever nothing finished. Compute them only when at
            # least one env is done; the select below is then a no-op copy.
            #
            # Once episodes desynchronise, finishes are Poisson at
            # num_envs/max_steps per step = 512/4500 = 0.114, so the branch
            # fires 1 - exp(-0.114) ~ 11% of the time and is skipped ~89%.
            # (Both branch comments predate the move to 15-minute episodes,
            # which made this MORE valuable, not less.)
            #
            # k_reset is split from rng BEFORE the branch, so the RNG stream
            # advances identically whether or not the reset is taken -- that is
            # what makes this behaviour-preserving rather than merely close.
            # NOTE: this relies on the predicate being a scalar. If make_chunk
            # were ever vmapped, lax.cond degrades to a select that executes
            # both branches and the saving silently vanishes.
            def do_reset(_):
                reset_state, reset_obs = env_reset(jax.random.split(k_reset, cfg.num_envs))
                return (_tree_where(done, reset_state, env_state),
                        _tree_where(done, reset_obs, obs))

            def no_reset(_):
                return env_state, obs

            env_state, obs = jax.lax.cond(jnp.any(done), do_reset, no_reset, None)

            transition = Transition(
                done=done, action=action, value=value, reward=reward,
                log_prob=logp, obs=last_obs, phi=phi_pre,
            )
            metrics = {
                "reward": reward,
                "done": done,
                "terminated": terminated,
                "fraction_removed": info["fraction_removed"],
                "worst_residual": info["worst_residual"],
                "fraction_clean": info["fraction_clean"],
                "standoff": info["standoff"],
                "tilt": info["tilt"],
                "drained_kg": info["drained_kg"],
                # Issue #3: delivered (reached trough, incl. pond) vs drained
                # (left the building). Conflating them overstates progress.
                "delivered_kg": info["delivered_kg"],
                "trough_grit_kg": info["trough_grit_kg"],
                # Phase masses (T3a): cut-and-abandon shows as adhered
                # falling while deposited climbs and drained stays flat --
                # invisible in total reward, so it needs its own curves.
                "adhered_kg": info["adhered_kg"],
                "deposited_kg": info["deposited_kg"],
                "suspended_kg": info["suspended_kg"],
            }
            return RunnerState(train_state, env_state, obs, rng), (transition, metrics)

        runner, (traj, metrics) = jax.lax.scan(
            env_step_fn, runner, None, length=cfg.num_steps
        )

        # ---------------- advantages ----------------
        # GAE seed uses the carried-forward (post-reset-selected) state, which
        # is where the next rollout step acts from -- consistent with runner.obs.
        _, _, f_last = network.apply(runner.train_state.params, runner.obs)
        last_value = (f_last - cfg.potential_baseline_alpha * REWARD_SCALE
                      * runner.env_state.potential)

        def gae_step(carry, t):
            adv, next_value = carry
            not_done = 1.0 - t.done
            delta = t.reward + cfg.gamma * next_value * not_done - t.value
            adv = delta + cfg.gamma * cfg.gae_lambda * not_done * adv
            return (adv, t.value), adv

        _, advantages = jax.lax.scan(
            gae_step, (jnp.zeros_like(last_value), last_value), traj, reverse=True
        )
        targets = advantages + traj.value

        # ---------------- optimise ----------------
        def epoch(carry, _):
            train_state, rng = carry
            rng, k_perm = jax.random.split(rng)

            # traj.phi rides the same flatten/shuffle/minibatch path with the
            # same perm, so mb_phi stays aligned with mb_obs (both pre-step).
            batch = (traj.obs, traj.action, traj.log_prob, traj.value,
                     traj.phi, advantages, targets)
            flat = jax.tree.map(lambda x: x.reshape((cfg.batch_size,) + x.shape[2:]), batch)
            perm = jax.random.permutation(k_perm, cfg.batch_size)
            shuffled = jax.tree.map(lambda x: jnp.take(x, perm, axis=0), flat)
            minibatches = jax.tree.map(
                lambda x: x.reshape((cfg.num_minibatches, -1) + x.shape[1:]), shuffled
            )

            def minibatch_step(train_state, mb):
                mb_obs, mb_action, mb_logp, mb_value, mb_phi, mb_adv, mb_target = mb

                def loss_fn(params):
                    mean, log_std, f_pred = network.apply(params, mb_obs)
                    # Full value in V-space so traj.value/targets/EV are
                    # untouched; the network only ever learns the residual f.
                    value = f_pred - cfg.potential_baseline_alpha * REWARD_SCALE * mb_phi
                    logp = log_prob(mean, log_std, mb_action)

                    ratio = jnp.exp(logp - mb_logp)
                    adv_std = mb_adv.std()
                    adv = (mb_adv - mb_adv.mean()) / (adv_std + 1e-8)
                    pg = -jnp.minimum(
                        ratio * adv,
                        jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv,
                    ).mean()

                    # Plain MSE value loss. SB3 defaults value clipping to
                    # None, and Andrychowicz et al. 2020 find PPO-style value
                    # clipping hurts regardless of threshold -- it binds exactly
                    # when the critic most needs to move (our value targets
                    # grew ~3x with gamma 0.999). Revisit only on evidence.
                    v_loss = 0.5 * ((value - mb_target) ** 2).mean()

                    ent = entropy(log_std).mean()
                    total = pg + cfg.vf_coef * v_loss - cfg.ent_coef * ent

                    approx_kl = ((ratio - 1.0) - jnp.log(ratio)).mean()
                    clip_frac = (jnp.abs(ratio - 1.0) > cfg.clip_eps).mean()
                    return total, (pg, v_loss, ent, approx_kl, clip_frac, adv_std)

                grads, aux = jax.grad(loss_fn, has_aux=True)(train_state.params)
                return train_state.apply_gradients(grads=grads), aux

            train_state, aux = jax.lax.scan(minibatch_step, train_state, minibatches)
            return (train_state, rng), aux

        (train_state, rng), aux = jax.lax.scan(
            epoch, (runner.train_state, runner.rng), None, length=cfg.update_epochs
        )
        runner = runner._replace(train_state=train_state, rng=rng)

        pg, v_loss, ent, approx_kl, clip_frac, adv_std = aux

        # What the network itself has to learn, once the analytic offset is
        # taken out: f = V + SCALE*Phi. `explained_variance` below is now
        # dominated by that offset and reads ~1 by construction, so it can no
        # longer tell a working critic from a drowned one. EV_f is the one that
        # can -- it is EV measured against the residual the network predicts.
        traj_f = traj.value + cfg.potential_baseline_alpha * REWARD_SCALE * traj.phi
        targets_f = targets + cfg.potential_baseline_alpha * REWARD_SCALE * traj.phi
        summary = {
            "reward_mean": metrics["reward"].mean(),
            "episodes": metrics["done"].sum(),
            "finished_clean": metrics["terminated"].sum(),
            "fraction_removed": metrics["fraction_removed"].mean(),
            "worst_residual": metrics["worst_residual"].mean(),
            "fraction_clean": metrics["fraction_clean"].mean(),
            "standoff_mean": metrics["standoff"].mean(),
            "tilt_mean": metrics["tilt"].mean(),
            "drained_kg": metrics["drained_kg"].mean(),
            "delivered_kg": metrics["delivered_kg"].mean(),
            "trough_grit_kg": metrics["trough_grit_kg"].mean(),
            "adhered_kg": metrics["adhered_kg"].mean(),
            "deposited_kg": metrics["deposited_kg"].mean(),
            "suspended_kg": metrics["suspended_kg"].mean(),
            "policy_loss": pg.mean(),
            "value_loss": v_loss.mean(),
            # EV against the FULL value, which is now mostly the analytic
            # offset -REWARD_SCALE*Phi (~+540 on a dirty floor, measured
            # 2026-09-19). Since that term is exact by construction, this reads
            # ~1 from the first update and is NO LONGER a measure of critic
            # learning -- do not read it as one. Kept only for continuity with
            # run 1, where it was the signal that something was wrong.
            # Use `explained_variance_f` and `adv_std_global` instead.
            "explained_variance": (
                1.0 - jnp.var(targets - traj.value)
                / jnp.maximum(jnp.var(targets), 1e-8)
            ),
            # Pre-normalisation advantage std per minibatch (issue #4 stop
            # signal). The loss rescales mb_adv to unit variance, so a
            # collapsing adv_std means the policy gradient keeps full magnitude
            # while becoming directionless -- invisible in reward_mean/EV.
            "adv_std": adv_std.mean(),
            # The exact pre-shuffle figure, not an average of minibatch stds.
            # This is the number to threshold on: it is the raw scale of the
            # learning signal before normalisation rescales it to 1.
            "adv_std_global": jnp.std(advantages),
            # Offset wiring check, cheap and worth having in every run: with a
            # zeroed value head these must satisfy value_mean == -SCALE*phi_mean
            # exactly. A drift between them means the offset has come unstuck
            # from the potential it is supposed to cancel.
            "value_mean": traj.value.mean(),
            "phi_mean": traj.phi.mean(),
            # EV against the residual. THIS is the critic diagnostic now.
            "explained_variance_f": (
                1.0 - jnp.var(targets_f - traj_f) / jnp.maximum(jnp.var(targets_f), 1e-8)
            ),
            "entropy": ent.mean(),
            "approx_kl": approx_kl.mean(),
            "clip_fraction": clip_frac.mean(),
        }
        return runner, summary

    @jax.jit
    def chunk(runner: RunnerState):
        return jax.lax.scan(update, runner, None, length=cfg.updates_per_chunk)

    return chunk


def greedy_action(network_params, env: CleaningEnv, obs: Obs):
    """Deterministic action (distribution mean) for evaluation and rendering."""
    network = ActorCritic(action_dim=env.action_dim)
    mean, _, _ = network.apply(network_params, obs)
    return jnp.clip(mean, -1.0, 1.0)
