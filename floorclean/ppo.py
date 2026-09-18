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

from .env import CleaningEnv, Obs
from .networks import ActorCritic, entropy, log_prob


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    num_envs: int = 512
    num_steps: int = 64  # rollout length per environment per update
    updates_per_chunk: int = 20  # compiled together; checkpoint between chunks
    total_timesteps: int = 200_000_000

    lr: float = 3e-4
    anneal_lr: bool = True
    # 0.2 s per step. 0.999 gives a ~1000-step (~3.5 min) effective horizon,
    # which is what a 15-minute episode needs; 0.997 (~110 s) could not see far
    # enough ahead to value pushing slurry the length of the bay.
    # MUST match CleaningEnv(discount=...) -- the shaping term uses it.
    gamma: float = 0.999
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
    value: jnp.ndarray
    reward: jnp.ndarray  # already bootstrap-adjusted for truncation
    log_prob: jnp.ndarray
    obs: Obs


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
    rng, k_net, k_reset = jax.random.split(rng, 3)

    network = ActorCritic(action_dim=env.action_dim)
    env_state, obs = jax.vmap(env.reset)(jax.random.split(k_reset, cfg.num_envs))
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
    network = ActorCritic(action_dim=env.action_dim)
    env_reset = jax.vmap(env.reset)
    env_step = jax.vmap(env.step)

    def update(runner: RunnerState, _):
        # ---------------- rollout ----------------
        def env_step_fn(runner: RunnerState, _):
            train_state, env_state, last_obs, rng = runner
            rng, k_act, k_reset = jax.random.split(rng, 3)

            mean, log_std, value = network.apply(train_state.params, last_obs)
            action = mean + jnp.exp(log_std) * jax.random.normal(k_act, mean.shape)
            logp = log_prob(mean, log_std, action)

            env_state, obs, reward, terminated, truncated, info = env_step(env_state, action)

            # Bootstrap through the time limit. `obs` here is still the real
            # final observation -- the reset has not happened yet.
            _, _, final_value = network.apply(train_state.params, obs)
            reward = reward + cfg.gamma * final_value * truncated * (1.0 - terminated)

            done = jnp.logical_or(terminated, truncated)
            reset_state, reset_obs = env_reset(jax.random.split(k_reset, cfg.num_envs))
            env_state = _tree_where(done, reset_state, env_state)
            obs = _tree_where(done, reset_obs, obs)

            transition = Transition(
                done=done, action=action, value=value, reward=reward,
                log_prob=logp, obs=last_obs,
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
        _, _, last_value = network.apply(runner.train_state.params, runner.obs)

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

            batch = (traj.obs, traj.action, traj.log_prob, traj.value, advantages, targets)
            flat = jax.tree.map(lambda x: x.reshape((cfg.batch_size,) + x.shape[2:]), batch)
            perm = jax.random.permutation(k_perm, cfg.batch_size)
            shuffled = jax.tree.map(lambda x: jnp.take(x, perm, axis=0), flat)
            minibatches = jax.tree.map(
                lambda x: x.reshape((cfg.num_minibatches, -1) + x.shape[1:]), shuffled
            )

            def minibatch_step(train_state, mb):
                mb_obs, mb_action, mb_logp, mb_value, mb_adv, mb_target = mb

                def loss_fn(params):
                    mean, log_std, value = network.apply(params, mb_obs)
                    logp = log_prob(mean, log_std, mb_action)

                    ratio = jnp.exp(logp - mb_logp)
                    adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg = -jnp.minimum(
                        ratio * adv,
                        jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv,
                    ).mean()

                    # Clipped value loss, as in the original PPO implementation.
                    v_clipped = mb_value + jnp.clip(
                        value - mb_value, -cfg.clip_eps, cfg.clip_eps
                    )
                    v_loss = 0.5 * jnp.maximum(
                        (value - mb_target) ** 2, (v_clipped - mb_target) ** 2
                    ).mean()

                    ent = entropy(log_std).mean()
                    total = pg + cfg.vf_coef * v_loss - cfg.ent_coef * ent

                    approx_kl = ((ratio - 1.0) - jnp.log(ratio)).mean()
                    clip_frac = (jnp.abs(ratio - 1.0) > cfg.clip_eps).mean()
                    return total, (pg, v_loss, ent, approx_kl, clip_frac)

                grads, aux = jax.grad(loss_fn, has_aux=True)(train_state.params)
                return train_state.apply_gradients(grads=grads), aux

            train_state, aux = jax.lax.scan(minibatch_step, train_state, minibatches)
            return (train_state, rng), aux

        (train_state, rng), aux = jax.lax.scan(
            epoch, (runner.train_state, runner.rng), None, length=cfg.update_epochs
        )
        runner = runner._replace(train_state=train_state, rng=rng)

        pg, v_loss, ent, approx_kl, clip_frac = aux
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
            "adhered_kg": metrics["adhered_kg"].mean(),
            "deposited_kg": metrics["deposited_kg"].mean(),
            "suspended_kg": metrics["suspended_kg"].mean(),
            "policy_loss": pg.mean(),
            "value_loss": v_loss.mean(),
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
