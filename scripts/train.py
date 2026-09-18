#!/usr/bin/env python3
"""PPO training driver: the outer loop only.

`floorclean/ppo.py` already compiles rollout + GAE + optimisation into one
`lax.scan` (`make_chunk`); this script calls it in a loop, logs metrics to CSV,
and checkpoints with orbax so a RunPod pod can be resumed or restarted without
babysitting. No wandb -- everything here runs offline.

    .venv/bin/python scripts/train.py --run-name base
    .venv/bin/python scripts/train.py --run-name base --resume

Each row of the CSV is one PPO update (num_envs * num_steps transitions).
A chunk is `updates_per_chunk` updates compiled together -- the unit of
progress between checkpoint, log flush, and wall-clock prints.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import tyro

from floorclean.config import Config
from floorclean.env import CleaningEnv
from floorclean.ppo import PPOConfig, RunnerState, init_runner, make_chunk


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)
    run_name: str = "base"
    checkpoint_dir: str = "checkpoints"
    csv_path: str = "training_logs"
    checkpoint_every_chunks: int = 1
    resume: bool = False

    # Fixed for now: the observation config is part of the environment and the
    # network shape follows from it. Change it and old checkpoints are void.
    seed: int = 0


CSV_COLUMNS = [
    "update", "timesteps", "reward_mean", "episodes", "finished_clean",
    "fraction_removed", "worst_residual", "fraction_clean",
    "standoff_mean", "tilt_mean", "policy_loss", "value_loss",
    "entropy", "approx_kl", "clip_fraction", "seconds",
]


def payload_template(runner: RunnerState, update: jnp.ndarray) -> dict:
    """The pytree we checkpoint. TrainState is split into plain arrays so the
    file never depends on flax struct internals."""
    ts = runner.train_state
    return {
        "update": update,
        "params": ts.params,
        "opt_state": ts.opt_state,
        "step": ts.step,
        "env_state": runner.env_state,
        "obs": runner.obs,
        "rng": runner.rng,
    }


def main():
    tcfg = tyro.cli(TrainConfig)
    ppo = tcfg.ppo

    ckpt_dir = pathlib.Path(tcfg.checkpoint_dir) / tcfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = pathlib.Path(tcfg.csv_path)
    csv_path.mkdir(parents=True, exist_ok=True)
    csv_path = csv_path / f"{tcfg.run_name}.csv"

    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(tcfg), f, indent=2)

    env = CleaningEnv(Config())
    rng = jax.random.PRNGKey(tcfg.seed)
    runner = init_runner(env, ppo, rng)
    chunk = make_chunk(env, ppo)

    ckptr = ocp.PyTreeCheckpointer()
    ckpt_path = str(ckpt_dir / "state")

    start_update = 0
    if tcfg.resume:
        template = payload_template(runner, jnp.array(0))
        restored = ckptr.restore(ckpt_path, ocp.args.PyTreeRestore(item=template))
        restored = jax.tree.map(jnp.asarray, restored)
        runner = runner._replace(
            train_state=runner.train_state.replace(
                params=restored["params"],
                opt_state=restored["opt_state"],
                step=restored["step"],
            ),
            env_state=restored["env_state"],
            obs=restored["obs"],
            rng=restored["rng"],
        )
        start_update = int(restored["update"])
        print(f"resumed from update {start_update}")

    fresh = not csv_path.exists() or start_update == 0
    csv_f = open(csv_path, "a" if not fresh else "w")
    if fresh:
        csv_f.write(",".join(CSV_COLUMNS) + "\n")

    def save(runner, update):
        ckptr.save(ckpt_path, ocp.args.PyTreeSave(
            item=payload_template(runner, jnp.asarray(update))
        ))

    updates_total = ppo.num_updates
    print(f"{ppo.num_envs} envs x {ppo.num_steps} steps = {ppo.batch_size} per update, "
          f"{ppo.updates_per_chunk} updates per chunk, "
          f"{updates_total} updates total "
          f"({ppo.total_timesteps / 1e6:.0f}M timesteps)")

    t_start = time.time()
    update = start_update
    try:
        while update < updates_total:
            t0 = time.time()
            runner, summary = chunk(runner)
            wall = time.time() - t0

            n = ppo.updates_per_chunk
            for i in range(n):
                row = [update + 1 + i, (update + 1 + i) * ppo.batch_size]
                row += [float(jnp.asarray(summary[k][i])) for k in CSV_COLUMNS[2:-1]]
                row.append(wall / n)
                csv_f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v)
                                     for v in row) + "\n")
            csv_f.flush()
            update += n

            last = -1
            print(
                f"upd {update:>6}/{updates_total}  "
                f"r {float(summary['reward_mean'][last]):9.2f}  "
                f"removed {float(summary['fraction_removed'][last]):5.3f}  "
                f"clean {float(summary['fraction_clean'][last]):5.3f}  "
                f"H {float(summary['entropy'][last]):5.3f}  "
                f"kl {float(summary['approx_kl'][last]):6.4f}  "
                f"eps {wall / n / ppo.batch_size * 1e6:6.0f} us/step  "
                f"{(update - start_update) * ppo.batch_size / (time.time() - t_start):8.0f} steps/s"
            )

            if (update // ppo.updates_per_chunk) % tcfg.checkpoint_every_chunks == 0:
                save(runner, update)
    except KeyboardInterrupt:
        print("interrupted -- saving")
    finally:
        save(runner, update)
        csv_f.close()
        print(f"checkpoint at update {update} -> {ckpt_path}")


if __name__ == "__main__":
    main()
