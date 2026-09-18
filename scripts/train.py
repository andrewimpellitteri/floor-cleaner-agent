#!/usr/bin/env python3
"""PPO training driver: the outer loop only.

`floorclean/ppo.py` already compiles rollout + GAE + optimisation into one
`lax.scan` (`make_chunk`); this script calls it in a loop, logs metrics to CSV
(plus Weights & Biases when configured -- see below), and checkpoints with
orbax so a RunPod pod can be resumed or restarted without babysitting.

W&B is strictly optional: logging activates only if `wandb` is installed and
`WANDB_API_KEY` is set (or `WANDB_MODE=offline`), otherwise every call no-ops
and the run is identical to a run without it. The CSV stays primary.

    .venv/bin/python scripts/train.py --run-name base
    .venv/bin/python scripts/train.py --run-name base --resume

Each row of the CSV is one PPO update (num_envs * num_steps transitions).
A chunk is `updates_per_chunk` updates compiled together -- the unit of
progress between checkpoint, log flush, and wall-clock prints.
"""

from __future__ import annotations

import dataclasses
import json
import os
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
from floorclean.rollout import NeuralPolicy, run_episode
from floorclean.render import render_episode, save_still
from floorclean.wandblog import WandbLog, git_tags


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)
    run_name: str = "base"
    checkpoint_dir: str = "checkpoints"
    csv_path: str = "training_logs"
    checkpoint_every_chunks: int = 1
    resume: bool = False

    # W&B visibility. Strictly opt-in by environment: active only when
    # `wandb` is installed and WANDB_API_KEY is set (or WANDB_MODE=offline).
    # `--no-wandb` forces it off even then. CSV logging is unaffected either
    # way and stays the primary record.
    wandb_project: str = "floorclean"
    wandb_every_chunks: int = 5  # greedy-eval stills cadence (0 = no renders)
    eval_seconds: float = 150.0  # simulated seconds per eval rollout
    no_wandb: bool = False

    # Fixed for now: the observation config is part of the environment and the
    # network shape follows from it. Change it and old checkpoints are void.
    seed: int = 0


CSV_COLUMNS = [
    "update", "timesteps", "reward_mean", "episodes", "finished_clean",
    "fraction_removed", "worst_residual", "fraction_clean",
    "standoff_mean", "tilt_mean", "policy_loss", "value_loss",
    "entropy", "approx_kl", "clip_fraction",
    # T3a — the cut-and-abandon panel: adhered falling while deposited climbs
    # and drained stays flat is the characteristic failure, invisible in
    # reward. Logged everywhere (CSV + W&B) so pinning them to one dashboard
    # panel is one click.
    "drained_kg", "adhered_kg", "deposited_kg", "suspended_kg",
    "seconds",
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

    ckpt_dir = pathlib.Path(tcfg.checkpoint_dir).expanduser().resolve() / tcfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = pathlib.Path(tcfg.csv_path).expanduser().resolve()
    csv_path.mkdir(parents=True, exist_ok=True)
    csv_path = csv_path / f"{tcfg.run_name}.csv"

    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(tcfg), f, indent=2)

    # W&B first, before any JAX-heavy work: it spawns its service process and
    # must not fork after XLA thread pools are up. On resume, the stored run
    # id continues the same curve instead of starting a second run.
    wandb_id_file = ckpt_dir / "wandb_id"
    resume_id = None
    if tcfg.resume and wandb_id_file.is_file():
        resume_id = wandb_id_file.read_text().strip() or None
    wlog = WandbLog.from_env(
        project=None if tcfg.no_wandb else tcfg.wandb_project,
        run_name=tcfg.run_name,
        config={"train": dataclasses.asdict(tcfg)},
        tags=["stage:ppo-train", f"seed:{tcfg.seed}"] + git_tags(),
        resume_id=resume_id,
    )
    if wlog.enabled and wlog.run_id and not tcfg.resume:
        wandb_id_file.write_text(wlog.run_id)

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
        # force=True: one rolling checkpoint per run, overwritten each chunk.
        ckptr.save(ckpt_path, ocp.args.PyTreeSave(
            item=payload_template(runner, jnp.asarray(update))
        ), force=True)

    updates_total = ppo.num_updates
    print(f"{ppo.num_envs} envs x {ppo.num_steps} steps = {ppo.batch_size} per update, "
          f"{ppo.updates_per_chunk} updates per chunk, "
          f"{updates_total} updates total "
          f"({ppo.total_timesteps / 1e6:.0f}M timesteps)")

    t_start = time.time()
    update = start_update
    media_dir = ckpt_dir / "media"
    last_states = None
    eval_idx = 0
    try:
        while update < updates_total:
            t0 = time.time()
            runner, summary = chunk(runner)
            wall = time.time() - t0

            n = ppo.updates_per_chunk
            for i in range(n):
                upd = update + 1 + i
                row = {"update": upd,
                       "timesteps": upd * ppo.batch_size}
                row.update({k: float(jnp.asarray(summary[k][i]))
                            for k in CSV_COLUMNS[2:-1]})
                row["seconds"] = wall / n
                csv_f.write(",".join(
                    f"{row[k]:.6g}" if isinstance(row[k], float) else str(row[k])
                    for k in CSV_COLUMNS) + "\n")
                wlog.log_update({k: row[k] for k in CSV_COLUMNS[1:]}, step=upd)
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

            # Greedy-eval stills on a fixed floor, so the images show learning
            # rather than floor lottery. Skipped entirely when W&B is off, and
            # never allowed to kill a paid-for run: any failure here degrades
            # to curves-only for this eval.
            if wlog.enabled and tcfg.wandb_every_chunks > 0 and (
                    update // ppo.updates_per_chunk) % tcfg.wandb_every_chunks == 0:
                try:
                    media_dir.mkdir(parents=True, exist_ok=True)
                    policy = NeuralPolicy(runner.train_state.params, env.action_dim)
                    res = run_episode(env, policy,
                                      jax.random.PRNGKey(10_000 + eval_idx),
                                      max_seconds=tcfg.eval_seconds,
                                      record_every=25)
                    last_states = res.states
                    n_states = len(res.states)
                    for name, frac in (("start", 0.0), ("mid", 0.5), ("end", 0.999)):
                        still = (media_dir /
                                 f"eval{eval_idx:04d}_{name}.png")
                        save_still(env, res.states[min(n_states - 1,
                                                       int(frac * n_states))],
                                   str(still))
                        wlog.log_image(str(still),
                                       caption=f"eval {eval_idx} {name}: "
                                               f"{res.seconds_to_clean:.0f}s to clean, "
                                               f"{res.final_remaining_kg:.2f}kg left",
                                       step=update)
                    eval_idx += 1
                except Exception as e:
                    print(f"[!] eval renders failed ({type(e).__name__}: {e}); "
                          f"continuing curves-only")
    except KeyboardInterrupt:
        print("interrupted -- saving")
    finally:
        save(runner, update)
        if wlog.enabled and last_states is not None:
            mp4 = media_dir / "sweep_final.mp4"
            try:
                render_episode(env, last_states, str(mp4), fps=12)
                wlog.log_video(str(mp4))
            except Exception as e:
                print(f"[!] final mp4 failed ({e}); stills kept in {media_dir}")
        bucket = os.environ.get("S3_BUCKET")
        if wlog.enabled and bucket:
            prefix = os.environ.get("S3_PREFIX", "floorclean")
            wlog.log_s3_reference(
                f"s3://{bucket}/{prefix}/runs/{tcfg.run_name}/checkpoints/",
                name=f"{tcfg.run_name}-checkpoints")
        wlog.finish()
        csv_f.close()
        print(f"checkpoint at update {update} -> {ckpt_path}")


if __name__ == "__main__":
    main()
