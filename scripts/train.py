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
import math
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
from floorclean.macro import MacroEnv
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
    # Simulated seconds per eval rollout. This was 150 s (2.5 min), far too
    # short to say anything about a job that takes 30+ minutes -- no policy can
    # finish in 2.5 min, so `eval/finished` was structurally 0 and
    # `eval/seconds_to_clean` never became finite.
    #
    # It was then 1800 s, to match the 30-minute scripted benchmark. That was
    # wrong in the other direction: an episode truncates at
    # sim.episode_seconds = 900 s, and the observation carries
    # step/max_steps (env.py:436), so minutes 15-30 fed the policy a feature
    # value of up to 2.0 that training never produced. The scripted baselines
    # read no observation, so the 30-minute comparison was biased against the
    # learned policy specifically. 0 means "one full training episode", which is
    # the only horizon where both sides are in distribution; `run_episode` now
    # also freezes at truncation so a larger number cannot silently reintroduce
    # the overrun. The 30-minute scripted references must be re-measured at this
    # horizon to stay comparable -- see results/.
    eval_seconds: float = 0.0

    # Pinned eval floors. This was PRNGKey(10_000 + eval_idx), a DIFFERENT floor
    # at every eval, while the comment at the call site claimed a fixed floor.
    # BEST_UPDATE was therefore ranking updates partly by floor lottery, and the
    # lottery is not small: far_to_near's own seed-to-seed spread is +-0.070
    # clean fraction (results/shaping_gradient_ablation.txt), wider than most of
    # the differences being selected between. Four floors, fixed for the life of
    # the run, scored on the mean.
    eval_floors: int = 4

    # Train over STROKES rather than 0.2 s wrist commands. See
    # results/action_leverage.txt: the low-level action space gives each
    # decision 0.19% of the between-floor variation, which is not learnable at
    # this batch size; the macro space gives 3.24%.
    macro: bool = False
    no_wandb: bool = False

    # Snapshot retention for model selection (S3-cost bounded by design):
    # every Nth chunk is kept under checkpoints/<run>/snapshots/, and at the
    # end only best-eval + last survive (see BEST_UPDATE) -- the rolling
    # `state` checkpoint alone cannot answer "which update was best".
    # 0 disables snapshots; the rolling checkpoint is unaffected.
    snapshot_every_chunks: int = 10

    # Fixed for now: the observation config is part of the environment and the
    # network shape follows from it. Change it and old checkpoints are void.
    seed: int = 0


CSV_COLUMNS = [
    "update", "timesteps", "reward_mean", "episodes", "finished_clean",
    "fraction_removed", "worst_residual", "fraction_clean",
    "standoff_mean", "tilt_mean", "policy_loss", "value_loss",
    "entropy", "approx_kl", "clip_fraction", "explained_variance",
    # Issue #4 stop signal: pre-norm advantage std. Collapses when the critic
    # memorises the drizzle and the normalised advantage becomes noise.
    # `adv_std_global` is the exact figure; `adv_std` averages minibatch stds.
    # `explained_variance_f` is EV against the residual the network actually
    # learns -- plain `explained_variance` now reads ~1 by construction and
    # says nothing. `value_mean`/`phi_mean` are the offset wiring check.
    "adv_std", "adv_std_global", "explained_variance_f",
    "value_mean", "phi_mean",
    # T3a — the cut-and-abandon panel: adhered falling while deposited climbs
    # and drained stays flat is the characteristic failure, invisible in
    # reward. Logged everywhere (CSV + W&B) so pinning them to one dashboard
    # panel is one click.
    # Issue #3: delivered (reached trough, incl. pond) vs drained (left).
    "drained_kg", "delivered_kg", "trough_grit_kg",
    "adhered_kg", "deposited_kg", "suspended_kg",
    "seconds",
]


def payload_template(runner: RunnerState, update: jnp.ndarray) -> dict:
    """The pytree we checkpoint. TrainState is split into plain arrays so the
    file never depends on flax struct internals."""
    ts = runner.train_state
    return {
        "update": update,
        # What the value head MEANS. Since issue #4 the network predicts the
        # residual f in V(s) = f(s) - REWARD_SCALE*Phi(s), not the full value.
        # A pre-#4 checkpoint holds a head trained to output V directly (~540
        # on a dirty floor); restoring it here would silently give
        # V = 540 - SCALE*Phi ~ 1080 and quietly poison the run. Orbax matches
        # the tree structure on restore, so an old checkpoint without this key
        # fails loudly -- which is the point. Bump it if the meaning changes
        # again.
        "value_head_semantics": jnp.array(2, dtype=jnp.int32),
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
    if tcfg.macro:
        # One decision per STROKE instead of per 0.2 s control step. Measured
        # justification in results/action_leverage.txt: a low-level action is
        # worth 0.19% of the between-floor variation, a macro decision 3.24%
        # -- 17x, which is the difference between an effective batch SNR of
        # 0.36 and 5.8.
        env = MacroEnv(env)
        # A macro-step spans K control steps, so the trainer must discount by
        # gamma**K. Overriding rather than asking the user to pass it: the
        # correct value depends on MACRO_STEPS and getting it wrong silently
        # rescales every value estimate, which the guard below would not catch
        # if the user simply passed a self-consistent wrong pair.
        ppo = dataclasses.replace(ppo, gamma=env.discount)
        print(f"[macro] one decision = {env.macro_steps} control steps "
              f"({env.macro_steps * env.cfg.sim.control_dt:.1f}s); "
              f"{env.cfg.sim.max_steps // env.macro_steps} decisions/episode; "
              f"gamma {env.discount:.6f} (= {CleaningEnv().discount}^{env.macro_steps})")

    # The shaping term F = gamma*Phi(s') - Phi(s) is policy-invariant only for
    # the gamma the optimiser discounts with. The unit test pins the defaults;
    # this pins the actual run, since tyro overrides can split them silently.
    if abs(env.discount - ppo.gamma) > 1e-12:
        sys.exit(f"env discount {env.discount} != ppo gamma {ppo.gamma}: "
                 f"shaping would bias every step by (1-gamma)*Phi. "
                 f"Pass matching values.")
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
        semantics = int(restored.get("value_head_semantics", 1))
        if semantics != 2:
            sys.exit(
                f"checkpoint value_head_semantics={semantics}, expected 2. This "
                f"checkpoint predates the Wiewiora offset (issue #4): its value "
                f"head outputs the FULL value, but the trainer now treats that "
                f"output as the residual f in V = f - REWARD_SCALE*Phi. "
                f"Resuming would double-count the offset. Start a fresh run.")
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

    def save_snapshot(runner, update):
        snap = str(ckpt_dir / "snapshots" / f"update_{update:06d}")
        ckptr.save(snap, ocp.args.PyTreeSave(
            item=payload_template(runner, jnp.asarray(update))
        ), force=True)

    def eval_key(results):
        """Model-selection score over the pinned eval floors.

        Ranks on the MEAN across floors, not on one floor: with a single floor
        this was comparing a lottery draw (see TrainConfig.eval_floors). Runs
        that clean every floor sort ahead of runs that clean some, which sort
        ahead of runs that clean none; within a tier, by mean time-to-clean or
        mean grit left. Tuple comparison does the right thing across all three.
        """
        times = [r.seconds_to_clean for r in results]
        n_finished = sum(1 for t in times if math.isfinite(t))
        mean_remaining = sum(float(r.final_remaining_kg) for r in results) / len(results)
        if n_finished == len(results):
            return (0, sum(times) / len(times))
        if n_finished:
            # More floors finished is better, hence the negation.
            return (1, -n_finished, mean_remaining)
        return (2, mean_remaining)

    updates_total = ppo.num_updates
    print(f"{ppo.num_envs} envs x {ppo.num_steps} steps = {ppo.batch_size} per update, "
          f"{ppo.updates_per_chunk} updates per chunk, "
          f"{updates_total} updates total "
          f"({ppo.total_timesteps / 1e6:.0f}M timesteps)")

    t_start = time.time()
    update = start_update
    media_dir = ckpt_dir / "media"
    best_update_file = ckpt_dir / "BEST_UPDATE"
    last_states = None
    eval_idx = 0
    # Best-eval tracking survives resume via the file (update + score tuple),
    # so a restarted run never "forgets" an early winner. The leading "v2" is
    # load-bearing: eval_key used to return a 2-tuple scored on ONE floor and
    # now returns a 2- or 3-tuple scored on the mean of several, so a v1 file
    # would be compared against incommensurable numbers. An unrecognised or
    # missing tag means "no previous best", which costs one eval, not a wrong
    # model choice.
    best_key = None
    best_update = None
    if tcfg.resume and best_update_file.is_file():
        try:
            parts = best_update_file.read_text().split()
            if parts[0] != "v2":
                raise ValueError(f"unrecognised BEST_UPDATE format: {parts[0]!r}")
            best_update = int(parts[1])
            best_key = (int(parts[2]), *(float(x) for x in parts[3:]))
        except (ValueError, IndexError):
            best_key, best_update = None, None
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
                f"drained {float(summary['drained_kg'][last]):6.3f}  "
                f"adv_std {float(summary['adv_std'][last]):7.4f}  "
                f"H {float(summary['entropy'][last]):5.3f}  "
                f"kl {float(summary['approx_kl'][last]):6.4f}  "
                f"eps {wall / n / ppo.batch_size * 1e6:6.0f} us/step  "
                f"{(update - start_update) * ppo.batch_size / (time.time() - t_start):8.0f} steps/s"
            )

            if (update // ppo.updates_per_chunk) % tcfg.checkpoint_every_chunks == 0:
                save(runner, update)

            # Retained snapshots for model selection: best-eval + last survive
            # the end-of-run prune (see BEST_UPDATE + train.sh), the rest are
            # transient S3 traffic, not storage.
            if tcfg.snapshot_every_chunks > 0 and (
                    update // ppo.updates_per_chunk) % tcfg.snapshot_every_chunks == 0:
                save_snapshot(runner, update)

            # Greedy-eval stills on a fixed floor, so the images show learning
            # rather than floor lottery. Skipped entirely when W&B is off, and
            # never allowed to kill a paid-for run: any failure here degrades
            # to curves-only for this eval.
            if wlog.enabled and tcfg.wandb_every_chunks > 0 and (
                    update // ppo.updates_per_chunk) % tcfg.wandb_every_chunks == 0:
                try:
                    media_dir.mkdir(parents=True, exist_ok=True)
                    policy = NeuralPolicy(runner.train_state.params, env.action_dim)
                    # fresh=True is REQUIRED here. Without it run_episode
                    # starts at a random point through the job (matching the
                    # training reset distribution), so every eval scalar below
                    # is measured on a floor that was already part-cleaned for
                    # free. That is what made training report fraction_clean
                    # ~0.42 for a policy that scores 0.031 from a fresh floor:
                    # the metric was reporting the reset lottery, not the
                    # policy. The scripted baselines are all benchmarked fresh,
                    # so this is also what makes the numbers comparable.
                    # 0 means "one full training episode" -- the only horizon
                    # where the learned policy and the scripted baselines are
                    # both in distribution. See TrainConfig.eval_seconds.
                    eval_seconds = (tcfg.eval_seconds
                                    or env.cfg.sim.episode_seconds)
                    results = [
                        run_episode(env, policy,
                                    jax.random.PRNGKey(10_000 + k),
                                    max_seconds=eval_seconds,
                                    record_every=250,
                                    fresh=True)
                        for k in range(tcfg.eval_floors)
                    ]
                    # Stills always come from floor 0, so the image strip shows
                    # the policy changing rather than the floor changing.
                    res = results[0]
                    last_states = res.states
                    # Scalar eval metrics: the model-selection instrument.
                    # Averaged over the pinned floors, because a single floor's
                    # score is dominated by which floor it is. The spread is
                    # logged too: if it stays comparable to the between-update
                    # differences, model selection is still mostly noise and the
                    # floor count needs raising.
                    remaining_kg = [float(r.final_remaining_kg) for r in results]
                    finished = [math.isfinite(r.seconds_to_clean) for r in results]
                    eval_metrics = {
                        "eval/finished": sum(finished) / len(finished),
                        "eval/remaining_kg": sum(remaining_kg) / len(remaining_kg),
                        "eval/remaining_kg_spread": max(remaining_kg) - min(remaining_kg),
                    }
                    # seconds_to_clean is inf when unfinished. Average over the
                    # floors that did finish, and only log it when some did, so
                    # the curve stays plottable.
                    times = [r.seconds_to_clean for r in results
                             if math.isfinite(r.seconds_to_clean)]
                    if times:
                        eval_metrics["eval/seconds_to_clean"] = sum(times) / len(times)
                    wlog.log_update(eval_metrics, step=update)
                    key = eval_key(results)
                    if best_key is None or key < best_key:
                        best_key, best_update = key, update
                        best_update_file.write_text(
                            "v2 " + " ".join([str(best_update), str(key[0])]
                                             + [f"{x:.6g}" for x in key[1:]]))
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
