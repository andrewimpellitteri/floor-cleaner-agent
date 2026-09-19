# Wash-Bay Floor Cleaning — physics simulation + RL optimisation

Which pressure-washing technique cleans the shop floor fastest, settled with
physics instead of opinion. See `ORIENTATION.md` first (domain facts, physics,
invariants), `WORKBOARD.md` for the task list, and `ANSWER.md` for the current
best answer to the practical question.

`math_analysis.pdf` derives the closed-form scaling laws behind those answers
(impingement pressure `P ~ cos^4(theta) d^-2`, the soft cutting threshold, film
depth `h ~ Q^(3/5)`, and the cutting surplus `Da ~ 42` that makes the job
coverage-limited). Its constants check out against `floorclean/config.py`; note
it assumes a 0.3 m/s walking pace, which `results/walking_speed.md` has since
measured at 0.24. Build it with `pdflatex math_analysis.tex`.

## Stack

End-to-end JAX: the `floorclean/` environment, rollouts, GAE and PPO updates
all compile into one GPU program. (An earlier Stable-Baselines3 + numba
prototype lives on as `cleaning_room.py` / `main.py` / `fluid_sim.py` /
`feature_extraction.py` — superseded, pending removal per WORKBOARD T9.)

## Setup

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest tests/ -q      # physics must pass
.venv/bin/python scripts/calibrate.py     # read this before believing anything
```

GPU training: `uv pip install -e ".[cuda]"`.

`uv.lock` is committed, so `uv sync` reproduces the exact resolution this was
developed and trained against. The pod resolves from the same lock.

**System dependencies**, which pip cannot install for you:

| tool | needed by | why |
|---|---|---|
| `ffmpeg` | `floorclean/render.py`, `scripts/media/` | writes the MP4 renders and extracts video frames |
| `awscli` | `scripts/jobs/train.sh` | mirrors checkpoints and the CSV to S3 from the pod |

**Credentials** are only needed to launch a RunPod run; everything local runs
without them. Copy `.env.example` to `.env` and fill it in — `.env` is
gitignored. `scripts/runpod_launch.py` reads the first of `--env-file`, `./.env`,
`~/.config/floorclean/.env`, and anything already exported wins over all three.

## Usage

```bash
.venv/bin/python scripts/calibrate.py                          # physical behaviour in checkable numbers
.venv/bin/python scripts/run_to_completion.py --strategy far_to_near --seed 0   # T1: time a baseline to completion
.venv/bin/python scripts/train.py --run-name base              # PPO training driver (CSV + checkpoints)
.venv/bin/python scripts/train.py --run-name base --resume     # resume from checkpoint
.venv/bin/python scripts/experiments.py --suite matrix --seeds 8 --out experiments.jsonl  # batched baseline comparison
```

Episodes are full 15-minute jobs (`sim.max_steps = 4500` at
`control_dt = 0.2 s`); the benchmark runs strategies to completion past the
training horizon. Do not rank policies by total return — compare fraction of
cells clean, time to finish, and grit delivered to the trough.

## License

This project is licensed under the MIT License.
