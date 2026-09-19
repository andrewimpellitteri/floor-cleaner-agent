# Wash-Bay Floor Cleaning — physics simulation + RL optimisation

Which pressure-washing technique cleans the shop floor fastest, settled with
physics instead of opinion. See `ORIENTATION.md` first (domain facts, physics,
invariants), `WORKBOARD.md` for the task list, and `ANSWER.md` for the current
best answer to the practical question.

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
