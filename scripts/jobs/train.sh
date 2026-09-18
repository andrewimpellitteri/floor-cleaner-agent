#!/usr/bin/env bash
# Pod-side training job for the floor_clean wash-bay project.
#
# Runs inside the RunPod container after BOOTSTRAP (see scripts/runpod_launch.py)
# has cloned the repo to $BOOTSTRAP_REPO and checked out $GIT_REF. It installs
# a CUDA JAX runtime, runs scripts/train.py with S3 log mirroring, uploads
# checkpoints + CSV, and terminates the pod -- success or failure.
#
# Environment (all forwarded by the launcher; see --dry-run):
#   RUN_NAME       run name -> scripts/train.py --run-name (default: base)
#   TRAIN_ARGS     extra args appended to scripts/train.py (tyro overrides)
#   JOB_TIMEOUT    seconds for the whole job; empty/0 = none (default: none).
#                  A pod-deadline kill is a hard kill, so results are synced to
#                  S3 in the background every 10 min (SYNC_SECS below), not
#                  just at the end: a kill loses at most one interval, not
#                  the run.
#   RESUME_FROM_S3 set to 1 with TRAIN_ARGS="--resume" to continue a previous
#                  pod's run: pulls its checkpoints + CSV down before starting.
#   S3_BUCKET      results destination (required)
#   S3_PREFIX      key prefix (default: floorclean)
#   RUNPOD_API_KEY + RUNPOD_POD_ID  self-termination (warn-only if missing:
#                  the account-side terminateAfter deadline still applies)
#
# runpod: min-vram-gb=20
# runpod: job-timeout-env=JOB_TIMEOUT:0
set -u
REPO="${BOOTSTRAP_REPO:-/tmp/repo}"
RUN_NAME="${RUN_NAME:-base}"
S3_PREFIX="${S3_PREFIX:-floorclean}"
LOG="/tmp/train.log"
LIVE_KEY="s3://${S3_BUCKET:-}/$S3_PREFIX/runs/$RUN_NAME/train.live.log"

fail() { echo "[train.sh] FATAL: $*" | tee -a "$LOG" >&2; final_upload; self_terminate; exit 1; }

s3() { aws s3 "$@" >>"$LOG" 2>&1; }

SYNC_SECS=600
sync_now() {
  # Periodic mirror: checkpoints (incl. media/ stills), CSV, and the log.
  # Runs in the background during the job and once at the end, so a hard kill
  # at the pod deadline loses at most one interval.
  [ -n "${S3_BUCKET:-}" ] || return 0
  dest="s3://$S3_BUCKET/$S3_PREFIX/runs/$RUN_NAME"
  s3 sync "$REPO/checkpoints/$RUN_NAME" "$dest/checkpoints/" --quiet || true
  [ -f "$REPO/training_logs/$RUN_NAME.csv" ] \
    && s3 cp "$REPO/training_logs/$RUN_NAME.csv" "$dest/training.csv" --quiet || true
  s3 cp "$LOG" "$LIVE_KEY" --quiet || true
}

final_upload() {
  [ -n "${S3_BUCKET:-}" ] || return 0
  dest="s3://$S3_BUCKET/$S3_PREFIX/runs/$RUN_NAME"
  echo "[train.sh] uploading results to $dest" | tee -a "$LOG"
  s3 cp "$LOG" "$dest/train.log" || true
  [ -d "$REPO/checkpoints/$RUN_NAME" ] && s3 sync "$REPO/checkpoints/$RUN_NAME" "$dest/checkpoints/" || true
  [ -f "$REPO/training_logs/$RUN_NAME.csv" ] && s3 cp "$REPO/training_logs/$RUN_NAME.csv" "$dest/training.csv" || true
}

self_terminate() {
  [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ] || {
    echo "[train.sh] no RUNPOD_API_KEY/POD_ID; pod ends at its terminateAfter deadline"
    return 0
  }
  pod_mutation=$(printf 'mutation{podTerminate(input:{podId:"%s}")}' "$RUNPOD_POD_ID")
  for _ in 1 2 3; do
    if curl -fsS --max-time 15 -A "floorclean-trainsh/1.0" \
        -X POST https://api.runpod.io/graphql \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"$pod_mutation\"}" >>"$LOG" 2>&1; then
      echo "[train.sh] self-terminated pod $RUNPOD_POD_ID"; return 0
    fi
    sleep 5
  done
  echo "[train.sh] SELF-TERMINATION FAILED; delete $RUNPOD_POD_ID manually" | tee -a "$LOG"
}

[ -n "${S3_BUCKET:-}" ] || fail "S3_BUCKET not set"
[ -d "$REPO/scripts" ] || fail "repo checkout missing at $REPO"
cd "$REPO" || fail "cannot cd to $REPO"
echo "[train.sh] source HEAD = $(git rev-parse HEAD 2>/dev/null || echo unknown)" | tee "$LOG"
echo "[train.sh] run=$RUN_NAME ref=${GIT_REF:-?} timeout=${JOB_TIMEOUT:-none}" | tee -a "$LOG"

command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a "$LOG"

# --- runtime: uv-managed Python 3.12 + CUDA JAX (pyproject `cuda` extra) ----
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >>"$LOG" 2>&1 \
  || fail "uv install failed"
VENV=/tmp/fcvenv
uv python install 3.12 >>"$LOG" 2>&1 || fail "uv python install 3.12 failed"
uv venv --python 3.12 "$VENV" >>"$LOG" 2>&1 || fail "venv creation failed"
# shellcheck disable=SC1091
. "$VENV/bin/activate"
uv pip install -q --python "$VENV/bin/python" -e "$REPO[cuda,wandb]" >>"$LOG" 2>&1 \
  || fail "pip install -e .[cuda,wandb] failed"
uv pip install -q --python "$VENV/bin/python" awscli >>"$LOG" 2>&1 \
  || echo "[train.sh] awscli install failed; S3 mirroring disabled" | tee -a "$LOG"

# Cross-pod resume: pull the previous pod's checkpoints + CSV before starting.
# train.py --resume then continues locally (incl. the same W&B run via wandb_id).
if [ "${RESUME_FROM_S3:-0}" = "1" ]; then
  echo "[train.sh] pulling previous run from S3" | tee -a "$LOG"
  s3 sync "s3://$S3_BUCKET/$S3_PREFIX/runs/$RUN_NAME/checkpoints/" \
    "$REPO/checkpoints/$RUN_NAME/" || echo "[train.sh] no checkpoints on S3 (fresh run?)" | tee -a "$LOG"
  s3 cp "s3://$S3_BUCKET/$S3_PREFIX/runs/$RUN_NAME/training.csv" \
    "$REPO/training_logs/$RUN_NAME.csv" || true
fi

# A CPU-only JAX here would silently train ~100x slower while billing GPU
# rates. Refuse to run unless a CUDA device is visible.
"$VENV/bin/python" -c "import jax; ds=jax.devices(); print(ds); assert any(d.platform=='cuda' or 'gpu' in str(d).lower() for d in ds), 'no CUDA device'" \
  >>"$LOG" 2>&1 || fail "no CUDA device visible to JAX"

# --- background S3 sync (replaces the old live-log-only mirror) -------------
sync_loop() {
  while true; do sleep "$SYNC_SECS"; sync_now; done
}
sync_loop &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null || true' EXIT

# --- the job ----------------------------------------------------------------
CMD="$VENV/bin/python scripts/train.py --run-name \"$RUN_NAME\" ${TRAIN_ARGS:-}"
echo "[train.sh] $CMD" | tee -a "$LOG"
if [ -n "${JOB_TIMEOUT:-}" ] && [ "${JOB_TIMEOUT:-0}" != "0" ]; then
  # shellcheck disable=SC2086
  eval timeout -s KILL "$JOB_TIMEOUT" "$CMD" 2>&1 | tee -a "$LOG"
  code=${PIPESTATUS[0]}
else
  # shellcheck disable=SC2086
  eval "$CMD" 2>&1 | tee -a "$LOG"
  code=${PIPESTATUS[0]}
fi
echo "[train.sh] train.py exited with code $code" | tee -a "$LOG"

kill $SYNC_PID 2>/dev/null || true
final_upload
self_terminate
exit "$code"
