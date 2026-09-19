#!/usr/bin/env bash
# Pod-side batched baseline experiments for the floor_clean wash-bay project.
#
# Runs inside the RunPod container after BOOTSTRAP (see scripts/runpod_launch.py)
# has cloned the repo to $BOOTSTRAP_REPO and checked out $GIT_REF. Installs a
# CUDA JAX runtime, runs scripts/experiments.py for each requested suite,
# mirrors results to S3, and terminates the pod -- success or failure.
#
# This is the CHEAP-EVIDENCE job, not training: no checkpoints, no resume, no
# W&B. It exists because the single-seed CPU matrix cannot distinguish a real
# 2-point ranking change from noise, and vmapping seeds on a GPU makes
# confidence intervals affordable.
#
# Environment (all forwarded by the launcher; see --dry-run):
#   RUN_NAME    run name -> S3 key segment (default: experiments)
#   SUITES      space-separated suites to run (default: "matrix lanes")
#   EXP_ARGS    extra args appended to every scripts/experiments.py invocation
#   JOB_TIMEOUT seconds for the whole job; empty/0 = none (default: none)
#   S3_BUCKET   results destination (required)
#   S3_PREFIX   key prefix (default: floorclean)
#   RUNPOD_API_KEY + RUNPOD_POD_ID  self-termination (warn-only if missing:
#               the account-side terminateAfter deadline still applies)
#
# runpod: min-vram-gb=20
# runpod: job-timeout-env=JOB_TIMEOUT:0
set -u
REPO="${BOOTSTRAP_REPO:-/tmp/repo}"
RUN_NAME="${RUN_NAME:-experiments}"
S3_PREFIX="${S3_PREFIX:-floorclean}"
SUITES="${SUITES:-matrix lanes}"
LOG="/tmp/experiments.log"
OUTDIR="/tmp/expout"
LIVE_KEY="s3://${S3_BUCKET:-}/$S3_PREFIX/experiments/$RUN_NAME/run.live.log"
SYNC_SECS=300

s3() { aws s3 "$@" >>"$LOG" 2>&1; }

sync_now() {
  [ -n "${S3_BUCKET:-}" ] || return 0
  dest="s3://$S3_BUCKET/$S3_PREFIX/experiments/$RUN_NAME"
  s3 sync "$OUTDIR" "$dest/" --quiet || true
  s3 cp "$LOG" "$LIVE_KEY" --quiet || true
}

final_upload() {
  [ -n "${S3_BUCKET:-}" ] || return 0
  dest="s3://$S3_BUCKET/$S3_PREFIX/experiments/$RUN_NAME"
  echo "[experiments.sh] uploading results to $dest" | tee -a "$LOG"
  s3 sync "$OUTDIR" "$dest/" || true
  s3 cp "$LOG" "$dest/run.log" || true
}

self_terminate() {
  [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ] || {
    echo "[experiments.sh] no RUNPOD_API_KEY/POD_ID; pod ends at terminateAfter"
    return 0
  }
  pod_mutation=$(printf 'mutation{podTerminate(input:{podId:"%s}")}' "$RUNPOD_POD_ID")
  for _ in 1 2 3; do
    if curl -fsS --max-time 15 -A "floorclean-experiments/1.0" \
        -X POST https://api.runpod.io/graphql \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"$pod_mutation\"}" >>"$LOG" 2>&1; then
      echo "[experiments.sh] self-terminated pod $RUNPOD_POD_ID"; return 0
    fi
    sleep 5
  done
  echo "[experiments.sh] SELF-TERMINATION FAILED; delete $RUNPOD_POD_ID manually" \
    | tee -a "$LOG"
}

fail() {
  echo "[experiments.sh] FATAL: $*" | tee -a "$LOG" >&2
  final_upload; self_terminate; exit 1
}

[ -n "${S3_BUCKET:-}" ] || fail "S3_BUCKET not set"
[ -d "$REPO/scripts" ] || fail "repo checkout missing at $REPO"
cd "$REPO" || fail "cannot cd to $REPO"
mkdir -p "$OUTDIR"
echo "[experiments.sh] source HEAD = $(git rev-parse HEAD 2>/dev/null || echo unknown)" | tee "$LOG"
echo "[experiments.sh] run=$RUN_NAME ref=${GIT_REF:-?} suites='$SUITES'" | tee -a "$LOG"

command -v nvidia-smi >/dev/null && \
  nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a "$LOG"

# --- runtime: uv-managed Python 3.12 + CUDA JAX (pyproject `cuda` extra) ----
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >>"$LOG" 2>&1 \
  || fail "uv install failed"
VENV=/tmp/fcvenv
uv python install 3.12 >>"$LOG" 2>&1 || fail "uv python install 3.12 failed"
uv venv --python 3.12 "$VENV" >>"$LOG" 2>&1 || fail "venv creation failed"
# shellcheck disable=SC1091
. "$VENV/bin/activate"
uv pip install -q --python "$VENV/bin/python" -e "$REPO[cuda]" >>"$LOG" 2>&1 \
  || fail "pip install -e .[cuda] failed"
uv pip install -q --python "$VENV/bin/python" awscli >>"$LOG" 2>&1 \
  || echo "[experiments.sh] awscli install failed; S3 mirroring disabled" | tee -a "$LOG"

# The pip CUDA wheels must win over the base image's CUDA (see train.sh).
NVDIRS=$("$VENV/bin/python" - <<'PY'
import glob, os, site
base = site.getsitepackages()[0]
dirs = sorted(glob.glob(os.path.join(base, "nvidia", "*", "lib")))
print(":".join(dirs))
PY
)
if [ -n "$NVDIRS" ]; then
  export LD_LIBRARY_PATH="$NVDIRS:${LD_LIBRARY_PATH:-/usr/local/cuda/lib64}"
  echo "[experiments.sh] prepended nvidia wheel lib dirs" | tee -a "$LOG"
fi

# A CPU-only JAX here would silently run ~100x slower while billing GPU rates.
# Test the BACKEND, not the device's spelling (see train.sh for the history).
"$VENV/bin/python" - >>"$LOG" 2>&1 <<'PY' || fail "no CUDA device visible to JAX"
import jax
backend = jax.default_backend()
print(f"devices={jax.devices()} backend={backend}")
assert backend != "cpu", f"JAX backend is {backend!r}, expected an accelerator"
PY

sync_loop() { while true; do sleep "$SYNC_SECS"; sync_now; done; }
sync_loop &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null || true' EXIT

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION:-0.9}"
export JAX_LOG_COMPILES=0
export PYTHONUNBUFFERED=1

rc=0
for suite in $SUITES; do
  echo "[experiments.sh] === suite: $suite ===" | tee -a "$LOG"
  # shellcheck disable=SC2086
  "$VENV/bin/python" scripts/experiments.py --suite "$suite" \
      --out "$OUTDIR/$suite.jsonl" ${EXP_ARGS:-} 2>&1 | tee -a "$LOG"
  s=${PIPESTATUS[0]}
  [ "$s" -eq 0 ] || { echo "[experiments.sh] suite $suite FAILED ($s)" | tee -a "$LOG"; rc=$s; }
  sync_now
done

final_upload
self_terminate
exit "$rc"
