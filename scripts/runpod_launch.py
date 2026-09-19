#!/usr/bin/env python3
"""Launch a floor_clean training run on RunPod and walk away.

Adapted from ../4o_clone/scripts/runpod_launch.py: same proven shape (GraphQL
deploy with the account-side `terminateAfter` cost guard, wait-healthy
startup check, S3 log mirroring), trimmed to what this project needs.

The pod clones this repo at --ref, installs a CUDA JAX runtime, runs
scripts/jobs/train.sh (which runs scripts/train.py), streams checkpoints and
the CSV to S3, and terminates itself. terminateAfter is MANDATORY -- this
script refuses to launch without it.

Usage:
  set -a; source ~/Documents/dev/4o_clone/.env; set +a   # RUNPOD_API_KEY, S3_BUCKET, AWS keys

  python scripts/runpod_launch.py --dry-run                # show the payload, spend nothing
  python scripts/runpod_launch.py --run-name base          # launch (default: current branch)
  python scripts/runpod_launch.py --status                 # what is running, and cost
  python scripts/runpod_launch.py --logs <pod-id>          # tail the S3-mirrored log
  python scripts/runpod_launch.py --kill <pod-id>

Push first: the pod clones from origin, NOT from this working tree, so a
launch runs whatever origin has. --ref defaults to your current branch and the
launcher refuses a ref that is missing from origin or behind your local
branch (override with --allow-stale-ref only to reproduce a published state).
"""

import argparse
import datetime
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

# REST API v2 remains the read/delete surface. Pod creation uses RunPod's
# documented GraphQL mutation because its input has the account-side
# `terminateAfter` cost guard and REST v2's CreatePodRequest does not.
REST = "https://api.runpod.io/v2"
GRAPHQL = "https://api.runpod.io/graphql"

# Minutes to wait for a container to actually start before terminating the pod.
# ON BY DEFAULT: a pod that never starts bills at the full rate until its
# deadline. wait_healthy() returns as soon as the container is up, so this
# costs a healthy launch seconds.
WAIT_HEALTHY_DEFAULT = 6.0
USER_AGENT = "floorclean-runpod-launcher/1.0"
GITHUB_REPO = "andrewimpellitteri/floor-cleaner-agent"
# CUDA 12.8 runtime on Ubuntu 24.04 (ships Python 3.12; uv installs the rest).
# JAX's pip CUDA wheels need a 12.x driver on the host; if the prestart hook
# rejects it, --wait-healthy kills the pod instead of billing for a wedge.
IMAGE = "docker.io/nvidia/cuda:12.8.1-runtime-ubuntu24.04"
DEFAULT_TERMINATE_AFTER_HOURS = 12.0
DEPLOY_MUTATION = """
mutation Deploy($input: PodFindAndDeployOnDemandInput) {
  podFindAndDeployOnDemand(input: $input) {
    id
    name
    costPerHr
    desiredStatus
  }
}
"""

# Secrets forwarded into the pod environment. GH_PAT is optional: the repo is
# public, so the clone needs no credentials; forward it only if set (e.g. for
# a private fork). The pod needs S3 to mirror logs/checkpoints and
# RUNPOD_API_KEY to terminate itself when the job finishes.
FORWARD_ENV = ["GH_PAT", "S3_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
               "S3_ENDPOINT_URL", "RUNPOD_API_KEY", "WANDB_API_KEY"]

# Bootstrap: authenticated-free clone of the public repo at $GIT_REF, then run
# the job script from that checkout. NO single quotes anywhere in this string:
# cmd_launch wraps it in bash -lc '<text>' and enforces that below.
BOOTSTRAP = (
    "ok=0 ; prep=0 ; "
    "timeout -s KILL 300 apt-get update -qq >/dev/null && "
    "timeout -s KILL 300 apt-get install -y -qq git curl ca-certificates >/dev/null && prep=1 ; "
    "[ $prep = 1 ] && for i in 1 2 3 ; do "
    "timeout -s KILL 120 git -c http.lowSpeedLimit=5000 -c http.lowSpeedTime=20 "
    "clone -q --depth 1 "
    "https://github.com/andrewimpellitteri/floor-cleaner-agent.git /tmp/repo "
    "&& { ok=1 ; break ; } ; "
    "sleep 5 ; rm -rf /tmp/repo ; done ; "
    "[ $ok = 1 ] && ( cd /tmp/repo && "
    "timeout -s KILL 120 git -c http.lowSpeedLimit=5000 -c http.lowSpeedTime=20 "
    "fetch -q --depth 1 origin \"${GIT_REF:-main}\" && "
    "git checkout -q --detach FETCH_HEAD && "
    "echo \"[bootstrap] source HEAD = $(git rev-parse HEAD)\" ) && "
    "BOOTSTRAP_REPO=/tmp/repo timeout -s KILL 54000 "
    # `--job` is forwarded as JOB and normalised to a bare `name.sh` by
    # _normalise_job, which also proves the file exists in the checkout. This
    # used to hardcode train.sh, so every --job other than the default silently
    # ran training instead -- caught by --dry-run before it cost a pod.
    "bash /tmp/repo/scripts/jobs/${JOB:-train.sh} || "
    "{ echo \"[bootstrap] FAILED; self-terminating\" ; "
    "echo eyJxdWVyeSI6Im11dGF0aW9ue3BvZFRlcm1pbmF0ZShpbnB1dDp7cG9kSWQ6XCJfX1BPRElEX19cIn0pfSJ9 "
    "| base64 -d | sed s/__PODID__/$RUNPOD_POD_ID/ > /tmp/kill.json ; "
    "term_ok=0 ; for i in 1 2 3 4 5 ; do "
    "term_response=$(curl --fail --silent --show-error --max-time 10 "
    "-A \"floorclean-bootstrap/1.0\" -X POST https://api.runpod.io/graphql "
    "-H \"Authorization: Bearer $RUNPOD_API_KEY\" "
    "-H \"Content-Type: application/json\" -d @/tmp/kill.json 2>&1) "
    "|| term_response= ; case \"$term_response\" in "
    "*\\\"errors\\\"*) echo \"[bootstrap] termination attempt $i rejected\" ;; "
    "*\\\"data\\\"*) term_ok=1 ; break ;; "
    "*) echo \"[bootstrap] termination attempt $i had no verified success\" ;; "
    "esac ; sleep 5 ; done ; "
    "[ $term_ok = 1 ] || echo \"[bootstrap] SELF-TERMINATION FAILED; delete $RUNPOD_POD_ID manually\" ; }"
)


def api(method, path, payload=None, timeout=60):
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY not set (source the .env first)")
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"{REST}{path}", data=data, method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        detail = e.read()[:600].decode(errors="replace")
        sys.exit(f"RunPod API {method} {path} -> HTTP {e.code}\n{detail}")


def graphql_api(query, variables, timeout=60):
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY not set (source the .env first)")
    body = json.dumps({"query": query, "variables": variables}).encode()
    request = urllib.request.Request(
        GRAPHQL, data=body, method="POST",
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:600].decode(errors="replace")
        sys.exit(f"RunPod GraphQL -> HTTP {exc.code}\n{detail}")
    except urllib.error.URLError as exc:
        sys.exit(f"RunPod GraphQL request failed: {exc.reason}")
    if result.get("errors"):
        messages = [str(e.get("message", "unknown error")) for e in result["errors"][:5]]
        sys.exit("RunPod GraphQL rejected pod creation: " + "; ".join(messages))
    data = result.get("data")
    if not isinstance(data, dict):
        sys.exit("RunPod GraphQL returned no data")
    return data


def pod_uptimes():
    """{pod_id: uptime_seconds_or_None} for every pod on the account."""
    body = {"query": "query{myself{pods{id runtime{uptimeInSeconds}}}}"}
    req = urllib.request.Request(
        GRAPHQL, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {os.environ.get('RUNPOD_API_KEY', '')}",
                 "Content-Type": "application/json", "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return {}
    out = {}
    for pod in ((data.get("data") or {}).get("myself") or {}).get("pods") or []:
        rt = pod.get("runtime")
        out[pod.get("id")] = (rt or {}).get("uptimeInSeconds") if rt else None
    return out


def _uptime_label(seconds):
    if seconds is None:
        return "NOT STARTED"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    return f"{seconds / 3600:.1f}h"


STARTUP_GRACE_SECONDS = 6 * 60


def _pod_age_seconds(pod, now=None):
    created = pod.get("createdAt")
    if not created:
        return None
    try:
        created_at = datetime.datetime.fromisoformat(created.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return max(0.0, (now - created_at).total_seconds())


def cmd_status(now=None):
    pods = api("GET", "/pods")
    if isinstance(pods, dict):
        pods = pods.get("pods", [])
    if not pods:
        print("No pods. Nothing is burning credit.")
        return
    uptimes = pod_uptimes()
    print(f"{'id':22s} {'name':26s} {'gpu':22s} {'status':10s} "
          f"{'uptime':>11s} {'$/hr':>6s}")
    print("-" * 100)
    total = 0.0
    dead = []
    for p in pods:
        rate = p.get("cost") or 0
        total += float(rate)
        gpu = (p.get("gpu") or {}).get("id", "?")
        pid = p.get("id", "")
        up = uptimes.get(pid)
        if pid in uptimes and up is None:
            age = _pod_age_seconds(p, now=now)
            if age is not None and age < STARTUP_GRACE_SECONDS:
                label = "STARTING"
            else:
                label = "NOT STARTED"
                dead.append((pid, float(rate)))
        else:
            label = _uptime_label(up) if pid in uptimes else "?"
        print(f"{pid:22s} {(p.get('name') or '')[:26]:26s} "
              f"{str(gpu)[:22]:22s} "
              f"{p.get('status', ''):10s} {label:>11s} {float(rate):6.3f}")
    print("-" * 100)
    print(f"{'TOTAL':>73s} {total:6.3f} $/hr  (${total * 24:.2f}/day if left running)")
    if dead:
        print()
        for pid, rate in dead:
            print(f"[!] {pid} is billing ${rate:.3f}/hr but its container "
                  f"has NEVER STARTED. Kill it:")
            print(f"    python scripts/runpod_launch.py --kill {pid}")


def cmd_logs(pod_id, run_name="base", prefix="floorclean"):
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        sys.exit("S3_BUCKET not set")
    for label, key in (
        ("completed", f"s3://{bucket}/{prefix}/runs/{run_name}/train.log"),
        ("live", f"s3://{bucket}/{prefix}/runs/{run_name}/train.live.log"),
        ("legacy", f"s3://{bucket}/logs/{pod_id}.log"),
    ):
        result = subprocess.run(["aws", "s3", "cp", key, "-"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[*] {label} log: {key}\n")
            print(result.stdout)
            return
    print("(no S3 log is available yet)")
    print("    If the container is up, the job has not reached its first S3 "
          "log flush yet -- check again shortly.")
    print(f"    Or inspect the pod in the RunPod console: {pod_id}")


def termination_deadline(hours, now=None):
    """Return RunPod's absolute UTC terminateAfter value. None only if 0."""
    if not math.isfinite(hours) or hours < 0:
        sys.exit("--hours must be a finite non-negative number")
    if hours == 0:
        return None
    now = now or datetime.datetime.now(datetime.timezone.utc)
    deadline = now + datetime.timedelta(hours=hours)
    return deadline.replace(microsecond=0).isoformat().replace("+00:00", "Z")


GPU_VRAM_GB = {
    "NVIDIA GeForce RTX 4090": 24,
    "NVIDIA RTX A5000": 24,
    "NVIDIA L4": 24,
    "NVIDIA A40": 48,
    "NVIDIA L40": 48,
    "NVIDIA L40S": 48,
    "NVIDIA RTX A6000": 48,
    "NVIDIA RTX 6000 Ada Generation": 48,
    "NVIDIA A100 80GB PCIe": 80,
    "NVIDIA A100-SXM4-80GB": 80,
    "NVIDIA H100 80GB HBM3": 80,
    "NVIDIA H100 PCIe": 80,
    "NVIDIA H100 NVL": 94,
    "NVIDIA H200": 141,
}

JOB_DIRECTIVE = re.compile(r"^#\s*runpod:\s*min-vram-gb\s*=\s*(\d+)", re.M)
JOB_TIMEOUT_DIRECTIVE = re.compile(
    r"^#\s*runpod:\s*job-timeout-env\s*=\s*([A-Z_][A-Z0-9_]*)\s*:\s*(\d+)", re.M)

JOBS_DIR = pathlib.Path(__file__).resolve().parent / "jobs"


def resolve_job(job):
    """Normalise --job to a bare `name.sh` and prove the file exists."""
    name = pathlib.Path(str(job)).name
    if not name.endswith(".sh"):
        name += ".sh"
    path = JOBS_DIR / name
    if not path.is_file():
        available = sorted(
            f.name for f in JOBS_DIR.glob("*.sh") if not f.name.startswith("_"))
        sys.exit(
            f"--job {job!r} does not resolve to a job script.\n"
            f"Looked for: {path}\n"
            f"Available: {', '.join(available)}")
    return name, path


def job_min_vram_gb(job_path):
    try:
        text = pathlib.Path(job_path).read_text()
    except OSError:
        return None
    m = JOB_DIRECTIVE.search(text)
    return int(m.group(1)) if m else None


def job_timeout_seconds(job_path, environ=None):
    """The job's own kill timeout, from `# runpod: job-timeout-env=VAR:default`."""
    environ = os.environ if environ is None else environ
    try:
        text = pathlib.Path(job_path).read_text()
    except OSError:
        return None
    m = JOB_TIMEOUT_DIRECTIVE.search(text)
    if not m:
        return None
    var, default = m.group(1), int(m.group(2))
    raw = environ.get(var)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def parse_env_overrides(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            sys.exit(f"--env expects KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        out[key] = value
    return out


def check_runtime_budget(job_path, hours, environ=None):
    """Refuse a launch whose pod deadline would cut the job's own timeout.

    A falsy job timeout (0/empty = none) means checkpoints stream to S3 every
    chunk, so a pod-deadline kill loses at most one chunk -- nothing to guard.
    """
    need = job_timeout_seconds(job_path, environ=environ)
    if not need or not hours:
        return
    pod_seconds = float(hours) * 3600.0
    margin = 1800.0  # 30 min for pull, bootstrap and install
    if pod_seconds < need + margin:
        sys.exit(
            f"{os.path.basename(str(job_path))} can run for {need}s "
            f"({need / 3600:.1f} h) but --hours {hours} terminates the pod at "
            f"{pod_seconds / 3600:.1f} h.\n"
            f"The pod deadline would fire first, hard-killing the container "
            f"before the job's cleanup trap can upload final results.\n"
            f"Raise --hours to at least {(need + margin) / 3600:.1f}.")
    print(f"[*] runtime budget ok: job timeout {need / 3600:.1f} h, "
          f"pod deadline {pod_seconds / 3600:.1f} h", file=sys.stderr)


def gpu_vram_gb(gpu_id):
    if gpu_id in GPU_VRAM_GB:
        return GPU_VRAM_GB[gpu_id]
    body = {"query": "query{gpuTypes{id memoryInGb}}"}
    req = urllib.request.Request(
        GRAPHQL, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {os.environ.get('RUNPOD_API_KEY', '')}",
                 "Content-Type": "application/json", "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return None
    for g in (data.get("data") or {}).get("gpuTypes") or []:
        if g.get("id") == gpu_id:
            return g.get("memoryInGb")
    return None


def check_vram(job_path, gpu_id):
    need = job_min_vram_gb(job_path)
    if need is None:
        return
    have = gpu_vram_gb(gpu_id)
    if have is None:
        print(f"[!] job declares min-vram-gb={need} but VRAM of {gpu_id!r} "
              f"could not be determined; continuing unchecked.", file=sys.stderr)
        return
    if have < need:
        sys.exit(f"job declares min-vram-gb={need}, but {gpu_id} has "
                 f"{have} GB. Pick a bigger card with --gpu.")
    print(f"[*] VRAM ok: {gpu_id} has {have} GB, job needs {need} GB", file=sys.stderr)


def current_branch():
    try:
        r = subprocess.run(["git", "branch", "--show-current"],
                           capture_output=True, text=True, timeout=10)
        name = r.stdout.strip()
        return name or "main"
    except (OSError, subprocess.SubprocessError):
        return "main"


def check_ref_pushed(ref, allow_stale=False):
    """Refuse to launch a ref whose origin state is missing or behind local.

    The pod clones from origin, NOT from this working tree, so a launch runs
    whatever origin has. Uncommitted changes NEVER reach the pod -- not even
    with --allow-stale-ref.
    """
    try:
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        dirty = None
    if dirty is not None and dirty.stdout.strip():
        print("[!] working tree has uncommitted changes. The pod clones from "
              "origin and will NOT see them. Commit and push first.", file=sys.stderr)
    try:
        remote = subprocess.run(
            ["git", "ls-remote", "--exit-code", "origin", ref],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        print(f"[!] could not reach origin to verify {ref!r}; continuing "
              f"unchecked. The pod clones from origin, not this working tree.", file=sys.stderr)
        return
    if remote.returncode != 0:
        msg = (f"ref {ref!r} does not exist on origin.\n"
               f"The pod clones from GitHub, not from this working tree, so "
               f"there is nothing for it to check out.\n"
               f"Push it first:  git push -u origin {ref}")
        if allow_stale:
            print(f"[!] {msg}", file=sys.stderr)
            return
        sys.exit(msg)
    remote_sha = remote.stdout.split()[0] if remote.stdout.split() else ""
    local = subprocess.run(["git", "rev-parse", ref],
                           capture_output=True, text=True)
    if local.returncode != 0:
        return
    local_sha = local.stdout.strip()
    if local_sha == remote_sha:
        print(f"[*] ref ok: origin/{ref} == local {local_sha[:7]}", file=sys.stderr)
        return
    ahead = subprocess.run(
        ["git", "rev-list", "--count", f"{remote_sha}..{local_sha}"],
        capture_output=True, text=True)
    n = ahead.stdout.strip() if ahead.returncode == 0 else "?"
    msg = (f"local {ref} is {n} commit(s) AHEAD of origin.\n"
           f"  local  {local_sha[:7]}\n  origin {remote_sha[:7]}\n"
           f"The pod clones origin, so it would run the OLD code.\n"
           f"Push first:  git push origin {ref}\n"
           f"Or pass --allow-stale-ref to run origin's version deliberately.")
    if allow_stale:
        print(f"[!] {msg}", file=sys.stderr)
        return
    sys.exit(msg)


def pod_started(pod_id):
    """True once the container is actually up (bills from rent, not start)."""
    body = {"query": "query{myself{pods{id runtime{uptimeInSeconds}}}}"}
    req = urllib.request.Request(
        GRAPHQL, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {os.environ.get('RUNPOD_API_KEY', '')}",
                 "Content-Type": "application/json", "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return None
    for pod in ((data.get("data") or {}).get("myself") or {}).get("pods") or []:
        if pod.get("id") == pod_id:
            rt = pod.get("runtime")
            return bool(rt and (rt.get("uptimeInSeconds") or 0) > 0)
    return False


def wait_healthy(pod_id, minutes, poll=20):
    deadline = time.time() + minutes * 60
    saw_unknown = False
    while time.time() < deadline:
        started = pod_started(pod_id)
        if started:
            print(f"[+] {pod_id}: container is up")
            return True
        if started is None:
            saw_unknown = True
        time.sleep(poll)
    if pod_started(pod_id) is True:
        print(f"[+] {pod_id}: container is up")
        return True
    if saw_unknown and pod_started(pod_id) is None:
        print(f"[!] {pod_id}: could not read pod runtime; leaving it ALONE.")
        return False
    print(f"[!] {pod_id}: uptime still 0 after {minutes} min -- the container "
          f"never started. Terminating so it stops billing.")
    cmd_kill(pod_id)
    return False


def cmd_kill(pod_id):
    api("DELETE", f"/pods/{pod_id}")
    print(f"[+] terminated {pod_id}")


def default_run_name(ref):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", ref)[:32] or "base"


def live_pod_names():
    """{pod_name: (pod_id, desiredStatus)} for every pod on the account."""
    body = {"query": "query{myself{pods{id name desiredStatus}}}"}
    req = urllib.request.Request(
        GRAPHQL, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {os.environ.get('RUNPOD_API_KEY', '')}",
                 "Content-Type": "application/json", "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return {}
    out = {}
    for pod in ((data.get("data") or {}).get("myself") or {}).get("pods") or []:
        if pod.get("name"):
            out[pod["name"]] = (pod.get("id"), pod.get("desiredStatus"))
    return out


def s3_run_has_state(run_name, prefix="floorclean"):
    """Objects already under this run's S3 prefix, as a short list (or [])."""
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        return []
    uri = f"s3://{bucket}/{prefix}/runs/{run_name}/"
    try:
        out = subprocess.run(["aws", "s3", "ls", uri, "--recursive"],
                             capture_output=True, text=True, timeout=60)
    except Exception:
        return []
    if out.returncode != 0:
        return []
    return [ln for ln in out.stdout.splitlines() if ln.strip()][:5]


def check_run_name_free(args):
    """Refuse a run name that collides with a live pod or existing S3 state.

    Why this exists: `--run-name` is used verbatim for the pod name, the S3
    prefix AND the W&B run name, so relaunching after a kill silently produces
    a SECOND W&B run with the same display name pointing at the SAME S3 prefix.
    That happened on 2026-09-19 (two `macro-a10` runs); it was harmless only
    because the killed pod died before writing a checkpoint. Had it got further,
    the relaunch would have inherited or clobbered its state, and the resulting
    curve would have been a silent splice of two different configurations --
    the kind of thing that is very hard to notice afterwards and invalidates
    the comparison it was launched to make.
    """
    name = f"fc-train-{args.run_name}"[:48]
    pods = live_pod_names()
    if name in pods and not args.allow_duplicate_run:
        pid, status = pods[name]
        sys.exit(
            f"a pod named {name!r} already exists ({pid}, {status}).\n"
            f"    Launching now would give two runs the same S3 prefix "
            f"(runs/{args.run_name}/) and the same W&B name.\n"
            f"    Kill it first:   python scripts/runpod_launch.py --kill {pid}\n"
            f"    Or use a different --run-name, or pass --allow-duplicate-run "
            f"if you really want both.")

    existing = s3_run_has_state(args.run_name)
    if existing and not (args.resume_run or args.allow_duplicate_run):
        listing = "\n".join(f"      {ln}" for ln in existing)
        sys.exit(
            f"s3 already holds state for run {args.run_name!r}:\n{listing}\n"
            f"    A fresh launch would overwrite the checkpoint and splice the "
            f"CSV.\n"
            f"    Use a different --run-name, or --resume-run to continue it "
            f"deliberately, or --allow-duplicate-run to overwrite.")


def cmd_launch(args):
    if not os.environ.get("S3_BUCKET"):
        sys.exit("missing env: S3_BUCKET (source the .env first)")
    args.job, job_path = resolve_job(args.job)
    check_run_name_free(args)
    overrides = parse_env_overrides(args.env)
    check_runtime_budget(job_path, args.hours,
                         environ={**os.environ, **overrides})
    check_vram(job_path, args.gpu)
    check_ref_pushed(args.ref, args.allow_stale_ref)

    env = {k: os.environ[k] for k in FORWARD_ENV if os.environ.get(k)}
    print(f"[*] forwarding {len(env)} env var(s): "
          f"{', '.join(sorted(env))}", file=sys.stderr)
    env["JOB"] = args.job
    env["GIT_REF"] = args.ref
    env["RUN_NAME"] = args.run_name
    if args.job_timeout:
        env["JOB_TIMEOUT"] = args.job_timeout
    if args.train_args:
        env["TRAIN_ARGS"] = args.train_args
    env.update(overrides)

    boot = BOOTSTRAP
    if "{{" in boot or "}}" in boot:
        sys.exit("internal: BOOTSTRAP has leftover doubled braces")
    if "'" in boot:
        sys.exit("internal: BOOTSTRAP contains a single quote -- it is "
                 "wrapped as bash -lc '<text>'")
    chk = subprocess.run(["bash", "-n", "-lc", boot],
                         capture_output=True, text=True)
    if chk.returncode != 0:
        sys.exit(f"bootstrap failed `bash -n` parse check:\n{chk.stderr}")

    payload = {
        "name": args.name or f"fc-train-{args.run_name}"[:48],
        "imageName": args.image,
        "gpuTypeId": args.gpu,
        "gpuCount": args.gpu_count,
        "containerDiskInGb": args.disk,
        "volumeInGb": 0,
        "cloudType": args.cloud,
        "env": [{"key": k, "value": v} for k, v in env.items()],
        "dockerArgs": f"bash -lc '{boot}'",
        "startSsh": False,
        "startJupyter": False,
    }
    # MANDATORY cost guard: unlike any timeout inside the job, this still
    # fires when an image pull wedges or the container never starts.
    terminate_after = termination_deadline(args.hours)
    if not terminate_after:
        sys.exit("refusing to launch with --hours 0: the account-side "
                 "terminateAfter guard is mandatory for this project")
    payload["terminateAfter"] = terminate_after

    if args.dry_run:
        redacted = dict(payload)
        redacted["env"] = [
            {"key": i["key"],
             "value": "<set>" if i["key"] in FORWARD_ENV else i["value"]}
            for i in payload["env"]
        ]
        print(json.dumps(redacted, indent=2))
        print("\nDry run. Re-run without --dry-run to create the pod.",
              file=sys.stderr)
        return

    data = graphql_api(DEPLOY_MUTATION, {"input": payload})
    pod = data.get("podFindAndDeployOnDemand")
    if not isinstance(pod, dict) or not pod.get("id"):
        sys.exit("RunPod GraphQL returned no created pod")
    pod_id = pod.get("id", "?")
    print(f"[+] launched pod {pod_id} ({payload['name']})")
    print(f"    gpu   : {args.gpu} x{args.gpu_count}")
    print(f"    job   : scripts/jobs/{args.job} @ {args.ref}")
    print(f"    run   : {args.run_name}")
    print(f"    cost  : ${pod.get('costPerHr', '?')}/hr")
    print(f"    guard : account-side deletion at {terminate_after}")
    print(f"\n    watch : python scripts/runpod_launch.py --status")
    print(f"    logs  : python scripts/runpod_launch.py --logs {pod_id} "
          f"--run-name {args.run_name}")
    print(f"    kill  : python scripts/runpod_launch.py --kill {pod_id}")
    print("\n  The pod terminates itself when the job finishes.")

    if args.wait_healthy:
        print(f"\n[*] confirming the container starts (up to "
              f"{args.wait_healthy:g} min; --wait-healthy 0 to skip)")
        wait_healthy(pod_id, args.wait_healthy)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", default="train.sh",
                    help="script name inside scripts/jobs/ (default: train.sh)")
    ap.add_argument("--ref", default=None,
                    help="git ref to run (default: current branch)")
    ap.add_argument("--run-name", default=None,
                    help="run name for train.py + S3 paths (default: from --ref)")
    ap.add_argument("--train-args", default=None,
                    help="extra args appended to scripts/train.py, e.g. "
                         "'--ppo.total-timesteps 50000000'")
    ap.add_argument("--job-timeout", default=None, metavar="SECONDS",
                    help="JOB_TIMEOUT for the pod-side job (default: none)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--disk", type=int, default=80, help="container disk GB")
    ap.add_argument("--hours", type=float, default=DEFAULT_TERMINATE_AFTER_HOURS,
                    help="account-side terminate-after cost guard in hours "
                          f"(default: {DEFAULT_TERMINATE_AFTER_HOURS:g}; "
                          "0 is REFUSED)")
    ap.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    ap.add_argument("--image", default=IMAGE,
                    help="pod base image (default: CUDA 12.8 runtime on "
                         "Ubuntu 24.04)")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="extra environment variable for the job (repeatable)")
    ap.add_argument("--allow-duplicate-run", action="store_true",
                    help="permit a run name that collides with a live pod or "
                         "existing S3 state (overwrites)")
    ap.add_argument("--resume-run", action="store_true",
                    help="deliberately continue an existing S3 run prefix; "
                         "pair with the job's own resume flag")
    ap.add_argument("--allow-stale-ref", action="store_true",
                    dest="allow_stale_ref",
                    help="launch even though --ref is missing from origin or "
                         "behind your local branch (runs origin's version)")
    ap.add_argument("--wait-healthy", type=float, default=WAIT_HEALTHY_DEFAULT,
                    metavar="MIN", dest="wait_healthy",
                    help="poll until the container starts; terminate the pod "
                         "if it never does (default "
                         f"{WAIT_HEALTHY_DEFAULT:g} min; 0 disables)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--logs", metavar="POD_ID")
    ap.add_argument("--kill", metavar="POD_ID")
    args = ap.parse_args()

    if args.status:
        return cmd_status()
    if args.logs:
        return cmd_logs(args.logs, getattr(args, "run_name", None) or "base")
    if args.kill:
        return cmd_kill(args.kill)
    if args.ref is None:
        args.ref = current_branch()
    if args.run_name is None:
        args.run_name = default_run_name(args.ref)
    cmd_launch(args)


if __name__ == "__main__":
    main()
