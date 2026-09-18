"""Optional Weights & Biases logging for training runs.

Auto-activating and silent: every method no-ops unless initialisation
succeeded, which requires `wandb` installed AND (`WANDB_API_KEY` set or
`WANDB_MODE=offline`). Nothing here imports wandb at module level, so a run
without it configured is identical to a run from before this file existed --
the CSV stays the primary record and works with no network at all.
"""

from __future__ import annotations

import os
import subprocess
import sys


def git_tags() -> list[str]:
    """Provenance tags: short SHA, plus git:dirty on an unclean tree."""
    tags: list[str] = []
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if sha.returncode == 0 and sha.stdout.strip():
            tags.append(f"git:{sha.stdout.strip()}")
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5)
        if dirty.returncode == 0 and dirty.stdout.strip():
            tags.append("git:dirty")
    except Exception:
        pass
    return tags


class WandbLog:
    """Thin wandb wrapper with silent fallback. Construct via `from_env`."""

    def __init__(self) -> None:
        self.enabled = False
        self.run = None

    @classmethod
    def from_env(cls, project: str | None, run_name: str,
                 config: dict | None = None,
                 tags: list[str] | None = None,
                 resume_id: str | None = None,
                 entity: str | None = None) -> "WandbLog":
        """Activate iff configured: a project, and (a key or offline mode).

        Any failure -- not installed, no credentials, no network -- returns a
        disabled logger rather than raising. Callers must not branch on
        `enabled` except to skip expensive work (renders); all log methods
        no-op on their own.
        """
        self = cls()
        if not project:
            return self
        if not os.environ.get("WANDB_API_KEY") \
                and os.environ.get("WANDB_MODE") != "offline":
            return self
        try:
            import wandb
            # Initialise before any JAX-heavy work in the caller: wandb spawns
            # its service process and must not fork after XLA thread pools
            # are up.
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=dict(config or {}),
                tags=list(tags or []),
                group=run_name,
                job_type="train",
                id=resume_id,
                resume="allow",
            )
            self.enabled = True
            print(f"[*] wandb enabled: project={project} run={run_name}",
                  file=sys.stderr)
        except Exception as e:
            print(f"[!] wandb disabled ({type(e).__name__}: {e})",
                  file=sys.stderr)
            self.enabled = False
            self.run = None
        return self

    @property
    def run_id(self) -> str | None:
        return self.run.id if self.run is not None else None

    def log_update(self, metrics: dict, step: int) -> None:
        if not self.enabled:
            return
        try:
            import wandb
            wandb.log(dict(metrics), step=step)
        except Exception:
            self.enabled = False

    def log_image(self, path: str, key: str = "floor",
                  caption: str = "", step: int | None = None) -> None:
        if not self.enabled:
            return
        try:
            import wandb
            wandb.log({key: wandb.Image(str(path), caption=caption)},
                      step=step)
        except Exception:
            self.enabled = False

    def log_video(self, path: str, key: str = "sweep", fps: int = 12) -> None:
        if not self.enabled:
            return
        try:
            import wandb
            wandb.log({key: wandb.Video(str(path), fps=fps, format="mp4")})
        except Exception:
            self.enabled = False

    def log_s3_reference(self, uri: str, name: str) -> None:
        """Point the run at an S3 object (e.g. checkpoints) without uploading
        its bytes; W&B stores the pointer, S3 keeps the data."""
        if not self.enabled or not self.run:
            return
        try:
            import wandb
            artifact = wandb.Artifact(name=name, type="checkpoint")
            artifact.add_reference(str(uri))
            self.run.log_artifact(artifact)
        except Exception as e:
            print(f"[!] wandb S3 reference failed for {uri}: {e}",
                  file=sys.stderr)

    def finish(self) -> None:
        if not self.enabled or not self.run:
            return
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
