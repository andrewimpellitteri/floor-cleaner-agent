"""Tests for the optional W&B logger.

The contract: without configuration, everything no-ops and wandb is never
imported. With WANDB_MODE=offline, the full path (init/log/image/finish)
works against a local directory with no network and no key.
"""

from __future__ import annotations

import sys

import pytest

from floorclean.wandblog import WandbLog, git_tags

wandb = pytest.importorskip("wandb")


def test_disabled_without_key_or_offline(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "online")
    before = "wandb" in sys.modules
    wlog = WandbLog.from_env(project="floorclean", run_name="test")
    assert not wlog.enabled
    # All methods no-op; none of this may raise or import wandb.
    wlog.log_update({"a": 1.0}, step=1)
    wlog.log_image("/nonexistent.png")
    wlog.log_video("/nonexistent.mp4")
    wlog.log_s3_reference("s3://bucket/key", name="x")
    wlog.finish()
    assert ("wandb" in sys.modules) == before


def test_disabled_without_project(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    wlog = WandbLog.from_env(project=None, run_name="test")
    assert not wlog.enabled


def test_offline_roundtrip(monkeypatch, tmp_path):
    """Full path in offline mode: init, per-step curves, image, finish."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    still = tmp_path / "still.png"
    fig.savefig(still)
    plt.close(fig)

    wlog = WandbLog.from_env(
        project="floorclean-test", run_name="offline-probe",
        config={"seed": 0}, tags=["stage:test"])
    assert wlog.enabled
    assert wlog.run_id
    wlog.log_update({"reward_mean": 0.5}, step=1)
    wlog.log_update({"reward_mean": 0.6}, step=2)
    wlog.log_image(str(still), caption="probe", step=2)
    wlog.finish()
    assert any((tmp_path / "wandb").glob("offline-run-*"))


def test_git_tags_shape():
    tags = git_tags()
    assert isinstance(tags, list)
    assert all(t.startswith("git:") for t in tags)
