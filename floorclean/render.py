"""Rendering: watch what the policy is actually doing.

This exists to catch modelling errors, not to make pretty pictures. An early
version of the scripted baselines walked ALONG the fan's long axis, so every
pass cleared a three-centimetre strip instead of a twenty-centimetre one. That
was invisible in the reward curve and would have been obvious in a single frame.
Assume there are more bugs of that shape and look at the video before believing
a number.

It is also how the result gets communicated. A table of completion times settles
the argument on paper; a side-by-side video of two techniques is what actually
convinces somebody who cleans this floor every day.

Headless by design -- this runs on a GPU box with no display.

Axis convention: arrays are indexed [i=x, j=y], so everything is transposed for
display to put x on the horizontal. This module is the only place allowed to do
that (see ORIENTATION.md).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse

from .config import Config
from .env import CleaningEnv, EnvState
from .jet import jet_impact

# Dirty epoxy: clean grey-green floor through to heavy brown grit.
DIRT_CMAP = LinearSegmentedColormap.from_list(
    "bay_floor", ["#d9dcd6", "#b9a07a", "#8a6032", "#4a3018"]
)


def _extent(cfg: Config):
    return [0.0, cfg.floor.length_x, 0.0, cfg.floor.length_y]


def draw_frame(ax, env: CleaningEnv, state: EnvState, title: str = ""):
    """Draw one frame of the bay onto `ax`. Returns the artists it created."""
    cfg = env.cfg
    fields = state.fields

    residual = np.asarray(fields.bound + fields.deposited + fields.suspended)
    water = np.asarray(fields.h)
    trough = np.asarray(env.floor.trough)

    ax.clear()
    ax.imshow(
        residual.T,
        origin="lower",
        extent=_extent(cfg),
        cmap=DIRT_CMAP,
        vmin=0.0,
        vmax=cfg.dirt.load_mean * 1.6,
        interpolation="nearest",
        aspect="equal",
    )

    # Standing water as a translucent blue wash. Where the water is matters as
    # much as where the dirt is -- it is what carries slurry.
    wet = np.clip(water / 8.0e-3, 0.0, 1.0)
    blue = np.zeros(wet.shape + (4,))
    blue[..., 2] = 1.0
    blue[..., 1] = 0.45
    blue[..., 3] = 0.40 * wet
    ax.imshow(np.transpose(blue, (1, 0, 2)), origin="lower", extent=_extent(cfg),
              interpolation="nearest", aspect="equal")

    # The trough.
    ax.contour(
        np.linspace(0, cfg.floor.length_x, cfg.floor.nx),
        np.linspace(0, cfg.floor.length_y, cfg.floor.ny),
        trough.T,
        levels=[0.5],
        colors="#1b3a5c",
        linewidths=2.0,
    )

    # The operator, and -- separately -- where the jet actually lands. These are
    # not the same point: at a high tilt the impact is most of a metre ahead,
    # and conflating them hides aiming errors.
    impact = jet_impact(cfg, env.floor, state.tip_x, state.tip_y,
                        state.standoff, state.tilt, state.azimuth)
    tx, ty = float(state.tip_x), float(state.tip_y)
    ix, iy = float(impact.impact_x), float(impact.impact_y)

    ax.plot([tx, ix], [ty, iy], color="#00d0ff", lw=1.4, alpha=0.9, zorder=5)
    ax.plot(tx, ty, "o", color="#00d0ff", ms=7, zorder=6)

    # Footprint of the fan: long across the push direction, thin along it.
    slant = float(state.standoff) / max(np.cos(float(state.tilt)), 1e-3)
    width = 2.0 * slant * np.tan(cfg.washer.fan_angle * 0.5)
    thick = (cfg.washer.fan_thickness_0
             + 2.0 * slant * np.tan(cfg.washer.fan_spread_angle * 0.5))
    thick /= max(np.cos(float(state.tilt)), 1e-3)
    ax.add_patch(
        Ellipse((ix, iy), width=max(thick, 0.02), height=max(width, 0.02),
                angle=np.degrees(float(state.azimuth)),
                facecolor="#ffe066", edgecolor="#ff8800", alpha=0.75, zorder=7)
    )

    ax.set_xlim(0, cfg.floor.length_x)
    ax.set_ylim(0, cfg.floor.length_y)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9, family="monospace")


def frame_title(env: CleaningEnv, state: EnvState) -> str:
    cfg = env.cfg
    residual = state.fields.bound + state.fields.deposited + state.fields.suspended
    remaining = float(np.sum(np.asarray(residual))) * cfg.floor.cell_area
    t = float(state.step) * cfg.sim.control_dt
    return (
        f"t={t / 60:5.2f} min   grit {remaining:6.2f} kg   "
        f"standoff {float(state.standoff) * 100:4.0f} cm   "
        f"tilt {np.degrees(float(state.tilt)):3.0f} deg"
    )


def render_episode(env: CleaningEnv, rollout, path: str, fps: int = 20,
                   stride: int = 5, dpi: int = 110):
    """Write an mp4 of a rollout.

    `rollout` is a sequence of `EnvState` (single-environment, not batched).
    `stride` subsamples: at 5 Hz control, stride=5 gives 1 s of work per frame.
    """
    import imageio.v2 as imageio

    cfg = env.cfg
    aspect = cfg.floor.length_y / cfg.floor.length_x
    fig, ax = plt.subplots(figsize=(6.0, 6.0 * aspect * 0.55), dpi=dpi)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)

    frames = []
    for state in rollout[::stride]:
        draw_frame(ax, env, state, frame_title(env, state))
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(buf.copy())

    plt.close(fig)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=None)
    return path


def render_comparison(env: CleaningEnv, rollouts: dict, path: str,
                      fps: int = 20, stride: int = 5, dpi: int = 100):
    """Several strategies side by side, on the same clock and the same floor.

    This is the persuasive artefact: identical starting conditions, identical
    elapsed time, visibly different amounts of floor left.
    """
    import imageio.v2 as imageio

    cfg = env.cfg
    names = list(rollouts)
    n = len(names)
    aspect = cfg.floor.length_y / cfg.floor.length_x
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.0 * aspect * 0.55), dpi=dpi)
    if n == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.05)

    length = min(len(r) for r in rollouts.values())
    frames = []
    for k in range(0, length, stride):
        for ax, name in zip(axes, names):
            state = rollouts[name][k]
            draw_frame(ax, env, state, f"{name}\n{frame_title(env, state)}")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    plt.close(fig)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=None)
    return path


def save_still(env: CleaningEnv, state: EnvState, path: str, dpi: int = 130):
    cfg = env.cfg
    aspect = cfg.floor.length_y / cfg.floor.length_x
    fig, ax = plt.subplots(figsize=(6.0, 6.0 * aspect * 0.55), dpi=dpi)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)
    draw_frame(ax, env, state, frame_title(env, state))
    fig.savefig(path)
    plt.close(fig)
    return path
