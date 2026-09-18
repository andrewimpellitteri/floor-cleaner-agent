#!/usr/bin/env python3
"""Print the simulator's physical behaviour in terms you can check by eye.

Run this FIRST, and argue with the numbers. Every conclusion the trained policy
produces rests on these, so if the single-pass removal fraction or the jet
kickback does not match what the wand feels like in your hands, the thing to fix
is `floorclean/config.py`, not the policy.

    python scripts/calibrate.py

The knobs worth turning, in order of how much they matter:

  washer.fan_angle          which tip you actually use (25 deg green is assumed)
  dirt.entrainment_rate     how fast a close, upright pass strips grit
  dirt.yield_mean           how hard the grit is stuck to the epoxy
  dirt.deposit_threshold    how easily loosened slurry pushes along
  washer.velocity_decay_length   how fast the jet dies with distance
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import jax.numpy as jnp
import numpy as np

from floorclean.config import GPM, Config
from floorclean.geometry import build_floor
from floorclean.jet import jet_impact
from floorclean.physics import initial_fields, physics_substep


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    cfg = Config()
    floor = build_floor(cfg)
    w, d, fc = cfg.washer, cfg.dirt, cfg.floor

    section("Machine")
    print(f"  pressure            {w.pressure / 6894.757:8.0f} psi")
    print(f"  flow                {w.flow / GPM:8.1f} gpm")
    print(f"  jet velocity        {w.jet_velocity:8.0f} m/s")
    print(f"  thrust (kickback)   {w.momentum_flux:8.1f} N  "
          f"({w.momentum_flux / 4.448:.1f} lbf)")
    print(f"  tip                 {np.degrees(w.fan_angle):8.0f} deg fan")

    section("Bay")
    print(f"  worked section      {fc.length_x:.1f} m along x "
          f"x {fc.length_y:.1f} m across  ({fc.length_x * fc.length_y:.0f} m^2)")
    print(f"  wall to trough      {fc.length_y / 2:.1f} m "
          f"({fc.length_y / 2 / 0.3048:.0f} ft)")
    print(f"  grid                {fc.nx} x {fc.ny} cells at {fc.dx * 100:.0f} cm")
    print(f"  cross slope         {fc.cross_slope * 100:.1f} %")
    print(f"  starting grit       {d.load_mean:.2f} kg/m^2  "
          f"(~{d.load_mean * fc.length_x * fc.length_y:.0f} kg over the section)")
    print(f"  adhesion            {d.yield_mean / 1e3:.1f} kPa typical, "
          f"up to {(d.yield_mean + d.worn_lane_boost) / 1e3:.1f} kPa in worn lanes")

    section("Jet at the floor: impingement pressure (kPa) vs adhesion")
    print("  Values below ~1.0x adhesion cannot lift stuck grit at all.")
    tilts = [0.0, 0.5, 0.9, 1.2, 1.4]
    print("  standoff |" + "".join(f"{np.degrees(t):>10.0f} deg" for t in tilts))
    for s in (0.08, 0.15, 0.25, 0.40, 0.60, 1.00):
        row = f"  {s * 100:5.0f} cm |"
        for t in tilts:
            p = float(jet_impact(cfg, floor, jnp.array(2.0), jnp.array(3.0),
                                 jnp.array(s), jnp.array(t), jnp.array(0.0)).p_normal)
            row += f"{p / 1e3:>8.1f}({p / d.yield_mean:>3.1f}x)"
        print(row)

    section("Jet at the floor: tangential push (N) -- what moves the slurry")
    print("  standoff |" + "".join(f"{np.degrees(t):>10.0f} deg" for t in tilts))
    for s in (0.08, 0.15, 0.25, 0.40, 0.60, 1.00):
        row = f"  {s * 100:5.0f} cm |"
        for t in tilts:
            im = jet_impact(cfg, floor, jnp.array(2.0), jnp.array(3.0),
                            jnp.array(s), jnp.array(t), jnp.array(0.0))
            push = float(jnp.sum(jnp.sqrt(im.tau_x**2 + im.tau_y**2)) * fc.cell_area)
            row += f"{push:>14.1f}"
        print(row)

    section("Single pass: effective swath (cm) and depth of cut at the centre")
    print("  One steady pass over uniformly dirty floor. Swath is the width of a")
    print("  fully-stripped band that accounts for the mass actually removed.")
    print("  This is the number to argue with: does one pass at a foot off,")
    print("  at walking pace, really take a strip that wide down that far?")
    print()
    print("  speed    |" + "".join(f"{s * 100:>14.0f} cm" for s in (0.10, 0.20, 0.30, 0.50)))
    for speed in (0.20, 0.35, 0.60, 1.00):
        row = f"  {speed:4.2f} m/s |"
        for standoff in (0.10, 0.20, 0.30, 0.50):
            swath, peak = single_pass_removal(cfg, floor, standoff, speed)
            row += f"{swath * 100:>8.1f}cm {peak * 100:>3.0f}%"
        print(row)

    section("Push: walking a loose deposit 3 m in to the trough")
    print("  A jet held STILL only shifts slurry to the edge of its own patch and")
    print("  strands it -- transport length is u*h/v_settle, and both u and h")
    print("  collapse the moment the slurry leaves the jet. So the operator walks")
    print("  with the bow wave. This is how much of it actually arrives.")
    print()
    for standoff, tilt, speed in (
        (0.20, 0.95, 0.45), (0.30, 0.95, 0.45), (0.30, 1.25, 0.45),
        (0.50, 1.25, 0.45), (0.30, 1.25, 0.20), (0.30, 1.25, 0.80),
    ):
        delivered, secs = push_delivery(cfg, floor, standoff, tilt, speed)
        reach = standoff * np.tan(tilt)
        print(f"  {standoff * 100:3.0f} cm / {np.degrees(tilt):3.0f} deg / "
              f"{speed:4.2f} m/s  (lands {reach * 100:4.0f} cm ahead)  ->  "
              f"{delivered * 100:5.1f}% reaches the trough in {secs:4.1f} s")

    section("Implied job time")
    lane = 0.25
    passes = fc.length_x / lane
    per_pass = (fc.length_y / 2) / 0.45
    one_side = passes * (per_pass + (fc.length_y / 2) / 1.0)
    print(f"  ~{passes:.0f} passes per side at {lane * 100:.0f} cm per lane")
    print(f"  ~{one_side / 60:.0f} min per side, ~{2 * one_side / 60:.0f} min "
          f"for the section, single coverage")
    print("  (Real jobs need more than single coverage; the benchmark measures")
    print("   the real number rather than assuming this one.)")
    print()


def _tip_y_for_impact(standoff, tilt, impact_y, sign=-1.0):
    """Where to stand so the jet lands at `impact_y`.

    The beam leaves the tip at `tilt` off vertical, so it strikes the slab
    `standoff * tan(tilt)` further on. At a 72 deg tilt and a 30 cm standoff
    that is most of a metre ahead of the operator -- far too much to ignore
    when aiming, in the simulator or in the bay.
    """
    return impact_y - sign * standoff * np.tan(tilt)


def single_pass_removal(cfg, floor, standoff, speed, tilt=0.5):
    """Drive the jet across a uniformly dirty strip once.

    Reports the EFFECTIVE SWATH: the width of a fully-stripped band that would
    account for the mass actually removed. That is the honest measure, because a
    close pass takes a narrow strip down to bare floor while a high one takes a
    wide strip down only part way, and a single "fraction removed" inside an
    arbitrary window cannot tell those apart.
    """
    fc = cfg.floor
    shape = (fc.nx, fc.ny)
    state = initial_fields(*shape)._replace(
        h=jnp.full(shape, 1.0e-3), bound=jnp.full(shape, cfg.dirt.load_mean)
    )
    ys = jnp.full(shape, cfg.dirt.yield_mean)

    dt = cfg.sim.physics_dt
    travel = 2.0
    n = int(travel / (speed * dt))
    y0 = fc.trough_y + 3.0
    x_fixed = fc.length_x * 0.5
    start_mass = float(jnp.sum(state.bound)) * fc.cell_area

    for k in range(n):
        impact_y = y0 - travel * k / n
        tip_y = _tip_y_for_impact(standoff, tilt, impact_y)
        im = jet_impact(cfg, floor, jnp.array(x_fixed), jnp.array(tip_y),
                        jnp.array(standoff), jnp.array(tilt), jnp.array(-jnp.pi / 2))
        state = physics_substep(cfg, floor, state, ys, im.water_source, im.coverage,
                                im.intensity, im.p_normal, im.tau_x, im.tau_y, dt)

    removed = start_mass - float(jnp.sum(state.bound)) * fc.cell_area
    swath = removed / (cfg.dirt.load_mean * travel)

    # Peak depth of cut along the centre line of the pass.
    centre = jnp.abs(floor.x - x_fixed) < fc.dx
    band = centre & (floor.y > y0 - travel + 0.3) & (floor.y < y0 - 0.3)
    peak = 1.0 - float(jnp.min(jnp.where(band, state.bound, 1e9))) / cfg.dirt.load_mean
    return swath, peak


def push_delivery(cfg, floor, standoff, tilt, speed, start_dist=3.0):
    """Walk a laid-over jet from `start_dist` out, in to the trough, and see how
    much of a loose deposit actually arrives.

    This, rather than a stationary jet, is the sweeping half of the job. A jet
    held still only moves slurry to the edge of its own patch and then strands
    it, because transport length is u*h/v_settle and both u and h collapse the
    moment the slurry leaves the jet. So the operator has to WALK with the bow
    wave, continuously re-entraining what has just dropped out. Reports what
    fraction of the deposit reaches the trough and what fraction is left
    stranded behind.
    """
    fc = cfg.floor
    shape = (fc.nx, fc.ny)
    y_blob = fc.trough_y + start_dist
    blob = (jnp.abs(floor.y - y_blob) < 0.3) & (jnp.abs(floor.x - fc.length_x / 2) < 0.5)

    state = initial_fields(*shape)._replace(
        h=jnp.where(blob, 2.0e-3, 0.0),
        deposited=jnp.where(blob, cfg.dirt.load_mean, 0.0),
    )
    ys = jnp.full(shape, cfg.dirt.yield_mean)
    start_mass = float(jnp.sum(state.deposited)) * fc.cell_area

    dt = cfg.sim.physics_dt
    # Walk the impact point from just behind the blob all the way to the trough.
    travel = start_dist + 0.4
    n = int(travel / (speed * dt))
    for k in range(n):
        impact_y = (y_blob + 0.4) - travel * k / n
        tip_y = _tip_y_for_impact(standoff, tilt, impact_y)
        im = jet_impact(cfg, floor, jnp.array(fc.length_x / 2), jnp.array(tip_y),
                        jnp.array(standoff), jnp.array(tilt), jnp.array(-jnp.pi / 2))
        state = physics_substep(cfg, floor, state, ys, im.water_source, im.coverage,
                                im.intensity, im.p_normal, im.tau_x, im.tau_y, dt)

    delivered = float(state.drained) / start_mass
    seconds = n * dt
    return delivered, seconds


if __name__ == "__main__":
    main()
