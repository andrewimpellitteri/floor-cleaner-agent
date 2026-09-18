"""Impingement model for a fan-tip pressure washer.

This is the module that decides what technique is optimal, so it is worth being
explicit about the physics.

The wand tip sits at horizontal position `(tip_x, tip_y)`, a height `standoff`
above the slab, tilted `tilt` away from vertical, pointed along `azimuth`. The
beam therefore travels a slant distance

    L = standoff / cos(tilt)

and lands at a horizontal offset `standoff * tan(tilt)` in the azimuth
direction. Two things happen at the impact patch:

  * NORMAL momentum digs grit out of the concrete texture. The fan spreads
    linearly with L, so the pressure delivered to the slab falls off roughly as
    1/L^2, and the oblique incidence costs another factor of cos(tilt): the
    normal pressure goes as cos^2(tilt) / L^2. Standing the wand up and getting
    close is how you break grit loose.

  * TANGENTIAL momentum pushes the water film, and that scales as sin(tilt).
    Laying the wand over is how you move a slurry to the trough.

You cannot have both at once, and that tension is the whole problem. The
optimal policy has to interleave the two -- which is exactly the argument worth
settling with a simulation rather than in the break room.

The impact patch of a 40 deg fan at a 30 cm standoff is about 22 cm x 3 cm,
which is thinner than one 5 cm grid cell. So pressure is computed from the TRUE
physical footprint area and only then rasterised onto the grid via an area-
preserving coverage fraction. Results are therefore independent of `dx`, which
they would not be if the footprint were simply painted onto cells.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .config import Config
from .geometry import Floor


class JetImpact(NamedTuple):
    water_source: jnp.ndarray  # (nx, ny) film depth added, m/s
    coverage: jnp.ndarray  # (nx, ny) fraction of cell under the impact patch
    tau_x: jnp.ndarray  # (nx, ny) tangential traction on the film, N/m^2
    tau_y: jnp.ndarray
    p_normal: jnp.ndarray  # scalar, impingement pressure at the patch, Pa
    impact_x: jnp.ndarray  # scalar, m
    impact_y: jnp.ndarray


# Fraction of the normal momentum that turns into radial wall-jet outflow
# instead of being absorbed by the slab. A vertical jet blows a visible crater
# in a puddle; this is that effect.
WALL_JET_COEFF = 0.35


def jet_impact(
    cfg: Config,
    floor: Floor,
    tip_x: jnp.ndarray,
    tip_y: jnp.ndarray,
    standoff: jnp.ndarray,
    tilt: jnp.ndarray,
    azimuth: jnp.ndarray,
) -> JetImpact:
    """Compute the footprint, pressure and traction of the jet on the slab.

    All of `tip_x, tip_y, standoff, tilt, azimuth` are scalars. Use `vmap` for
    a batch of environments.
    """
    wc = cfg.washer
    dx = cfg.floor.dx

    cos_t = jnp.cos(tilt)
    sin_t = jnp.sin(tilt)
    # Guard the grazing-incidence singularity: past tilt_max the beam would
    # never reach the floor at a usable distance.
    cos_t = jnp.maximum(cos_t, jnp.cos(wc.tilt_max))

    slant = standoff / cos_t
    impact_x = tip_x + standoff * (sin_t / cos_t) * jnp.cos(azimuth)
    impact_y = tip_y + standoff * (sin_t / cos_t) * jnp.sin(azimuth)

    # Jet velocity after air drag, and the momentum it still carries.
    v_local = wc.jet_velocity * jnp.exp(-slant / wc.velocity_decay_length)
    momentum = cfg.water_density * wc.flow * v_local  # N

    # Physical footprint: a fan sheet, long across the azimuth and thin along
    # it, stretched by 1/cos(tilt) because it strikes obliquely.
    width = 2.0 * slant * jnp.tan(wc.fan_angle * 0.5)
    thickness = wc.fan_thickness_0 + 2.0 * slant * jnp.tan(wc.fan_spread_angle * 0.5)
    thickness = thickness / cos_t

    # Gaussian stand-in for the fan's intensity profile: +/-2 sigma spans the
    # nominal extent. sigma_u lies across the azimuth, sigma_v along it.
    sigma_u_phys = jnp.maximum(width * 0.25, 1e-4)
    sigma_v_phys = jnp.maximum(thickness * 0.25, 1e-4)
    area_phys = 2.0 * jnp.pi * sigma_u_phys * sigma_v_phys

    # Normal pressure delivered to the slab over the true footprint.
    p_normal = momentum * cos_t / area_phys

    # Rasterisation: widen the kernel to at least half a cell so the patch is
    # representable on the grid, but keep the total covered area equal to the
    # physical one. Pressure above was already computed from the physical area,
    # so the grid resolution does not change how hard the jet hits.
    sigma_u = jnp.maximum(sigma_u_phys, 0.5 * dx)
    sigma_v = jnp.maximum(sigma_v_phys, 0.5 * dx)

    d_x, d_y = jnp.cos(azimuth), jnp.sin(azimuth)  # along the push direction
    p_x, p_y = -jnp.sin(azimuth), jnp.cos(azimuth)  # across the fan

    rx = floor.x - impact_x
    ry = floor.y - impact_y
    u = rx * p_x + ry * p_y
    v = rx * d_x + ry * d_y

    # Area-normalised kernel: integrates to 1 over the slab.
    kernel = jnp.exp(-0.5 * ((u / sigma_u) ** 2 + (v / sigma_v) ** 2)) / (
        2.0 * jnp.pi * sigma_u * sigma_v
    )

    # Fraction of each cell under the patch, capped at full coverage.
    coverage = jnp.minimum(kernel * area_phys, 1.0)

    # Water delivery: the pump's whole flow lands inside the patch.
    water_source = wc.flow * kernel

    # Tangential traction: the in-plane component of the jet's momentum, plus
    # the share of the normal component that turns into radial wall-jet flow.
    tau_along = momentum * sin_t * kernel  # N/m^2
    tau_x = tau_along * d_x
    tau_y = tau_along * d_y

    radial = momentum * cos_t * WALL_JET_COEFF * kernel
    rr = jnp.sqrt(rx**2 + ry**2) + 1e-6
    tau_x = tau_x + radial * (rx / rr)
    tau_y = tau_y + radial * (ry / rr)

    return JetImpact(
        water_source=water_source,
        coverage=coverage,
        tau_x=tau_x,
        tau_y=tau_y,
        p_normal=p_normal,
        impact_x=impact_x,
        impact_y=impact_y,
    )
