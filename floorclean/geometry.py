"""Static floor geometry and randomised initial dirt fields.

The bay section is sloped in two directions into a central trough, matching the
real floor: fall from both side walls toward the middle, and a gentler fall
along the trough toward the drain end.

All functions here are pure JAX and safe to `jit`/`vmap`. Fields are indexed
`[i, j]` with `x = i*dx`, `y = j*dx` (see config.py).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import Config


class Floor(NamedTuple):
    """Static, episode-invariant geometry. Built once and shared by every env."""

    z: jnp.ndarray  # (nx, ny) floor elevation, m
    dzdx: jnp.ndarray  # (nx, ny) floor slope, m/m
    dzdy: jnp.ndarray
    trough: jnp.ndarray  # (nx, ny) in [0,1], fraction of cell inside the trough
    x: jnp.ndarray  # (nx, ny) cell-centre coordinates, m
    y: jnp.ndarray


def build_floor(cfg: Config) -> Floor:
    """Construct the elevation field for the bay section."""
    fc = cfg.floor
    nx, ny, dx = fc.nx, fc.ny, fc.dx

    xs = (jnp.arange(nx) + 0.5) * dx
    ys = (jnp.arange(ny) + 0.5) * dx
    x, y = jnp.meshgrid(xs, ys, indexing="ij")

    dist_from_trough = jnp.abs(y - fc.trough_y)

    # Fall toward the trough from both sides, and along the trough toward the
    # drain at the +x end.
    z = fc.cross_slope * dist_from_trough + fc.trough_slope * (fc.length_x - x)

    # Carve the trough. A 15 cm channel is only three cells wide, so a sharp
    # edge is not resolvable and would only feed the flow solver a cliff.
    # Smooth it over ~1.5 cells, which is honest at this resolution.
    edge = 1.5 * dx
    basin = 0.5 * (1.0 + jnp.tanh((0.5 * fc.trough_width - dist_from_trough) / edge))
    z = z - fc.trough_depth * basin

    dzdx, dzdy = _gradient(z, dx)
    return Floor(z=z, dzdx=dzdx, dzdy=dzdy, trough=basin, x=x, y=y)


def _gradient(field: jnp.ndarray, dx: float):
    """Central-difference gradient with one-sided differences at the edges."""
    gx = jnp.gradient(field, dx, axis=0)
    gy = jnp.gradient(field, dx, axis=1)
    return gx, gy


def smooth_random_field(
    key: jax.Array,
    shape: tuple[int, int],
    correlation_length: float | tuple[float, float],
    dx: float,
) -> jnp.ndarray:
    """Zero-mean, unit-variance Gaussian random field with a given correlation length.

    Built in Fourier space so the cost is independent of the correlation length
    and the result is smooth rather than pixel noise. Pass a `(Lx, Ly)` pair for
    an anisotropic field -- used for the worn traffic lanes, which are long
    along the bay and narrow across it.
    """
    nx, ny = shape
    noise = jax.random.normal(key, (nx, ny))

    if isinstance(correlation_length, tuple):
        lx, ly = correlation_length
    else:
        lx = ly = correlation_length

    kx = jnp.fft.fftfreq(nx, d=dx) * 2.0 * jnp.pi
    ky = jnp.fft.fftfreq(ny, d=dx) * 2.0 * jnp.pi

    # Gaussian-correlated field: amplitude filter exp(-(kx^2 Lx^2 + ky^2 Ly^2)/4).
    filt = jnp.exp(-0.25 * (kx[:, None] ** 2 * lx**2 + ky[None, :] ** 2 * ly**2))
    field = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(noise) * filt))

    std = jnp.std(field)
    return field / jnp.maximum(std, 1e-8)


def episode_elevation(key: jax.Array, cfg: Config, floor: Floor) -> jnp.ndarray:
    """Floor elevation for one episode: the design slope plus real-world undulation.

    A slab two years past its recoat is not a plane. It has settled and worn
    into flat spots that hold standing water. Those matter in both directions:
    the water they hold is what carries slurry, and the absence of slope is why
    dirt settles there and stays. A policy trained on a perfect plane would have
    no idea they exist.

    The undulation is resampled per episode so the policy has to READ where the
    water is sitting rather than memorise one floor. Pin the episode seed to
    model one specific bay.
    """
    fc = cfg.floor
    bumps = smooth_random_field(
        key, (fc.nx, fc.ny), fc.flat_spot_length, fc.dx
    )
    # Leave the trough itself alone: it is formed, not poured flat.
    return floor.z + fc.flat_spot_amplitude * bumps * (1.0 - floor.trough)


def initial_water(key: jax.Array, cfg: Config, z: jnp.ndarray) -> jnp.ndarray:
    """Standing water at the start of an episode.

    A uniform working film, plus ponding wherever the slab dips below its
    surroundings. Ponding is estimated as the depth below a locally smoothed
    elevation, which is a cheap stand-in for a proper depression-filling pass
    and puts the water in the right places.
    """
    fc = cfg.floor
    k_film, k_var = jax.random.split(key)

    level = jax.random.uniform(k_film, (), minval=0.4, maxval=1.6)
    film = fc.initial_film_mean * level + fc.initial_film_std * smooth_random_field(
        k_var, (fc.nx, fc.ny), 1.5, fc.dx
    )

    smoothed = _box_blur(z, radius=max(1, int(round(0.8 / fc.dx))))
    ponding = jnp.clip(smoothed - z, 0.0, fc.max_ponding)

    return jnp.maximum(film, 0.0) + ponding


def _box_blur(field: jnp.ndarray, radius: int) -> jnp.ndarray:
    """Separable moving average with edge replication."""
    k = 2 * radius + 1
    kernel = jnp.ones(k) / k
    padded = jnp.pad(field, ((radius, radius), (0, 0)), mode="edge")
    out = jnp.apply_along_axis(lambda c: jnp.convolve(c, kernel, mode="valid"), 0, padded)
    padded = jnp.pad(out, ((0, 0), (radius, radius)), mode="edge")
    return jnp.apply_along_axis(lambda c: jnp.convolve(c, kernel, mode="valid"), 1, padded)


def initial_dirt(key: jax.Array, cfg: Config, floor: Floor) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample the starting bound-grit loading and the yield-stress map.

    Returns (bound_kg_per_m2, yield_stress_pa), both shape (nx, ny).

    The loading is patchy rather than uniform -- grit collects where the pieces
    sit and gets tracked around -- and the trough starts clean because anything
    that reaches it is already gone.
    """
    dc = cfg.dirt
    shape = (cfg.floor.nx, cfg.floor.ny)
    dx = cfg.floor.dx
    k_load, k_yield, k_lane = jax.random.split(key, 3)

    # Patchy loading, correlation length ~0.6 m: blotches a couple of feet
    # across, which is what tracked-in grit actually looks like.
    load = dc.load_mean + dc.load_std * smooth_random_field(k_load, shape, 0.6, dx)
    load = jnp.maximum(load, 0.0)

    # Anything in the trough has already drained away.
    load = load * (1.0 - floor.trough)

    # Base yield stress: how deep grit sits in the anti-slip aggregate varies
    # over inches, not feet.
    ys = dc.yield_mean + dc.yield_std * smooth_random_field(k_yield, shape, 0.25, dx)

    # Worn traffic lanes. The epoxy is overdue for a recoat and is worn through
    # where pieces get dragged, so those strips hold grit much harder. Lanes run
    # along the bay, so the field is stretched in x and narrow in y.
    lane = smooth_random_field(
        k_lane, shape, (dc.worn_lane_length_x, dc.worn_lane_length_y), dx
    )
    # Only the upper tail counts as "worn through", giving distinct lanes rather
    # than a smooth wobble across the whole floor.
    ys = ys + dc.worn_lane_boost * jnp.clip(lane - 0.5, 0.0, None)

    ys = jnp.maximum(ys, dc.yield_min)

    return load, ys
