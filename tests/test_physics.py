"""Physics correctness tests.

The point of these is to pin down the properties that make the reward signal
meaningful. The previous simulation advected dirt with a non-conservative
scheme, so total dirt drifted on its own and the agent was partly rewarded for
numerical error. Conservation is tested to tight tolerance here so that a
"dirt removed" reward can only be earned by actually moving grit to the trough.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from floorclean.config import Config, FloorConfig
from floorclean.geometry import Floor, build_floor, initial_dirt
from floorclean.jet import jet_impact
from floorclean.physics import initial_fields, physics_substep, total_dirt


def _sealed(floor: Floor) -> Floor:
    """Same floor with the trough sink switched off, for conservation tests."""
    return floor._replace(trough=jnp.zeros_like(floor.trough))


def _zero_jet(shape):
    z = jnp.zeros(shape)
    return z, z, z, jnp.array(0.0), z, z  # source, coverage, intensity, p_normal, tau_x, tau_y


def test_water_is_conserved_when_sealed():
    cfg = Config()
    floor = _sealed(build_floor(cfg))
    shape = (cfg.floor.nx, cfg.floor.ny)

    key = jax.random.PRNGKey(0)
    h0 = 2.0e-3 + 1.0e-3 * jax.random.uniform(key, shape)
    state = initial_fields(*shape)._replace(h=h0)
    src, cov, inten, p, tx, ty = _zero_jet(shape)
    ys = jnp.full(shape, cfg.dirt.yield_mean)

    start = jnp.sum(state.h)
    for _ in range(200):
        state = physics_substep(cfg, floor, state, ys, src, cov, inten, p, tx, ty,
                                cfg.sim.physics_dt)

    rel = abs(float(jnp.sum(state.h) - start) / float(start))
    assert rel < 1e-5, f"water mass drifted by {rel:.2e}"


def test_sediment_is_conserved():
    """bound + suspended + drained must be invariant, trough included."""
    cfg = Config()
    floor = build_floor(cfg)
    shape = (cfg.floor.nx, cfg.floor.ny)

    key = jax.random.PRNGKey(1)
    bound, ys = initial_dirt(key, cfg, floor)
    h0 = jnp.full(shape, 3.0e-3)
    state = initial_fields(*shape)._replace(h=h0, bound=bound)

    # Run a real jet over the floor so entrainment, advection, deposition and
    # the trough sink are all exercised. Work just uphill of the trough and aim
    # at it, so grit actually reaches the sink inside the test's horizon.
    start = float(total_dirt(state, cfg))
    y_work = cfg.floor.trough_y + 0.5

    for k in range(400):
        tip_x = jnp.array(1.0 + 0.004 * k)
        # azimuth -pi/2 points toward -y, i.e. down the slope to the trough.
        impact = jet_impact(cfg, floor, tip_x, jnp.array(y_work), jnp.array(0.25),
                            jnp.array(0.9), jnp.array(-jnp.pi / 2))
        state = physics_substep(
            cfg, floor, state, ys, impact.water_source, impact.coverage,
            impact.intensity, impact.p_normal, impact.tau_x, impact.tau_y,
            cfg.sim.physics_dt)

    end = float(total_dirt(state, cfg)) + float(state.drained)
    rel = abs(end - start) / start
    assert rel < 1e-4, f"sediment budget drifted by {rel:.2e} ({start} -> {end})"
    # And the run must actually have done something, or the test is vacuous.
    assert float(state.drained) > 0.0, "no grit reached the trough; test is vacuous"


def test_water_runs_downhill_into_the_trough():
    cfg = Config()
    floor = build_floor(cfg)
    shape = (cfg.floor.nx, cfg.floor.ny)

    # A puddle up on the side slope, 1.5 m out from the trough. Placed by
    # coordinate rather than index so the test survives a resize of the bay.
    y_start = cfg.floor.trough_y + 1.5
    patch = (jnp.abs(floor.y - y_start) < 0.3) & (jnp.abs(floor.x - 2.0) < 0.5)
    h = jnp.where(patch, 6.0e-3, 0.0)
    assert float(jnp.sum(h)) > 0.0, "test patch landed outside the domain"

    state = initial_fields(*shape)._replace(h=h)
    src, cov, inten, p, tx, ty = _zero_jet(shape)
    ys = jnp.full(shape, cfg.dirt.yield_mean)

    y_centroid_0 = float(jnp.sum(state.h * floor.y) / jnp.sum(state.h))
    # Sheet flow on a 1.5% grade runs at roughly 0.2 m/s, so 1.5 m of travel
    # needs on the order of 10 s of simulated time.
    for _ in range(1600):
        state = physics_substep(cfg, floor, state, ys, src, cov, inten, p, tx, ty,
                                cfg.sim.physics_dt)

    remaining = float(jnp.sum(state.h))
    if remaining > 1e-9:
        y_centroid_1 = float(jnp.sum(state.h * floor.y) / jnp.sum(state.h))
        assert y_centroid_1 < y_centroid_0, "water did not move toward the trough"
    assert remaining < float(jnp.sum(h)), "no water drained away"


def test_jet_pressure_falls_with_standoff():
    """Closer is harder-hitting: the core of 'standoff matters most'."""
    cfg = Config()
    floor = build_floor(cfg)
    args = dict(tilt=jnp.array(0.0), azimuth=jnp.array(0.0))

    pressures = [
        float(jet_impact(cfg, floor, jnp.array(2.0), jnp.array(1.5),
                         jnp.array(s), **args).p_normal)
        for s in (0.10, 0.25, 0.50, 1.00)
    ]
    assert all(a > b for a, b in zip(pressures, pressures[1:])), pressures
    # The fall-off must be steep enough to matter: 10 cm should hit far harder
    # than 1 m, not marginally harder.
    assert pressures[0] / pressures[-1] > 20.0, pressures


def test_tilt_trades_digging_for_pushing():
    """Standing the wand up digs; laying it over pushes."""
    cfg = Config()
    floor = build_floor(cfg)

    upright = jet_impact(cfg, floor, jnp.array(2.0), jnp.array(1.5),
                         jnp.array(0.3), jnp.array(0.0), jnp.array(0.0))
    laid_over = jet_impact(cfg, floor, jnp.array(2.0), jnp.array(1.5),
                           jnp.array(0.3), jnp.array(1.2), jnp.array(0.0))

    assert float(upright.p_normal) > float(laid_over.p_normal)

    push_up = float(jnp.sum(jnp.abs(upright.tau_x)))
    push_over = float(jnp.sum(jnp.abs(laid_over.tau_x)))
    assert push_over > push_up, (push_up, push_over)


def test_jet_pressure_is_grid_independent():
    """Pressure comes from the physical footprint, not the cell size.

    If this fails, every calibrated number becomes a function of `dx` and the
    conclusions would not survive a resolution change.
    """
    pressures = []
    for dx in (0.08, 0.05, 0.025):
        cfg = Config(floor=FloorConfig(dx=dx))
        floor = build_floor(cfg)
        p = jet_impact(cfg, floor, jnp.array(2.0), jnp.array(1.5),
                       jnp.array(0.3), jnp.array(0.5), jnp.array(0.0)).p_normal
        pressures.append(float(p))

    spread = (max(pressures) - min(pressures)) / np.mean(pressures)
    assert spread < 1e-6, f"pressure depends on dx: {pressures}"


def test_water_delivery_matches_pump_flow():
    """The rasterised source must deliver exactly the pump's gpm, no more."""
    cfg = Config()
    floor = build_floor(cfg)
    impact = jet_impact(cfg, floor, jnp.array(2.0), jnp.array(1.5),
                        jnp.array(0.3), jnp.array(0.4), jnp.array(1.0))
    delivered = float(jnp.sum(impact.water_source) * cfg.floor.cell_area)
    assert delivered == pytest.approx(cfg.washer.flow, rel=2e-2), (
        delivered, cfg.washer.flow)


def test_stalled_slurry_redeposits():
    """Grit pushed somewhere and abandoned must settle back out."""
    cfg = Config()
    floor = _sealed(build_floor(cfg))
    shape = (cfg.floor.nx, cfg.floor.ny)

    state = initial_fields(*shape)._replace(
        h=jnp.full(shape, 1.0e-3), suspended=jnp.full(shape, 0.1)
    )
    src, cov, inten, p, tx, ty = _zero_jet(shape)
    # Adhesion irrelevant here; the film is still so nothing is re-entrained.
    ys = jnp.full(shape, 1e9)

    for _ in range(100):
        state = physics_substep(cfg, floor, state, ys, src, cov, inten, p, tx, ty,
                                cfg.sim.physics_dt)

    # It must land in the LOOSE layer, not re-bond to the epoxy: grit that has
    # been blasted off does not stick again just because the water stopped.
    assert float(jnp.sum(state.deposited)) > 0.0, "suspended grit never settled"
    assert float(jnp.sum(state.bound)) == 0.0, "settled grit wrongly re-adhered"
    assert float(jnp.sum(state.suspended)) < float(0.1 * shape[0] * shape[1])


def test_four_way_outflow_creates_no_water():
    """The CFL clip must be positivity-preserving in 2D.

    A cell draining through all four faces at the clip limit loses at most one
    cell volume per substep; the old per-face `cfl` bound allowed 4*cfl*h
    (1.8*h) and the positivity clamp manufactured water. Regression test for
    WORKBOARD B4: an adversarial radial burst over one wet cell must conserve
    water exactly on a sealed floor.
    """
    from floorclean.physics import flow_substep
    cfg = Config()
    floor = _sealed(build_floor(cfg))
    shape = (cfg.floor.nx, cfg.floor.ny)
    cx, cy = shape[0] // 2, shape[1] // 2

    h = jnp.zeros(shape).at[cx, cy].set(2e-3)
    state = initial_fields(*shape)._replace(h=h)
    big = 5000.0  # Pa, saturates the clip on every face around the cell
    tx = jnp.zeros(shape).at[cx + 1, cy].set(big).at[cx - 1, cy].set(-big)
    ty = jnp.zeros(shape).at[cx, cy + 1].set(big).at[cx, cy - 1].set(-big)

    start = float(jnp.sum(state.h))
    h1, _, _ = flow_substep(cfg, floor, state, tx, ty, jnp.zeros(shape),
                            cfg.sim.physics_dt)
    rel = abs(float(jnp.sum(h1)) - start) / start
    assert rel < 1e-6, f"clip manufactured water: rel drift {rel:.2e}"


def test_four_way_outflow_creates_no_sediment():
    """Same as above for grit, through the full substep (WORKBOARD B4).

    The sediment guarantee comes from the flow clip: if `flow_substep` hands
    `_advect_sediment` face flows whose combined outflow exceeds the donor's
    mass, the positivity clamp manufactures grit. An adversarial radial burst
    over a loaded cell must conserve bound + deposited + suspended + drained.
    """
    cfg = Config()
    floor = _sealed(build_floor(cfg))
    shape = (cfg.floor.nx, cfg.floor.ny)
    cx, cy = shape[0] // 2, shape[1] // 2

    h = jnp.zeros(shape).at[cx, cy].set(2e-3)
    susp = jnp.zeros(shape).at[cx, cy].set(0.5)
    state = initial_fields(*shape)._replace(h=h, suspended=susp)
    big = 5000.0  # Pa, saturates the clip on every face around the cell
    tx = jnp.zeros(shape).at[cx + 1, cy].set(big).at[cx - 1, cy].set(-big)
    ty = jnp.zeros(shape).at[cx, cy + 1].set(big).at[cx, cy - 1].set(-big)
    src, cov, inten = jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape)
    p = jnp.array(0.0)
    ys = jnp.full(shape, 1e9)  # nothing re-entrains; pure advection + settling

    start = float(total_dirt(state, cfg))
    for _ in range(5):
        state = physics_substep(cfg, floor, state, ys, src, cov, inten, p, tx, ty,
                                cfg.sim.physics_dt)
    end = float(total_dirt(state, cfg)) + float(state.drained)
    rel = abs(end - start) / start
    assert rel < 1e-6, f"clip manufactured grit: rel drift {rel:.2e}"


def test_entrainment_integrates_patch_excess():
    """Bound removal equals the analytic patch integral (WORKBOARD B1).

    For a Gaussian peak P over a threshold Y, INT max(0, P*e - Y) dA over the
    above-threshold region is exactly A_patch * (P - Y - Y*ln(P/Y)). Spread
    over cells with `coverage` (which integrates to the physical patch area),
    one substep must remove rate * G * sum(coverage)*cell_area * dt in total.
    The old `max(P - Y, 0)` form overestimated this by ~3x at working heights
    (it gave the wings full peak excess); a below-threshold jet must remove
    nothing at all.
    """
    import math
    cfg = Config()
    floor = _sealed(build_floor(cfg))
    shape = (cfg.floor.nx, cfg.floor.ny)

    impact = jet_impact(cfg, floor, jnp.array(2.0), jnp.array(7.5),
                        jnp.array(0.30), jnp.array(0.5), jnp.array(0.0))
    peak = float(impact.p_normal)
    assert peak > cfg.dirt.yield_mean, "test jet cannot cut at all"

    def removed_after_one_step(yield_value):
        state = initial_fields(*shape)._replace(
            h=jnp.full(shape, 1.0e-3),
            bound=jnp.full(shape, cfg.dirt.load_mean),
        )
        ys = jnp.full(shape, yield_value)
        start = float(jnp.sum(state.bound)) * cfg.floor.cell_area
        state = physics_substep(
            cfg, floor, state, ys, impact.water_source, impact.coverage,
            impact.intensity, impact.p_normal, impact.tau_x, impact.tau_y,
            cfg.sim.physics_dt)
        return start - float(jnp.sum(state.bound)) * cfg.floor.cell_area

    Y = cfg.dirt.yield_mean
    G = peak - Y - Y * math.log(peak / Y)
    area = float(jnp.sum(jnp.asarray(impact.coverage))) * cfg.floor.cell_area
    expected = cfg.dirt.entrainment_rate * G * area * cfg.sim.physics_dt
    # 5e-3 tolerance: float32 accumulation on a 4e-4 kg quantity. The old
    # max(P - Y, 0) form sits ~40% higher, far outside this band.
    assert removed_after_one_step(Y) == pytest.approx(expected, rel=5e-3)
    # Below threshold: not a gram moves.
    assert removed_after_one_step(peak * 2.0) == pytest.approx(0.0, abs=1e-9)
