"""Overland sheet flow and three-phase sediment transport on the bay floor.

WATER: the local inertial form of the shallow-water equations (Bates, Horritt &
Fewtrell 2010), on a staggered grid, with friction taken semi-implicitly.
Discharge per unit width q = h*u is carried on cell faces and updated as

    q*   = q - g * h_face * dt * d(z+h)/dx  +  dt * tau_jet / rho
    q^n+1 = q* / (1 + g * dt * n^2 * |q*| / h_face^(7/3))

and depth follows from exact mass conservation, dh/dt = -div q + source.

Retaining INERTIA is not a refinement here, it is the point. An earlier version
of this module used the diffusive-wave approximation, which drops the
acceleration term and solves directly for a friction-balanced velocity. That
makes water leaving the jet decelerate instantly to its gravity-driven speed,
so the transport length u*h/v_settle collapses from over a metre to a few
centimetres the moment slurry passes out of the impact patch -- and a pushed
slurry could never travel more than a few inches, no matter the technique. The
whole question being asked here is how far a push carries, so a scheme that
cannot represent coasting cannot answer it.

The semi-implicit friction term is what keeps this stable in thin films, where
an explicit treatment would need an impractically small timestep.

SEDIMENT: three phases, following Hairsine-Rose.

    bound      grit adhered to the epoxy. Held by ADHESION, so only the jet's
               normal impingement breaks it loose -- kilopascals.
    deposited  grit knocked loose and settled again. Lying ON the floor, so
               ordinary tangential stress moves it -- pascals.
    suspended  grit carried by the film, advected with the water.

Suspended grit settles back into the DEPOSITED layer, never into the bound one:
grit blasted off the epoxy does not re-bond. So a stalled push costs transport,
not the cutting work already done.

Every transport term is in conservative flux form with upwinding, so water and
grit are conserved to machine precision and every gram leaving the domain is
accounted for at the trough. `tests/test_physics.py` pins this down, because the
reward's correctness depends on it: if grit could vanish into numerical
diffusion, the agent would be paid for nothing.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .config import Config
from .geometry import Floor

# Time constant for water and slurry disappearing down the trough once they
# reach it. Fast compared with everything else: the trough is the sink.
TROUGH_DRAIN_TAU = 0.4  # s


class FieldState(NamedTuple):
    """The mutable physical fields."""

    h: jnp.ndarray  # (nx, ny) film depth, m
    qx: jnp.ndarray  # (nx+1, ny) discharge per unit width on x-faces, m^2/s
    qy: jnp.ndarray  # (nx, ny+1) ditto on y-faces
    suspended: jnp.ndarray  # (nx, ny) grit in the film, kg/m^2
    bound: jnp.ndarray  # (nx, ny) grit adhered to the epoxy, kg/m^2
    deposited: jnp.ndarray  # (nx, ny) grit lying loose on the floor, kg/m^2
    drained: jnp.ndarray  # scalar, cumulative grit removed at the trough, kg


def initial_fields(nx: int, ny: int) -> FieldState:
    z2 = jnp.zeros((nx, ny))
    return FieldState(
        h=z2,
        qx=jnp.zeros((nx + 1, ny)),
        qy=jnp.zeros((nx, ny + 1)),
        suspended=z2,
        bound=z2,
        deposited=z2,
        drained=jnp.array(0.0),
    )


def _face_depth(z_l, z_r, h_l, h_r):
    """Effective flow depth at an interface (Bates et al.).

    The depth water can actually move through between two cells: the higher
    water surface minus the higher bed. Zero when the upstream surface is below
    the downstream bed, which is what stops water climbing a step it cannot
    reach.
    """
    return jnp.maximum(z_l + h_l, z_r + h_r) - jnp.maximum(z_l, z_r)


def flow_substep(cfg: Config, floor: Floor, state: FieldState,
                 tau_x, tau_y, water_source, dt: float):
    """Advance depth and discharge one step. Returns (h, qx, qy)."""
    fc = cfg.floor
    g, rho, n = cfg.gravity, cfg.water_density, fc.manning_n
    dx = fc.dx
    h, qx, qy = state.h, state.qx, state.qy

    surface = floor.z + h

    # --- x-faces (interior only; domain boundaries stay at zero flux) ------
    hf_x = _face_depth(floor.z[:-1, :], floor.z[1:, :], h[:-1, :], h[1:, :])
    hf_x = jnp.maximum(hf_x, 0.0)
    slope_x = (surface[1:, :] - surface[:-1, :]) / dx
    tau_face_x = 0.5 * (tau_x[:-1, :] + tau_x[1:, :])

    q_int = qx[1:-1, :]
    q_star = q_int - g * hf_x * dt * slope_x + dt * tau_face_x / rho
    denom = 1.0 + g * dt * n**2 * jnp.abs(q_star) / jnp.maximum(hf_x, fc.h_min) ** (7.0 / 3.0)
    q_new_x = q_star / denom
    # A face with no water carries no flow.
    q_new_x = jnp.where(hf_x > fc.h_min, q_new_x, 0.0)
    qx = jnp.zeros_like(qx).at[1:-1, :].set(q_new_x)

    # --- y-faces -----------------------------------------------------------
    hf_y = _face_depth(floor.z[:, :-1], floor.z[:, 1:], h[:, :-1], h[:, 1:])
    hf_y = jnp.maximum(hf_y, 0.0)
    slope_y = (surface[:, 1:] - surface[:, :-1]) / dx
    tau_face_y = 0.5 * (tau_y[:, :-1] + tau_y[:, 1:])

    q_int = qy[:, 1:-1]
    q_star = q_int - g * hf_y * dt * slope_y + dt * tau_face_y / rho
    denom = 1.0 + g * dt * n**2 * jnp.abs(q_star) / jnp.maximum(hf_y, fc.h_min) ** (7.0 / 3.0)
    q_new_y = q_star / denom
    q_new_y = jnp.where(hf_y > fc.h_min, q_new_y, 0.0)
    qy = jnp.zeros_like(qy).at[:, 1:-1].set(q_new_y)

    # --- limit the implied velocity for advective stability ---------------
    # The gravity-wave speed in a millimetre film is tiny, so the binding CFL
    # constraint is the advective one. Clip discharge rather than let a
    # close-range jet outrun the timestep.
    #
    # The limit must use the UPWIND depth, not the larger of the two
    # neighbours. A face between a shallow cell and a deep one would otherwise
    # be allowed to carry more than the shallow cell actually holds, driving it
    # negative; clamping that back to zero then CREATES water and grit out of
    # nothing, which is exactly the class of bug that made the old simulator's
    # reward signal meaningless.
    #
    # The per-face Courant number is at most `cfl / 2`, not `cfl`: in 2D a cell
    # can drain through FOUR faces at once, so a per-face limit of `cfl` allows
    # up to 4*cfl*h of outflow per substep (1.8*h at cfl=0.45) and the
    # positivity clamp below starts manufacturing water and sediment. Halving
    # keeps total outflow under one cell volume however the faces combine.
    u_max = cfg.sim.cfl * dx / dt / 2.0
    hx = _upwind_depth_x(h, qx)
    hy = _upwind_depth_y(h, qy)
    qx = jnp.clip(qx, -u_max * hx, u_max * hx)
    qy = jnp.clip(qy, -u_max * hy, u_max * hy)

    # --- continuity (exactly conservative) ---------------------------------
    div = (qx[1:, :] - qx[:-1, :]) / dx + (qy[:, 1:] - qy[:, :-1]) / dx
    h = h - dt * div + dt * water_source
    h = jnp.maximum(h, 0.0)

    return h, qx, qy


def _upwind_depth_x(h, q):
    """Depth of the cell each x-face is draining FROM, given the flow direction.

    Boundary faces carry no flux, so their value is irrelevant; the neighbouring
    cell's depth is used to keep the array shape right.
    """
    inner = jnp.where(q[1:-1, :] > 0.0, h[:-1, :], h[1:, :])
    return jnp.concatenate([h[:1, :], inner, h[-1:, :]], axis=0)


def _upwind_depth_y(h, q):
    inner = jnp.where(q[:, 1:-1] > 0.0, h[:, :-1], h[:, 1:])
    return jnp.concatenate([h[:, :1], inner, h[:, -1:]], axis=1)


def _advect_sediment(susp, h, qx, qy, dx, h_min, dt):
    """Move depth-integrated sediment with the face discharges, conservatively.

    The transported quantity is the volumetric concentration susp/h; multiplying
    by the face discharge q (m^2/s) gives a sediment flux per unit width. Upwind
    so it stays positive, and in flux form so interior terms telescope and the
    total can only change through the (zero) domain boundaries.
    """
    conc = susp / jnp.maximum(h, h_min)

    c_up_x = jnp.where(qx[1:-1, :] > 0.0, conc[:-1, :], conc[1:, :])
    flux_x = jnp.zeros_like(qx).at[1:-1, :].set(qx[1:-1, :] * c_up_x)

    c_up_y = jnp.where(qy[:, 1:-1] > 0.0, conc[:, :-1], conc[:, 1:])
    flux_y = jnp.zeros_like(qy).at[:, 1:-1].set(qy[:, 1:-1] * c_up_y)

    div = (flux_x[1:, :] - flux_x[:-1, :]) / dx + (flux_y[:, 1:] - flux_y[:, :-1]) / dx
    return jnp.maximum(susp - dt * div, 0.0)


def physics_substep(
    cfg: Config,
    floor: Floor,
    state: FieldState,
    yield_stress: jnp.ndarray,
    water_source: jnp.ndarray,
    coverage: jnp.ndarray,
    intensity: jnp.ndarray,
    p_normal: jnp.ndarray,
    tau_x: jnp.ndarray,
    tau_y: jnp.ndarray,
    dt: float,
) -> FieldState:
    """Advance the fields by one explicit substep."""
    fc, dc = cfg.floor, cfg.dirt
    susp, bound, dep = state.suspended, state.bound, state.deposited

    # --- water -------------------------------------------------------------
    h, qx, qy = flow_substep(cfg, floor, state, tau_x, tau_y, water_source, dt)

    # --- sediment advection ------------------------------------------------
    susp = _advect_sediment(susp, state.h, qx, qy, fc.dx, fc.h_min, dt)

    # --- breaking ADHERED grit loose ---------------------------------------
    # Only the jet's normal impingement can do this, and only where it lands.
    #
    # Two things have to be true at once and they pull against each other. The
    # Gaussian wings of the patch sit below the adhesion threshold and must not
    # cut; but the patch is far SMALLER than a grid cell (a 25 deg fan is about
    # 4 mm thick against a 70 mm cell), so nothing about it can be evaluated at
    # cell centres. Sampling the profile at the cell centre puts the sample ~10
    # sigma off the patch, reads essentially zero, and stops the jet cutting at
    # all -- while using the peak pressure everywhere instead lets the wings
    # cut when they should not.
    #
    # Both are avoided by integrating analytically over the patch. For a
    # Gaussian of peak P and threshold Y, the excess pressure integrated over
    # the region where it exceeds the threshold has a closed form:
    #
    #   INT max(0, P*exp(-s/2) - Y) dA  =  A_patch * (P - Y - Y*ln(P/Y))
    #
    # (integrating to s_max = 2*ln(P/Y), where the profile drops to Y). So the
    # effective excess is G = P - Y - Y*ln(P/Y), which falls smoothly to zero as
    # P approaches Y and tends to P when P >> Y. Spreading that over cells with
    # `coverage` -- which integrates to the physical patch area -- makes the
    # total removal exactly right and independent of dx.
    #
    # NOTE: `intensity` (the true per-cell profile) is intentionally NOT used
    # here -- the patch is far thinner than a cell, so per-cell sampling is a
    # grid-alignment lottery. It is carried for render/diagnostics only.
    ratio = p_normal / jnp.maximum(yield_stress, 1.0)
    g_excess = jnp.where(
        ratio > 1.0,
        p_normal - yield_stress - yield_stress * jnp.log(jnp.maximum(ratio, 1.0)),
        0.0,
    )
    from_bound = jnp.minimum(dc.entrainment_rate * g_excess * coverage * dt, bound)

    # --- sweeping up the LOOSE deposited layer ------------------------------
    # This responds to tangential stress: the film's own bed shear plus
    # whatever the jet is pushing with. A laid-over wand generates a large
    # traction over a wide patch, which is the sweeping action, and it works at
    # a threshold three orders of magnitude below adhesion.
    speed = _cell_speed(h, qx, qy, fc.h_min)
    tau_bed = rho_g_n2(cfg) * speed**2 / jnp.maximum(h, fc.h_min) ** (1.0 / 3.0)
    tau_total = tau_bed + jnp.sqrt(tau_x**2 + tau_y**2)

    wet = jnp.clip(h / (2.0 * fc.h_min), 0.0, 1.0)
    excess_tan = jnp.maximum(tau_total - dc.deposit_threshold, 0.0)
    from_dep = jnp.minimum(dc.deposit_entrainment_rate * excess_tan * wet * dt, dep)

    # --- deposition --------------------------------------------------------
    # Settling velocity times volumetric concentration. A film that thins or
    # stalls dumps its load, which is what punishes pushing slurry further than
    # its water will carry it. It returns to the LOOSE layer, not the adhered
    # one: blasted-off grit does not re-bond.
    concentration = susp / jnp.maximum(h, fc.h_min)
    settling = jnp.minimum(dc.settling_velocity * concentration * dt, susp)

    bound = bound - from_bound
    dep = dep - from_dep + settling
    susp = susp + from_bound + from_dep - settling

    # --- the trough: everything that reaches it is gone --------------------
    drain = floor.trough * (1.0 - jnp.exp(-dt / TROUGH_DRAIN_TAU))

    # Settled grit in the trough only leaves when there is water running over it
    # to flush it away. Without this gate, grit lying at the lip would drain on
    # its own -- free progress for an agent that does nothing.
    flush = jnp.clip(h / (4.0 * fc.h_min), 0.0, 1.0)
    removed = susp * drain + dep * drain * flush

    susp = susp - susp * drain
    dep = dep - dep * drain * flush
    h = h - h * drain

    drained = state.drained + jnp.sum(removed) * fc.cell_area

    return FieldState(h=h, qx=qx, qy=qy, suspended=susp, bound=bound,
                      deposited=dep, drained=drained)


def rho_g_n2(cfg: Config):
    return cfg.water_density * cfg.gravity * cfg.floor.manning_n**2


def _cell_speed(h, qx, qy, h_min):
    """Cell-centred flow speed from the staggered discharges."""
    ux = 0.5 * (qx[:-1, :] + qx[1:, :]) / jnp.maximum(h, h_min)
    uy = 0.5 * (qy[:, :-1] + qy[:, 1:]) / jnp.maximum(h, h_min)
    return jnp.sqrt(ux**2 + uy**2)


def cell_velocity(state: FieldState, cfg: Config):
    """Cell-centred velocity components, for rendering and diagnostics."""
    hm = cfg.floor.h_min
    h = jnp.maximum(state.h, hm)
    return 0.5 * (state.qx[:-1, :] + state.qx[1:, :]) / h, \
           0.5 * (state.qy[:, :-1] + state.qy[:, 1:]) / h


def residual_map(state: FieldState) -> jnp.ndarray:
    """Areal grit loading per cell, kg/m^2 -- what 'clean' is judged on.

    All three phases count: grit lying loose is still dirt, and grit in
    suspension settles back out the moment the water stops.
    """
    return state.bound + state.deposited + state.suspended


def total_dirt(state: FieldState, cfg: Config) -> jnp.ndarray:
    """Grit still on the floor, kg."""
    return jnp.sum(residual_map(state)) * cfg.floor.cell_area
