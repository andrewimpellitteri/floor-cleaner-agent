"""Experiment A: seeded mobility ladder + water-source geometry A/B.

A1: mobile fraction vs ambient gpm, 8 seeds, uniform source. Re-pins (or kills)
    the 0.26/0.37/0.51/0.69/0.80 numbers with mean+-SD, the stricter predicate
    (tau>thr AND h>2*h_min) and the trough excluded.
A2: at EQUAL total Q, uniform vs two-tap (room midpoints) vs line source
    (awnings draining on the bar). Same seeds, same z. Tests whether geometry
    changes the wetted structure, not just the amount.
"""
import dataclasses, sys, time
import jax, jax.numpy as jnp, numpy as np
from floorclean.config import Config
from floorclean.geometry import build_floor, episode_elevation, initial_water, initial_dirt
from floorclean.physics import initial_fields, physics_substep, rho_g_n2, _cell_speed

cfg = Config(); floor0 = build_floor(cfg)
fc = cfg.floor
AREA = fc.length_x * fc.length_y
GPM_M3 = 6.30902e-5
SPIN = 600.0                      # s of sim to reach steady state
NSEED = 8
land = np.asarray(floor0.trough) < 0.5      # exclude the trough itself

X = np.asarray(floor0.x); Y = np.asarray(floor0.y)

def src_uniform(gpm):
    return jnp.full((fc.nx, fc.ny), gpm * GPM_M3 / AREA)

def _gauss(cx, cy, sig):
    g = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sig ** 2))
    return g / (g.sum() * fc.cell_area)          # integrates to 1 m^-2

def src_two_tap(gpm, sig=0.15):
    # "two taps at the midpoints of the room": mid-length (x), at each wall (y).
    f = _gauss(fc.length_x * 0.5, 0.3, sig) + _gauss(fc.length_x * 0.5,
                                                     fc.length_y - 0.3, sig)
    return jnp.asarray(f * 0.5 * gpm * GPM_M3)

def src_line(gpm, sig=0.25):
    # awnings draining on the bar: a line source along x, at one wall.
    g = np.exp(-((Y - 0.4) ** 2) / (2 * sig ** 2)) * np.ones_like(X)
    g = g / (g.sum() * fc.cell_area)
    return jnp.asarray(g * gpm * GPM_M3)

SOURCES = {"uniform": src_uniform, "two_tap": src_two_tap, "line": src_line}

def steady(seed, src):
    kz, kw, kd = jax.random.split(jax.random.PRNGKey(seed), 3)
    z = episode_elevation(kz, cfg, floor0)
    bound, ys = initial_dirt(kd, cfg, floor0)
    f = initial_fields(fc.nx, fc.ny)._replace(bound=bound,
                                              h=initial_water(kw, cfg, z))
    fl = floor0._replace(z=z)
    zero = jnp.zeros((fc.nx, fc.ny))
    @jax.jit
    def ctrl(f):
        def sub(f, _):
            return physics_substep(cfg, fl, f, ys, src, zero, zero,
                                   zero, zero, zero, cfg.sim.physics_dt), None
        return jax.lax.scan(sub, f, None, length=cfg.sim.physics_substeps)[0]
    for _ in range(int(SPIN / cfg.sim.control_dt)):
        f = ctrl(f)
    h = np.asarray(f.h)
    spd = _cell_speed(f.h, f.qx, f.qy, fc.h_min)
    tau = np.asarray(rho_g_n2(cfg) * spd ** 2 / jnp.maximum(f.h, fc.h_min) ** (1 / 3))
    mob = (tau > cfg.dirt.deposit_threshold) & (h > 2 * fc.h_min)
    return dict(mobile=mob[land].mean(), mean_h=h[land].mean() * 1e3,
                med_h=np.median(h[land]) * 1e3,
                p2mm=(h[land] > 2e-3).mean(),
                gini=float(np.abs(np.subtract.outer(h[land], h[land])).mean()
                           / (2 * h[land].mean() + 1e-12)))

def report(tag, rows):
    ks = ["mobile", "mean_h", "med_h", "p2mm", "gini"]
    m = {k: np.mean([r[k] for r in rows]) for k in ks}
    s = {k: np.std([r[k] for r in rows]) for k in ks}
    print(f"{tag:>22}  " + "  ".join(
        f"{k}={m[k]:.3f}+-{s[k]:.3f}" for k in ks), flush=True)

print(f"=== A1: mobility ladder, uniform source, {NSEED} seeds, spin {SPIN:.0f}s ===",
      flush=True)
print(f"{'gpm':>22}  mobile / mean_h(mm) / med_h(mm) / P(h>2mm) / gini", flush=True)
for gpm in [0.0, 4.0, 8.0, 20.0, 45.0, 90.0]:
    t0 = time.time()
    rows = [steady(s, src_uniform(gpm)) for s in range(NSEED)]
    report(f"{gpm:.0f} gpm", rows)
    print(f"{'':>22}  ({time.time()-t0:.0f}s)", flush=True)

print(f"\n=== A2: source GEOMETRY at equal Q = 8 gpm, {NSEED} seeds ===", flush=True)
for name, fn in SOURCES.items():
    t0 = time.time()
    rows = [steady(s, fn(8.0)) for s in range(NSEED)]
    report(name, rows)
    print(f"{'':>22}  ({time.time()-t0:.0f}s)", flush=True)
print("DONE_A", flush=True)
