"""Experiment B: is chase_worst's failure COVERAGE or ROUTING?

My claim has been that chase_worst collapses because every jump abandons its
own water, so it never builds a film. The rival explanation is duller: it
simply covers less floor. Water-immunity is consistent with both, so the
single-seed matrix cannot separate them. These two metrics can:

  coverage_frac  fraction of land cells the impact patch EVER touched.
                 If chase_worst's coverage is low, it is a coverage failure.
  h_at_patch     time-averaged film depth WHERE THE JET IS WORKING
                 (coverage-weighted). If chase_worst works on systematically
                 drier floor at equal coverage, it is a routing failure.

Plus eta = drained / cut, the delivered fraction of what was loosened, and the
stranded loose layer left behind.
"""
import dataclasses, sys, time
import jax, jax.numpy as jnp, numpy as np
from floorclean.config import Config
from floorclean.env import CleaningEnv
from floorclean.jet import jet_impact
from floorclean.baselines import PushSweep, ChaseWorst, RandomPolicy

CAP_MIN, NSEED = 30.0, 4
GPMS = [8.0, 20.0]
base = Config()
TOTAL = int(CAP_MIN * 60 / base.sim.control_dt)


def diagnose(env, policy, seeds):
    cfg = env.cfg
    land = (env.floor.trough < 0.5).astype(jnp.float32)
    nland = jnp.sum(land)

    def single(key):
        st = env.fresh_state(key)
        carry = policy.init(env, st)
        bound0 = jnp.sum(st.fields.bound) * cfg.floor.cell_area

        def step(cs, _):
            c, s = cs
            c, a = policy.act(env, s, c)
            imp = jet_impact(cfg, env.floor, s.tip_x, s.tip_y,
                             s.standoff, s.tilt, s.azimuth)
            cov = imp.coverage * land
            wet = jnp.sum(cov * s.fields.h) / jnp.maximum(jnp.sum(cov), 1e-9)
            s2, _o, _r, _t, _tr, _i = env.step(s, a)
            return (c, s2), (cov > 0.01, wet)

        (_c, sf), (hits, wets) = jax.lax.scan(step, (carry, st), None, length=TOTAL)
        visited = jnp.any(hits, axis=0).astype(jnp.float32) * land
        boundf = jnp.sum(sf.fields.bound) * cfg.floor.cell_area
        dep = jnp.sum(sf.fields.deposited) * cfg.floor.cell_area
        cut = bound0 - boundf
        return dict(coverage=jnp.sum(visited) / nland,
                    h_at_patch=jnp.mean(wets) * 1e3,
                    cut=cut, drained=sf.fields.drained,
                    eta=sf.fields.drained / jnp.maximum(cut, 1e-9),
                    stranded=dep,
                    frac_clean=jnp.sum((
                        (sf.fields.bound + sf.fields.deposited + sf.fields.suspended)
                        < cfg.dirt.clean_threshold).astype(jnp.float32) * land) / nland)

    return jax.jit(jax.vmap(single))(seeds)


seeds = jax.vmap(jax.random.PRNGKey)(jnp.arange(NSEED))
strategies = [PushSweep(name="far_to_near"), PushSweep(name="near_to_far",
                                                       near_to_far=True),
              ChaseWorst(), RandomPolicy()]
print(f"cap {CAP_MIN:.0f}min  seeds {NSEED}  ({TOTAL} steps)", flush=True)
print(f"{'gpm':>5} {'strategy':>14} {'coverage':>9} {'h@patch mm':>11} "
      f"{'cut kg':>8} {'drained':>8} {'eta':>6} {'stranded':>9} {'clean':>7}", flush=True)
for gpm in GPMS:
    cfg = dataclasses.replace(base,
        floor=dataclasses.replace(base.floor, ambient_inflow_gpm=gpm))
    env = CleaningEnv(cfg)
    for p in strategies:
        t0 = time.time()
        r = {k: np.asarray(v) for k, v in diagnose(env, p, seeds).items()}
        m = {k: v.mean() for k, v in r.items()}
        s = {k: v.std() for k, v in r.items()}
        print(f"{gpm:5.0f} {p.name:>14} {m['coverage']:6.3f}+-{s['coverage']:.3f} "
              f"{m['h_at_patch']:8.3f}+-{s['h_at_patch']:.2f} "
              f"{m['cut']:8.2f} {m['drained']:8.2f} {m['eta']:6.3f} "
              f"{m['stranded']:9.2f} {m['frac_clean']:7.3f}   ({time.time()-t0:.0f}s)",
              flush=True)
print("DONE_B", flush=True)
