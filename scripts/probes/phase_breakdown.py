"""T1 phase breakdown at the CURRENT dirt loading and the CURRENT physics.

Issue #5: every "transport is the bottleneck" / "cutting is the bottleneck"
verdict in WORKBOARD T1 was measured at 2.90-8.90 kg of grit. The corrected
loading is ~1.13 kg, and since then the water sources (#1) and the ponding
trough (#3) have both changed. So nobody currently knows where the job
actually binds. This measures it.

Reports, per strategy and water level, the fate of the grit that started on
the floor: cut off the epoxy, delivered to the trough, left lying loose, and
what is actually still stuck down.
"""
import dataclasses, time
import jax, jax.numpy as jnp, numpy as np
from floorclean.config import Config
from floorclean.env import CleaningEnv
from floorclean.baselines import PushSweep, BlastThenSweep

CAP_MIN, NSEED = 30.0, 6
base = Config()
TOTAL = int(CAP_MIN * 60 / base.sim.control_dt)
seeds = jax.vmap(jax.random.PRNGKey)(jnp.arange(NSEED))


def measure(env, policy):
    cell = env.cfg.floor.cell_area

    def one(key):
        st = env.fresh_state(key)
        c = policy.init(env, st)
        bound0 = jnp.sum(st.fields.bound) * cell

        def step(cs, _):
            c, s = cs
            c, a = policy.act(env, s, c)
            s, _o, _r, _t, _tr, i = env.step(s, a)
            return (c, s), jnp.stack([i["fraction_clean"], i["worst_residual"]])

        (_c, sf), tr = jax.lax.scan(step, (c, st), None, length=TOTAL)
        f = sf.fields
        return dict(
            start_kg=bound0,
            adhered_left=jnp.sum(f.bound) * cell,
            loose_left=jnp.sum(f.deposited + f.suspended) * cell,
            drained=f.drained,
            cut_frac=1.0 - (jnp.sum(f.bound) * cell) / jnp.maximum(bound0, 1e-9),
            delivered_frac=f.drained / jnp.maximum(bound0, 1e-9),
            clean=tr[-1, 0],
            worst=tr[-1, 1],
        )

    return jax.jit(jax.vmap(one))(seeds)


print(f"cap {CAP_MIN:.0f} min, {NSEED} seeds, layout={base.floor.ambient_layout}, "
      f"retain={base.floor.trough_retain_depth*1e3:.0f}mm", flush=True)
print(f"{'gpm':>4} {'strategy':>17} {'start':>7} {'cut%':>11} {'deliv%':>11} "
      f"{'stuck':>7} {'loose':>7} {'clean%':>11}", flush=True)
for gpm in [8.0, 20.0]:
    cfg = dataclasses.replace(
        base, floor=dataclasses.replace(base.floor, ambient_inflow_gpm=gpm))
    env = CleaningEnv(cfg)
    for p in [PushSweep(name="far_to_near"), BlastThenSweep()]:
        t0 = time.time()
        r = {k: np.asarray(v) for k, v in measure(env, p).items()}
        m = {k: v.mean() for k, v in r.items()}
        s = {k: v.std() for k, v in r.items()}
        print(f"{gpm:4.0f} {p.name:>17} {m['start_kg']:7.3f} "
              f"{m['cut_frac']*100:7.1f}+-{s['cut_frac']*100:3.1f} "
              f"{m['delivered_frac']*100:7.1f}+-{s['delivered_frac']*100:3.1f} "
              f"{m['adhered_left']:7.3f} {m['loose_left']:7.3f} "
              f"{m['clean']*100:7.1f}+-{s['clean']*100:3.1f}   ({time.time()-t0:.0f}s)",
              flush=True)
print("DONE_PHASE", flush=True)
