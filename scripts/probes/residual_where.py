"""Is the irreducible 0.088 kg a PRESSURE problem or a COVERAGE problem?

If the grit that never comes off sits in high-yield-stress cells, the jet
cannot break it and the answer is pressure (standoff/tilt/dwell). If it is
scattered across ordinary cells, the passes simply never went there and the
answer is coverage (lane pattern). The two call for opposite fixes, so T1
should not be retargeted until we know which.
"""
import dataclasses
import jax, jax.numpy as jnp, numpy as np
from floorclean.config import Config
from floorclean.env import CleaningEnv
from floorclean.baselines import PushSweep

cfg = Config()
env = CleaningEnv(cfg)
TOTAL = int(30 * 60 / cfg.sim.control_dt)
p = PushSweep(name="far_to_near")
land = np.asarray(env.floor.trough) < 0.5

def run(seed):
    st = env.fresh_state(jax.random.PRNGKey(seed))
    c = p.init(env, st)
    def step(cs, _):
        c, s = cs
        c, a = p.act(env, s, c)
        s, _o, _r, _t, _tr, _i = env.step(s, a)
        return (c, s), 0.0
    (_c, sf), _ = jax.jit(lambda cs: jax.lax.scan(step, cs, None, length=TOTAL))((c, st))
    return (np.asarray(sf.fields.bound), np.asarray(st.fields.bound),
            np.asarray(st.yield_stress))

thr = cfg.dirt.clean_threshold
rows = [run(s) for s in range(4)]
print(f"clean_threshold = {thr}  yield_mean = {cfg.dirt.yield_mean:.0f} Pa\n")
for i, (bound_f, bound_0, ys) in enumerate(rows):
    m = land
    stuck = bound_f[m]
    y = ys[m]
    start = bound_0[m]
    hi = y > np.quantile(y, 0.80)          # worn lanes: top-quintile yield stress
    frac_stuck_in_lanes = stuck[hi].sum() / max(stuck.sum(), 1e-12)
    frac_area_lanes = hi.mean()
    # how much of each population survived
    surv_lane = stuck[hi].sum() / max(start[hi].sum(), 1e-12)
    surv_norm = stuck[~hi].sum() / max(start[~hi].sum(), 1e-12)
    dirty = stuck > thr
    print(f"seed {i}: stuck mass in top-20% yield cells = {frac_stuck_in_lanes*100:5.1f}% "
          f"(those cells are {frac_area_lanes*100:.0f}% of area)   "
          f"survival: lanes {surv_lane*100:5.1f}% vs rest {surv_norm*100:4.1f}%   "
          f"cells>thr {dirty.mean()*100:5.1f}%")
    if i == 0:
        q = np.quantile(stuck[dirty], [0.5, 0.9, 1.0]) / thr
        print(f"         failing-cell residual, x threshold: "
              f"median {q[0]:.1f}  p90 {q[1]:.1f}  max {q[2]:.1f}")
