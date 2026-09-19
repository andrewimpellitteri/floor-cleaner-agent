"""Best technique as a function of water supply.

Runs the scripted suite to completion at several ambient rinse rates. The
question is not only "does more water help" but "does the RANKING move" --
i.e. is the optimal technique itself a function of how wet the bay is.
"""
import dataclasses, sys, time, json
import jax, jax.numpy as jnp, numpy as np
from floorclean.config import Config
from floorclean.env import CleaningEnv
from floorclean.baselines import default_suite

CAP_MIN = 30.0
GPMS = [8.0, 20.0, 45.0, 90.0]
SEEDS = [0]
out_path = sys.argv[1] if len(sys.argv) > 1 else "water_matrix.jsonl"

def run_one(env, policy, key, total_steps, clean_thr):
    state = env.fresh_state(key)
    carry = policy.init(env, state)
    def one(cs, _):
        c, s = cs
        c, a = policy.act(env, s, c)
        s, _o, _r, _t, _tr, info = env.step(s, a)
        return (c, s), jnp.stack([info["worst_residual"], info["remaining_kg"],
                                  info["drained_kg"], info["fraction_clean"],
                                  info["water_m3"]])
    @jax.jit
    def go(cs):
        return jax.lax.scan(one, cs, None, length=total_steps)
    (c, s), tr = go((carry, state))
    tr = np.asarray(tr)
    worst, remaining, drained, frac_clean, water = tr.T
    clean = worst < clean_thr
    idx = int(np.argmax(clean)) if clean.any() else -1
    return dict(
        finished=bool(clean.any()),
        minutes=(idx + 1) * 0.2 / 60 if clean.any() else float("nan"),
        frac_clean_end=float(frac_clean[-1]),
        worst_end=float(worst[-1]),
        drained_end=float(drained[-1]),
        remaining_end=float(remaining[-1]),
        wand_gal=float(water[idx if clean.any() else -1] * 264.172),
        initial_kg=float(state.initial_mass),
    )

base = Config()
total_steps = int(CAP_MIN * 60 / base.sim.control_dt)
print(f"cap {CAP_MIN:.0f} sim-min ({total_steps} steps)  gpms={GPMS}  seeds={SEEDS}",
      flush=True)
fh = open(out_path, "w")
for gpm in GPMS:
    cfg = dataclasses.replace(base,
        floor=dataclasses.replace(base.floor, ambient_inflow_gpm=gpm))
    env = CleaningEnv(cfg)
    # ambient delivered to the whole bay over the capped window, for context
    amb_gal = gpm * CAP_MIN
    for policy in default_suite():
        for seed in SEEDS:
            t0 = time.time()
            r = run_one(env, policy, jax.random.PRNGKey(seed), total_steps,
                        cfg.dirt.clean_threshold)
            r.update(gpm=gpm, strategy=policy.name, seed=seed,
                     ambient_gal_over_cap=amb_gal, wall_s=round(time.time()-t0, 1))
            fh.write(json.dumps(r) + "\n"); fh.flush()
            fin = f"{r['minutes']:.1f}min" if r["finished"] else "  --  "
            print(f"{gpm:5.0f}gpm {policy.name:>16}  {fin}  "
                  f"clean {r['frac_clean_end']*100:5.1f}%  "
                  f"drained {r['drained_end']:.2f}kg  worst {r['worst_end']:.4f}  "
                  f"({r['wall_s']}s)", flush=True)
fh.close()
print("DONE", flush=True)
