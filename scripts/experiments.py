"""Batched baseline experiments: water x technique, with seeds.

Everything here runs the SCRIPTED strategies to completion under `lax.scan`,
vmapped across seeds, so a cell that costs ~50 s serially on CPU costs one
batched pass on the GPU. That is what makes confidence intervals affordable:
the single-seed matrix this replaces could not distinguish a real 2-point
ranking change from noise.

Two suites:

  matrix  -- default_suite() x ambient gpm x seeds. The T4 table.
  lanes   -- standoff x lane_width x ambient gpm x seeds. Resolves the
             confound in the single-seed standoff ladder: `standoff_variants`
             overrides ONLY standoff, so lane_width stays 0.18 m while the fan
             width changes by ~4x across the ladder. The apparent optimum at
             0.50 m may be lane-spacing match rather than standoff.

             Lane width is set as a FRACTION of the swath actually measured at
             that standoff (`calibrate.single_pass_removal` at the strategy's
             own tilt and push speed), not as an absolute width -- an absolute
             ladder confounds differently at every standoff, which is the bug
             being investigated. f<1 overlaps passes, f>1 leaves stripes.

Results stream to JSONL, one row per (cell, seed), so a killed pod loses at
most the cell in flight.
"""
from __future__ import annotations

import argparse, dataclasses, json, time
import jax, jax.numpy as jnp, numpy as np

from floorclean.config import Config
from floorclean.env import CleaningEnv
from floorclean.baselines import PushSweep, default_suite


def run_cell(env, policy, seeds, total_steps, clean_thr):
    """One (strategy, gpm) cell, vmapped across `seeds`. Fresh floors."""
    def single(key):
        state = env.fresh_state(key)
        carry = policy.init(env, state)

        def one(cs, _):
            c, s = cs
            c, a = policy.act(env, s, c)
            s, _o, _r, _t, _tr, info = env.step(s, a)
            return (c, s), jnp.stack([info["worst_residual"], info["drained_kg"],
                                      info["fraction_clean"], info["water_m3"]])

        (_c, s), tr = jax.lax.scan(one, (carry, state), None, length=total_steps)
        worst, drained, frac, water = tr.T
        clean = worst < clean_thr
        ever = clean.any()
        # first step index at which the whole floor is under threshold
        idx = jnp.argmax(clean)
        return dict(
            finished=ever,
            steps_to_clean=jnp.where(ever, idx + 1, -1),
            frac_clean_end=frac[-1],
            worst_end=worst[-1],
            drained_end=drained[-1],
            drained_at_clean=jnp.where(ever, drained[idx], drained[-1]),
            water_m3=jnp.where(ever, water[idx], water[-1]),
            initial_kg=state.initial_mass,
        )

    return jax.jit(jax.vmap(single))(seeds)


def measure_swaths(cfg, standoffs, tilt, speed):
    """Effective cut swath (m) at each standoff, at the strategy's own posture."""
    from floorclean.geometry import build_floor
    from scripts.calibrate import single_pass_removal
    floor = build_floor(cfg)
    out = {}
    for so in standoffs:
        swath, peak = single_pass_removal(cfg, floor, so, speed, tilt=tilt)
        out[so] = float(swath)
        print(f"  swath(standoff={so:.2f}) = {swath*100:5.1f} cm  peak_cut={peak:.2f}",
              flush=True)
    return out


def cells(suite, gpms, standoffs, fracs, swaths):
    if suite == "matrix":
        for gpm in gpms:
            for p in default_suite():
                yield dict(gpm=gpm, strategy=p.name), p
    else:
        for gpm in gpms:
            for so in standoffs:
                for f in fracs:
                    lw = max(swaths[so] * f, 0.02)
                    p = PushSweep(name=f"so{so:.2f}_f{f:.2f}",
                                  standoff=so, lane_width=lw)
                    yield dict(gpm=gpm, strategy=p.name, standoff=so,
                               lane_frac=f, lane_width=lw,
                               swath=swaths[so]), p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="matrix", choices=["matrix", "lanes"])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--cap-minutes", type=float, default=30.0)
    ap.add_argument("--gpms", default="8,20,45,90")
    ap.add_argument("--standoffs", default="0.10,0.20,0.30,0.40,0.50,0.65,0.80")
    ap.add_argument("--lane-fracs", default="0.70,1.00,1.35",
                    help="lane width as a fraction of the MEASURED swath")
    ap.add_argument("--out", default="experiments.jsonl")
    args = ap.parse_args()

    gpms = [float(v) for v in args.gpms.split(",")]
    standoffs = [float(v) for v in args.standoffs.split(",")]
    fracs = [float(v) for v in args.lane_fracs.split(",")]
    seeds = jax.vmap(jax.random.PRNGKey)(jnp.arange(args.seeds))

    base = Config()
    total_steps = int(args.cap_minutes * 60 / base.sim.control_dt)
    print(f"suite={args.suite} seeds={args.seeds} cap={args.cap_minutes:.0f}min "
          f"({total_steps} steps) gpms={gpms} devices={jax.devices()}", flush=True)

    envs = {}
    fh = open(args.out, "w")
    swaths = {}
    if args.suite == "lanes":
        print("measuring swaths at tilt=0.55 speed=0.45 ...", flush=True)
        swaths = measure_swaths(base, standoffs, tilt=0.55, speed=0.45)
    todo = list(cells(args.suite, gpms, standoffs, fracs, swaths))
    for i, (meta, policy) in enumerate(todo, 1):
        gpm = meta["gpm"]
        if gpm not in envs:
            cfg = dataclasses.replace(
                base, floor=dataclasses.replace(base.floor, ambient_inflow_gpm=gpm))
            envs[gpm] = CleaningEnv(cfg)
        env = envs[gpm]
        t0 = time.time()
        res = run_cell(env, policy, seeds, total_steps, env.cfg.dirt.clean_threshold)
        res = {k: np.asarray(v) for k, v in res.items()}
        dt = base.sim.control_dt
        for s in range(args.seeds):
            row = dict(meta)
            row.update(seed=s,
                       finished=bool(res["finished"][s]),
                       minutes=(float(res["steps_to_clean"][s]) * dt / 60
                                if res["finished"][s] else None),
                       frac_clean_end=float(res["frac_clean_end"][s]),
                       worst_end=float(res["worst_end"][s]),
                       drained_end=float(res["drained_end"][s]),
                       drained_at_clean=float(res["drained_at_clean"][s]),
                       gal=float(res["water_m3"][s]) * 264.172,
                       initial_kg=float(res["initial_kg"][s]))
            fh.write(json.dumps(row) + "\n")
        fh.flush()
        fc = res["frac_clean_end"]
        nfin = int(res["finished"].sum())
        print(f"[{i:3d}/{len(todo)}] {gpm:5.0f}gpm {meta['strategy']:>18}  "
              f"clean {fc.mean()*100:5.1f}+-{fc.std()*100:4.1f}%  "
              f"drained {res['drained_end'].mean():.2f}kg  "
              f"finished {nfin}/{args.seeds}  ({time.time()-t0:.0f}s)", flush=True)
    fh.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
