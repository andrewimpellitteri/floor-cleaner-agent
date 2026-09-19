# Next steps — floor_clean

Ordered by what blocks what. Open board: #1–#5, #8–#11 (all OPEN), PR #6 (OPEN).
Last updated 2026-09-19, after the #4 implementation + issues #8–#11 filed.

## 0. #4 landed, gate slice FAILED 2/3 -- CLOSED with verdict (no action)

Merged via PR #7; suite green (33/33). Gate slice `gate4` (800 upd / 26.2M
steps, ~$0.35): `adv_std_global` collapsed 16x (0.634 -> 0.059), entropy
monotone throughout, `finished_clean` 0. Only the instrumentation criterion
passed (`EV_f` 0.166 -> 0.793 -- informative, and it diagnoses the loop:
entropy up -> random policy -> predictable returns -> adv shrinks -> entropy
term dominates -> entropy up; `approx_kl` fell 23x, gradient doing nothing).
The offset made the failure visible at update 300 instead of inferred at 1500.
Next: `ent_coef` ablation {0.003, 0.0003, 0.0} x 800 upd (~$0.55) -- earned by
evidence; if that fails, suspect `REWARD_SCALE = 200` signal scale itself.

## 1. One shop visit settles #9, #10, #11 (no compute, highest realism/$)

Three photos, each closing one issue. Do them together:

- [ ] **#9:** taps+bar running, no washing — steady-state wet bands photo.
      Fit `bar_fraction` (0.25/0.5/0.75 first) in `config.py:147` against the
      `scripts/probes/water_sources.py` signature (mobile fraction / median /
      P(h>2mm) / gini). Sigmas (`tap_sigma`, `bar_sigma`) only if channel width
      mismatches.
- [ ] **#10:** ruler photo of the standing pond at the trough base low spot.
      Set `trough_retain_depth` (`config.py:75`, currently 5 mm guess), re-run
      the two pond tests. Profile along x only as a followup if it varies.
- [ ] **#11:** one real pass at recorded standoff/tilt/speed; stripped width +
      bare-floor photo vs `scripts/calibrate.py` single-pass table (±20%).
      Adjust `fan_*`/decay first on mismatch; `entrainment_rate` only per T1
      knob order; never `worn_lane_boost` (#5) or reward.

## 2. T1 completion run (minutes of CPU, unblocks the benchmark)

- [ ] `PushSweep(far_to_near)` to completion on the current config (real layout,
      ponding trough, 1.13 kg loading): expect 25–35 sim-min per the recalibrated
      band (#2, #5). Report `time_to_90` alongside `time_to_100` — the tail is T4.
- [ ] If it lands far outside the band, tune in T1 order (`entrainment_rate` →
      `settling_velocity` → `deposit_entrainment_rate`), one knob per commit message.

## 3. ent_coef ablation slices (gate failed -- this replaces the run-2 plan)

- [ ] Three x 800-update slices, `ent_coef` in {0.003 control, 0.0003, 0.0},
      watching `adv_std_global`, entropy, `fraction_clean`. ~45 min, ~$0.55.
- [ ] If `adv_std` stabilises and `clean` stops declining at lower `ent_coef`,
      that is the fix. If it degrades anyway, the problem is the advantage
      signal itself -- next suspect `REWARD_SCALE = 200` per-step scale.
- [ ] No full run until a slice holds `adv_std` flat with non-monotone entropy.
      Start any full run fresh (run 1's checkpoint carries nothing).

## 4. Then, in order

- [ ] T4 benchmark table (scripted suite + trained policy, fixed-floor
      completions with confidence intervals) → T6 plain-language advice.
- [ ] #8 perception rung 1 (fog/noise the global map) — only after run 2 learns.
      Rungs 2–3 (visited memory, FOV cone + recurrence) post-T4.
- [ ] T7 two operators (labour-minutes, not wall-clock), T8 bucket dumps —
      extensions Andrew asked about, after the one-operator answer exists.

## Explicitly deferred

- Hyperparameter grid search: pointless until #4 lands and T1 calibrates; then
  targeted ablations only (λ ladder first, T stays 64).
- `worn_lane_boost` / yield changes (domain ground truth), reward-term additions
  (invariant 5), training through solver resolution (grid-independence pinned).
