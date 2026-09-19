# Next steps — floor_clean

Ordered by what blocks what. Open board: #1–#5, #8–#11 (all OPEN), PR #6 (OPEN).
Last updated 2026-09-19, after the #4 implementation + issues #8–#11 filed.

## 0. Land the #4 working tree (blocks everything, ~30 min, no GPU)

The Option C fix is implemented but **uncommitted** (`floorclean/ppo.py`,
`floorclean/env.py`, `scripts/train.py`, `tests/test_env.py` — a second agent
landed parallel edits in the same files, so review carefully):

- [ ] `git diff` review: merged summary keys (`adv_std`, `adv_std_global`,
      `value_mean`, `phi_mean`, `explained_variance_f`), `Transition.phi`
      threading with the four distinct reads (`phi_pre` / `phi_next_pre_reset` /
      selected / `mb_phi`), `delivered_kg` plumbing, cancellation + chunk-wiring
      tests. All 17 `test_env.py` + 12 `test_physics.py` pass as of this writing.
- [ ] Commit on a branch, open PR (or stack on PR #6 if still open).
- [ ] CPU smoke already covered by the tiny-chunk tests; no pod yet.

**Gate for any GPU spend:** committed + green suite. Nothing below needs a pod.

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

## 3. #4-gated short pod slice (first GPU spend since run 1)

- [ ] Small slice (not full 150M): watch `adv_std_global` (must stay non-collapsed),
      `explained_variance_f` (must be informative, not ~1-by-construction),
      entropy (must not climb monotonically), greedy-eval stills (must move).
- [ ] Stop rule: `adv_std` collapse → stop, same as cut-and-abandon divergence
      (`adhered` down + `deposited` up + `drained` flat → stop).
- [ ] Only on green slice: launch run 2. Start fresh — run 1's checkpoint
      (entropy 5.62, worse-than-init policy) carries nothing worth keeping.

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
