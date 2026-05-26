# Freeze Ledger

This is the decision record for tuning keys that should stay fixed, stay
narrow, or remain open. The purpose is to reduce config blowout without
forgetting why a key was constrained.

Do not treat this as permanent truth. Reopen a frozen key only when a focused
test gives evidence that the old decision is stale.


########################################################################
## Evidence Base
########################################################################

Current ledger state is based mainly on:

- `outputs/tuning/phase9-cluster-onoff-overnight`
- `outputs/tuning/phase9-no-1h-cluster-overnight`
- `outputs/tuning/phase9-arch-cluster-fixed`
- `outputs/tuning/phase9-arch-no1h-relax`
- earlier wallet ablations recorded in `archive/AblationResults.md`

Earlier broad comparison:

- 1h cluster on best holdout: `+128.04%` gross-vs-hodl
- 1h cluster off best holdout: `+120.63%` gross-vs-hodl
- 1h cluster on worst multistart: `+104.21%`
- 1h cluster off worst multistart: `+95.42%`

Latest focused comparison:

- 1h cluster best holdout: `+129.62%` gross-vs-hodl
- no-1h stats holdout: `+142.61%` gross-vs-hodl
- 1h cluster best worst multistart: `+103.74%`
- no-1h stats worst multistart: `+112.60%`

Latest no-1h robust map:

- no-1h `phase9-no1h-divergence-map` best holdout: `+129.53%`
- no-1h `phase9-no1h-divergence-map` stats holdout: `+133.45%`
- no-1h `phase9-no1h-divergence-map` robust04 holdout: `+148.95%`
- no-1h `phase9-no1h-divergence-map` robust04 worst multistart: `+115.52%`
- robust03/04/05 formed a tight holdout family around `+148.5%` to
  `+149.0%`, all with 1h clustering disabled and daily posture enabled.

Conclusion: 1h clustering is quarantined for normal sweeps. The no-1h
architecture is the current candidate family, but the holdout result must be
treated as architecture evidence rather than a fresh untouched deploy score.

Current promoted candidate:

- `inputs/profiles/user/results/peaklock-config.json`
- live mirror: `inputs/profiles/user/live-config.json`
- posture source:
  `inputs/clustering/promoted/link_pid_snappy_confirm4.csv`
- normal tune/backtest runtime no longer carries 1h cluster gate axes or
  1h cluster metadata columns
- selected deployment surface carries the current `PEAK_LOCK_*` PID
  supervisor keys


########################################################################
## Freeze In Normal Sweeps
########################################################################

These keys should be fixed unless the sweep is explicitly testing the named
area. Before deleting code paths, confirm the freeze with robust-region
candidates, not only peak tune rows.

| Key | Freeze Value | Reason |
| --- | --- | --- |
| `p1/p2/p3` | `12/20/55` | Stable base engine shape. |
| `MACRO_INTERVAL` | `1d` | Daily macro is the active useful structure. |
| `MACRO_P1` | `7` | Stable daily macro fast EMA. |
| `MACRO_GRAD_PERIOD` | `7` | Active gradient sensor. |
| `MACRO_P3` | `150` | Stable daily macro slow EMA. |
| Trend gate | structural on | Gate is structural, not experimental. |
| Grad1 buy gate | structural on | Gate is structural, not experimental. |
| Grad1 sell gate | structural on | Gate is structural, not experimental. |
| Cooldown gate | structural on | Gate is structural, not experimental. |
| Macro move gate | structural on | Macro gate is load-bearing. |
| Local-HODL defense | removed/inactive | Defense branch is currently inactive. |
| `PHASE_BUY_PORTIONS` | `3` | Stable wallet shape. |
| `PHASE_SELL_PORTIONS` | `5` | Dominated top rows; `7` underperformed. |
| `FINAL_PORTION_PCT` | `0.5` | Stable final sizing. |
| `COOLDOWN` | `5` | Stable trade-rate control. |


########################################################################
## Keep Narrow
########################################################################

These keys still affect results, but should not be allowed to explode broad
sweeps.

| Key | Normal Range | Reason |
| --- | --- | --- |
| `GRAD1_SELL_Z_MIN` | `0.8, 1.0, 1.2` | `0.8` led holdout robustness; `1.0` led peak tune; `1.2` remains a control family. |
| `GRAD1_SELL_WIN_DAYS` | `60, 75, 90, 105, 120, 135, 150` | Robust04 hit the previous upper bound at `105`; boundary test must find where it breaks. |
| `MACRO_NRG_WIN_DAYS` | `175, 210, 245, 280, 315` | Robust04 hit the previous upper bound at `245`; boundary test must find where it breaks. |
| `MACRO_DYN_PCT_MAX` | `23, 27` | Both useful; `31` is weaker control only. |
| `MACRO_GRAD_WIN_DAYS` | `45, 60, 75, 90` | Timing still affects robustness; robust04 selected `45`. |
| `MACRO_MULT_GRAD_MAX` | `1.0` | Divergence-map kept this fixed and improved; treat as frozen unless testing macro looseness directly. |
| `MACRO_SELL_RELAX_PCT` | `0, 5, 10, 15, 20` | Robust04 selected `10`; boundary test checks if nearby values keep the plateau. |

The no-1h robust family now mostly selects `MACRO_MULT_GRAD_MAX=1.0`.
Freeze it to `1.0` in divergence-map profiles. Use `0.75` only as a
specific regression/control test, and avoid larger values unless testing
macro looseness directly.


########################################################################
## 1h Clustering
########################################################################

Quarantined from normal sweeps and normal result schemas.

The standalone research/export code remains available, but the normal
profile>tune>holdout path no longer accepts or emits 1h cluster gate keys.

If it is tested again, keep it only in this narrow family:

| Clustering Config Field | Value |
| --- | --- |
| `policyTarget` | `24` |
| `featureFamilies` | current runtime feature family |

| Clustering Config Field | Range |
| --- | --- |
| `clusters` | `5, 6` |

Remove from large sweeps:

- `policyTarget=12`
- `policyTarget=48`
- broad `clusters` exploration
- any profile-level `CLUSTER_*` gate keys

Reasoning:

- `heuristic_forward / 24h / k5` produced the strongest current holdout.
- `k6` is close enough to keep as a robustness control.
- target `12` and target `48` did not justify their sweep cost.
- `flag_outcome` previously looked like tune overfit via sell suppression.
- the focused no-1h `stats` candidate beat the focused cluster `best`
  candidate on holdout, worst multistart, and drawdown.


########################################################################
## 1d Daily Posture
########################################################################

Daily posture clustering should remain enabled in normal architecture tests.

Freeze the current daily cluster artifact and wallet constants unless the
sweep is explicitly a daily-posture test.

Open work:

- replace hard-coded daily cluster ids with labels derived from daily cluster
  stats
- keep 1d daily posture separate from 1h required-move clustering


########################################################################
## Conditional Keys
########################################################################

`MACRO_SELL_RELAX_PCT` should not be swept everywhere.

Use:

- no 1h cluster track: `0, 10, 20, 25`
- 1h cluster track: start with `0`, if explicitly retesting clustering

Reasoning:

- `10` helped the no-1h-cluster run.
- `20` was selected by the no-1h `stats` row that transferred best to
  holdout in the focused comparison.
- the best 1h-cluster run used `0`.


########################################################################
## Rejected Or Quarantined
########################################################################

Do not include these in normal sweeps:

- run-up sell grace
- old defense floor branches
- old neutralised spacing and micro-energy branches
- `flag_outcome` 1h cluster policy
- broad 1h cluster target exploration
- `PHASE_SELL_PORTIONS=7`
- `MACRO_MULT_GRAD_MAX` above `1.25`


########################################################################
## Still Open
########################################################################

These are not freeze decisions yet:

- robust-region selector versus peak tune selector
- exact width of the no-1h robust plateau
- whether daily posture cluster ids can be derived automatically
- whether no-1h-cluster plus tuned `MACRO_SELL_RELAX_PCT` keeps beating
  clustered candidates after robust-region holdout reruns
