# Ablation Results

This document records the useful knowledge produced by the previous
`plan.md` workstream. It is not the active task list. Current work is tracked
in `../TODO.md`.


########################################################################
## Objective
########################################################################

Build a LINKUSDT trading profile that compounds materially against HODL while
controlling the main observed failure mode:

- asset exposure dwindles during a strong run
- price spikes while the bot is underexposed
- `GrossVsHODL` falls sharply even if local swing trading looked reasonable

The system does not need to stay above HODL on every candle. The target is
meaningful improvement over time against HODL with controlled underexposure
risk.


########################################################################
## Stable Infrastructure
########################################################################

- C owns grouped tuning sweeps, CSV row emission, progress, and winner
  selection.
- Python owns profile parsing, kline prep, host-spec generation, charts,
  holdout reruns, and result artifacts.
- Profile backtests are split-aware. Active replay starts after
  `primer_days + training_days + tuner_days`.
- Holdout reruns support multi-start offsets in the first 0-20 percent of the
  holdout period by 5 percent steps.
- Historical runs can be anchored with `TUNE_ANCHOR_MS`,
  `TUNE_ANCHOR_DATE`, `BACKTEST_ANCHOR_MS`, or `BACKTEST_ANCHOR_DATE`.
- Macro-to-micro alignment is causal and uses last-known macro values.
- Macro required-move smoothing reduced abrupt jumps in the dynamic required
  percentage.
- Chart wallet markers now separate normal trades, seed buys, and daily
  posture lock sells.
- Marker audit found no unexplained wallet trades on the then-current winning
  profile.


########################################################################
## Removed Or Neutralised Logic
########################################################################

The following logic did not earn its complexity and has been removed or
neutralised in the active build:

- run-up sell grace
- daily strong asset floor
- general/core asset floor
- dead spacing and micro-energy branches
- old oracle/cache tuning path
- old Python row-by-row tuner path

The main tune path is now C sweep plus Python orchestration/post-processing.


########################################################################
## Wallet And Daily Posture Findings
########################################################################

Daily 1d posture clustering is currently the strongest wallet-side signal.

Earlier wallet ablation run:

```text
.codex_tmp/wallet-ablation-0-40/
```

Baseline:

- base gross: `+150.83%`
- worst 0-40 start: `+67.23%`
- trades/year: `87.3`
- daily locks: `12`
- minimum asset exposure: `0.00%`

Measured effects:

- Removing daily posture collapsed the base run to `+36.17%`.
- Removing daily lock dropped the base run to `+82.72%`.
- Daily strong floor and core floor were inert or harmful.
- Run-up sell grace was inert.
- `PHASE_BUY_PORTIONS=7` improved worst start to `+97.07%` while only
  slightly lowering base return.

Working interpretation:

- keep 1d posture and daily lock
- treat 1d clustering as wallet/posture logic, not micro signal generation
- do not sweep daily posture while testing the 1h clustering architecture
- eventually replace hard-coded daily cluster ids with labels derived from
  daily cluster stats


########################################################################
## 1h Clustering Findings
########################################################################

1h clustering is a required-move multiplier. It does not create trades.

Useful quick-run result:

```text
outputs/tuning/phase9-macro-cluster-quick/
```

The better forward candidate was the stats row:

- `CLUSTER_K=5`
- `CLUSTER_POLICY_MODE=heuristic_forward`
- `CLUSTER_POLICY_TARGET=24`
- tune gross-vs-hodl: `+214.43%`
- holdout gross-vs-hodl: `+124.30%`
- holdout worst start: `+102.49%`
- trades: `256`

This was the main evidence that 1h clustering might help robustness.

Bad focused result:

```text
outputs/tuning/phase9-cluster-policy-focused/
```

The best/stats rows selected:

- `CLUSTER_K=4`
- `CLUSTER_POLICY_MODE=flag_outcome`
- `CLUSTER_POLICY_TARGET=48`

Holdout was worse:

- best holdout gross-vs-hodl: `+90.94%`
- best worst start: `+78.59%`
- stats holdout gross-vs-hodl: `+73.98%`
- stats worst start: `+64.73%`

The selected `flag_outcome_t48/k4` policy mostly acted as a sell suppressor:

```text
best gate audit:
  BUY cluster_mult:  0 -> 0
  SELL cluster_mult: 222 -> 5

stats gate audit:
  BUY cluster_mult:  0 -> 0
  SELL cluster_mult: 357 -> 3
```

Conclusion:

- `flag_outcome` can win tune by blocking sells and is not trusted.
- The only 1h cluster policy still worth validating is `heuristic_forward`.
- The active overnight comparison is a go/no-go test for 1h clustering.
- If heuristic 1h clustering does not beat no-1h-cluster on holdout,
  multistart worst case, trade count, and tune/holdout parity, quarantine or
  remove 1h clustering.


########################################################################
## Cluster Artifact Fix
########################################################################

Run-local cluster artifacts originally collided when sweeping
`CLUSTER_POLICY_MODE` or `CLUSTER_POLICY_TARGET`. Different policy rows for
the same feature set and `k` wrote to the same artifact path.

Fix:

```text
outputs/tuning/<label>/ml/cluster/<featureSet>/k05/<policyMode>_t24/
```

Post-fix validation:

- `phase9-cluster-policy-focused` produced `25` cluster profiles.
- duplicate model/policy references: `0`
- result rows now point at the policy file they actually used.


########################################################################
## Macro And Sell Gate Findings
########################################################################

The macro required-move gate remains load-bearing.

Quick no-macro control:

- disabling macro move caused much higher trade counts
- tune score was materially worse than macro-on configs
- conclusion: do not remove macro move

The important macro uncertainty is not the structural macro engine. Freeze:

- `MACRO_INTERVAL=1d`
- `MACRO_P1=7`
- `MACRO_GRAD_PERIOD=7`
- `MACRO_P3=150`

Still worth testing narrowly:

- `MACRO_MULT_GRAD_MAX`
- `MACRO_DYN_PCT_MIN`
- `MACRO_DYN_PCT_MAX`
- `MACRO_GRAD_WIN_DAYS`
- `MACRO_GRAD_Z_MAX`
- `MACRO_SELL_RELAX_PCT`
- `GRAD1_SELL_Z_MIN`
- `GRAD1_SELL_WIN_DAYS`

The quick cluster run suggested `MACRO_MULT_GRAD_MAX=1.25` was useful. The
later `flag_outcome` run overfit lower values such as `0.75`; those results
should not be promoted without stronger holdout support.


########################################################################
## Active Comparison Runs
########################################################################

Two overnight runs define the next architecture decision:

```text
phase9-cluster-onoff-overnight
```

Purpose:

- daily 1d clustering fixed on
- compare 1h clustering off versus `heuristic_forward` 1h clustering on
- include `k=4,5,6` and targets `12,24,48`
- exclude `flag_outcome`

```text
phase9-no-1h-cluster-overnight
```

Purpose:

- assume 1h clustering is deleted
- daily 1d clustering fixed on
- sweep macro/sell shape only
- include `MACRO_SELL_RELAX_PCT`

Decision rule:

- keep 1h clustering only if `heuristic_forward` clearly improves holdout,
  multistart worst case, trade count sanity, and tune/holdout parity
- otherwise quarantine or remove 1h clustering and simplify profiles/code


########################################################################
## Robust Selection Insight
########################################################################

Current selection is too peak-seeking. Several runs show high tune maxima that
underperform on holdout.

The better direction is a robust-region selector over `results.csv`:

- score parameter neighbourhoods, not isolated rows
- prefer broad plateaus over sharp peaks
- use local median, lower quartile, standard deviation, drawdown, and trade
  count sanity
- holdout-test robust plateau candidates alongside `best` and `stats`
- penalise tune/holdout divergence and multistart spread after holdout rerun

This is likely more valuable than adding more wallet logic.


########################################################################
## Keys Usually Safe To Freeze
########################################################################

The active freeze/narrow/open decision record now lives in
`../FreezeLedger.md`. This section is the historical source that led to that
ledger.

In normal future sweeps, freeze:

- `p1/p2/p3 = 12/20/55`
- `MACRO_INTERVAL = 1d`
- `MACRO_P1 = 7`
- `MACRO_GRAD_PERIOD = 7`
- `MACRO_P3 = 150`
- `GATE_TREND_ENABLE = 1`
- `GATE_GRAD1_BUY_ENABLE = 1`
- `GATE_GRAD1_SELL_ENABLE = 1`
- `GATE_COOLDOWN_ENABLE = 1`
- `GATE_MACRO_MOVE_ENABLE = 1`
- `DEFENSE_ENABLE = 0`
- `PHASE_BUY_PORTIONS = 3`
- `FINAL_PORTION_PCT = 0.5`
- `COOLDOWN = 5`
- current daily posture and daily lock constants unless explicitly testing
  daily posture


########################################################################
## Tools Produced
########################################################################

- `src/tools/marker_audit.py`
- `src/tools/wallet_logic_probe.py`
- `src/tools/lock_sweep.py`
- `src/tools/wallet_ablation.py`

These remain useful for validating wallet behaviour, lock behaviour, and
marker/trade alignment.
