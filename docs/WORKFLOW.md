# Workflow

## Tune

Run a parameter sweep from `scripts/`:

```bash
./run_tune.sh <profile.json> <label>
```

Writes a tuning run under `outputs/tuning/<label>/`:
- `results.csv` (all evaluated combos)
- `best-configs/best-config.json` (return-maximized pick)
- `best-configs/beststats-config.json` (optional; skipped if same as best)
- `charts/tune/` (scatter + equity curves)

Selection:
- `best`: maximizes gross % vs buy-and-hold on the tuning window.
- `beststats` ("risk minimizes worst-window"): maximizes:
  `riskScore = 0.10*(CAGR/MDD) + 0.45*min(sharpe4w, sharpe13w)`
  `+ 0.45*min(sortino4w, sortino13w)`

## Holdout

Given a tuning run directory, print holdout summaries and write holdout charts:

```bash
./run_backtest.sh ../outputs/tuning/<label> holdout
```

Outputs under `outputs/tuning/<label>/charts/holdout/`.

## Cross-asset tolerance (optional)

If a profile has multiple `tickers`, the first is treated as the base ticker.
During tuning, candidates can be replayed across the other tickers and filtered
by an error tolerance (`TUNE_ERROR`).
