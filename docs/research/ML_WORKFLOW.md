# Quarantined ML Workflow

This is the old supervised ML posture lane. It is retained as historical
research under `src/research/quarantine/` and is not an active source for
current tuning or live posture unless it is deliberately revalidated.

Current active strategy direction:

- clustering remains the primary ML technique.
- supervised models are comparison/distillation tools for posture labels,
  not direct BUY/SELL or return predictors.
- random forest is acceptable for narrow posture or rare-risk experiments,
  but should be judged by drawdown reduction and multi-anchor holdout, not
  by single-window tune edge.
- direct price prediction, direct signal classification, RL, and deep models
  are out of scope until the clustering/posture/DSP boundary is simpler.

## Flow

1. Run clustering to create role labels:

```bash
scripts/run_clustering.sh \
  inputs/clustering/current/linkusdt-6h-ema-fast-posture.json
```

2. Train ML posture candidates:

```bash
MPLCONFIGDIR=.mplconfig PYTHONPATH=src python3 \
  -m research.quarantine.ml_posture \
  --config inputs/research/quarantine/ml/linkusdt-6h-role-classifier.json
```

3. Review ML metrics:

```text
outputs/ml/<name>/model_scores.csv
outputs/ml/<name>/model_rank.csv
outputs/ml/<name>/class_metrics.csv
outputs/ml/<name>/confusion_matrices.csv
outputs/ml/<name>/confusion_matrices.pdf
outputs/ml/<name>/metrics/confusion_matrix_<model>_<partition>.csv
outputs/ml/<name>/metrics/confusion_matrix_<model>_<partition>.pdf
```

4. Point a tuning profile at a selected posture CSV:

```json
"DAILY_CLUSTER_PATH": "outputs/ml/<name>/posture/logreg-posture.csv"
```

5. Run the normal tuner and holdout:

```bash
scripts/run_tune.sh <profile.json> <label>
```

## Partitions

The ML script splits valid rows in time order:

```text
train  -> fit the classifier
select -> rank ML candidates before expensive C tuning
holdout -> report untouched ML classification performance
```

The trading tuner should still make final decisions through the existing
tune and holdout workflow. The ML `holdout` metrics are classification
diagnostics, not the final trading holdout.

Targets are shifted forward by `targetShiftBars` in the ML config. The
default 6h config uses `1`, so features at candle `T` predict the role at
the next 6h candle.

When pairing an ML posture CSV with a tuning profile, keep the profile's
active tune window out of the ML `train` partition. The intended split is:

```text
profile primer/training -> ML train
profile tuner          -> ML select
profile holdout        -> final trading holdout
```

## Models

Built-in model names:

```text
logreg
randomForest
histGradientBoost
xgboost
lightgbm
```

The default config uses only sklearn models. `xgboost` and `lightgbm` are
available names when those packages are installed in the Python environment.

## Cleanup Boundary

Keep the production tune/trace path simple:

- `DAILY_CLUSTER_PATH` supplies prepared posture labels.
- `ULTRA_SELL_MULT`, `ULTRA_EXPOSURE_TARGET`, and `DAILY_DOWN_BUY_MULT`
  are the main posture execution controls.
- `ULTRA_EXIT_*` keys are optional and should be omitted for no-forced-exit
  profiles.
- legacy `DAILY_STRONG_*` and `DAILY_LOCK_*` profiles should be archived or
  migrated before use; the current profile validator rejects them.
