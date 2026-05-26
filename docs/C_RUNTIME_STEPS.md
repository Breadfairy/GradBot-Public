# C Runtime Boundary

This file describes the active C-to-Python split. It is not a roadmap file.

## Current Boundary

The active tune runtime is two layers:

- C owns the sweep, progress printing, CSV writing, and best/stats row
  selection.
- Python owns profile parsing, local kline preparation, host-spec writing,
  post-run artifacts, charts, and holdout reruns.

This is the boundary to preserve.

## Active Path

1. `scripts/run_tune.sh`
2. `src/tune/run.py`
3. `src/data/prepare_klines.py`
4. `src/tune/host.py`
5. `build/native/gradbot_tune`
6. `src/tune/post.py`
7. `src/tune/trace.py`

## What C Owns

- sweep loops across all combinations
- grouped interval and macro evaluation reuse
- progress and ETA printing
- `results.csv` emission
- `best-row.csv` emission
- `stats-row.csv` emission
- best row selection by lifecycle edge score
- stats row selection by risk score
- fixed clustering-model inference during tune when
  `DAILY_CLUSTER_MODEL_PATH` is supplied
- PID/peak-lock wallet supervision through the `PEAK_LOCK_*` row surface

## What Python Owns

- JSON profile loading and validation
- rejection of legacy keys
- local kline preparation in `inputs/klines/`
- packing host-spec files
- reading winner rows back into Python
- writing best/stats config JSON artifacts
- chart reruns for tune outputs
- selected-config trace over holdout plus holdout charts
- fingerprints and text summaries
- reusable analysis tools under `src/tools/`

## Host-Spec Contract

`src/tune/host.py` writes `outputs/tuning/<label>/host-spec/`:

- `meta.txt`
  - run metadata and scalar knobs that are not swept in grouped files
  - window split:
    `primerDays`, `trainingDays`, `tunerDays`, `holdoutDays`
- `interval_groups.csv`
  - interval and micro period groups
- `macro_groups.csv`
  - macro interval, macro periods, macro dyn, and grad groups
- `axes.txt`
  - independent sweep axes for the remaining keys

`build/native/gradbot_tune` consumes that directory and writes:

- `results.csv`
- `best-row.csv`
- `stats-row.csv`

Ultra posture and PID/peak-lock parameters are part of the tunable row surface.
C consumes either prepared posture label arrays or a flattened fixed cluster
model supplied through the host spec and evaluates the same wallet-level
posture effects during sweeps.
`ULTRA_EXPOSURE_TARGET` can force-buy up to a target asset allocation when
the aligned posture cluster is strong. `ULTRA_GAIN_MIN_PCT`,
`ULTRA_GAIN_MAX_PCT`, `ULTRA_EXIT_DEPTH`, and `ULTRA_EXIT_HOLD_DAYS` map
confirmed ultraBull entry-to-exit gain into post-ultra exit depth and hold
duration when that optional exit surface is enabled. Omit those keys, or keep
`ULTRA_EXIT_DEPTH=0`, when ultra exit should return straight to normal DSP
trading. Keep confirmed bear regions neutral in the posture CSV when they
should remain DSP buy-in zones.

Training/exporting the posture cluster model remains Python-owned. For tune
runs, `src/tune/host.py` flattens `DAILY_CLUSTER_MODEL_PATH` JSON into
`host-spec/daily_cluster_model.txt`; the C host then infers clusters from
closed posture-interval candles using the fixed scaler/PCA/centroids. Live
runtime under `src/live/` still uses `inputs/live/model/cluster_model.json`
for the same fixed-model inference path in Python.

## Winner Selection

Best row:

- selected in C by highest lifecycle edge score over the tune equity curve

Stats row:

- selected in C by
  `lifecycleEdgeScore`
  `+ 2.0 * (CAGR / MDD)`
  `+ 4.0 * min(sharpe4w, sharpe13w)`
  `+ 4.0 * min(sortino4w, sortino13w)`
  `- 100.0 * (0.35 * MDD + 1.25 * max(MDD - 0.55, 0))`
  `- 0.03 * max(trades - 500, 0)`

The Python side treats those row files as authoritative and turns them into
full JSON configs and charts.

Peak-lock/PID is no longer a Python post-run selector in the active tune path.
`PEAK_LOCK_*` scalars, arrays, and ranges are expanded into native axes and
scored with DSP, macro, and posture logic in the same C row.

Posture selection also remains Python-owned. A scalar `DAILY_CLUSTER_PATH`
runs one lane. A list of paths runs no-chart posture lanes, prefers each
lane's robust candidate when present, then stats/risk, then best lifecycle,
and copies the tune-selected lane's artifacts back to the root run directory.
If
`DAILY_CLUSTER_MODEL_PATH` is supplied, C model inference takes precedence over
the static label CSV for the host sweep.

`src/data/causality_audit.py` runs before the host sweep. It verifies active
tune/holdout candle separation and last-known daily/macro alignment before
active decisions. `src/tune/host.py` also rejects model artifacts with
`fitEndMs` after the tune-window start.

## Files To Read For Tune Runtime Changes

If a change affects the tune pipeline, read these first:

- `src/config/profile.py`
- `src/tune/schema.py`
- `src/tune/axes.py`
- `src/tune/host.py`
- `src/native/host/tuneSpec.h`
- `src/native/host/tuneSpec.c`
- `src/native/host/tuneHost.c`
- `src/native/engine/engine.h`
- `src/native/engine/batch.c`
- `src/tune/artifacts.py`
- `src/tune/post.py`
- `src/tune/trace.py`

## Files To Touch When Adding Or Removing A Tunable Key

Usually all of these are part of the edit set:

- `src/config/profile.py`
  - validation and legacy-key handling
- `src/tune/schema.py`
  - shared key metadata, row field lists, and host axis-name mapping
- `src/tune/axes.py`
  - sweep axis registration
- `src/tune/host.py`
  - host-spec packing and row parsing
- `src/native/host/tuneSpec.h`
  - C-side parameter structs
- `src/native/host/tuneSpec.c`
  - parsing and axis loading
- `src/native/host/tuneHost.c`
  - CSV headers and row packing
- `src/native/engine/engine.h`
  - runtime parameter structs if the key is consumed in the engine
- `src/native/engine/batch.c`
  - batch evaluation and CSV row writing
- `src/tune/artifacts.py`
  - config reconstruction from row data
- `inputs/profiles/user/*.json` or `inputs/profiles/codex/*.json`
  - checked-in profiles and templates

If the key affects the live gate math, also inspect:

- `src/runtime/diag.py`
- `src/runtime/gates.py`
- `src/engine/core.py`
- `src/engine/macro_view.py`
- `src/analysis/charting.py`

If the key affects wallet execution, also inspect:

- `src/portfolio/wallet.py`
- `src/runtime/posture_feed.py`
- `src/analysis/reporting.py`
- `src/tools/marker_audit.py`

## Things That Should Not Move Back Into C By Default

- JSON parsing
- chart rendering
- holdout orchestration
- local kline fetch/update logic

Those are outside the core sweep boundary.
