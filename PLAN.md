# Wide Source Rehome Architecture Plan

This plan replaces the older tuning-first migration plan. The tuning-first
work is now a completed prerequisite: offline backtest behavior lives inside
`src/tune/trace.py`, `scripts/run_tune.sh` calls `python3 -m tune.run`, and
shared Python helpers have already been grouped under focused `src/`
packages.

The wide source rehome is complete: `src/` now means "repo source code", not
"tuning Python code".

Do not edit the root README. Read `docs/README.md` before broad code scans.


## Current Native Unification

The active tune engine now treats DSP, posture clustering, and PID/peak-lock
as one row-scored system:

- Python still trains and exports clustering artifacts.
- `src/tune/host.py` flattens a promoted `DAILY_CLUSTER_MODEL_PATH` JSON into
  `host-spec/daily_cluster_model.txt`.
- `src/native/host/tuneHost.c` can infer fixed posture clusters from closed
  posture candles during the tune sweep.
- `src/native/engine/batch.c` evaluates `PEAK_LOCK_*` PID/peak-lock behavior
  inside the wallet loop for every C row.
- Python post-run now reconstructs best/stats/robust configs and traces
  holdout; it no longer runs a separate peak-lock selector.

Remaining follow-up:

- Promote or regenerate clustering artifacts with explicit `fitEndMs` so the
  host-spec causality check can enforce model training before the tune window
  on every walk-forward run.
- Add a compact parity fixture comparing Python live cluster inference against
  C host inference for the promoted model.


## Mental Model

The repo has three active applications and one native runtime layer:

- `src/tune/`: offline tuning, selection, and trace asset generation.
- `src/live/`: live and paper trading runtime.
- `src/clustering/`: posture artifact and model generation.
- `src/native/`: C engine and C host tuner.

Shared Python logic stays in sibling packages:

- `src/engine/`: rolling-window math and signal kernels.
- `src/runtime/`: flag gates and diagnostics.
- `src/strategy/`: posture, sizing, signal state, PID/peak-lock supervisor.
- `src/portfolio/`: wallet, accounting, and tax helpers.
- `src/data/`: kline I/O, kline preparation, time bounds, causality audit.
- `src/config/`: profile parsing and parameter normalization.
- `src/analysis/`: charts, metrics, reports, summaries.
- `src/tools/`: deterministic checks and targeted diagnostics.
- `src/research/quarantine/`: historical, non-runtime experiments.

Non-source runtime material does not belong in `src/`:

- profiles, live configs, clustering configs, and model inputs go in
  `inputs/`
- generated runs, live sessions, clustering outputs, and research artifacts
  go in `outputs/`
- shell entrypoints go in `scripts/`
- build products go outside source directories


## Target Layout

```text
src/
  tune/
  live/
  clustering/
  native/
    engine/
    host/
  engine/
  runtime/
  strategy/
  portfolio/
  data/
  config/
  analysis/
  tools/
  research/quarantine/

scripts/
  run_tune.sh
  run_live.sh
  run_install.sh
  run_clustering.sh
  get_klines.sh
  update_klines.sh
  lib/

inputs/
  profiles/
    tuning/
    live/
    reference/
    results/
  live/
    model/
    config.ini
  clustering/
    current/
    experiments/
    promoted/
    reference/
  klines/

outputs/
  tuning/
  live/
    sessions/
    latest_active.json
  clustering/
  research/
  codex/
  user/

requirements/
  live.txt
```

Root-level source and research spillover has been retired. Historical docs
belong under `docs/`, old supervised ML configs belong under
`inputs/research/quarantine/`, and scratch/cache outputs are ignored instead
of tracked.

The exact profile subdirectory names can be adjusted during migration, but the
source/runtime split should not.


## Entrypoint Contract

The final user-facing shell surface should be:

```bash
scripts/run_tune.sh PROFILE LABEL [KEY=VALUE ...]
scripts/run_live.sh [warm-start args]
scripts/run_install.sh
scripts/run_clustering.sh [CONFIG]
scripts/get_klines.sh TICKER INTERVAL [DAYS]
scripts/update_klines.sh
```

Python module entrypoints should be:

```bash
PYTHONPATH=src python3 -m tune.run ...
PYTHONPATH=src python3 -m live.live ...
PYTHONPATH=src python3 -m clustering.run_cluster ...
PYTHONPATH=src python3 -m data.prepare_klines ...
```

`scripts/run_live.sh` should automatically call `scripts/run_install.sh` when
the live venv or required packages are missing. Keep `run_install.sh` as a
direct maintenance entrypoint, but normal use should be `run_live.sh`.


## Completed Prerequisites

- Standalone backtest surface retired.
- `src/tune/run.py` owns tune orchestration.
- `src/tune/trace.py` owns selected-config trace behavior.
- `scripts/run_backtest.sh` removed.
- Old supervised ML posture lane quarantined under research paths.
- Current clustering lane identified as active posture artifact generation.
- Shared Python code grouped under `engine`, `strategy`, `portfolio`,
  `runtime`, `data`, `config`, and `analysis`.


## Goals

1. Put active source code under `src/`.
2. Make tune, live, clustering, and native C sibling source domains.
3. Keep runtime inputs and generated outputs out of source directories.
4. Keep the shell entrypoint surface small and explicit.
5. Let live import shared source packages without `sys.path` surgery.
6. Let tune call the native host binary through a stable build/output path.
7. Keep live behavior, tune behavior, and clustering output semantics stable.
8. Update docs so they describe actual active workflows only.


## Non-Goals

- Do not move live inside `src/tune/`.
- Do not move clustering inside `src/tune/`.
- Do not put profiles, models, credentials, venvs, logs, or generated outputs
  under `src/`.
- Do not change trading math, PID/peak-lock behavior, posture semantics, or
  tune/holdout selection rules during the rehome.
- Do not retrain or replace the live clustering model as part of the move.
- Do not edit the root README.


## Final Target

The repo should read as:

```text
scripts/run_tune.sh
  -> src/tune/run.py
  -> build/native/gradbot_tune
  -> outputs/tuning/

scripts/run_live.sh
  -> src/live/live.py
  -> inputs/profiles/live/live-config.json
  -> inputs/live/model/
  -> outputs/live/

scripts/run_clustering.sh
  -> src/clustering/run_cluster.py
  -> inputs/clustering/
  -> outputs/clustering/
```

All active source code lives under `src/`. Runtime inputs and generated
outputs stay under `inputs/` and `outputs/`. Shell entrypoints stay under
`scripts/`.
