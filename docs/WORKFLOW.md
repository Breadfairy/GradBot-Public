# Workflow

This is the end-to-end map of the active runtime.

## Tune Flow

Run from `scripts/`:

```bash
./run_tune.sh <profile.json> <label>
./run_tune.sh <profile.json> <label> TUNE_ANCHOR_DATE=YYYY-MM-DD
```

The actual sequence is:

1. `run_tune.sh`
2. `src/tune/run.py`
3. `src/config/profile.py`
4. `src/data/prepare_klines.py`
5. `src/data/causality_audit.py`
6. `src/tune/host.py`
7. `build/native/gradbot_tune`
8. `src/tune/post.py`
9. `src/tune/trace.py`

## What Each Stage Does

### 1. Shell runner

`scripts/run_tune.sh`:

- resolves the profile path
- creates the run directory
- protects against clobbering an existing run label
- sets `MPLCONFIGDIR`
- uses the runtime clock when no anchor env var is set
- forwards `TUNE_ANCHOR_DATE` into `--anchor-date` when set
- `TUNE_ANCHOR_MS` remains available for old runs
- calls `python3 -m tune.run`
- holdout trace is now called by Python orchestration, not shell

### 2. Python pre-run

`src/tune/run.py`:

- loads the profile
- normalizes final portion fields
- validates the profile for tuner mode
- ensures all required klines exist locally for the requested anchor
- writes `causality-audit.json` before the sweep and stops on failed active
  split/alignment checks
- runs a posture outer sweep when `DAILY_CLUSTER_PATH` is a list; each posture
  path gets its own no-chart tune lane and the root run adopts the best
  tune-selected lane
- honors `TUNE_FLASH=1` by disabling chart generation for smoke sweeps
- writes the host-spec directory
- runs the host tuner binary

The time split is:

- `primer_days`: prepare rolling statistics before train/tune scoring.
- `training_days`: reserve pre-tune data for stable window shape.
- `tuner_days`: score profile sweeps on the tune window.
- `holdout_days`: replay the chosen winner as future-like data.

### 3. C sweep

`build/native/gradbot_tune`:

- reads `host-spec/`
- runs the grouped sweep in C
- infers fixed posture clusters from `DAILY_CLUSTER_MODEL_PATH` when supplied
- evaluates `PEAK_LOCK_*` PID/peak-lock settings as part of each row
- prints progress during the sweep
- writes `results.csv`
- writes `best-row.csv`
- writes `stats-row.csv`

### 4. Python post-run

`src/tune/post.py`:

- reconstructs best/stats configs from the winner rows
- scores robust-region candidates from `results.csv`
- writes `best-configs/*.json`
- writes `robust-candidates.csv` and `robust-row.csv`
- reruns chart traces for best/stats
- writes tune charts and summaries
- writes run fingerprints
- syncs result profiles back into the matching
  `inputs/profiles/user/results/` or `inputs/profiles/codex/results/`

### 5. Holdout

`src/tune/trace.py`:

- loads `best-config.json`
- loads `beststats-config.json` if it is distinct
- loads `bestrobust*-config.json` candidates when present
- reuses the run anchor from `fingerprint.json` when present
- traces the selected configs on the holdout window
- prints one compact comparison table across configured start offsets
- writes holdout charts

For one-off holdout diagnostics, invoke `tune.trace` directly.

Holdout start offsets come from the selected config:
`HOLDOUT_START_MIN_PCT`, `HOLDOUT_START_MAX_PCT`, and
`HOLDOUT_START_STEP_PCT`. CLI arguments can override them for one-off runs.
Tune and holdout chart output can be trimmed from the selected config with
`CHARTS_TIMEVAL` for equity/allocation charts and `CHARTS_TRADES` for
chunked trade/price charts. Omitted keys default to the historical behavior
of writing both chart families.

## Output Layout

Primary run directory:

- `outputs/tuning/<label>/results.csv`
- `outputs/tuning/<label>/best-row.csv`
- `outputs/tuning/<label>/stats-row.csv`
- `outputs/tuning/<label>/robust-row.csv`
- `outputs/tuning/<label>/robust-candidates.csv`
- `outputs/tuning/<label>/best-configs/`
- `outputs/tuning/<label>/causality-audit.json`
- `outputs/tuning/<label>/posture-sweep-summary.csv` when posture paths are
  swept
- `outputs/tuning/<label>/charts/tune/`
- `outputs/tuning/<label>/charts/holdout/`
- `outputs/tuning/<label>/host-spec/`
- `outputs/tuning/<label>/fingerprint.json`

Synced profile outputs:

- `inputs/profiles/user/results/best-config.json` for user profiles, or
  `inputs/profiles/codex/results/best-config.json` for codex profiles
- matching `stats-config.json` and robust configs in the same `results/`
  directory

Artifact landing folders:

- use `outputs/codex/` for Codex-created research outputs worth keeping
  around for review
- use `outputs/user/` for manually generated runs or curated user artifacts
- normal `outputs/tuning/<label>/` runs remain generated artifacts; keep only
  deliberate summaries or examples under version control

## Selection Rules

Best:

- highest lifecycle edge score on the tuning window
- the lifecycle score evaluates the whole strategy-vs-HODL equity curve,
  not only the final value

Stats:

- highest lifecycle-plus-risk score using lifecycle edge score, CAGR/MDD,
  Sharpe worst-window, Sortino worst-window, and explicit MDD penalties
  above a 55% soft limit
- excessive trade count is penalized after 500 trades so fee/tax churn does
  not win on curve shape alone

Robust:

- Python post-run ranks local parameter regions from `results.csv`
- candidates use tune-only information before holdout reruns
- the score prefers high drawdown-adjusted lifecycle score, high local lower
  quartile, lower local variance, and smaller gap between the row score and
  its neighbourhood
- the robust-region pre-score also penalizes excessive trade count after 500
  trades
- robust candidates are comparison candidates, not replacements for C's
  authoritative `best` and `stats` rows

Peak-lock:

- `PEAK_LOCK_*` scalars, arrays, and ranges are native C axes
- PID/peak-lock behavior is selected in the same row as DSP, macro, and
  posture behavior
- ultra posture uses `ULTRA_EXPOSURE_TARGET` as a sell floor for DSP sells;
  active peak-lock caps that floor until the re-entry ladder or release target
  lifts the cap
- holdout reruns replay the selected row; they do not select PID parameters

Posture:

- a scalar `DAILY_CLUSTER_PATH` runs one posture lane
- a list of `DAILY_CLUSTER_PATH` values runs an outer posture sweep
- lane selection compares each lane's best, stats, and first robust candidate
  on the same tune-side drawdown/trade-adjusted score
- holdout columns are transfer diagnostics only

## Single Config Flow

Standalone backtest is retired. For a single configuration, use a tuning
profile with scalar keys. The same tune path runs selection, traces the
selected config, writes charts, and reuses the run anchor from
`fingerprint.json` for holdout traces.

## Live Trading Flow

Run from the repo root:

```bash
scripts/run_live.sh
```

The deployment runtime is `src/live/` and does not call the tuner. Its launch
sequence is:

1. load `inputs/profiles/user/live-config.json`
2. create a fresh session directory, or resume the last active unclosed one
3. load `inputs/live/config.ini` when paper trading is disabled
4. fetch fresh 1h, macro, and posture candles
5. infer the latest posture from the closed posture-interval candles
6. run the warmup readiness check
7. hydrate the dashboard with the latest applied causal posture
8. restore `*_state.csv` only when resuming an active unclosed session
9. start the dashboard and websocket stream
10. evaluate each new closed 1h candle with the ready context

The live runtime applies `inputs/live/model/cluster_model.json` to closed
posture candles. It does not train, remap, or regenerate the posture model at
launch. Tune runs can apply the same fixed-model contract in C by supplying
`DAILY_CLUSTER_MODEL_PATH`; Python still owns training/export.

If the process restarts or reconnects after a stream disconnect, it reloads
fresh candles and checks missed closed 1h candles. In paper mode it replays
missed candles in order through the normal paper wallet and trade logs. In
live and dry-run modes it still summarizes missed candles without retroactive
orders.

A clean `quit` marks the live session `closed`. The next `run_live` creates a
new epoch under `outputs/live/sessions/<session-id>/`. If the prior
process died without a clean quit, that session remains `active` and the next
run resumes its state/log directory.

Live logging writes:

- `session.json`: active/closed lifecycle metadata
- `out`: dashboard/account snapshots
- `*_state.csv`: one-row rolling restart state
- `*_decisions.csv`: one row per new closed 1h candle
- `*_trades.csv`: fills plus core decision context
- `*_events.csv`: runtime and command history

Decision rows are not chunked. They include the applied causal daily posture,
macro state, micro trend/z-score state, accepted flags, final action, reason
text, and run identity hashes.

## Live Promotion Flow

Live config promotion is deliberate. Tuning syncs selected configs into
`inputs/profiles/user/results/` or `inputs/profiles/codex/results/`,
but it does not overwrite the live runtime
profile automatically.

The normal promotion path is:

```text
inputs/profiles/user/results/best-config.json
  -> inputs/profiles/user/live-config.json
```

Before running live, keep the live-only fields in the destination profile:

- `PAPER_TRADING`
- `LIVE_DRY_RUN`
- `history_days`
- `config_path`
- `DAILY_CLUSTER_MODEL_PATH`
- `out`

## Analysis Tools

Reusable one-off evaluation helpers live in `src/tools/`.

- `marker_audit.py`: checks generated flags against wallet trade markers and
  separates seed buys / daily lock sells from signal trades.
- `lock_sweep.py`: runs a compact daily profit-lock sweep and writes top
  configs/charts plus `summary.csv`.

## Marker Semantics

Charts can contain both signal-backed wallet trades and wallet-only actions.

- normal wallet BUY/SELL markers come from accepted generated flags
- seed buy markers are initial-position wallet actions
- purple/magenta lock markers are executed `daily_posture_lock` sells

Daily lock sells affect fees, tax accounting, equity curves, trade counts,
and `GrossVsHODL`, but they are not generated `SELL` flags.

## Cross-Asset Filter

If a tuning profile contains multiple tickers:

- the first ticker is the base tuning asset
- the others are replay assets used for tolerance filtering
- filtering is controlled by `TUNE_ERROR`

This happens during the tuning sweep, not in post processing.
