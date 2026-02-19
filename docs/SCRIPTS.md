# Scripts

All scripts live in `scripts/`. Typical usage:

```bash
cd scripts
```

## Tuning + holdout

- `./run_tune.sh PROFILE LABEL`
  - Runs `src/tune_pipeline.py`
  - Writes `outputs/tuning/<LABEL>/`
- `./run_backtest.sh PROFILE NAME`
  - Runs a backtest and writes `outputs/tuning/<NAME>/`
  - If `PROFILE` is a tuning run directory, runs holdout summaries for the
    tuned configs instead.

## Klines cache

- `./get_klines.sh TICKER INTERVAL [DAYS]`: create a cache file under
  `inputs/klines/` if missing (requires `src/config.ini`).
- `./update_klines.sh`: forward-fill all cached klines to now (requires
  `src/config.ini`).

## Cache cleanup

- `./clean_cache.sh`: purge/trim cached ctx/signals artifacts.

## Oracles (experimental)

- `./run_oracle.sh LABEL [oracle-config.json] [--add]`
  - Writes generated profiles under `inputs/profiles/`
  - Writes charts/stats under `outputs/oracles/<LABEL>/<TICKER>/`
