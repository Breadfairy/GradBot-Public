# Oracles (experimental)

The oracle flow is a non-causal labeling pass used to generate wide tuner
profiles for a new asset. It is intentionally separate from the core causal
backtest engine.

## Run

From `scripts/`:

```bash
./run_oracle.sh <label> [oracle-config.json] [--add]
```

## Outputs

- Generated profiles under `inputs/profiles/`:
  - `*-full-config.json`, `*-mid-config.json`
- Per-ticker artifacts under `outputs/oracles/<label>/<TICKER>/`:
  - `charts/`
  - `stats/`
