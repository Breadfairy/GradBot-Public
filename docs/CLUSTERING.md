# Clustering Posture Lane

This directory is the active posture artifact lane. It is Python-only and
stays outside `src/tune/` because it generates inputs for both tuning and
live deployment rather than running inside either runtime.

Current default:

- ticker: `LINKUSDT`
- interval: `6h`
- rolling window: `240` candles
- clusters: `4`
- config: `inputs/clustering/current/linkusdt-6h-ema-fast-posture.json`

Run from the repo root:

```bash
scripts/run_clustering.sh
```

Outputs are written under:

```text
outputs/clustering/<run-name>/<view>/
```

For the `engine` view, each `kXX` directory also writes:

```text
cluster_flag_summary.csv
cluster_policy.csv
cluster_transition_summary.csv
clustered_features.csv
```

The active tune/runtime path consumes promoted posture CSVs through
`DAILY_CLUSTER_PATH`. 1h cluster models are not part of the normal
profile>tune>holdout path. The older 1d regime configs are retained under
`inputs/clustering/reference/` for reference checks and comparisons.

6h research sweeps can set `periodCombos` to run the same cluster counts and
feature families across several fast/mid/slow EMA triplets. Outputs are nested
under an `emaXXX_YYY_ZZZ` directory when multiple period combos are present.
Set `clusterMethods` to compare model families such as `kmeans` and `gmm`.
When multiple methods are present, outputs are nested under the method name.

Supervised cluster imitation can be run after a sweep:

```bash
PYTHONPATH=src python3 -m clustering.train_cluster_classifier \
  --features outputs/clustering/RUN/engine/ema013_052_136/kmeans/FAMILY/k04/clustered_features.csv \
  --out outputs/clustering/RUN/engine/ema013_052_136/kmeans/FAMILY/k04/supervised_cluster_classifier
```

This writes `model_scores.csv`, `confusion_matrices.csv`, and
`confusion_matrices.pdf`. It also writes probability-filtered DSP flag
summaries, cluster-confidence flag summaries, and classified row outputs.

Cluster charts can be rendered from an existing output:

```bash
PYTHONPATH=src python3 -m clustering.render_cluster_charts \
  --features outputs/clustering/RUN/engine/ema013_052_136/kmeans/FAMILY/k04/clustered_features.csv \
  --out outputs/clustering/RUN/engine/ema013_052_136/kmeans/FAMILY/k04/charts/rendered \
  --yearly
```

Dual-cluster state-machine research can be run from paired raw/event outputs:

```bash
PYTHONPATH=src python3 -m clustering.cluster_state_machine \
  --regime-features outputs/clustering/RUN/engine/ema013_052_136/kmeans/raw_market_ema_expanded/k04/clustered_features.csv \
  --event-features outputs/clustering/RUN/engine/ema013_052_136/kmeans/capitulation_state/k04/clustered_features.csv \
  --parent-features outputs/clustering/DAILY_RUN/engine/reduced_no_gate_inputs/k04/clustered_features.csv \
  --out outputs/clustering/state_machine/RUN_NAME
```

Use `cluster_state_machine_sweep.py` to score all matching raw/event pairs
inside a clustering run.

Daily parent labels can also be previewed from current 6h features with a
causal supervised classifier:

```bash
PYTHONPATH=src python3 -m clustering.train_parent_preview \
  --regime-features outputs/clustering/RUN/engine/ema013_052_136/kmeans/raw_market_ema_expanded/k04/clustered_features.csv \
  --parent-features outputs/clustering/DAILY_RUN/engine/reduced_no_gate_inputs/k04/clustered_features.csv \
  --out outputs/clustering/state_machine/PARENT_PREVIEW_RUN
```

Or from combined closed 1h and 6h intraday evidence:

```bash
PYTHONPATH=src python3 -m clustering.train_parent_intraday_preview \
  --regime-features outputs/clustering/RUN/engine/ema013_052_136/kmeans/raw_market_ema_expanded/k04/clustered_features.csv \
  --one-hour-klines inputs/klines/LINKUSDT/linkusdt_1h.csv \
  --six-hour-klines inputs/klines/LINKUSDT/linkusdt_6h.csv \
  --daily-klines inputs/klines/LINKUSDT/linkusdt_1d.csv \
  --parent-features outputs/clustering/DAILY_RUN/engine/reduced_no_gate_inputs/k04/clustered_features.csv \
  --out outputs/clustering/state_machine/PARENT_INTRADAY_PREVIEW_RUN
```

Detailed state-machine runs write `window_scores.csv`, `window_trades.csv`,
and `timeVals.csv`. The `timeVals.csv` file contains the model equity curve,
HODL benchmark, exposure, target reason, and cluster roles for the original
fit/holdout split plus trailing one, two, three, and four year windows.
When `--parent-features` is supplied, the daily parent role is aligned by
closed candle time using last-known daily labels only. Phase-machine policies
then add ultra/chop hysteresis and a post-ultra crab phase for DSP swing
trading after profit lock. Lifecycle-machine policies are also scored; these
require a sustained ultra regime, wait for a confirmed ultra exit, then run a
short profit-lock and DSP-led crab phase.

Render state-machine PNGs from the time-values artifact with:

```bash
PYTHONPATH=src python3 -m clustering.render_state_machine_charts \
  --timevals outputs/clustering/state_machine/RUN_NAME/timeVals.csv \
  --out outputs/clustering/state_machine/RUN_NAME/charts/timeVals
```

Render simplified post-hysteresis regime bands and band-quality metrics with:

```bash
PYTHONPATH=src python3 -m clustering.render_regime_bands \
  --timevals outputs/clustering/state_machine/RUN_NAME/timeVals.csv \
  --model MODEL_NAME \
  --out outputs/clustering/state_machine/RUN_NAME/charts/regime_bands \
  --min-run-bars 12
```

Use `--partitions all --holdout-open-ms MS` for one long-history chart with
the holdout boundary marked. Use `--split-crab-clusters` to split the crab
band by the underlying regime cluster and post-ultra crab phase.
Use `--causal-confirm-bars N` for live-like forward-only confirmation instead
of diagnostic smoothing. A state change appears only after `N` consecutive raw
bars confirm the new state.

Compare raw cluster bands with causal cooldown-smoothed bands from any
`clustered_features.csv`. The bundled 1h config is quarantined under
`inputs/research/quarantine/clustering/` and should be treated as research
only:

```bash
PYTHONPATH=src python3 -m clustering.cluster_cooldown_bands \
  --features outputs/clustering/linkusdt-1h-cooldown-band-check/engine/raw_market_ema_expanded/k05/clustered_features.csv \
  --out outputs/clustering/state_machine/linkusdt_1h_raw_vs_cooldown_k05 \
  --confirm-bars 3 \
  --cooldown-bars 6
```

This is intended for 1h experiments where raw clustering may react faster
than 6h posture, but short-run state chop needs causal damping.

Export a confirmed-regime posture CSV for the tuner/trace daily-posture
bridge with:

```bash
PYTHONPATH=src python3 -m clustering.export_regime_posture \
  --timevals outputs/clustering/state_machine/RUN_NAME/timeVals.csv \
  --model MODEL_NAME \
  --confirm-bars 12 \
  --out outputs/clustering/state_machine/RUN_NAME/regime_posture.csv
```

The export maps confirmed `ultraBull` to strong cluster `2` and maps bear,
flush, and crab states to neutral cluster `1`. This keeps confirmed bear
regions available as DSP buy-in zones instead of applying the old down-buy
shrink.

## Causality

Feature rows are built from past and current candles only. The row at candle
`i` means:

```text
the previous 60 daily candles ending at candle i looked like this
```

Forward returns are written only for evaluation after clustering. They are not
used as model inputs.

Discrete MA derivatives are causal differences. Do not replace them with
`np.gradient`, because centered gradients leak future candles.

## Runtime Boundary

The runtime model is deliberately narrower than the exploratory feature set.
The `engine` view clusters C-compatible causal posture features and writes a
daily/posture label file plus a fixed model artifact. Runtime uses those labels
or fixed model outputs for wallet posture and profit-lock behavior; it does not
create BUY or SELL candidates.

For tuning, `DAILY_CLUSTER_MODEL_PATH` is flattened into a host-spec text
artifact and the C host applies the fixed scaler/PCA/centroids to closed
posture candles. `DAILY_CLUSTER_PATH` remains a static-label fallback. A model
artifact with `fitEndMs` after the tune-window start is rejected by the host
spec writer.
