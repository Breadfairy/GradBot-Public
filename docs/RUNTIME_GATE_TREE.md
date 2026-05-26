# Runtime Gate Decision Tree

This is the current plain-language map of how runtime gates and daily 1d
posture execution interact. The 1h cluster required-move gate is not part of
the normal profile>tune>holdout runtime.


########################################################################
## Runtime Decision Tree
########################################################################

```text
For each candle
|
|-- Build micro state
|   |-- EMA p1 / p2 / p3
|   |-- trendCode
|   |-- GRAD1 z-scores
|
|-- Build macro state
|   |-- macro EMA p1 / grad period / p3
|   |-- macro direction
|   |-- macro momentum
|   |-- macro dynamic required move %
|
|-- BUY lane
|   |
|   |-- require micro trendCode == -1
|   |
|   |-- require BUY grad z-score >= GRAD1_BUY_Z_MIN
|   |
|   |-- require enough candles since previous BUY
|   |
|   |-- Required move
|       |-- base = macro dyn %
|       |-- require price drop from anchor >= final required move
|       |-- accept BUY
|
|-- SELL lane
|   |
|   |-- require micro trendCode == 1
|   |
|   |-- require SELL grad z-score >= GRAD1_SELL_Z_MIN
|   |
|   |-- require enough candles since previous SELL
|   |
|   |-- Required move
|       |-- base = macro dyn %
|       |-- require price rise from anchor >= final required move
|       |-- accept SELL
|
|-- Same-candle conflict
|   |-- SELL wins
|
|-- Wallet execution
    |-- seed initial asset position
    |-- scale phase portions
    |-- apply floor / daily posture
    |-- execute daily lock sell if forced
```

Short form:

```text
micro trend chooses side
GRAD1 confirms pressure
cooldown prevents spam
macro decides required move
accepted flag becomes trade input
wallet decides final trade size or wallet-only lock actions
```


########################################################################
## Wallet Execution Controls
########################################################################

Flags are candidate trade inputs. The wallet layer makes final execution
decisions.

- Daily posture source
  - tune/trace profiles use `DAILY_CLUSTER_PATH` to read prepared posture
    cluster labels.
  - tune/trace alignment uses the latest completed posture `closeMs` at
    each micro candle open. That means posture candle `D` applies from the
    next micro open.
  - live deployment uses `DAILY_CLUSTER_MODEL_PATH` to infer cluster labels
    from closed posture-interval candles.
  - the live bundled `clustered_features.csv` file is a validation fixture,
    not the source for fresh live posture.

- Daily strong-up posture
  - the current 60d model treats cluster `2` as strong-up.
  - can shrink ordinary sells through `ULTRA_SELL_MULT`.
  - can buy up to `ULTRA_EXPOSURE_TARGET` while ultraBull is active.

- Daily weak/down posture
  - the current 60d model treats clusters `0` and `3` as weak/down.
  - can shrink buys through `DAILY_DOWN_BUY_MULT`.

- Optional post-ultra exit
  - maps confirmed ultraBull entry-to-exit gain into exit depth and
    post-ultra hold duration.
  - uses `ULTRA_GAIN_MIN_PCT`, `ULTRA_GAIN_MAX_PCT`,
    `ULTRA_EXIT_DEPTH`, and `ULTRA_EXIT_HOLD_DAYS`.
  - omit these keys, or keep `ULTRA_EXIT_DEPTH=0`, when ultra exit should
    return directly to normal DSP trading without a forced transition sell.
  - forced lock sells are tagged `daily_posture_lock`.
  - they are real executed sells, but not generated `SELL` flags.

Chart markers now distinguish:

- normal signal-backed wallet buys/sells
- seed buys
- optional daily posture lock sells

Live state CSVs are restart helpers, not trading history authority. Trade
history still comes from the runtime trade/event logs and, in non-paper mode,
from exchange/account records.

The live runtime also writes `*_decisions.csv`, one row per new closed 1h
candle. Decision rows capture the applied causal daily posture, macro state,
micro trend/z-score state, accepted flags, final action, and reason text.
Trade rows carry the core decision fields as well, so fills can be audited
directly.


########################################################################
## Current Tune Flow
########################################################################

```text
profile config
|
|-- Python writes host-spec CSV/TXT files
|
|-- C evaluates interval/macro/parameter groups
|
|-- C writes results.csv
|
|-- C chooses global best/stats across all profiles
|
|-- Python reconstructs best/stats configs and holdout charts
```
