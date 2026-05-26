# GradBot Parameters (JSON profiles)

This document describes JSON profile keys used by tuning and selected-config
trace runs. Keys are UPPERCASE unless noted.

Conventions
- Single-config trace profiles use scalars. Tuner profiles may accept
  lists/ranges.
- `tickers` is required and always a list; the first entry is the base ticker.
- `intervals` may be a string ("1h") or a list (["15m", "1h"]).
  Single-config traces use the first entry.
- Window split: `primer_days`, `training_days`, `tuner_days`,
  `holdout_days` (ints).
  Legacy `days` / `HOLDOUT_DAYS` are not supported.

Required (base)
- tickers: list of symbols, e.g., ["LINKUSDT","ETHUSDT"].
- intervals: candle interval(s), e.g., ["15m"] or "15m".
- p1, p2, p3: MA periods (ints). Typically p1<p2<p3.
- primer_days, training_days, tuner_days, holdout_days: window split (ints).

Gradient gate (g1(p1) z-score)
- GRAD1_BUY_Z_MIN / GRAD1_SELL_Z_MIN: z-thresholds (>0) for BUY/SELL.
- GRAD1_BUY_WIN_DAYS / GRAD1_SELL_WIN_DAYS: rolling lookbacks (days)
  for BUY/SELL.
- Z-scores use rolling mean/std (causal), not center-referenced gradients.

Macro dyn% (dynamic pct gate)
- MACRO_INTERVAL: macro candle interval (e.g., "6h"). Set to "" to disable.
- MACRO_P1 / MACRO_GRAD_PERIOD / MACRO_P3: macro EMA periods.
- MACRO_NRG_WIN_DAYS: z-score window for macro spacing%.
- MACRO_NRG_Z_MIN / MACRO_NRG_Z_MAX: z window to map into pct.
- MACRO_DYN_PCT_MIN / MACRO_DYN_PCT_MAX: output bounds for dyn% mapping.
- MACRO_GRAD_WIN_DAYS: z-score window for abs(EMAgrad gradient).
- MACRO_GRAD_Z_MIN / MACRO_GRAD_Z_MAX: z window (of abs(EMAgrad gradient))
  mapped into a multiplier.
- MACRO_MULT_GRAD_MIN / MACRO_MULT_GRAD_MAX: multiplier bounds applied to
  macro dyn% (then clipped to pct min/max).
- Implementation summary:
  - spacing13Pct: abs(EMA1-EMA3)/EMA3 * 100 on the macro interval
  - baseMag: positive z-score of spacing13Pct mapped to pct max
  - warmup: linear ramp within each bull/bear regime over the z-score window
  - damp: linear multiplier from spread/peak(spread) within the regime
  - gradMult: z-score(abs(EMAgrad gradient)) multiplier for steep moves
  - alignment: last-known macro sample aligned to micro candles

Mechanics
- COOLDOWN: per-side minimum separation (candles) between signals.
- PHASE_BUY_PORTIONS / PHASE_SELL_PORTIONS: number of portions per phase.
- FINAL_PORTION_PCT: fraction (0..1) applied for the last portion in a phase.

1h signal clustering is not part of normal runtime. Standalone clustering
research remains under `clustering/`, but tune profiles should not include
1h cluster keys.

Daily cluster posture
- DAILY_CLUSTER_PATH: CSV with `closeMs`, `close`, and `cluster` columns.
  A non-empty path enables 1d posture. In tuner profiles this can be a list;
  the pipeline runs a posture outer sweep and selects by tune metrics only.
- The 60d daily model identity is fixed: cluster `2` is strong-up, and
  clusters `0` and `3` are weak/down.
- ULTRA_SELL_MULT: sell scale multiplier while ultraBull is active.
- ULTRA_EXPOSURE_TARGET: target asset exposure while ultraBull is active.
- DAILY_DOWN_BUY_MULT: buy scale multiplier in weak/down clusters.
- ULTRA_EXIT_DEPTH: maximum forced post-ultra exposure reduction.
- ULTRA_GAIN_MIN_PCT / ULTRA_GAIN_MAX_PCT: entry-to-exit ultraBull gain
  range mapped into the forced exit depth.
- ULTRA_EXIT_HOLD_DAYS: maximum post-ultra floor hold duration.
- Omit `ULTRA_EXIT_*`, or keep `ULTRA_EXIT_DEPTH` at `0`, to return
  directly to normal DSP trading after ultraBull exits.
- Daily lock sells are real wallet trades tagged `daily_posture_lock`.
  They are not generated `SELL` flags, so charting marks them separately.
- PEAK_LOCK_RELEASE_TARGET_PCT: when a confirmed ultraBull re-enters while
  peak-lock is active, raise the peak-lock cap to at least this exposure.
  This prevents stale peak-lock from blocking ultraBull re-entry.
- PEAK_LOCK_ULTRA_GRACE_DAYS: optional suppression window after ultraBull
  exits before a new peak-lock can arm. It is available for experiments but
  is not included in the default peak-lock overlay sweep.

Tax and wallet (trace required keys)
- WALLET_SEED_QUOTE: starting quote balance (e.g., 10000.0).
- WALLET_SEED_ASSET_PCT: starting quote fraction immediately deployed into
  the asset. `1.0` preserves the old all-in seed behavior.
- WALLET_FEE_RATE: trade fee rate (e.g., 0.001 for 0.1%).
- QUOTE_TO_AUD_RATE: used for AUD conversions in prints (e.g., 0.64).
- TAX_MODE: "cgt" or "income".
- Income-tax mode uses a fixed annual base of `36000.0`.

Charts and output
- CHART_CHUNK_SIZE: approx days per chart image.
- CHARTS_TIMEVAL: enable tune/holdout timVal equity/allocation charts. Defaults
  to true when omitted.
- CHARTS_TRADES: enable tune/holdout chunked trade/price charts. Defaults to
  true when omitted.
- TUNE_FLASH=1: environment flag for no-PNG smoke sweeps. It disables tune
  charts, writes generated configs with both holdout chart keys false, and
  uses a smaller peak-lock base-row set.
- HOLDOUT_START_MIN_PCT / HOLDOUT_START_MAX_PCT /
  HOLDOUT_START_STEP_PCT: holdout start-offset sweep, expressed as
  percentages of the holdout window. Defaults are 0, 20, and 5.
- out: tuner output CSV path (tuner only).
- Seed buys and daily posture lock sells have separate chart markers.
- Purple/magenta lock markers are executed `daily_posture_lock` sells.

Selection metrics
- scoreMetric: lifecycle edge score. It evaluates the full
  strategy-vs-HODL equity curve, including median edge, lower-quartile edge,
  worst edge, time below HODL, time tracking HODL too closely, underwater
  severity, and edge drawdown.
- Best row selection uses lifecycle edge score instead of final gross edge.
- Stats row selection starts from lifecycle edge score and adds risk terms for
  CAGR/MDD, worst-window Sharpe/Sortino, and explicit MDD penalties.

Gate behavior (AND logic)
- BUY requires ALL of: regime = BEAR; g1_p1 z-score ≥ BUY
  threshold; cooldown; pct move since last BUY/phase anchor ≥
  |macro dyn%| * macro mult(BUY).
- SELL requires ALL of: regime = BULL; g1_p1 z-score ≥ SELL threshold;
  cooldown; pct move since last SELL/phase anchor ≥ |macro dyn%| *
  macro mult(SELL); SELL wins ties at same index.
- Accepted flags then pass into wallet execution, where daily posture and
  lock logic can resize or add wallet-only lock actions.

Quick disables (set to 0)
- Macro dyn%: set `MACRO_INTERVAL` → "" (or set `MACRO_DYN_PCT_MAX` → 0.0).
- Cooldown: COOLDOWN → 0 (allow back‑to‑back signals).

Example (single-config trace, scalars)
{
  "tickers": ["LINKUSDT"],
  "primer_days": 365,
  "training_days": 365,
  "tuner_days": 365,
  "holdout_days": 0,
  "intervals": ["15m"],
  "p1": 12,
  "p2": 20,
  "p3": 55,
  "GRAD1_BUY_Z_MIN": 1.5,
  "GRAD1_SELL_Z_MIN": 1.5,
  "GRAD1_BUY_WIN_DAYS": 365,
  "GRAD1_SELL_WIN_DAYS": 365,
  "MACRO_INTERVAL": "6h",
  "MACRO_P1": 12,
  "MACRO_GRAD_PERIOD": 20,
  "MACRO_P3": 55,
  "MACRO_NRG_WIN_DAYS": 90,
  "MACRO_NRG_Z_MIN": 0.0,
  "MACRO_NRG_Z_MAX": 1.0,
  "MACRO_DYN_PCT_MIN": 3.0,
  "MACRO_DYN_PCT_MAX": 15.0,
  "MACRO_GRAD_WIN_DAYS": 90,
  "MACRO_GRAD_Z_MIN": 0.0,
  "MACRO_GRAD_Z_MAX": 1.0,
  "MACRO_MULT_GRAD_MIN": 0.7,
  "MACRO_MULT_GRAD_MAX": 2.0,
  "PHASE_BUY_PORTIONS": 5,
  "PHASE_SELL_PORTIONS": 5,
  "COOLDOWN": 8,
  "FINAL_PORTION_PCT": 1.0,
  "WALLET_SEED_QUOTE": 10000.0,
  "WALLET_SEED_ASSET_PCT": 1.0,
  "WALLET_FEE_RATE": 0.001,
  "QUOTE_TO_AUD_RATE": 0.64,
  "TAX_MODE": "income",
  "CHART_CHUNK_SIZE": 30,
  "CHARTS_TIMEVAL": true,
  "CHARTS_TRADES": false,
  "HOLDOUT_START_MIN_PCT": 0,
  "HOLDOUT_START_MAX_PCT": 20,
  "HOLDOUT_START_STEP_PCT": 5
}

Sweeps and risk metrics
- Sharpe/Sortino/MDD/CAGR use a mark-to-market equity curve with trade deltas
  at trade indices.
