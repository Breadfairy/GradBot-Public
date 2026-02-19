# GradBot Parameters (JSON profiles)

This document describes JSON profile keys used by backtests, tuning, holdout,
and oracle runs. Keys are UPPERCASE unless noted.

Conventions
- Backtest profiles use scalars. Tuner profiles may accept lists/ranges.
- `tickers` is required and always a list; the first entry is the base ticker.
- `intervals` may be a string ("1h") or a list (["15m","1h"]). Backtest uses
  the first entry.
- Window split: `primer_days`, `tuner_days`, `holdout_days` (ints).
  Legacy `days` / `HOLDOUT_DAYS` are not supported.

Required (base)
- tickers: list of symbols, e.g., ["LINKUSDT","ETHUSDT"].
- intervals: candle interval(s), e.g., ["15m"] or "15m".
- p1, p2, p3: MA periods (ints). Typically p1<p2<p3.
- primer_days, tuner_days, holdout_days: window split (ints).

Gradient gate (g1(p1) z-score)
- GRAD1_BUY_Z_MIN / GRAD1_SELL_Z_MIN: z-thresholds (>0) for BUY/SELL.
- GRAD1_BUY_WIN_DAYS / GRAD1_SELL_WIN_DAYS: rolling lookbacks (days).
- Z-scores use rolling mean/std (causal), not center-referenced gradients.

Spacing gate (z-score + optional energy)
- SPACING_Z_MIN_12 / SPACING_Z_MIN_23 (float >0): per-leg z mins.
- SPACING_WIN_DAYS_12 / SPACING_WIN_DAYS_23 (int >0): per-leg windows.
- MICRO_NRG_MODEL (str):
  - "sum": requires both legs (ma12, ma23) to meet energy z-thresholds
  - "sum23": requires only ma23
  - "none": disables energy gate
- MICRO_NRG_WIN_DAYS (int >0): rolling window (days) for energy z-score.
- MICRO_NRG_MIN_12 / MICRO_NRG_MIN_23 (float >0): energy z mins.

Macro dyn% (dynamic pct gate)
- MACRO_INTERVAL: macro candle interval (e.g., "6h"). Set to "" to disable.
- MACRO_P1 / MACRO_P2 / MACRO_P3: macro EMA periods.
- MACRO_NRG_WIN_DAYS: z-score window for macro spacing%.
- MACRO_NRG_Z_MIN / MACRO_NRG_Z_MAX: z window to map into pct.
- MACRO_DYN_PCT_MIN / MACRO_DYN_PCT_MAX: output bounds for dyn% mapping.
- MACRO_GRAD_WIN_DAYS: z-score window for abs(EMA2 gradient).
- MACRO_GRAD_Z_MIN / MACRO_GRAD_Z_MAX: z window (of abs(EMA2 gradient))
  mapped into a multiplier.
- MACRO_MULT_GRAD_MIN / MACRO_MULT_GRAD_MAX: multiplier bounds applied to
  macro dyn% (then clipped to pct min/max).
- Macro dyn% requirement multipliers (applied in flags):
  - Base multipliers by macro backbone:
    `MACRO_BUY_MULT_BULL`, `MACRO_BUY_MULT_BEAR`,
    `MACRO_SELL_MULT_BULL`, `MACRO_SELL_MULT_BEAR`
  - Extra multipliers in transition zones:
    `MACRO_BUY_MULT_REV`, `MACRO_BUY_MULT_ROLL`,
    `MACRO_SELL_MULT_REV`, `MACRO_SELL_MULT_ROLL`
  - Transition definitions (macro EMAs):
    - REV: EMA1 < EMA3 and EMA1 > EMA2 (bear rally / early reversal)
    - ROLL: EMA1 > EMA3 and EMA1 < EMA2 (bull pullback / early rollover)
- Implementation summary:
  - spacing13Pct: abs(EMA1-EMA3)/EMA3 * 100 on the macro interval
  - baseMag: positive z-score of spacing13Pct mapped to pct max
  - warmup: linear ramp within each bull/bear regime over the z-score window
  - damp: quadratic multiplier from spread/peak(spread) within the regime
  - gradMult: multiplier from z-score(abs(EMA2 gradient)) for early steep moves
  - alignment: last-known macro sample aligned to micro candles

Mechanics
- COOLDOWN: per-side minimum separation (candles) between signals.
- PHASE_BUY_PORTIONS / PHASE_SELL_PORTIONS: number of portions per phase.
- FINAL_PORTION_PCT: fraction (0..1) applied for the last portion in a phase.

Tax and wallet (backtest only; required keys)
- SUMMARY_LABEL: "" (full summary), "BEST", "STATS", or a 1-letter tag.
- WALLET_SEED_QUOTE: starting quote balance (e.g., 10000.0).
- WALLET_FEE_RATE: trade fee rate (e.g., 0.001 for 0.1%).
- QUOTE_TO_AUD_RATE: used for AUD conversions in prints (e.g., 0.64).
- TAX_MODE: "cgt" or "income".
- ANNUAL_INCOME_BASE: base salary for income-tax mode.
- (CGT) marginal rate is derived from ANNUAL_INCOME_BASE via AU brackets.
- PROFIT_SWEEP_INTERVAL: "month" or "" to disable.
- PROFIT_SWEEP_SHARE: 0.0..1.0 share of realized gains to lock.

Charts and output
- CHART_CHUNK_SIZE: approx days per chart image.
- out: tuner output CSV path (tuner only).

Gate behavior (AND logic)
- BUY requires ALL of: regime = BEAR; g1_p1 z-score ≥ BUY
  threshold; spacing per-leg z-score mask; cooldown; pct move since last
  BUY/phase anchor ≥ |macro dyn%| * macro mult(BUY).
- SELL requires ALL of: regime = BULL; g1_p1 z-score ≥ SELL threshold; spacing
  per-leg z-score mask; cooldown; pct move since last SELL/phase anchor ≥
  |macro dyn%| * macro mult(SELL); SELL wins ties at same index.

Quick disables (set to 0)
- Spacing gate (not recommended): set per-leg `SPACING_Z_MIN_12/_23`
  very low (≈0.1) to approximate an always-on gate.
- Spacing energy: set `MICRO_NRG_MODEL` → "none" or thresholds ≤0.
- Macro dyn%: set `MACRO_INTERVAL` → "" (or set `MACRO_DYN_PCT_MAX` → 0.0).
- Cooldown: COOLDOWN → 0 (allow back‑to‑back signals).

Example (backtest, scalars)
{
  "tickers": ["LINKUSDT"],
  "primer_days": 365,
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
  "SPACING_Z_MIN_12": 1.5,
  "SPACING_Z_MIN_23": 1.5,
  "SPACING_WIN_DAYS_12": 365,
  "SPACING_WIN_DAYS_23": 365,
  "MICRO_NRG_MODEL": "sum23",
  "MICRO_NRG_WIN_DAYS": 365,
  "MICRO_NRG_MIN_12": 1.5,
  "MICRO_NRG_MIN_23": 1.5,
  "MACRO_INTERVAL": "6h",
  "MACRO_P1": 12,
  "MACRO_P2": 20,
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
  "SUMMARY_LABEL": "",
  "WALLET_SEED_QUOTE": 10000.0,
  "WALLET_FEE_RATE": 0.001,
  "QUOTE_TO_AUD_RATE": 0.64,
  "TAX_MODE": "income",
  "ANNUAL_INCOME_BASE": 36000,
  "PROFIT_SWEEP_INTERVAL": "month",
  "PROFIT_SWEEP_SHARE": 0.0,
  "CHART_CHUNK_SIZE": 30
}

Sweeps and risk metrics
- Sharpe/Sortino/MDD/CAGR use a mark-to-market equity curve with trade deltas
  at trade indices. Monthly profit sweeps are not deducted from this curve.
  For research, prefer PROFIT_SWEEP_SHARE=0.0 for consistent comparisons.
