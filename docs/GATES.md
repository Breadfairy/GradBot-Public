# Gate Logic

This file is the plain-language gate reference for the active runtime.

## One-Sentence Summary

A BUY or SELL only survives if:

- the micro EMA ordering allows that side,
- the `GRAD1_*` z-score gate allows that side,
- cooldown allows another signal,
- and price has moved far enough from the current phase reference to beat
  the macro-adjusted required move.

Macro logic does not create the trade by itself. It changes how hard the
final move threshold is.

After flags are accepted, wallet execution can still resize or add
wallet-only actions through daily posture and profit-lock logic. Those wallet
decisions are not new signal flags.

## Core Pieces

### 1. Micro regime

Source: `src/engine/core.py` and `src/runtime/diag.py`

- `trendCode == -1` means full bear ordering on the micro EMAs.
  - Only BUY candidates are allowed there.
- `trendCode == 1` means full bull ordering on the micro EMAs.
  - Only SELL candidates are allowed there.
- `trendCode == 2` or `-2` is half-state structure.
  - No BUY or SELL is allowed there in the active flagger.

Very basic meaning:

- BUYs are only considered when the micro backbone is fully bearish.
- SELLs are only considered when the micro backbone is fully bullish.

### 2. GRAD1 gate

Source: `src/runtime/gates.py`, consumed by `src/runtime/diag.py`

- `GRAD1_BUY_WIN_DAYS` / `GRAD1_BUY_Z_MIN`
- `GRAD1_SELL_WIN_DAYS` / `GRAD1_SELL_Z_MIN`

The code takes the gradient of the fast micro EMA, turns it into a rolling
z-score, signs it per side, and checks it against the configured minimum.

Very basic meaning:

- BUY only stays alive if the fast EMA has fallen hard enough for the BUY
  z-score threshold.
- SELL only stays alive if the fast EMA has risen hard enough for the SELL
  z-score threshold.

This is the only active micro threshold layer now.

### 3. Cooldown

Source: `src/runtime/gates.py`, consumed by `src/runtime/diag.py`

- `COOLDOWN`

After the raw BUY or SELL candidates are found, the code drops later signals
that are too close to the last accepted one.

Very basic meaning:

- even if the raw conditions keep passing, signals cannot repeat too quickly.

### 4. Macro required move

Sources: `src/engine/macro_view.py`, `src/engine/core.py`,
`src/runtime/diag.py`

This is the last gate.

The flagger computes a required percent move and compares current price
movement against it:

- BUY compares drop from reference price to current price.
- SELL compares rise from reference price to current price.

If the move is smaller than the required threshold, the candidate is dropped.

Very basic meaning:

- the macro layer decides how much extra move must happen before the signal
  is allowed.

## Macro Layers

### Backbone sensor

- `MACRO_INTERVAL`
- `MACRO_P1`
- `MACRO_P3`

The macro backbone is the fast-vs-slow EMA spread on the macro timeframe.
The core spread is `EMA(P1) - EMA(P3)`.

This defines:

- macro direction (`bull` or `bear`)
- base macro pressure from spread size

### Base macro pressure

- `MACRO_NRG_WIN_DAYS`
- `MACRO_NRG_Z_MIN`
- `MACRO_NRG_Z_MAX`
- `MACRO_DYN_PCT_MIN`
- `MACRO_DYN_PCT_MAX`

The code converts the `P1 vs P3` spread size into a rolling z-score, maps
that into a dyn percent, then constrains it between the configured min/max.

Important behaviors:

- It warms in over `MACRO_NRG_WIN_DAYS` inside a new regime.
- It fades as the spread collapses from its own peak.
- That fade uses `ratio`, so it tapers linearly with spread collapse.

Very basic meaning:

- strong, unusual macro spread means a larger required move
- weak or collapsing spread means a smaller required move

### Gradient layer

- `MACRO_GRAD_PERIOD`
- `MACRO_GRAD_WIN_DAYS`
- `MACRO_GRAD_Z_MIN`
- `MACRO_GRAD_Z_MAX`
- `MACRO_MULT_GRAD_MIN`
- `MACRO_MULT_GRAD_MAX`

`MACRO_GRAD_PERIOD` is the middle macro EMA. It is a sensor, not a
multiplier by itself.

The code measures the slope of that EMA, turns it into a z-score, maps that
to a multiplier, and applies the multiplier to the base macro dyn.

Very basic meaning:

- steep macro slope can tighten the macro requirement
- flat macro slope can relax the macro requirement

## Cascading Logic Tree

```text
START
|
+-- idx < startIdx
|   +-- reject
|
+-- trendCode == -1
|   +-- BUY lane active
|   +-- GRAD1 BUY z-score passes?
|   |   +-- no -> reject
|   |   +-- yes
|   +-- cooldown passed?
|   |   +-- no -> reject
|   |   +-- yes
|   +-- choose BUY reference price
|   |   +-- last accepted BUY in this phase, else
|   |   +-- first candle of current BUY phase
|   +-- build required BUY move
|   |   +-- base dyn from macro P1 vs P3 spread z-score
|   |   +-- warm in over MACRO_NRG_WIN_DAYS
|   |   +-- damp by spread peak ratio
|   |   +-- scale by MACRO_GRAD_PERIOD slope multiplier
|   +-- current BUY drop >= required BUY move?
|       +-- yes -> BUY
|       +-- no  -> reject
|
+-- trendCode == 1
|   +-- SELL lane active
|   +-- GRAD1 SELL z-score passes?
|   |   +-- no -> reject
|   |   +-- yes
|   +-- cooldown passed?
|   |   +-- no -> reject
|   |   +-- yes
|   +-- choose SELL reference price
|   |   +-- last accepted SELL in this phase, else
|   |   +-- first candle of current SELL phase
|   +-- build required SELL move
|   |   +-- base dyn from macro P1 vs P3 spread z-score
|   |   +-- warm in over MACRO_NRG_WIN_DAYS
|   |   +-- damp by spread peak ratio
|   |   +-- scale by MACRO_GRAD_PERIOD slope multiplier
|   +-- current SELL rise >= required SELL move?
|       +-- yes -> SELL
|       +-- no  -> reject
|
+-- trendCode in {2, -2, 0}
    +-- reject
```

## Mental Model

The final required move is roughly:

`required move = base macro dyn * grad multiplier`

Where:

- base macro dyn comes from the macro `P1 vs P3` spread
- grad multiplier comes from the slope of `MACRO_GRAD_PERIOD`

## Overlap Areas

There is intentional overlap in macro strictness:

- spread-collapse damping
- gradient-based tightening or relaxing

That means a late trend can soften in more than one way at once.

## Code Ownership

- Final BUY/SELL gate: `src/runtime/diag.py`
- Shared gate helpers: `src/runtime/gates.py`
- Macro dyn math: `src/engine/core.py`
- Macro interval loading and alignment: `src/engine/macro_view.py`
- Trace runtime that consumes the flags: `src/tune/trace.py`
- Wallet execution gates: `src/portfolio/wallet.py`
- Daily posture / profit lock: `src/strategy/posture.py` and
  `src/strategy/supervisor.py`, with adapters in `src/runtime/posture_feed.py` and
  `src/portfolio/wallet.py`
- Daily clustering posture lane: `src/clustering/run_cluster.py`

## Wallet Execution Layer

Accepted flags become wallet inputs, not guaranteed fixed-size trades.

The wallet can:

- seed the initial position (`seed_buy`)
- scale BUY/SELL portions by phase behavior
- shrink sells during daily strong-up posture
- shrink buys during daily weak/down posture
- optionally execute forced post-ultra posture sells (`daily_posture_lock`)

Posture lock sells only exist when the profile enables the `ULTRA_EXIT_*`
surface. They are real executed sells and count in portfolio value, fees,
tax accounting, trade counts, and `GrossVsHODL`. They are not generated
`SELL` flags, so charts mark them separately.
