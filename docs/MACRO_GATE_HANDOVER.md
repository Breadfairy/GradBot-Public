# Macro Gate Handover

Historical note:

- This document records the earlier `rev/roll` exploration branch.
- The current engine no longer uses the macro state multiplier family:
  `MACRO_*_MULT_BULL`, `MACRO_*_MULT_BEAR`, `MACRO_*_MULT_REV`, or
  `MACRO_*_MULT_ROLL`.
- For current live behavior, prefer `docs/GATES.md`.

This note is the handover for that earlier macro gate testing sequence.

## Goal

The target behavior is:

- identify flags just before a significant increase
- identify flags just before a significant decrease
- make the macro gate keys meaningfully active instead of sweeping dead or
  mostly redundant ranges
- understand whether missed ramp exposure is a gating problem, a sizing
  problem, or both

The practical concern behind this work is missing full investment exposure
before a price ramp. The likely next design area is buy sizing / buy phase
weighting, but first the gate ranges need to be in a responsive zone.

## Active Code Path

Relevant files:

- `src/engine/core.py`
- `src/native/engine/engine.c`
- `src/engine/macro_view.py`
- `src/runtime/diag.py`
- `src/runtime/gates.py`
- `src/tune/trace.py`
- `src/portfolio/wallet.py`

Important runtime facts:

- micro structure creates candidate BUY / SELL lanes
- the macro layer does not create trades by itself
- the macro layer changes the required move threshold
- buy sizing still happens later in `src/portfolio/wallet.py`

## Historical Macro Logic

At the time of this exploration, macro required move was described as:

1. `EMA(P1) - EMA(P3)` spread on the macro timeframe
2. rolling z-score of spread size
3. mapping that z-score into a dyn percent using:
   - `MACRO_NRG_WIN_DAYS`
   - `MACRO_NRG_Z_MIN`
   - `MACRO_NRG_Z_MAX`
   - `MACRO_DYN_PCT_MIN`
   - `MACRO_DYN_PCT_MAX`
4. spread-collapse damping
5. gradient multiplier layer from:
   - `MACRO_GRAD_PERIOD`
   - `MACRO_GRAD_WIN_DAYS`
   - `MACRO_GRAD_Z_MAX`
   - `MACRO_MULT_GRAD_MAX`
6. state multipliers:
   - bull / bear
   - rev / roll

Current code still uses linear spread-collapse damping:

- Python: `src/engine/core.py`
- C: `src/native/engine/engine.c`

This was changed from `ratio^2` to `ratio`.

## Historical State Definitions

Earlier flag experiments used:

- `bull`: `P1 > P3` and not roll
- `bear`: `P1 < P3` and not rev
- `rev`: bear backbone, but `P1 > GRAD_PERIOD`
- `roll`: bull backbone, but `P1 < GRAD_PERIOD`

Interpretation:

- `rev` = early upside reversal inside bearish macro structure
- `roll` = early downside rollover inside bullish macro structure

## What Was Learned From The Huge Sweep

Run:

- `outputs/tuning/huge`

Important result:

- the rev / roll multiplier axes were completely dead in that sweep

Measured inactivity:

- `MACRO_BUY_MULT_REV`: 100% inactive
- `MACRO_BUY_MULT_ROLL`: 100% inactive
- `MACRO_SELL_MULT_REV`: 100% inactive
- `MACRO_SELL_MULT_ROLL`: 100% inactive

That did not mean the code path was broken. It meant the tested ranges and
state construction were not producing decision-sensitive rev / roll behavior.

Macro sensitivity from that run:

- strongest:
  - `MACRO_DYN_PCT_MIN`
  - `MACRO_GRAD_WIN_DAYS`
  - `MACRO_GRAD_Z_MAX`
- meaningful:
  - `MACRO_P3`
  - `MACRO_MULT_GRAD_MAX`
  - `MACRO_DYN_PCT_MAX`
  - `MACRO_NRG_WIN_DAYS`
- weak:
  - `MACRO_NRG_Z_MAX`

Read:

- `MACRO_NRG_WIN_DAYS` was active, but not dominant
- `MACRO_NRG_Z_MAX` was too often on a flat part of the mapping
- the tested `Z_MAX` bounds likely missed the threshold region where the
  decision boundary moves

## Highscore Config From Huge

Derived post-selection config:

- `outputs/tuning/huge/best-configs/highscore-config.json`

This was chosen because it held up better than the raw tune winner on the
reconstructed holdout window.

## State Occupancy Report

The old one-off `src/macro_state_report.py` utility was removed during the
strategy consolidation cleanup. Recreate state occupancy diagnostics from
`src/runtime/diag.py` if this analysis is needed again.

Important finding on the huge highscore config:

- `rev = 0`
- `roll = 0`

So the rev / roll multipliers were structurally unused on that window.

## Focused Rev / Roll Test

New focused profile:

- `inputs/profiles/codex/link-revroll-linear.json`

Design intent:

- keep the search small
- separate `MACRO_GRAD_PERIOD` from `MACRO_P1`
- use linear damping
- force rev / roll states to exist

Sweep size:

- `19,683` combos

Run:

- `outputs/tuning/revroll-linear`

## What Was Learned From Revroll-Linear

This run succeeded in activating rev / roll.

For the tune winner:

- `bull 20.26%`
- `roll 13.36%`
- `bear 47.42%`
- `rev 18.95%`

For the stats winner:

- `bull 20.82%`
- `roll 12.81%`
- `bear 47.97%`
- `rev 18.40%`

All four rev / roll multiplier axes became active:

- `MACRO_BUY_MULT_REV`
- `MACRO_BUY_MULT_ROLL`
- `MACRO_SELL_MULT_REV`
- `MACRO_SELL_MULT_ROLL`

Measured mean effect ranges from that sweep:

- `MACRO_BUY_MULT_REV`: ~3.41
- `MACRO_BUY_MULT_ROLL`: ~12.44
- `MACRO_SELL_MULT_REV`: ~7.68
- `MACRO_SELL_MULT_ROLL`: ~10.04

Read:

- roll matters more than rev in the tested region
- sell-side roll also matters materially
- rev / roll are now real levers, but they still are not the strongest
  macro levers

Stronger than the rev / roll multipliers in that run:

- `MACRO_GRAD_PERIOD`
- `MACRO_DYN_PCT_MIN`

This means the focused rev / roll test proved usefulness, but it still did
not isolate the transition multipliers completely.

## Current Interpretation

This section is historical. The active build no longer uses the rev/roll
state multiplier family documented above. Current macro behavior is covered
in `docs/GATES.md`:

- macro dyn percent from the macro spread
- macro gradient multiplier
- optional sell relax
- optional cluster required-move multiplier

Daily posture profit locking now lives in wallet execution, not in macro
state multipliers.

## Historical Interpretation

The sequence so far suggests:

1. dead axes were caused by the tested state construction, not necessarily by
   missing code
2. linear damping plus separated `MACRO_GRAD_PERIOD` makes rev / roll appear
3. `ROLL` currently looks more influential than `REV`
4. base macro bounds still dominate a lot of the result spread
5. missed ramp capture may still be partly a buy sizing problem after the
   gate is improved

## Historical Next Recommended Steps

These steps are retained for context only. They are not the current plan.
The current plan is in `plan.md`.

1. Freeze the stronger base macro levers more tightly:
   - `MACRO_GRAD_PERIOD`
   - `MACRO_DYN_PCT_MIN`
   - `MACRO_P3`
   - `MACRO_GRAD_WIN_DAYS`

2. Run a tighter multiplier-only or near-multiplier-only sweep on:
   - `MACRO_BUY_MULT_REV`
   - `MACRO_BUY_MULT_ROLL`
   - `MACRO_SELL_MULT_REV`
   - `MACRO_SELL_MULT_ROLL`

3. Add diagnostics aimed at ramp capture, not just gross return:
   - time from first accepted BUY in a phase to local ramp start
   - fraction of available cash deployed before a strong upside move
   - fraction of base position exited before a strong downside move
   - phase-level MFE / MAE after first accepted signal
   - capture ratio vs buy-and-hold over major trend legs

4. Once gates are in a responsive zone, examine buy weighting in
   `src/portfolio/wallet.py`

The likely design question is:

- should buy phases stay equal-weight
- or should early accepted BUYs in a bear/rev context get more capital so the
  strategy captures more exposure before the ramp accelerates

## Practical Commands

Run focused tune:

```bash
scripts/run_tune.sh codex/link-revroll-linear.json revroll-linear
```

## Output Directories To Review

- `outputs/tuning/huge`
- `outputs/tuning/revroll-linear`

Most useful files:

- `results.csv`
- `best-row.csv`
- `stats-row.csv`
- `best-configs/best-config.json`
- `best-configs/beststats-config.json`
- `run.log`
- `host-spec/meta.txt`
