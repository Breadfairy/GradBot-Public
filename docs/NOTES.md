# Notes

## Macro dynamics (high level)

- Computed on `MACRO_INTERVAL` using EMA1/EMA2/EMA3.
- Produces a signed dyn% that is aligned to micro candles using last-known
  macro samples (no interpolation).
- Key knobs:
  - `MACRO_NRG_*`, `MACRO_DYN_PCT_*`
  - `MACRO_GRAD_*`, `MACRO_MULT_GRAD_*`
  - `MACRO_BUY_MULT_*`, `MACRO_SELL_MULT_*`
- Code:
  - core math: `src/engine_core.py` (`macroDynFromMas`)
  - wrapper/alignment: `src/dynamics.py`

## Micro gates

- Gradient gate: `GRAD1_*` (rolling z-score threshold on g1(p1)).
- Spacing gate: `SPACING_*` (rolling z-score thresholds on ma12/ma23 legs).
- Energy: `MICRO_NRG_*` (regime-summed accumulator + rolling z-score).
