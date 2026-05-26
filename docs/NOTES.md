# Notes

This file is for active caveats that are easy to miss while reading code.

## Runtime shape

- The active tune path is C sweep plus Python post handling.
- There is no active Python row-by-row tuner loop.
- There is no active cache/oracle path in the current tuning runtime.
- The active deployment runtime is isolated under `src/live/`.

## Macro alignment

- Macro values are computed on `MACRO_INTERVAL`.
- Macro values are aligned to micro candles with last-known-sample carry.
- There is no interpolation from macro candles into micro candles.

## Active micro gates

- The active micro gate is `GRAD1_*`.
- The active side regime gate is the full micro trend ordering.
- The active repeat limiter is `COOLDOWN`.
- The active spacing gate is macro dyn percent since the phase anchor or
  last accepted same-side flag.
- Optional clustering context multiplies that same spacing requirement in C.
- There is no active micro-energy gate in the Python flagger.

## Macro overlap

Macro strictness can change in multiple layers at once:

- spread-collapse damping from the `P1 vs P3` backbone
- grad-based scaling from `MACRO_GRAD_PERIOD`
- explicit bull / bear side multipliers

That overlap is real. Do not assume one macro knob is acting alone near a
transition.

## Key naming

- `MACRO_P2` is legacy.
- The active middle macro EMA key is `MACRO_GRAD_PERIOD`.
- `src/config/profile.py` intentionally rejects `MACRO_P2`.

## Holdout

- Holdout is a post-tune check, not a tuning loop.
- Repeatedly adjusting config from the same holdout turns it into training.
- `fingerprint.json` persists the concrete run anchor in `anchorMs`.
- `fingerprint.json` also marks whether that anchor came from runtime,
  `TUNE_ANCHOR_DATE`, or legacy `TUNE_ANCHOR_MS`.
- Single-profile work now uses a scalar one-combination tuning profile.
- Anchored single-profile traces should use `TUNE_ANCHOR_DATE` or
  `TUNE_ANCHOR_MS`.

## Authority

Inside a tuning run, these files are the authoritative outputs:

- `results.csv`
- `best-row.csv`
- `stats-row.csv`
- `best-configs/best-config.json`
- `best-configs/beststats-config.json`

## Charts

- `timVal` now has an allocation subplot for `EDGE Asset %` and
  `EDGE USDT %`.
- Use it to spot partial-liquidation behavior before ramps.
- Price charts distinguish normal wallet buys/sells from seed buys and daily
  posture lock sells.
- Purple/magenta daily lock markers are executed `daily_posture_lock` sells,
  not generated `SELL` flags. They appear only when the profile enables the
  `ULTRA_EXIT_*` surface.

## Wallet posture

- Floors, daily posture, and optional post-ultra locks live in wallet
  execution, not in the ordinary flag generator.
- Post-ultra lock sells are real trades and affect fees, tax accounting,
  equity, trade counts, and `GrossVsHODL`.
- Marker audit should show zero unexplained wallet trades; seed buys and
  daily lock sells are expected synthetic wallet actions.

## Live deployment

- `inputs/profiles/user/live-config.json` is set up around the current scalar PID posture
  profile while keeping the bundled live cluster model paths.
- Live posture inference uses `inputs/live/model/cluster_model.json` and
  closed posture-interval candles.
- Tune/trace posture now aligns shifted-open: the latest completed
  posture `closeMs` is selected at each 1h candle open, so posture candle
  `D` applies from the open of `D+1`.
- The model inference was checked against the exported clustered posture
  feature artifact and reproduced the historical labels exactly.
- Launch runs a warmup readiness check before the first live 1h assessment.
- `PAPER_TRADING` is currently true in the bundled live profile.
- `inputs/live/config.ini` is the local live credential path. Live exchange
  trading only starts after the profile is changed out of paper mode.
- Live runtime logs are now per-session under
  `outputs/live/sessions/<session-id>/`.
- A clean `quit` closes the session. The next `run_live` starts a fresh epoch.
- Only an unclosed active session is resumed on a later `run_live`, which is
  the crash/timeout recovery path.
- Startup hydrates the dashboard's current posture from the latest valid
  causal posture after warmup readiness.
- The live runtime writes snapshot, trade, decision, event, and rolling state
  CSVs beside the session-local `out` path.
- `*_decisions.csv` is a single append-only closed-candle audit file. It logs
  every new closed 1h candle, including paused periods, with applied daily
  posture, macro state, micro trend/z-score state, accepted flags, final
  action, and reason text.
- Trade rows carry the same core decision context as the decision row so fills
  remain self-explaining during post-run review.
- Live signal generation now imports the shared runtime flagger and gate
  helpers instead of maintaining a separate copy of the flag logic.
- Live peak-lock/PID support is wired through the shared strategy supervisor
  and the bundled profile carries the selected `PEAK_LOCK_*` keys.
- The live dashboard history pane shows trade rows only. Commands/runtime
  messages remain in `*_events.csv`.
- The live dashboard has an `OPEN ORDERS` pane. Pending orders remain visible;
  filled orders linger for one minute, and `cls ord N` cancels or clears a
  numbered stale row.
- The rolling `*_state.csv` file stores the latest dashboard/runtime state so
  an unclosed active session can restore local phase context. In paper mode,
  missed closed candles are replayed into the paper wallet/trade log; live
  modes still summarize them only. It is not a replacement for exchange
  records.
- The rolling state CSV also persists peak-lock/PID supervisor state when that
  supervisor is active, so restart resumes the controller state rather than
  recreating it from price history alone.

## Analysis tools

- `src/tools/marker_audit.py`: flag-vs-wallet marker validation.
- `src/tools/lock_sweep.py`: compact daily profit-lock sweep.
