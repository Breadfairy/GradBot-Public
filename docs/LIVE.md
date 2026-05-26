# Live Runtime

Lean live runtime for spot trading with the same core signal/flag logic.

## Files
- `src/live/live.py`: live event loop and order execution.
- `inputs/profiles/user/live-config.json`: runtime profile.
- `inputs/live/model/linkusdt-6h-ema-fast-posture-k04-clustered_features.csv`:
  optimized posture cluster artifact used as an inference validation fixture.
- `inputs/live/model/cluster_model.json`: trained posture model used live.
- `inputs/live/model/cluster_policy.csv`: cluster id to posture policy
  reference.
- `inputs/live/config.ini`: local Binance API credentials INI.
- `outputs/live/sessions/<session-id>/`: per-run session directories containing
  snapshots, state, decisions, events, fills, and `session.json`.
- `outputs/live/latest_active.json`: pointer to the last active or closed
  session.

## Run
```bash
scripts/run_live.sh
```

`scripts/run_live.sh` calls `scripts/run_install.sh` automatically when the
live environment or required packages are missing. `run_install.sh` creates
`.venv/live`, installs `requirements/live.txt`, and ensures `tmux` is present
when Homebrew or apt is available.

`run_live` starts the paper/live runtime inside a `tmux` session named
`gradbot-live`. It wraps the process with `caffeinate` on macOS, or
`systemd-inhibit` on Linux when available.

The live build does not train or remap clustering artifacts. It applies the
exported `cluster_model.json` to fresh closed posture-interval candles.

At launch the runtime fetches enough 1h, macro, and posture candle context to
satisfy the warmup. If that context is not ready, startup fails before the
dashboard begins trading decisions.

Each fresh `run_live` creates a new session directory unless the last session
is still marked `active`, which means the prior process stopped uncleanly. A
clean `quit` marks the session `closed`, so the next `run_live` starts a new
epoch from scratch. Websocket reconnects inside one running process keep the
same session directory and restore that session state.

To start only the runtime against the current profile:
```bash
PYTHONPATH=src .venv/live/bin/python -m live.live
```

## Profile Notes
- `PAPER_TRADING`: `true` simulates fills against a paper wallet and never
  sends exchange orders.
- `LIVE_DRY_RUN`: `true` reads live balances and quotes but does not send
  exchange orders. It is ignored when `PAPER_TRADING` is `true`.
- `tickers[0]` and `intervals[0]` are used.
- `history_days` defines the fetched live candle context.
- `primer_days` defines the warmup before live signals are considered valid.
- Flag, phase, macro, and wallet posture keys mirror the tuned profile.
- `DAILY_CLUSTER_MODEL_PATH` points at the bundled trained posture model.
- Posture inference uses only closed posture rows and infers the current
  posture from the trained model at runtime. Startup also hydrates the dashboard
  posture immediately after the warmup check.
- The dashboard shows asset/account state, the current applied daily posture,
  static trade count, elapsed runtime, a command line, then newest-first
  command and trade history.
- The command menu supports `quit`, `seed`, `pause`, `resume`, manual
  `buy`/`sell`, and `cls ord N` for clearing or canceling a numbered open
  order. The old `status` command is intentionally removed.
- Tips rotate every 10 seconds with examples like `buy 55%`, `sell $400`,
  and `cls ord 4`.
- Visible dashboard values are rounded for readability. The top price line
  shows the current symbol price plus percent move from the latest closed 1h
  candle.
- The lower pane shows `OPEN ORDERS` above `HISTORY`. Pending orders remain
  visible until canceled or filled; filled rows linger for one minute.
- Visible history rows show trades only. Command/runtime history still goes
  to `*_events.csv` but is not printed in the dashboard history pane.
- Trade rows show base quantity plus quote value and hide raw exchange/order
  ids.
- Non-paper modes start with auto trading paused. Run `resume` explicitly
  after checking balances, profile, and seed state.
- The dashboard heartbeat marks stream, candle, and quote data as `STALE`
  when live updates stop arriving.
- Startup and live stream disconnects retry forever with backoff. On reconnect,
  paper mode replays missed closed candles in order and writes simulated fills
  into the normal trade log. Live and dry-run modes still summarize missed
  candles without retroactive orders.
- `WALLET_SEED_QUOTE` sets starting paper quote cash when `PAPER_TRADING` is
  enabled. Paper mode does not auto-buy the asset on startup.
- `WALLET_FEE_RATE` is applied to simulated fills.
- `config_path` is only needed when `PAPER_TRADING` is `false`.
- `out` names the snapshot CSV file inside the current session directory;
  trades are written beside it as `*_trades.csv`.
- Closed-candle decision audit rows are written beside `out` as
  `*_decisions.csv`. This file is one append-only CSV, not monthly chunked.
- Decision rows are written for every new closed 1h candle, including paused
  periods. They contain OHLCV, account values, daily posture, macro state,
  trend/z-score state, accepted flags, final action, and reason text.
- Trade rows include the same core decision context so each fill can be
  reviewed without joining back to the decision log.
- Rolling state is written beside `out` as `*_state.csv`. It is restored only
  when resuming an unclosed active session.
- Event history is written beside `out` as `*_events.csv`.
- Snapshot, trade, decision, and event rows include `profile_hash`,
  `model_hash`, `code_version`, and `session_id` for later analysis.

## Current Deployment Profile

- Source family: `inputs/profiles/user/results/best-config.json`.
- Selected live profile: current scalar PID posture config.
- Symbol/interval: `LINKUSDT` on `1h`.
- Posture model: `inputs/live/model/cluster_model.json`.
- Paper mode is enabled until `inputs/profiles/user/live-config.json` is
  deliberately changed.

Live profile promotion is manual. Copy the selected tuning result into
`inputs/profiles/user/live-config.json`, then preserve the live-only keys for
paper/live mode, credentials path, history window, model path, and output name.
