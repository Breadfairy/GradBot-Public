# TODO

## Active Decisions

- Current active candidate:
  - `inputs/profiles/user/results/best-config.json`
  - mirrored for deployment by `inputs/profiles/user/live-config.json`
  - current surface is no-1h-cluster, posture clustering, and
    `PEAK_LOCK_*` PID supervisor keys

- 1h signal clustering is quarantined from normal tune/backtest runtime.
  - Daily 1d clustering remains active wallet/execution logic.
  - Standalone clustering research code is retained under `src/clustering/`;
    supervised ML comparison code is quarantined under
    `src/research/quarantine/`.

- Stale profile clutter has been removed from `inputs/profiles/`.
  Keep only `codex/`, `codex/results/`, `user/`, and `user/results/`.

## Robust Selection

- Robust-region selector is implemented in `src/tune/robust.py`.
  - It exports `robust-candidates.csv`.
  - It exports `robust-row.csv`.
  - It writes `bestrobust*-config.json` candidates.
  - Holdout now reruns robust candidates when present.

- Suggested robust score ingredients:
  - tune gross-vs-hodl
  - local median gross-vs-hodl
  - local p25 gross-vs-hodl
  - local standard deviation
  - trade-count band penalty
  - max drawdown penalty
  - tune-to-holdout parity penalty after holdout rerun
  - multistart spread penalty after holdout rerun

## Clustering

- `flag_outcome` 1h cluster policy is quarantined.
  - Current evidence: `flag_outcome_t48/k4` can win tune by blocking sells
    and then underperform holdout.

- If 1h clustering is retested:
  - Keep it narrow around `heuristic_forward / 24h / k5-k6`.
  - Add explicit checks that the cluster gate is not just suppressing sells.

- 1h cluster tune axes/runtime branches have been removed from the normal
  C/Python profile>tune>holdout path.

- Improve daily 1d cluster handling.
  - Replace hard-coded posture-role ids with labels derived from daily
    cluster stats.
  - Keep current daily artifact fixed until the 1h clustering decision is
    closed.

## Sweep Simplification

- Use `FreezeLedger.md` as the active freeze/narrow/open record before
  building any large sweep.

- Freeze settled keys in normal sweeps:
  - `p1/p2/p3 = 12/20/55`
  - `MACRO_INTERVAL = 1d`
  - `MACRO_P1 = 7`
  - `MACRO_GRAD_PERIOD = 7`
  - `MACRO_P3 = 150`
  - trend/grad/cooldown/macro gates enabled
  - defense disabled
  - phase buy `3`
  - final portion `0.5`
  - cooldown `5`
  - daily lock and daily posture constants fixed unless explicitly testing
    daily posture.

- Keep future sweeps staged:
  - architecture switches first
  - macro/sell shape second
  - wallet/daily posture third
  - robustness validation last

- Cross-asset robustness build:
  - Stage 1: per-asset EMA period discovery to fit native cycle timing.
  - Stage 2: freeze the selected periods, then tune feature configurables
    around GRAD1, macro dyn, daily posture, and wallet lock behaviour.
  - Stage 3: build asset-native daily posture clustering and remap cluster
    labels into runtime semantics before using `DAILY_CLUSTER_PATH`.
  - Validate frozen configs by cross-asset holdout before allowing any
    asset-specific tune to influence shared defaults.

## Code Cleanup

- Continue deleting redundant branches rather than leaving inert toggles.

- Reduce profile/config clutter.
  - Keep a small set of named active profiles.
  - Move or delete stale exploratory profiles once their conclusions are
    recorded.

- Continue simplifying wallet logic.
  - Keep only wallet behaviours with measured impact.
  - Avoid reintroducing overlapping sell blockers.

- Keep docs current after each architecture decision.
  - Update `docs/README.md`, `docs/GATES.md`, `docs/CLUSTERING.md`, and
    `docs/RUNTIME_GATE_TREE.md` when clustering or gate logic changes.

- Align documentation after the overnight 1h clustering decision.
  - Keep `docs/RUNTIME_GATE_TREE.md`, `docs/CLUSTERING.md`,
    `scripts/PARAMS.md`, and `FreezeLedger.md` aligned with the simplified
    no-1h runtime.
