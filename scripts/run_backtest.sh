#!/usr/bin/env bash
set -euo pipefail

# Simple backtest runner. Intended to be run from the scripts/ directory.
# Creates outputs/tuning/<name>/ with charts and a transactions log.

if [[ $# -lt 2 ]]; then
  echo "usage: run_backtest.sh <profile.json|tune_run_dir> <name>" >&2
  echo "example: scripts/run_backtest.sh inputs/profiles/btc.json demo-run" >&2
  exit 1
fi

PROFILE="$1"
NAME="$2"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
LIB_DIR="$SCRIPT_DIR/lib"
source "$LIB_DIR/profile_path.sh"
OUTPUTS_DIR="$ROOT_DIR/outputs/tuning"
if [[ -z "$NAME" ]]; then
  NAME="backtest-$(date +%Y%m%d-%H%M%S)"
fi

OUT_DIR="$OUTPUTS_DIR/$NAME"
mkdir -p "$OUT_DIR"

# Holdout mode: if given a tuning run directory, run best/stats holdout summaries.
if [[ -d "$PROFILE" ]] && [[ -f "$PROFILE/best-configs/best-config.json" ]]; then
  BEST_CFG="$PROFILE/best-configs/best-config.json"
  STATS_CFG="$PROFILE/best-configs/beststats-config.json"
  CHARTS_ROOT="$PROFILE/charts/holdout"
  mkdir -p "$CHARTS_ROOT"
  python3 "$ROOT_DIR/src/holdout.py" \
    --best "$BEST_CFG" \
    --stats "$STATS_CFG" \
    --charts-root "$CHARTS_ROOT"
  exit 0
fi

# Allow short profile names from inputs/profiles
PROFILES_DIR="$ROOT_DIR/inputs/profiles"
PROFILE_ABS="$(resolve_profile_path "$PROFILE" "$PROFILES_DIR")"

echo "Running backtest..."
CHARTS_OUT_DIR="$OUT_DIR/charts" \
python3 "$ROOT_DIR/src/main.py" \
  backtest --profile "$PROFILE_ABS" --charts 2>&1 | tee "$OUT_DIR/transactions.txt"
echo "Artifacts generated"
