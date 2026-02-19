#!/usr/bin/env bash
set -euo pipefail

# Fetch klines for a ticker+interval into inputs/klines, creating the CSV
# if missing or extending history backward/forward if it already exists.
# Usage: ./get_klines.sh BTCUSDT 30m [DAYS]

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <TICKER> <INTERVAL> [DAYS]" >&2
  exit 1
fi

TICKER="$1"
INTERVAL="$2"
WINDOW="${3:-365}"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

exec python3 "$ROOT_DIR/src/klines_tools.py" get "$TICKER" "$INTERVAL" "$WINDOW"
