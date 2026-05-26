#!/usr/bin/env bash
set -euo pipefail

# Update all local klines under inputs/klines by forward-filling to now.
# Usage: ./update_klines.sh

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

exec python3 -m data.prepare_klines update
