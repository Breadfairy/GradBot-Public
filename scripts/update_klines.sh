#!/usr/bin/env bash
set -euo pipefail

# Update all cached klines under inputs/klines by forward-filling to now.
# Usage: ./update_klines.sh

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

exec python3 "$ROOT_DIR/src/klines_tools.py" update
