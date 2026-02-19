#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

PYTHONPATH="$ROOT_DIR/src" python3 "$ROOT_DIR/src/tune_helper.py" backfill --root "$ROOT_DIR"
