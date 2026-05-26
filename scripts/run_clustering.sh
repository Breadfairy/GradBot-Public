#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
DEFAULT_CONFIG="$ROOT_DIR/inputs/clustering/current/linkusdt-6h-ema-fast-posture.json"
CONFIG_PATH="${1:-$DEFAULT_CONFIG}"
MPL_DIR="$ROOT_DIR/.mplconfig"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$MPL_DIR"
export MPLCONFIGDIR="$MPL_DIR"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-4}"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m clustering.run_cluster --config "$CONFIG_PATH"
