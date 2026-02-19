#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: run_oracle.sh <run_label> [oracle-config.json] [--add]" >&2
}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
LIB_DIR="$SCRIPT_DIR/lib"
source "$LIB_DIR/profile_path.sh"
DEFAULT_CFG="$ROOT_DIR/inputs/profiles/oracle-config.json"
PY_SCRIPT="$ROOT_DIR/src/oracles.py"
ORACLE_OUT="$ROOT_DIR/outputs/oracles"
PROFILES_DIR="$ROOT_DIR/inputs/profiles"

ADD_MODE=0
LABEL=""
CFG_INPUT=""
for arg in "$@"; do
  if [[ "$arg" == "--add" ]]; then
    ADD_MODE=1
    continue
  fi
  if [[ -z "$LABEL" ]]; then
    LABEL="$arg"
    continue
  fi
  if [[ -z "$CFG_INPUT" ]]; then
    CFG_INPUT="$arg"
    continue
  fi
  usage
  exit 1
done
if [[ -z "$LABEL" ]]; then
  usage
  exit 1
fi

if [[ -n "$CFG_INPUT" ]]; then
  CFG_PATH="$(resolve_profile_path "$CFG_INPUT" "$PROFILES_DIR")"
else
  CFG_PATH="$DEFAULT_CFG"
fi

if [[ ! -f "$CFG_PATH" ]]; then
  echo "[oracles] missing config file: $(basename "$CFG_PATH")" >&2
  exit 1
fi

OUT_DIR="$ORACLE_OUT/$LABEL"
if [[ -d "$OUT_DIR" ]]; then
  if [[ -t 1 ]]; then
    echo "[oracles] output directory exists: $LABEL"
    while [[ -d "$OUT_DIR" ]]; do
      read -r -p \
"[oracles] Enter a new label, 'E' (erase), '!' (abort): " NEW_LABEL
      if [[ "$NEW_LABEL" == "!" ]]; then
        echo "[oracles] aborted by user."
        exit 1
      fi
      if [[ "$NEW_LABEL" == "E" ]]; then
        rm -rf "$OUT_DIR"
        echo "[oracles] erased run '$LABEL'"
        break
      fi
      if [[ -z "$NEW_LABEL" ]]; then
        NEW_LABEL="${LABEL}-$(date +%Y%m%d-%H%M%S)"
      fi
      LABEL="$NEW_LABEL"
      OUT_DIR="$ORACLE_OUT/$LABEL"
    done
  else
    SUFFIX="$(date +%Y%m%d-%H%M%S)"
    LABEL="${LABEL}-${SUFFIX}"
    OUT_DIR="$ORACLE_OUT/$LABEL"
    echo "[oracles] output dir exists; using run '$LABEL'"
  fi
fi

mkdir -p "$OUT_DIR"

CFG_NAME="$(basename "$CFG_PATH")"
echo "[oracles] running oracle build with config: $CFG_NAME"
EXTRA_ARGS=""
if [[ "$ADD_MODE" -eq 1 ]]; then
  EXTRA_ARGS="--add"
fi
PYTHONPATH="$ROOT_DIR/src" python3 "$PY_SCRIPT" --config "$CFG_PATH" --label "$LABEL" $EXTRA_ARGS
echo "[oracles] configs written under inputs/profiles/ (*-full-config.json, *-mid-config.json)"
echo "[oracles] charts under outputs/oracles/$LABEL/<TICKER>/charts/"
echo "[oracles] stats + bells under outputs/oracles/$LABEL/<TICKER>/stats/"
