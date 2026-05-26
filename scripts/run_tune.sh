#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: run_tune.sh <profile.json> <run_label> [KEY=VALUE ...]" >&2
  echo "example: scripts/run_tune.sh user-config.json run1" >&2
  exit 1
fi

PROFILE_INPUT="$1"
RUN_LABEL="$2"
shift 2
if [[ $# -gt 0 ]]; then
  for kv in "$@"; do
    if [[ "$kv" == *=* ]]; then
      export "$kv"
    fi
  done
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
LIB_DIR="$SCRIPT_DIR/lib"
source "$LIB_DIR/profile_path.sh"
PROFILES_DIR="$ROOT_DIR/inputs/profiles"
OUTPUTS_DIR="$ROOT_DIR/outputs/tuning"
RUN_DIR="$OUTPUTS_DIR/$RUN_LABEL"
MPL_DIR="$ROOT_DIR/.mplconfig"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHON_BIN
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$ROOT_DIR/src:$PYTHONPATH"
else
  export PYTHONPATH="$ROOT_DIR/src"
fi

# Allow short profile names from inputs/profiles
PROFILE_ABS="$(resolve_profile_path "$PROFILE_INPUT" "$PROFILES_DIR")"

# Avoid clobbering existing runs: prompt for a new label or erase.
if [[ -d "$RUN_DIR" ]]; then
  if [[ -t 1 ]]; then
    echo "[tune] output directory exists: $RUN_LABEL"
    while [[ -d "$RUN_DIR" ]]; do
      read -r -p \
        "[tune] Enter a new run label, 'E' (erase), '!' (abort): " NEW_LABEL
      if [[ "$NEW_LABEL" == "!" ]]; then
        echo "[tune] aborted by user."
        exit 1
      fi
      if [[ "$NEW_LABEL" == "E" ]]; then
        rm -rf "$RUN_DIR"
        echo "[tune] erased run '$RUN_LABEL'"
        break
      fi
      if [[ -z "$NEW_LABEL" ]]; then
        NEW_LABEL="${RUN_LABEL}-$(date +%Y%m%d-%H%M%S)"
      fi
      RUN_LABEL="$NEW_LABEL"
      RUN_DIR="$OUTPUTS_DIR/$RUN_LABEL"
    done
  else
    SUFFIX="$(date +%Y%m%d-%H%M%S)"
    RUN_LABEL="${RUN_LABEL}-${SUFFIX}"
    RUN_DIR="$OUTPUTS_DIR/$RUN_LABEL"
    echo "[tune] output dir exists; using run '$RUN_LABEL'"
  fi
fi

mkdir -p "$RUN_DIR"
mkdir -p "$MPL_DIR"
export MPLCONFIGDIR="$MPL_DIR"

ANCHOR_ARGS=()
if [[ -n "${TUNE_ANCHOR_MS:-}" ]] && [[ -n "${TUNE_ANCHOR_DATE:-}" ]]; then
  echo "[tune] use only one of TUNE_ANCHOR_MS or TUNE_ANCHOR_DATE" >&2
  exit 1
fi
if [[ -n "${TUNE_ANCHOR_MS:-}" ]]; then
  ANCHOR_ARGS+=(--anchor-ms "$TUNE_ANCHOR_MS")
fi
if [[ -n "${TUNE_ANCHOR_DATE:-}" ]]; then
  ANCHOR_ARGS+=(--anchor-date "$TUNE_ANCHOR_DATE")
fi

if [[ ${#ANCHOR_ARGS[@]} -gt 0 ]]; then
  "$PYTHON_BIN" -m tune.run \
    --profile "$PROFILE_ABS" \
    --label "$RUN_LABEL" \
    --out "$RUN_DIR" \
    "${ANCHOR_ARGS[@]}"
else
  "$PYTHON_BIN" -m tune.run \
    --profile "$PROFILE_ABS" \
    --label "$RUN_LABEL" \
    --out "$RUN_DIR"
fi
