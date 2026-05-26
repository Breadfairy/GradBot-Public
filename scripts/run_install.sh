#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv/live}"
REQ_PATH="$ROOT_DIR/requirements/live.txt"

install_tmux() {
    if command -v tmux >/dev/null 2>&1; then
        return
    fi

    echo "tmux not found; attempting install"
    if command -v brew >/dev/null 2>&1; then
        brew install tmux
        return
    fi

    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y tmux
        return
    fi

    echo "tmux is required. Install it, then rerun scripts/run_install.sh." >&2
    exit 1
}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "python not found: $PYTHON_BIN" >&2
    exit 1
fi

install_tmux

if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y python3-venv
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    else
        echo "could not create Python venv at $VENV_DIR" >&2
        exit 1
    fi
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -r "$REQ_PATH"

for path in \
    "$ROOT_DIR/inputs/profiles/user/live-config.json" \
    "$ROOT_DIR/inputs/live/model/cluster_model.json"; do
    if [ ! -f "$path" ]; then
        echo "missing required live file: $path" >&2
        exit 1
    fi
done

echo "live install complete"
echo "venv: $VENV_DIR"
echo "next: scripts/run_live.sh"
