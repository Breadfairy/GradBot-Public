#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
SESSION="${LIVE_TMUX_SESSION:-gradbot-live}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv/live}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
INSTALL_PATH="$SCRIPT_DIR/run_install.sh"

needs_install() {
    if [ ! -x "$PYTHON_BIN" ]; then
        return 0
    fi
    if ! "$PYTHON_BIN" -c "import aiohttp, binance, certifi, numpy" \
        >/dev/null 2>&1; then
        return 0
    fi
    if ! command -v tmux >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

if needs_install; then
    "$INSTALL_PATH"
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    if [ "$#" -gt 0 ]; then
        echo "cannot pass warm-start args to existing tmux session" >&2
        echo "session: $SESSION" >&2
        exit 1
    fi
    echo "attaching existing tmux session: $SESSION"
    exec tmux attach-session -t "$SESSION"
fi

INHIBIT=""
if command -v caffeinate >/dev/null 2>&1; then
    INHIBIT="caffeinate -dimsu"
elif command -v systemd-inhibit >/dev/null 2>&1; then
    INHIBIT="systemd-inhibit --what=idle:sleep --why=gradbot-live"
else
    echo "no sleep inhibitor found; starting without caffeinate"
fi

LIVE_CMD="cd \"$ROOT_DIR\" && export PYTHONPATH=\"$ROOT_DIR/src\${PYTHONPATH:+:\$PYTHONPATH}\" && exec "
if [ -n "$INHIBIT" ]; then
    LIVE_CMD="$LIVE_CMD$INHIBIT "
fi
LIVE_CMD="${LIVE_CMD}\"$PYTHON_BIN\" -m live.live"
for arg in "$@"; do
    LIVE_CMD="$LIVE_CMD $(printf '%q' "$arg")"
done

tmux new-session -d -s "$SESSION" -c "$ROOT_DIR" "$LIVE_CMD"
exec tmux attach-session -t "$SESSION"
