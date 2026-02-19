#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
CACHE_DIR="${GRADBOT_CACHE_DIR:-$ROOT_DIR/cache}"
CACHE_LABEL="${CACHE_DIR#$ROOT_DIR/}"
if [[ "$CACHE_LABEL" == "$CACHE_DIR" ]]; then
  CACHE_LABEL="$(basename "$CACHE_DIR")"
fi

usage() {
  cat <<'EOF'
Usage: clean_cache.sh --all | --results | --older-than DAYS | --keep-size GB

Options
  --all           Remove the entire cache directory.
  --results       Remove only cached backtest results (cache/results).
  --older-than D  Delete cache shards older than D days.
  --keep-size G   Trim cache shards until total size <= G gigabytes.

GRADBOT_CACHE_DIR overrides the cache root (default: <repo>/cache).
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

if [[ ! -d "$CACHE_DIR" ]] && [[ "$1" != "--all" ]]; then
  echo "[cache] directory not found: $CACHE_LABEL"
  exit 0
fi

cmd="$1"

delete_dirs() {
  local -a dirs=("$@")
  if [[ ${#dirs[@]} -eq 0 ]]; then
    echo "[cache] nothing to delete."
    return
  fi
  for dir in "${dirs[@]}"; do
    rel="${dir#$CACHE_DIR/}"
    rm -rf "$dir"
    if [[ -n "$rel" && "$rel" != "$dir" ]]; then
      echo "[cache] removed $rel"
    else
      echo "[cache] removed $(basename "$dir")"
    fi
  done
}

case "$cmd" in
  --all)
    if [[ -d "$CACHE_DIR" ]]; then
      rm -rf "$CACHE_DIR"
      echo "[cache] cleared $CACHE_LABEL"
    else
      echo "[cache] directory not present: $CACHE_LABEL"
    fi
    ;;
  --results)
    results_dir="$CACHE_DIR/results"
    results_label="${results_dir#$CACHE_DIR/}"
    if [[ "$results_label" == "$results_dir" ]]; then
      results_label="$(basename "$results_dir")"
    fi
    if [[ -d "$results_dir" ]]; then
      rm -rf "$results_dir"
      echo "[cache] cleared result cache at $results_label"
    else
      echo "[cache] result cache not present: $results_label"
    fi
    ;;
  --older-than)
    if [[ $# -lt 2 ]]; then
      usage
      exit 1
    fi
    days="$2"
    targets=()
    while IFS= read -r -d '' path; do
      targets+=("$path")
    done < <(
      find "$CACHE_DIR" -mindepth 4 -maxdepth 4 -type d \
        -mtime +"$days" -print0
    )
    delete_dirs "${targets[@]}"
    ;;
  --keep-size)
    if [[ $# -lt 2 ]]; then
      usage
      exit 1
    fi
    python3 - "$CACHE_DIR" "$2" <<'PY'
import os
import shutil
import sys

root = sys.argv[1]
limit_gb = float(sys.argv[2])
limit_bytes = max(0, int(limit_gb * (1024 ** 3)))

shards = []
for ticker in os.listdir(root):
    t_dir = os.path.join(root, ticker)
    if not os.path.isdir(t_dir):
        continue
    for interval in os.listdir(t_dir):
        i_dir = os.path.join(t_dir, interval)
        if not os.path.isdir(i_dir):
            continue
        for days in os.listdir(i_dir):
            d_dir = os.path.join(i_dir, days)
            if not os.path.isdir(d_dir):
                continue
            for digest in os.listdir(d_dir):
                path = os.path.join(d_dir, digest)
                if not os.path.isdir(path):
                    continue
                size = 0
                for walk_root, _, files in os.walk(path):
                    for name in files:
                        fp = os.path.join(walk_root, name)
                        try:
                            size += os.path.getsize(fp)
                        except OSError:
                            pass
                shards.append((os.path.getmtime(path), size, path))

total = sum(size for _, size, _ in shards)
if total <= limit_bytes:
    print(f"[cache] total {total / (1024 ** 3):.3f} GB within limit.")
    sys.exit(0)

shards.sort()  # oldest first
for _, size, path in shards:
    if total <= limit_bytes:
        break
    shutil.rmtree(path, ignore_errors=True)
    total -= size
    rel = os.path.relpath(path, root)
    print(
        f"[cache] removed {rel} "
        f"({size / (1024 ** 2):.2f} MB)"
    )

print(f"[cache] final size {total / (1024 ** 3):.3f} GB")
PY
    ;;
  *)
    usage
    exit 1
    ;;
esac
