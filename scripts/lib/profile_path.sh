#!/usr/bin/env bash

# resolve_profile_path INPUT DIR
# Resolves short profile names within DIR to absolute paths.
resolve_profile_path() {
  local input="$1"
  local base_dir="$2"
  local candidate="$input"
  local subdir=""
  if [[ "$candidate" != /* ]] && [[ -f "$base_dir/$candidate" ]]; then
    candidate="$base_dir/$candidate"
  elif [[ "$candidate" != */* ]]; then
    for subdir in user user/results codex codex/results; do
      if [[ -f "$base_dir/$subdir/$candidate" ]]; then
        candidate="$base_dir/$subdir/$candidate"
        break
      fi
    done
  fi
  printf '%s/%s\n' "$(cd "$(dirname "$candidate")" && pwd)" "$(basename "$candidate")"
}
