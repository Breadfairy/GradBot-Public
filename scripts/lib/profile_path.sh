#!/usr/bin/env bash

# resolve_profile_path INPUT DIR
# Resolves short profile names within DIR to absolute paths.
resolve_profile_path() {
  local input="$1"
  local base_dir="$2"
  local candidate="$input"
  if [[ "$candidate" != /* ]] && [[ -f "$base_dir/$candidate" ]]; then
    candidate="$base_dir/$candidate"
  fi
  printf '%s/%s\n' "$(cd "$(dirname "$candidate")" && pwd)" "$(basename "$candidate")"
}
