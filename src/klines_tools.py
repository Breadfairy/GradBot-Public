#!/usr/bin/env python3
"""
klines_tools.py – manage on-disk klines cache under inputs/klines

Commands (run from scripts/ or anywhere in repo):
  - update: forward-fill all cached CSVs to now using Binance API
  - get <TICKER> <INTERVAL> [DAYS]: fetch and cache if missing

Requires src/config.ini with Binance read-only API keys.
"""

from __future__ import annotations

import os
import sys
import csv
import math
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from binance_io import (
    getClient,
    showStatus,
    getKlines,
    loadCachedKlines,
    normalizeInterval,
)


def _ticker_dir(ticker: str) -> str:
    return os.path.join(REPO_ROOT, "inputs", "klines", ticker.upper())


def _csv_path(ticker: str, interval: str) -> str:
    fname = f"{ticker.lower()}_{interval}.csv"
    return os.path.join(_ticker_dir(ticker), fname)


def _read_open_times(csv_path: str) -> tuple[int | None, int | None]:
    first = None
    last = None
    with open(csv_path, newline="") as fh:
        r = csv.reader(fh)
        for row in r:
            if not row:
                continue
            t = int(float(row[0]))
            if first is None:
                first = t
            last = t
    return first, last


def _days_span_since(ms: int) -> int:
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    delta_ms = max(0, now - ms)
    return int(math.ceil(delta_ms / (24 * 60 * 60 * 1000)))


def update_all() -> None:
    root = os.path.join(REPO_ROOT, "inputs", "klines")
    to_update: list[tuple[str, str]] = []
    for sym in sorted(os.listdir(root)):
        sym_dir = os.path.join(root, sym)
        if not os.path.isdir(sym_dir):
            continue
        for fn in sorted(os.listdir(sym_dir)):
            if not fn.endswith(".csv") or "_" not in fn:
                continue
            base, _ext = os.path.splitext(fn)
            parts = base.split("_")
            if len(parts) < 2:
                continue
            ticker_lower = parts[0]
            interval = parts[-1]
            to_update.append((ticker_lower.upper(), interval))

    client = getClient()
    showStatus(client)
    for ticker, interval in to_update:
        path = _csv_path(ticker, interval)
        first, last = _read_open_times(path)
        days = _days_span_since(first)
        print(
            f"[update] {ticker} {interval} → days={days} "
            f"(first={first}, last={last})"
        )
        # This will forward-fill missing bars and rewrite a deduped file
        getKlines(client, ticker, interval, days, minCandles=None)


def get_once(ticker: str, interval: str, days: int) -> None:
    """Fetch klines for a window and merge with existing CSV if present.

    Unlike the previous behavior, this function now extends history
    backwards when a CSV already exists, and forward-fills to now when
    needed. It relies on getKlines() to dedup + sort before writing.
    """
    interval = normalizeInterval(interval)
    path = _csv_path(ticker, interval)
    # Ensure directory exists; merging handled by getKlines()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    client = getClient()
    showStatus(client)
    print(
        f"[get] {ticker.upper()} {interval} days={days} (extend/merge if exists)"
    )
    getKlines(client, ticker.upper(), interval, days, minCandles=None)


def main(argv: list[str]) -> int:
    cmd = argv[0].lower()
    # Straight-line CLI: assume valid inputs and delegate directly
    if cmd == "update":
        update_all()
        return 0
    if cmd == "get":
        ticker = argv[1].upper()
        interval = argv[2]
        days = int(argv[3])
        get_once(ticker, interval, days)
        return 0
    raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
