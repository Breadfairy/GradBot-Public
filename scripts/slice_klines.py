#!/usr/bin/env python3
"""
slice_klines.py – slice cached klines into tune/holdout windows.

Usage (from repo root or scripts/):
  python3 scripts/slice_klines.py \
      --ticker LINKUSDT \
      --interval 15m \
      --window 730 \
      --holdout-days 365

Environment:
  KLINES_OUT_DIR  Base directory for sliced outputs. Within this directory
                  the script writes:
                    tune/<TICKER>/<ticker>_<interval>.csv
                    holdout/<TICKER>/<ticker>_<interval>.csv
                  If unset, defaults to cache/klines_slices under repo root.

Simplicity:
  - Assumes source CSVs exist and are valid.
  - Uses straight-line csv + slicing; no on-disk caches.
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone, timedelta

import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="slice_klines",
        description="Slice cached klines into tune/holdout CSVs.",
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Ticker symbol, e.g. LINKUSDT",
    )
    parser.add_argument(
        "--interval",
        required=True,
        help="Candle interval, e.g. 15m",
    )
    parser.add_argument(
        "--window",
        required=True,
        help="Total lookback window in days (e.g. 730)",
    )
    parser.add_argument(
        "--holdout-days",
        type=int,
        required=True,
        help="Holdout span in days (e.g. 365)",
    )
    return parser.parse_args()


def sourceCsvPath(ticker: str, interval: str) -> str:
    symbolDir = os.path.join(REPO_ROOT, "inputs", "klines", ticker.upper())
    fileName = f"{ticker.lower()}_{interval.lower()}.csv"
    return os.path.join(symbolDir, fileName)


def loadRows(csvPath: str) -> list[list[str]]:
    rows: list[list[str]] = []
    with open(csvPath, newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            rows.append(row)
    rows.sort(key=lambda r: int(float(r[0])))
    return rows


def sliceRows(
    rows: list[list[str]],
    windowSpec: str,
    holdoutDays: int,
) -> tuple[list[list[str]], list[list[str]]]:
    nowUtc = datetime.now(timezone.utc)
    windowDelta = timedelta(days=int(windowSpec))
    holdoutDelta = timedelta(days=int(holdoutDays))
    windowStartMs = int((nowUtc - windowDelta).timestamp() * 1000)
    holdoutStartMs = int((nowUtc - holdoutDelta).timestamp() * 1000)

    def openMs(row: list[str]) -> int:
        return int(float(row[0]))

    windowed = [row for row in rows if openMs(row) >= windowStartMs]
    tuneRows = [row for row in windowed if openMs(row) < holdoutStartMs]
    holdoutRows = [row for row in windowed if openMs(row) >= holdoutStartMs]
    return tuneRows, holdoutRows


def outRoots() -> tuple[str, str]:
    base = os.environ.get(
        "KLINES_OUT_DIR",
        os.path.join(REPO_ROOT, "cache", "klines_slices"),
    )
    tuneRoot = os.path.join(base, "tune")
    holdoutRoot = os.path.join(base, "holdout")
    return tuneRoot, holdoutRoot


def writeSlices(
    ticker: str,
    interval: str,
    tuneRows: list[list[str]],
    holdoutRows: list[list[str]],
) -> None:
    tuneRoot, holdoutRoot = outRoots()
    fileName = f"{ticker.lower()}_{interval.lower()}.csv"

    tuneDir = os.path.join(tuneRoot, ticker.upper())
    holdoutDir = os.path.join(holdoutRoot, ticker.upper())
    os.makedirs(tuneDir, exist_ok=True)
    os.makedirs(holdoutDir, exist_ok=True)

    tunePath = os.path.join(tuneDir, fileName)
    holdoutPath = os.path.join(holdoutDir, fileName)

    writeCsvAtomic(tunePath, tuneRows)
    writeCsvAtomic(holdoutPath, holdoutRows)

def writeCsvAtomic(path: str, rows: list[list[str]]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)
    os.replace(tmp, path)


def main() -> int:
    args = parseArgs()
    ticker = args.ticker
    interval = args.interval
    windowSpec = str(args.window).strip()
    holdoutDays = int(args.holdout_days)

    srcPath = sourceCsvPath(ticker, interval)
    rows = loadRows(srcPath)
    tuneRows, holdoutRows = sliceRows(rows, windowSpec, holdoutDays)
    writeSlices(ticker, interval, tuneRows, holdoutRows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
