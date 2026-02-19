#!/usr/bin/env python3
# prepare_klines.py – ensure required klines exist before tuning.

import argparse
import os
from typing import Any, Dict, List, Tuple

from klines_tools import (
    _csv_path,
    _read_open_times,
    _days_span_since,
    get_once,
    update_all,
)
from binance_io import normalizeInterval
from cache import profile_windows as profileWindows
import profile


def ensureKlinesForProfile(cfg: Dict[str, Any]) -> None:
    tickers = [str(t).strip().upper() for t in cfg["tickers"]]
    intervals = profile.intervalsFromConfig(cfg)
    _primer, _tuner, _holdout, totalDays = profileWindows(cfg)

    didGet = False
    for sym in tickers:
        for iv in intervals:
            intervalNorm = normalizeInterval(iv)
            csvPath = _csv_path(sym, intervalNorm)
            if not os.path.exists(csvPath):
                get_once(sym, intervalNorm, totalDays)
                didGet = True
                continue
            first, _last = _read_open_times(csvPath)
            haveDays = _days_span_since(first)
            if haveDays < totalDays:
                get_once(sym, intervalNorm, totalDays)
                didGet = True

    if didGet:
        update_all()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="prepare_klines",
        description="Ensure klines exist for a tuner profile.",
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="Path to tuner profile JSON",
    )
    args = parser.parse_args()
    cfg = profile.loadJson(args.profile)
    ensureKlinesForProfile(cfg)


if __name__ == "__main__":
    main()
