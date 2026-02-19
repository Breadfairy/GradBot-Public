#!/usr/bin/env python3
# main.py – CLI entrypoint
# Usage:
#   python3 src/main.py tune --profile <tune-config.json> --out <run_dir>
#   python3 src/main.py backtest --profile <profile.json> [--charts --prints --oracles]

import argparse
import os
import sys
import json
from typing import Tuple

from backtest import Backtest
from binance_io import loadCachedKlines
from tune import runTuner
import profile


# ---------------- Profile helpers ----------------
def _loadProfile(path: str) -> dict:
    if path == '-':
        return json.load(sys.stdin)
    return profile.loadJson(path)


def _requireIntScalar(cfg: dict, key: str) -> int:
    raw = cfg[key]
    val = profile.scalarValue(raw, 0)
    return int(val if val is not None else 0)


def _validateAndExtractBacktest(cfg: dict) -> Tuple[str, str, int, list[int], dict]:
    tickers = profile._requireTickers(cfg)  # reuse profile tickers normaliser
    ticker = tickers[0]
    intervals = profile.intervalsFromConfig(cfg)
    if not intervals:
        raise SystemExit("profile requires at least one interval")
    interval = intervals[0]
    _primer, _tuner, _holdout, days = profile.windowParts(cfg)
    p1 = _requireIntScalar(cfg, 'p1')
    p2 = _requireIntScalar(cfg, 'p2')
    p3 = _requireIntScalar(cfg, 'p3')

    base_fields = {
        'ticker', 'tickers', 'intervals', 'p1', 'p2', 'p3', 'out',
    }
    overrides = {
        key: value for key, value in cfg.items()
        if key not in base_fields
    }
    overridesNorm = profile.overrides(overrides)
    profile.validate(overridesNorm, kind='backtest')
    return ticker, interval, days, [p1, p2, p3], overridesNorm


def _load_klines_or_fetch(
    ticker: str,
    interval: str,
    days: int,
    periods: list[int],
) -> list:
    minCandles = max(periods) * 2 + 1
    klines = loadCachedKlines(
        ticker,
        interval,
        days,
        minCandles=minCandles,
    )
    if klines:
        return klines
    raise SystemExit(
        "cached klines missing under inputs/klines; use scripts/update_klines.sh "
        "or scripts/get_klines.sh to populate the cache"
    )


def _cmd_tune(args) -> None:
    cfg = _loadProfile(args.profile)
    profile.ensureFinalPortionPct(cfg)
    profile.validate(cfg, kind='tuner')
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    runTuner(cfg, out_dir)


def _cmd_backtest(args) -> None:
    cfg = _loadProfile(args.profile)
    profile.ensureFinalPortionPct(cfg)
    ticker, interval, days, periods, overrides = (
        _validateAndExtractBacktest(cfg)
    )
    klines = _load_klines_or_fetch(ticker, interval, days, periods)
    Backtest(
        ticker,
        klines,
        interval,
        periods,
        days=days,
        doOracles=bool(getattr(args, 'oracles', False)),
        showCharts=bool(getattr(args, 'charts', False)),
        showPrints=bool(getattr(args, 'prints', False)),
        overrides=overrides,
        holdoutDays=0,
    ).run()


def main():
    parser = argparse.ArgumentParser(
        prog="gradbot",
        description="GradBot CLI (profile-driven)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    tune_parser = subparsers.add_parser(
        "tune", help="Run the tuner with a JSON profile",
    )
    tune_parser.add_argument(
        "--profile",
        required=True,
        help="Path to tuner profile JSON",
    )
    tune_parser.add_argument(
        "--out",
        required=True,
        help="Destination directory for tuner outputs",
    )

    backtest_parser = subparsers.add_parser(
        "backtest", help="Run a backtest with optional charts",
    )
    backtest_parser.add_argument(
        "--profile",
        required=True,
        help="Path to backtest profile JSON",
    )
    backtest_parser.add_argument(
        "--charts",
        action="store_true",
        help="Write chart images (CHARTS_OUT_DIR env controls destination)",
    )
    backtest_parser.add_argument(
        "--prints",
        action="store_true",
        help="Verbose per-trade prints",
    )
    backtest_parser.add_argument(
        "--oracles",
        action="store_true",
        help="Enable oracle markers",
    )

    args = parser.parse_args()

    if args.command == 'tune':
        return _cmd_tune(args)
    if args.command == 'backtest':
        return _cmd_backtest(args)


if __name__ == "__main__":
    main()
