#!/usr/bin/env python3
# prepare_klines.py – host-client kline preparation helpers.

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict

from data.klines_io import (
    INTERVAL_MS,
    KLINES_ROOT,
    daysSpanSince,
    iterCachedKlines,
    klineCsvPath,
    normalizeInterval,
    readOpenTimes,
)
from config import profile
from data.time_bounds import resolveAnchorMs
from repo_paths import NATIVE_BINANCE_BIN, NATIVE_HOST_DIR
from tune.axes import axesFromConfig


HOST_DIR = str(NATIVE_HOST_DIR)
HOST_BIN = str(NATIVE_BINANCE_BIN)
DAY_MS = 24 * 60 * 60 * 1000


def _nowMs() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _requiredFetchDays(days: int, anchorMs: int | None = None) -> int:
    needDays = int(days)
    if anchorMs is None:
        return max(needDays, 1)
    deltaMs = max(_nowMs() - int(anchorMs), 0)
    extraDays = int((deltaMs + DAY_MS - 1) / DAY_MS)
    return max(needDays + extraDays, 1)


def _needsRefresh(
    firstMs: int | None,
    lastMs: int | None,
    interval: str,
    days: int,
    anchorMs: int | None = None,
) -> bool:
    nowMs = (
        int(anchorMs)
        if anchorMs is not None
        else _nowMs()
    )
    startMs = nowMs - (int(days) * DAY_MS)
    slackMs = INTERVAL_MS[interval] * 2
    if firstMs is None or lastMs is None:
        return True
    if firstMs > startMs + slackMs:
        return True
    if lastMs < nowMs - slackMs:
        return True
    return False


def ensureHostBin() -> None:
    subprocess.run(
        ["make", "gradbot_binance"],
        cwd=HOST_DIR,
        check=True,
    )


def fetchWindow(ticker: str, interval: str, days: int) -> None:
    subprocess.run(
        [
            HOST_BIN,
            str(ticker).upper(),
            normalizeInterval(interval),
            str(int(days)),
            KLINES_ROOT,
        ],
        check=True,
    )


def fetchKlines(ticker: str, interval: str, days: int) -> None:
    ensureHostBin()
    fetchWindow(ticker, interval, days)


def updateAllKlines() -> None:
    builtBin = False
    for ticker, interval, csvPath in iterCachedKlines():
        firstMs, lastMs = readOpenTimes(csvPath)
        days = max(daysSpanSince(firstMs), 1)
        if not _needsRefresh(firstMs, lastMs, interval, days):
            continue
        if not builtBin:
            ensureHostBin()
            builtBin = True
        print(f"[update] {ticker} {interval} → days={days}")
        fetchWindow(ticker, interval, days)


def updateSelectedKlines(
    tickers: list[str],
    intervals: list[str],
    days: int,
    anchorMs: int | None = None,
) -> None:
    builtBin = False
    symbols = list(
        dict.fromkeys(str(ticker).strip().upper() for ticker in tickers)
    )
    intervalKeys = list(
        dict.fromkeys(normalizeInterval(interval) for interval in intervals)
    )
    for ticker in symbols:
        for interval in intervalKeys:
            csvPath = klineCsvPath(ticker, interval)
            firstMs, lastMs = readOpenTimes(csvPath)
            fetchDays = _requiredFetchDays(
                max(daysSpanSince(firstMs), 1),
                anchorMs=anchorMs,
            )
            if firstMs is None or lastMs is None:
                fetchDays = _requiredFetchDays(days, anchorMs=anchorMs)
            if not _needsRefresh(
                firstMs,
                lastMs,
                interval,
                days,
                anchorMs=anchorMs,
            ):
                continue
            if not builtBin:
                ensureHostBin()
                builtBin = True
            print(f"[update] {ticker} {interval} → days={fetchDays}")
            fetchWindow(ticker, interval, fetchDays)


def ensureKlinesForProfile(
    cfg: Dict[str, Any],
    anchorMs: int | None = None,
) -> None:
    tickers = [str(t).strip().upper() for t in cfg["tickers"]]
    axes = axesFromConfig(cfg)
    intervals = list(
        dict.fromkeys(
            profile.intervalsFromConfig(cfg)
            + [str(iv) for iv in axes["macroIntervalValues"]]
        )
    )
    _primer, _training, _tuner, _holdout, totalDays = (
        profile.profileWindows(cfg)
    )

    builtBin = False
    for sym in tickers:
        for intervalValue in intervals:
            intervalNorm = normalizeInterval(intervalValue)
            csvPath = klineCsvPath(sym, intervalNorm)
            firstMs, lastMs = readOpenTimes(csvPath)
            fetchDays = _requiredFetchDays(totalDays, anchorMs=anchorMs)
            needFetch = (
                not os.path.exists(csvPath)
                or _needsRefresh(
                    firstMs,
                    lastMs,
                    intervalNorm,
                    int(totalDays),
                    anchorMs=anchorMs,
                )
            )
            if not needFetch:
                continue
            if not builtBin:
                ensureHostBin()
                builtBin = True
            fetchWindow(sym, intervalNorm, fetchDays)


def parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="prepare_klines",
        description="Manage local kline CSVs using the C host client.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    profileParser = sub.add_parser(
        "profile",
        help="Ensure all klines required by a tuner profile exist locally",
    )
    profileParser.add_argument(
        "--profile",
        required=True,
        help="Path to tuner profile JSON",
    )
    profileParser.add_argument(
        "--anchor-ms",
        type=int,
        default=None,
        help="Optional UTC millisecond anchor for historical runs",
    )
    profileParser.add_argument(
        "--anchor-date",
        default=None,
        help="Optional UTC anchor date for historical runs (YYYY-MM-DD)",
    )

    getParser = sub.add_parser(
        "get",
        help="Fetch or refresh one ticker/interval window",
    )
    getParser.add_argument("ticker")
    getParser.add_argument("interval")
    getParser.add_argument("days", type=int)

    sub.add_parser(
        "update",
        help="Refresh all cached kline CSVs to now",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parseArgs(argv)
    if args.cmd == "profile":
        cfg = profile.loadJson(args.profile)
        ensureKlinesForProfile(
            cfg,
            anchorMs=resolveAnchorMs(
                anchorMs=args.anchor_ms,
                anchorDate=args.anchor_date,
            ),
        )
        return 0
    if args.cmd == "get":
        fetchKlines(args.ticker, args.interval, args.days)
        return 0
    if args.cmd == "update":
        updateAllKlines()
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
