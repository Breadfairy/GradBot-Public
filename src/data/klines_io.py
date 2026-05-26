#!/usr/bin/env python3
# klines_io.py - raw kline file helpers for active tune/trace paths.

from __future__ import annotations

import csv
import hashlib
import os
import struct
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterator


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
KLINES_ROOT = os.path.join(REPO_ROOT, "inputs", "klines")
DAY_MS = 24 * 60 * 60 * 1000
INTERVAL_MS = {
    "1m": 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "2h": 2 * 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "6h": 6 * 60 * 60 * 1000,
    "8h": 8 * 60 * 60 * 1000,
    "12h": 12 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}


def normalizeInterval(interval: str) -> str:
    key = str(interval).strip().lower()
    alias = {
        "1min": "1m",
        "1minute": "1m",
        "5min": "5m",
        "5minute": "5m",
        "15min": "15m",
        "15minute": "15m",
        "30min": "30m",
        "30minute": "30m",
        "1hour": "1h",
        "1hr": "1h",
        "60m": "1h",
        "2hour": "2h",
        "2hr": "2h",
        "4hour": "4h",
        "4hr": "4h",
        "6hour": "6h",
        "6hr": "6h",
        "8hour": "8h",
        "8hr": "8h",
        "12hour": "12h",
        "12hr": "12h",
        "1day": "1d",
    }
    out = alias.get(key, key)
    if out not in INTERVAL_MS:
        raise SystemExit(f"Unsupported interval: {interval}")
    return out


def klineCsvPath(ticker: str, interval: str) -> str:
    key = normalizeInterval(interval)
    fileName = f"{str(ticker).lower()}_{key}.csv"
    return os.path.join(KLINES_ROOT, str(ticker).upper(), fileName)


def readOpenTimes(csvPath: str) -> tuple[int | None, int | None]:
    if not os.path.exists(csvPath):
        return None, None
    first = None
    last = None
    with open(csvPath, newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            value = int(float(row[0]))
            if first is None:
                first = value
            last = value
    return first, last


def daysSpanSince(ms: int | None) -> int:
    if ms is None:
        return 0
    nowMs = int(datetime.now(timezone.utc).timestamp() * 1000)
    deltaMs = max(0, nowMs - int(ms))
    return int((deltaMs + (24 * 60 * 60 * 1000) - 1) / (24 * 60 * 60 * 1000))


def iterCachedKlines() -> Iterator[tuple[str, str, str]]:
    if not os.path.isdir(KLINES_ROOT):
        return
    for sym in sorted(os.listdir(KLINES_ROOT)):
        symDir = os.path.join(KLINES_ROOT, sym)
        if not os.path.isdir(symDir):
            continue
        for name in sorted(os.listdir(symDir)):
            if not name.endswith(".csv") or "_" not in name:
                continue
            base, _ext = os.path.splitext(name)
            parts = base.split("_")
            if len(parts) < 2:
                continue
            ticker = parts[0].upper()
            interval = normalizeInterval(parts[-1])
            yield ticker, interval, os.path.join(symDir, name)


def _loadRows(csvPath: str) -> list[list[str]]:
    if not os.path.exists(csvPath):
        return []
    with open(csvPath, newline="") as fh:
        reader = csv.reader(fh)
        return [row for row in reader if row]


def _dedupAndSort(rows: list[list[str]]) -> list[list[str]]:
    unique: dict[int, list[str]] = {}
    for row in rows:
        key = int(float(row[0]))
        unique[key] = [str(part) for part in row]
    return [unique[key] for key in sorted(unique)]


def _rowOpenMs(row: list[str]) -> int:
    return int(float(row[0]))


def _prepareRows(rows: list[list[str]]) -> list[list[object]]:
    prepared: list[list[object]] = []
    for row in rows:
        out: list[object] = list(row)
        out[0] = int(float(row[0]))
        if len(out) > 6:
            out[6] = int(float(row[6]))
        if len(out) > 8:
            out[8] = int(float(row[8]))
        prepared.append(out)
    return prepared


def loadCachedKlines(
    ticker: str,
    interval: str,
    days: int,
    minCandles: int | None = None,
    anchorMs: int | None = None,
) -> list:
    csvPath = klineCsvPath(ticker, interval)
    rows = _dedupAndSort(_loadRows(csvPath))
    if not rows:
        return []
    nowMs = (
        int(anchorMs)
        if anchorMs is not None
        else int(datetime.now(timezone.utc).timestamp() * 1000)
    )
    startMs = nowMs - (int(days) * DAY_MS)
    windowed = [
        row for row in rows
        if startMs <= _rowOpenMs(row) <= nowMs
    ]
    if not windowed:
        return []
    if minCandles is not None and len(windowed) < int(minCandles):
        return []
    return _prepareRows(windowed)


def _trimHoldout(
    klines: list,
    holdoutDays: int,
    anchorMs: int | None = None,
) -> list:
    if int(holdoutDays) <= 0:
        return klines
    cutMs = (
        int(anchorMs)
        if anchorMs is not None
        else int(datetime.now(timezone.utc).timestamp() * 1000)
    ) - (int(holdoutDays) * DAY_MS)
    return [
        row for row in klines
        if int(float(row[0])) < cutMs
    ]


def loadWindowedKlines(
    ticker: str,
    interval: str,
    days: int,
    minCandles: int | None,
    holdoutDays: int = 0,
    anchorMs: int | None = None,
) -> list:
    klines = loadCachedKlines(
        ticker,
        interval,
        int(days),
        minCandles=minCandles,
        anchorMs=anchorMs,
    )
    if not klines:
        raise SystemExit(
            f"cached klines missing for {ticker} {interval} {int(days)}d; "
            "add data under inputs/klines or adjust days/interval"
        )
    klines = _trimHoldout(klines, holdoutDays, anchorMs=anchorMs)
    if minCandles is not None and len(klines) < int(minCandles):
        raise SystemExit(
            f"cached klines window too small for {ticker} {interval} "
            f"{int(days)}d (need {int(minCandles)}, have {len(klines)})"
        )
    return klines


def klinesMeta(klines: list) -> Dict[str, int | str]:
    if not klines:
        raise ValueError("klines list empty; metadata needs loaded rows")
    digest = hashlib.blake2b(digest_size=16)
    packQ = struct.Struct("<q")
    packD = struct.Struct("<d")
    for row in klines:
        openMs = int(float(row[0]))
        closePx = float(row[4])
        digest.update(packQ.pack(openMs))
        digest.update(packD.pack(closePx))
    return {
        "start": int(float(klines[0][0])),
        "end": int(float(klines[-1][0])),
        "count": len(klines),
        "digest": digest.hexdigest(),
    }


__all__ = [
    "INTERVAL_MS",
    "KLINES_ROOT",
    "REPO_ROOT",
    "daysSpanSince",
    "iterCachedKlines",
    "klineCsvPath",
    "klinesMeta",
    "loadCachedKlines",
    "loadWindowedKlines",
    "normalizeInterval",
    "readOpenTimes",
]
