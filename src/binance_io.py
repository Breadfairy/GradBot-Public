#!/usr/bin/env python3
# binance_io.py – Binance client, status, and kline fetch.

import os
import csv
import configparser
from datetime import datetime, timezone, timedelta
from binance.client import Client


CACHE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "inputs", "klines")
)

KLINE_INTERVAL_MAP = {
    "1m":  Client.KLINE_INTERVAL_1MINUTE,
    "5m":  Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "30m": Client.KLINE_INTERVAL_30MINUTE,
    "1h":  Client.KLINE_INTERVAL_1HOUR,
    "2h":  Client.KLINE_INTERVAL_2HOUR,
    "4h":  Client.KLINE_INTERVAL_4HOUR,
    "6h":  Client.KLINE_INTERVAL_6HOUR,
    "12h": Client.KLINE_INTERVAL_12HOUR,
    "1d":  Client.KLINE_INTERVAL_1DAY,
}

KLINE_INTERVAL_MS = {
    "1m":  60 * 1000,
    "5m":  5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h":  60 * 60 * 1000,
    "2h":  2 * 60 * 60 * 1000,
    "4h":  4 * 60 * 60 * 1000,
    "6h":  6 * 60 * 60 * 1000,
    "12h": 12 * 60 * 60 * 1000,
    "1d":  24 * 60 * 60 * 1000,
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
        "12hour": "12h",
        "12hr": "12h",
        "1day": "1d",
    }
    out = alias.get(key, key)
    if out not in KLINE_INTERVAL_MS:
        raise SystemExit(f"Unsupported interval: {interval}")
    return out


def _cache_paths(ticker: str, interval: str):
    symbolDir = os.path.join(CACHE_ROOT, ticker.upper())
    fileName = f"{ticker.lower()}_{interval.lower()}.csv"
    return symbolDir, os.path.join(symbolDir, fileName)


def _load_cached_rows(cachePath: str):
    if not os.path.exists(cachePath):
        return []
    rows = []
    with open(cachePath, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows.append(row)
    return rows


def _dedup_and_sort(rows):
    unique = {}
    for row in rows:
        if not row:
            continue
        # normalize to strings for storage, ints for keys for ordering
        key = int(float(row[0]))
        unique[key] = [str(part) for part in row]
    return [unique[k] for k in sorted(unique.keys())]


def _write_cache(cachePath: str, rows):
    if not rows:
        return
    symbolDir = os.path.dirname(cachePath)
    os.makedirs(symbolDir, exist_ok=True)
    tmpPath = f"{cachePath}.tmp"
    with open(tmpPath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    os.replace(tmpPath, cachePath)


def _row_open_ms(row):
    return int(float(row[0]))


def _prepare_rows(rows):
    prepared = []
    for row in rows:
        out = list(row)
        out[0] = int(float(out[0]))
        if len(out) > 6:
            out[6] = int(float(out[6]))
        if len(out) > 8:
            out[8] = int(float(out[8]))
        prepared.append(out)
    return prepared


def _loadConfig():
    cfg = configparser.ConfigParser()
    here = os.path.dirname(__file__)
    path = os.path.join(here, "config.ini")
    if not os.path.exists(path):
        raise SystemExit(
            "config.ini not found alongside src/; expected src/config.ini"
        )
    cfg.read(path)

    if (
        'binance' not in cfg
        or 'api_key' not in cfg['binance']
        or 'api_secret' not in cfg['binance']
    ):
        raise SystemExit("config.ini missing [binance] api_key/api_secret")
    return cfg


def getClient():
    cfg = _loadConfig()
    apiKey = cfg["binance"]["api_key"]
    apiSec = cfg["binance"]["api_secret"]
    return Client(apiKey, apiSec, {"timeout": 30})


def showStatus(client):
    st = client.get_system_status()
    ts = client.get_server_time()["serverTime"]
    dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
    print(f"\nServer status: {st['msg']}")
    print(f"Server time  : {dt:%Y-%m-%d %H:%M} UTC")


def loadCachedKlines(ticker, interval, days, minCandles=None):
    """Return cached klines for the requested window if available."""
    key = normalizeInterval(interval)
    _, cachePath = _cache_paths(ticker, key)
    cachedRows = _load_cached_rows(cachePath)
    if not cachedRows:
        return []

    cachedRows = _dedup_and_sort(cachedRows)
    nowUtc = datetime.now(timezone.utc)
    startMs = int((nowUtc - timedelta(days=days)).timestamp() * 1000)
    windowed = [
        row for row in cachedRows
        if _row_open_ms(row) >= startMs
    ]

    if not windowed:
        return []

    if minCandles is not None and len(windowed) < int(minCandles):
        return []

    return _prepare_rows(windowed)


def getKlines(client, ticker, interval, days, minCandles=None):
    key = normalizeInterval(interval)
    intervalMs = KLINE_INTERVAL_MS[key]
    _, cachePath = _cache_paths(ticker, key)
    cachedRows = _load_cached_rows(cachePath)
    updatedRows = list(cachedRows)

    nowUtc = datetime.now(timezone.utc)
    endMs = int(nowUtc.timestamp() * 1000)
    startMs = int((nowUtc - timedelta(days=days)).timestamp() * 1000)

    fetchSegments = []
    if not updatedRows:
        fetchSegments.append((startMs, None))
    else:
        earliestMs = _row_open_ms(updatedRows[0])
        lastMs = _row_open_ms(updatedRows[-1])

        if earliestMs > startMs:
            endOld = earliestMs - intervalMs
            if endOld >= startMs:
                fetchSegments.append((startMs, endOld))
        if lastMs < endMs - intervalMs:
            fetchSegments.append((lastMs + intervalMs, None))

    fetched = False
    for segStart, segEnd in fetchSegments:
        if segEnd is not None and segEnd < segStart:
            continue
        klSegment = client.get_historical_klines(
            ticker,
            KLINE_INTERVAL_MAP[key],
            int(segStart),
            int(segEnd) if segEnd is not None else None,
        )
        if klSegment:
            updatedRows.extend(klSegment)
            fetched = True

    if fetchSegments:
        updatedRows = _dedup_and_sort(updatedRows)
        if updatedRows:
            _write_cache(cachePath, updatedRows)
        if fetched:
            print(
                f"Refreshing cache for {ticker} {interval} with "
                f"{len(updatedRows)} rows."
            )
    else:
        # ensure deterministic order even without fetch
        updatedRows = _dedup_and_sort(updatedRows)

    # Trim to requested window
    windowed = [
        row for row in updatedRows
        if _row_open_ms(row) >= startMs
    ]

    if minCandles is not None:
        if not windowed or len(windowed) < int(minCandles):
            cnt = len(windowed) if windowed else 0
            raise ValueError(
                f"Only {cnt} candles available; need at least {minCandles}"
            )

    if not windowed:
        return []

    return _prepare_rows(windowed)
