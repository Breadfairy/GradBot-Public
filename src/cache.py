#!/usr/bin/env python3
# cache.py – ctx/signals artifact helpers + small in-memory LRU.

from __future__ import annotations

import csv
import json
import os
import struct
import hashlib
import threading
from datetime import datetime, timezone, timedelta
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from binance_io import loadCachedKlines
import profile


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CTX_MEM_CAP = 16
SIG_MEM_CAP = 32

_CTX_MEM: "OrderedDict[str, dict]" = OrderedDict()
_SIG_MEM: "OrderedDict[str, dict]" = OrderedDict()
_CTX_LOCK = threading.Lock()
_SIG_LOCK = threading.Lock()
_KLINES_META: Dict[int, Dict[str, int | str]] = {}
_KLINES_META_LOCK = threading.Lock()

RESULT_CACHE_VERSION = 1

_ENGINE_HASH: str | None = None
_ENGINE_META: dict | None = None
_ENGINE_RECORDED = False


def _engineFiles() -> tuple[str, ...]:
    return (
        "src/engine_core.py",
        "src/engine_shared.py",
        "src/flags.py",
        "src/dynamics.py",
        "src/wallet.py",
        "src/backtest.py",
    )


def _recordEngineBuild(meta: dict) -> None:
    """Persist engine build metadata for reproducible cache invalidation."""
    global _ENGINE_RECORDED
    if _ENGINE_RECORDED:
        return
    root = cacheRoot()
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "engine_builds.json")
    if os.path.exists(path):
        with open(path) as fh:
            data = json.load(fh)
    else:
        data = {"version": 1, "builds": {}}
    builds = data.get("builds")
    if not isinstance(builds, dict):
        builds = {}
        data["builds"] = builds
    key = str(meta.get("engineHash", ""))
    if key and key not in builds:
        builds[key] = meta
        tmpPath = f"{path}.tmp"
        with open(tmpPath, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmpPath, path)
    _ENGINE_RECORDED = True


def engineFingerprint() -> str:
    """Fingerprint engine code so caches auto-segregate per build."""
    global _ENGINE_HASH, _ENGINE_META
    if _ENGINE_HASH is not None:
        return _ENGINE_HASH
    digest = hashlib.blake2b(digest_size=16)
    fileMeta: list[dict] = []
    for relPath in _engineFiles():
        absPath = os.path.join(REPO_ROOT, relPath)
        with open(absPath, "rb") as fh:
            blob = fh.read()
        digest.update(blob)
        fileMeta.append(
            {
                "path": relPath,
                "bytes": len(blob),
                "hash": hashlib.blake2b(blob, digest_size=16).hexdigest(),
            }
        )
    _ENGINE_HASH = digest.hexdigest()
    _ENGINE_META = {
        "engineHash": _ENGINE_HASH,
        "cacheVersion": RESULT_CACHE_VERSION,
        "createdUtc": datetime.now(timezone.utc).isoformat(),
        "files": fileMeta,
    }
    _recordEngineBuild(_ENGINE_META)
    return _ENGINE_HASH


def cacheRoot() -> str:
    override = str(os.environ.get("GRADBOT_CACHE_DIR", "")).strip()
    base = override if override else os.path.join(REPO_ROOT, "cache")
    return os.path.abspath(base)


def _resultsRoot() -> str:
    return os.path.join(cacheRoot(), "results")


def _lruGet(store: "OrderedDict[str, dict]", key: str) -> dict | None:
    val = store.get(key)
    if val is not None:
        store.move_to_end(key)
    return val


def _lruPut(store: "OrderedDict[str, dict]", key: str, value: dict, cap: int) -> None:
    if cap <= 0:
        return
    store[key] = value
    store.move_to_end(key)
    while len(store) > cap:
        store.popitem(last=False)


def _klinesMeta(klines: list) -> Dict[str, int | str]:
    kid = id(klines)
    with _KLINES_META_LOCK:
        cached = _KLINES_META.get(kid)
    if (
        cached is not None
        and int(cached.get("count", -1)) == len(klines)
        and int(cached.get("start", -1)) == int(float(klines[0][0]))
        and int(cached.get("end", -1)) == int(float(klines[-1][0]))
    ):
        return cached
    if not klines:
        raise ValueError("klines list empty; cache expects preloaded data")
    opens = [int(float(row[0])) for row in klines]
    start_ms = opens[0]
    end_ms = opens[-1]
    count = len(opens)
    digest = hashlib.blake2b(digest_size=16)
    pack_q = struct.Struct("<q")
    pack_d = struct.Struct("<d")
    for row in klines:
        open_ms = int(float(row[0]))
        close_px = float(row[4])
        digest.update(pack_q.pack(open_ms))
        digest.update(pack_d.pack(close_px))
    meta = {
        "start": start_ms,
        "end": end_ms,
        "count": count,
        "digest": digest.hexdigest(),
    }
    with _KLINES_META_LOCK:
        _KLINES_META[kid] = meta
    return meta


def _hashSpec(spec: Dict[str, Any]) -> str:
    blob = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


def _ctxSpec(
    ticker: str,
    interval: str,
    days: int,
    periods: Iterable[int],
    klines: list,
) -> Tuple[Dict[str, Any], str]:
    stats = _klinesMeta(klines)
    spec = {
        "ticker": str(ticker).upper(),
        "interval": str(interval).lower(),
        "days": int(days),
        "periods": [int(p) for p in periods],
        "engineHash": engineFingerprint(),
        "start": stats["start"],
        "end": stats["end"],
        "count": stats["count"],
        "digest": stats["digest"],
    }
    return spec, _hashSpec(spec)


def ctxSpecHash(
    ticker: str,
    interval: str,
    days: int,
    periods: Iterable[int],
    klines: list,
) -> str:
    """Compute context spec hash without building context artifacts."""
    _spec, specHash = _ctxSpec(ticker, interval, days, periods, klines)
    return specHash


def _ctxDir(specHash: str, ticker: str, interval: str, days: int) -> str:
    return os.path.join(
        cacheRoot(),
        str(ticker).upper(),
        str(interval).lower(),
        f"days-{int(days)}",
        specHash,
    )


def _signalsName(lookbacks: Iterable[int]) -> Tuple[str, Tuple[int, ...]]:
    lbs = tuple(sorted(set(int(lb) for lb in lookbacks)))
    label = "-".join(str(lb) for lb in lbs)
    return f"signals_lb-{label}.npz", lbs


def _ensureDir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _resultDir(
    ticker: str,
    interval: str,
    days: int,
) -> str:
    return os.path.join(
        _resultsRoot(),
        str(ticker).upper(),
        str(interval).lower(),
        f"days-{int(days)}",
    )


def loadResultRow(spec: Dict[str, Any]) -> dict | None:
    """Load cached backtest row for a spec if present."""
    specLocal = dict(spec)
    specLocal["version"] = RESULT_CACHE_VERSION
    specLocal["engineHash"] = engineFingerprint()
    ticker = str(specLocal["ticker"]).upper()
    interval = str(specLocal["interval"]).lower()
    days = int(specLocal["days"])
    specHash = _hashSpec(specLocal)
    dirPath = _resultDir(ticker, interval, days)
    path = os.path.join(dirPath, f"{specHash}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        data = json.load(fh)
    row = data.get("row")
    return row if isinstance(row, dict) else None


def saveResultRow(spec: Dict[str, Any], row: dict) -> None:
    """Persist backtest row for a spec under cache/results."""
    specLocal = dict(spec)
    specLocal["version"] = RESULT_CACHE_VERSION
    specLocal["engineHash"] = engineFingerprint()
    ticker = str(specLocal["ticker"]).upper()
    interval = str(specLocal["interval"]).lower()
    days = int(specLocal["days"])
    specHash = _hashSpec(specLocal)
    dirPath = _resultDir(ticker, interval, days)
    _ensureDir(dirPath)
    path = os.path.join(dirPath, f"{specHash}.json")
    if os.path.exists(path):
        return
    tmpPath = f"{path}.tmp"
    payload = {"spec": specLocal, "row": row}
    with open(tmpPath, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"), sort_keys=True)
    os.replace(tmpPath, path)


def _saveNpz(path: str, payload: Dict[str, np.ndarray]) -> None:
    tmpPath = f"{path}.tmp"
    np.savez_compressed(tmpPath, **payload)
    src = tmpPath if tmpPath.endswith(".npz") else f"{tmpPath}.npz"
    os.replace(src, path)


def _ctxPayload(ctx: dict) -> Dict[str, np.ndarray]:
    mas = [np.asarray(arr, dtype=float) for arr in ctx["mas"]]
    masArr = np.stack(mas, axis=0)
    return {
        "closes": np.asarray(ctx["closes"], dtype=float),
        "opens": np.asarray(ctx["opens"], dtype=float),
        "ath": np.asarray(ctx["ath"], dtype=float),
        "mas": masArr,
        "periods": np.asarray(ctx["periods"], dtype=np.int64),
    }


def _loadCtx(path: str, klines: list) -> dict:
    with np.load(path) as data:
        closes = np.array(data["closes"])
        opens = np.array(data["opens"])
        ath = np.array(data["ath"])
        masArr = np.array(data["mas"])
        periods = np.array(data["periods"], dtype=int).tolist()
    mas = [masArr[i].copy() for i in range(masArr.shape[0])]
    ctx = {
        "klines": klines,
        "closes": closes,
        "opens": opens,
        "periods": periods,
        "mas": mas,
        "smoothMas": mas,
        "ath": ath,
    }
    return ctx


def _signalsPayload(signals: dict, lookbacks: Tuple[int, ...]) -> Dict[str, np.ndarray]:
    payload: Dict[str, np.ndarray] = {
        "g1P1": np.asarray(signals["g1P1"], dtype=float),
        "g1P3": np.asarray(signals["g1P3"], dtype=float),
        "s12": np.asarray(signals["s12"], dtype=float),
        "s23": np.asarray(signals["s23"], dtype=float),
        "trendCode": np.asarray(signals["trendCode"], dtype=int),
        "lookbacks": np.asarray(lookbacks, dtype=np.int64),
    }
    pctBelow = signals["pctBelow"]
    pctAbove = signals["pctAbove"]
    for lb in lookbacks:
        payload[f"pctBelow_lb{lb}"] = np.asarray(pctBelow[lb], dtype=float)
        payload[f"pctAbove_lb{lb}"] = np.asarray(pctAbove[lb], dtype=float)
    return payload


def _loadSignals(path: str) -> dict:
    with np.load(path) as data:
        g1p1 = np.array(data["g1P1"])
        g1p3 = np.array(data["g1P3"])
        s12 = np.array(data["s12"])
        s23 = np.array(data["s23"])
        trend = np.array(data["trendCode"], dtype=int)
        lookbacks = np.array(data["lookbacks"], dtype=int).tolist()
        pctBelow: Dict[int, np.ndarray] = {}
        pctAbove: Dict[int, np.ndarray] = {}
        for lb in lookbacks:
            pctBelow[lb] = np.array(data[f"pctBelow_lb{lb}"])
            pctAbove[lb] = np.array(data[f"pctAbove_lb{lb}"])
    return {
        "g1P1": g1p1,
        "g1P3": g1p3,
        "s12": s12,
        "s23": s23,
        "trendCode": trend,
        "pctBelow": pctBelow,
        "pctAbove": pctAbove,
    }


def getContext(
    ticker: str,
    interval: str,
    days: int,
    periods: List[int],
    klines: list,
    builder: Callable[[], dict],
) -> dict:
    spec, specHash = _ctxSpec(ticker, interval, days, periods, klines)
    memKey = f"ctx::{specHash}"
    with _CTX_LOCK:
        cached = _lruGet(_CTX_MEM, memKey)
    if cached is not None:
        cached["klines"] = klines
        return cached
    artifactDir = _ctxDir(specHash, ticker, interval, days)
    ctxPath = os.path.join(artifactDir, "ctx.npz")
    if os.path.exists(ctxPath):
        ctx = _loadCtx(ctxPath, klines)
        ctx["_cache"] = {
            "dir": artifactDir,
            "specHash": specHash,
            "ticker": ticker,
            "interval": interval,
            "days": days,
        }
        with _CTX_LOCK:
            _lruPut(_CTX_MEM, memKey, ctx, CTX_MEM_CAP)
        return ctx
    ctx = builder()
    ctx["klines"] = klines
    payload = _ctxPayload(ctx)
    _ensureDir(artifactDir)
    _saveNpz(ctxPath, payload)
    ctx["_cache"] = {
        "dir": artifactDir,
        "specHash": specHash,
        "ticker": ticker,
        "interval": interval,
        "days": days,
    }
    with _CTX_LOCK:
        _lruPut(_CTX_MEM, memKey, ctx, CTX_MEM_CAP)
    return ctx


def getSignals(
    ticker: str,
    interval: str,
    days: int,
    periods: List[int],
    lookbacks: Iterable[int],
    klines: list,
    builder: Callable[[], dict],
) -> dict:
    if not lookbacks:
        return builder()
    spec, specHash = _ctxSpec(ticker, interval, days, periods, klines)
    fileName, lbs = _signalsName(lookbacks)
    memKey = f"sig::{specHash}::{fileName}"
    with _SIG_LOCK:
        cached = _lruGet(_SIG_MEM, memKey)
    if cached is not None:
        return cached
    artifactDir = _ctxDir(specHash, ticker, interval, days)
    sigPath = os.path.join(artifactDir, fileName)
    if os.path.exists(sigPath):
        sigs = _loadSignals(sigPath)
        with _SIG_LOCK:
            _lruPut(_SIG_MEM, memKey, sigs, SIG_MEM_CAP)
        return sigs
    sigs = builder()
    payload = _signalsPayload(sigs, lbs)
    _ensureDir(artifactDir)
    _saveNpz(sigPath, payload)
    with _SIG_LOCK:
        _lruPut(_SIG_MEM, memKey, sigs, SIG_MEM_CAP)
    return sigs


__all__ = ["cacheRoot", "getContext", "getSignals", "klinesMeta"]


# --------------------- Z-stats and series artifacts -----------------------

def _ctxInfo(ctx) -> Tuple[str, str] | None:
    meta = ctx.get("_cache") if isinstance(ctx, dict) else None
    if not isinstance(meta, dict):
        return None
    d = meta.get("dir")
    h = meta.get("specHash")
    if isinstance(d, str) and isinstance(h, str) and d and h:
        return d, h
    return None


def getZStatsForSeries(
    ctx,
    seriesId: str,
    windowBars: int,
    builder: Callable[[], Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    info = _ctxInfo(ctx)
    if info is None:
        return builder()
    dirPath, _specHash = info
    fileName = f"zstats_ms_{seriesId}_win-{int(windowBars)}.npz"
    path = os.path.join(dirPath, fileName)
    if os.path.exists(path):
        with np.load(path) as data:
            mean = np.array(data["mean"])  # type: ignore[index]
            std = np.array(data["std"])  # type: ignore[index]
        return mean, std
    mean, std = builder()
    _ensureDir(dirPath)
    payload = {
        "mean": np.asarray(mean),
        "std": np.asarray(std),
    }
    _saveNpz(path, payload)
    return mean, std


def klinesMeta(klines: list) -> Dict[str, int | str]:
    """Public wrapper for kline fingerprinting."""
    return _klinesMeta(klines)


def getSeriesArray(
    ctx,
    seriesId: str,
    builder: Callable[[], np.ndarray],
) -> np.ndarray:
    info = _ctxInfo(ctx)
    if info is None:
        return builder()
    dirPath, _specHash = info
    path = os.path.join(dirPath, f"series_{seriesId}.npz")
    if os.path.exists(path):
        with np.load(path) as data:
            arr = np.array(data["arr"])  # type: ignore[index]
        return arr
    arr = builder()
    _ensureDir(dirPath)
    payload = {"arr": np.asarray(arr)}
    _saveNpz(path, payload)
    return arr


# ======================================================================
# Klines cache + results hydration (from cache_help)
# ======================================================================


RESULT_FIELD_NAMES = [
    'ticker', 'interval', 'days', 'p1', 'p2', 'p3',
    'GRAD1_BUY_Z_MIN', 'GRAD1_SELL_Z_MIN',
    'GRAD1_BUY_WIN_DAYS', 'GRAD1_SELL_WIN_DAYS',
    'SPACING_Z_MIN_12', 'SPACING_Z_MIN_23',
    'SPACING_WIN_DAYS_12', 'SPACING_WIN_DAYS_23',
    'MICRO_NRG_MODEL', 'MICRO_NRG_WIN_DAYS',
    'MICRO_NRG_MIN_12', 'MICRO_NRG_MIN_23',
    'PHASE_BUY_PORTIONS', 'PHASE_SELL_PORTIONS', 'FINAL_PORTION_PCT',
    'COOLDOWN',
    'MACRO_INTERVAL', 'MACRO_P1', 'MACRO_P2', 'MACRO_P3',
    'MACRO_NRG_WIN_DAYS', 'MACRO_NRG_Z_MIN', 'MACRO_NRG_Z_MAX',
    'MACRO_DYN_PCT_MIN', 'MACRO_DYN_PCT_MAX',
    'MACRO_GRAD_WIN_DAYS', 'MACRO_GRAD_Z_MIN', 'MACRO_GRAD_Z_MAX',
    'MACRO_MULT_GRAD_MIN', 'MACRO_MULT_GRAD_MAX',
    'MACRO_BUY_MULT_BULL', 'MACRO_BUY_MULT_BEAR',
    'MACRO_BUY_MULT_REV', 'MACRO_BUY_MULT_ROLL',
    'MACRO_SELL_MULT_BULL', 'MACRO_SELL_MULT_BEAR',
    'MACRO_SELL_MULT_REV', 'MACRO_SELL_MULT_ROLL',
    'TAX_MODE', 'ANNUAL_INCOME_BASE',
    'PROFIT_SWEEP_INTERVAL', 'PROFIT_SWEEP_SHARE',
    'preTaxEdge', 'postTaxEdge', 'netPctVsHodl',
    'simValue', 'simPostTax', 'benchValue', 'benchPostTax',
    'trades', 'fees', 'tax', 'lockedProfit',
    'potentialProfit', 'potentialProfitBench',
    'netAfterTaxProfit', 'netAfterTaxProfitBench',
    'grossEdgeVsBench', 'netEdgeVsBench', 'edgeVsBench',
    'sharpe', 'sortino', 'mdd', 'cagr',
    'sharpe4w', 'sortino4w', 'sharpe13w', 'sortino13w',
    'sharpe4wAbs', 'sortino4wAbs',
    'sharpe13wAbs', 'sortino13wAbs',
    'scoreMetric',
]

_KLINE_CACHE: Dict[tuple, tuple[int, list]] = {}
_CACHE_LOCK = threading.Lock()


def profileWindows(config: dict) -> tuple[int, int, int, int]:
    return profile.windowParts(config)


# Backward-compatible snake_case alias
profile_windows = profileWindows


def getKlinesCached(
    ticker: str,
    interval: str,
    days: int,
    minCandles: int | None,
    holdoutDays: int = 0,
) -> list:
    key = (ticker, interval, days, holdoutDays)
    required = minCandles if minCandles is not None else 0
    with _CACHE_LOCK:
        cached = _KLINE_CACHE.get(key)
        if cached is not None:
            cachedMin, klines = cached
            if cachedMin >= required:
                return klines

    windowDays = int(days)
    offline = loadCachedKlines(
        ticker,
        interval,
        windowDays,
        minCandles=minCandles,
    )
    if not offline:
        raise SystemExit(
            f"cached klines missing for {ticker} {interval} "
            f"{windowDays}d; add data under inputs/klines or "
            f"adjust days/interval"
        )

    if holdoutDays > 0:
        now = datetime.now(timezone.utc)
        cutTs = now - timedelta(days=int(holdoutDays))
        cutMs = int(cutTs.timestamp() * 1000)
        offline = [
            row for row in offline
            if int(float(row[0])) < cutMs
        ]

    if minCandles is not None and len(offline) < int(minCandles):
        raise SystemExit(
            f"cached klines window too small for {ticker} "
            f"{interval} {days}d (need {minCandles}, "
            f"have {len(offline)})"
        )

    with _CACHE_LOCK:
        previous = _KLINE_CACHE.get(key)
        cachedMin = len(offline)
        if previous is None or previous[0] < cachedMin:
            _KLINE_CACHE[key] = (cachedMin, offline)
        else:
            offline = previous[1]
    return offline


def _specHash(spec: dict) -> str:
    return _hashSpec(spec)


def _ctxHashForRow(
    row: dict,
    holdoutDays: int,
    getKlinesFn: Callable[[str, str, int, int | None, int], list],
) -> str:
    ticker = str(row['ticker'])
    interval = str(row['interval'])
    periods = [
        int(row['p1']),
        int(row['p2']),
        int(row['p3']),
    ]
    minCandles = (max(periods) * 2) + 1
    kl = getKlinesFn(
        ticker,
        interval,
        int(row['days']),
        minCandles,
        holdoutDays=holdoutDays,
    )
    return ctxSpecHash(
        ticker,
        interval,
        int(row['days']),
        periods,
        kl,
    )


def _specFromRow(
    row: dict,
    primerDays: int,
    holdoutDays: int,
    getKlinesFn: Callable[[str, str, int, int | None, int], list],
) -> dict:
    taxModeStr = str(row['TAX_MODE']).lower()
    sweepRaw = str(row['PROFIT_SWEEP_INTERVAL']).strip().lower()
    sweepIntervalOverride = (
        None if sweepRaw in ("", "none") else sweepRaw
    )
    sweepIntervalRow = (
        "" if sweepIntervalOverride is None else sweepIntervalOverride
    )
    ctxHash = _ctxHashForRow(row, holdoutDays, getKlinesFn)
    return {
        "ticker": str(row['ticker']),
        "interval": str(row['interval']),
        "days": int(row['days']),
        "primerDays": int(primerDays),
        "ctxSpecHash": ctxHash,
        "p1": int(row['p1']),
        "p2": int(row['p2']),
        "p3": int(row['p3']),
        "MACRO_INTERVAL": str(row['MACRO_INTERVAL']),
        "MACRO_NRG_WIN_DAYS": int(row['MACRO_NRG_WIN_DAYS']),
        "MACRO_NRG_Z_MIN": float(row['MACRO_NRG_Z_MIN']),
        "MACRO_NRG_Z_MAX": float(row['MACRO_NRG_Z_MAX']),
        "MACRO_DYN_PCT_MIN": float(row['MACRO_DYN_PCT_MIN']),
        "MACRO_DYN_PCT_MAX": float(row['MACRO_DYN_PCT_MAX']),
        "MACRO_P1": int(row['MACRO_P1']),
        "MACRO_P2": int(row['MACRO_P2']),
        "MACRO_P3": int(row['MACRO_P3']),
        "MACRO_GRAD_WIN_DAYS": int(row['MACRO_GRAD_WIN_DAYS']),
        "MACRO_GRAD_Z_MIN": float(row['MACRO_GRAD_Z_MIN']),
        "MACRO_GRAD_Z_MAX": float(row['MACRO_GRAD_Z_MAX']),
        "MACRO_MULT_GRAD_MIN": float(row['MACRO_MULT_GRAD_MIN']),
        "MACRO_MULT_GRAD_MAX": float(row['MACRO_MULT_GRAD_MAX']),
        "MACRO_BUY_MULT_BULL": float(row['MACRO_BUY_MULT_BULL']),
        "MACRO_BUY_MULT_BEAR": float(row['MACRO_BUY_MULT_BEAR']),
        "MACRO_BUY_MULT_REV": float(row['MACRO_BUY_MULT_REV']),
        "MACRO_BUY_MULT_ROLL": float(row['MACRO_BUY_MULT_ROLL']),
        "MACRO_SELL_MULT_BULL": float(row['MACRO_SELL_MULT_BULL']),
        "MACRO_SELL_MULT_BEAR": float(row['MACRO_SELL_MULT_BEAR']),
        "MACRO_SELL_MULT_REV": float(row['MACRO_SELL_MULT_REV']),
        "MACRO_SELL_MULT_ROLL": float(row['MACRO_SELL_MULT_ROLL']),
        "GRAD1_BUY_Z_MIN": float(row['GRAD1_BUY_Z_MIN']),
        "GRAD1_SELL_Z_MIN": float(row['GRAD1_SELL_Z_MIN']),
        "GRAD1_BUY_WIN_DAYS": int(row['GRAD1_BUY_WIN_DAYS']),
        "GRAD1_SELL_WIN_DAYS": int(row['GRAD1_SELL_WIN_DAYS']),
        "PHASE_BUY_PORTIONS": int(row['PHASE_BUY_PORTIONS']),
        "PHASE_SELL_PORTIONS": int(row['PHASE_SELL_PORTIONS']),
        "FINAL_PORTION_PCT": float(row['FINAL_PORTION_PCT']),
        "COOLDOWN": int(row['COOLDOWN']),
        "TAX_MODE": taxModeStr,
        "ANNUAL_INCOME_BASE": float(row['ANNUAL_INCOME_BASE']),
        "PROFIT_SWEEP_INTERVAL": sweepIntervalRow,
        "PROFIT_SWEEP_SHARE": float(row['PROFIT_SWEEP_SHARE']),
        "SPACING_Z_MIN_12": float(row['SPACING_Z_MIN_12']),
        "SPACING_Z_MIN_23": float(row['SPACING_Z_MIN_23']),
        "SPACING_WIN_DAYS_12": int(row['SPACING_WIN_DAYS_12']),
        "SPACING_WIN_DAYS_23": int(row['SPACING_WIN_DAYS_23']),
        "MICRO_NRG_MODEL": row['MICRO_NRG_MODEL'],
        "MICRO_NRG_WIN_DAYS": int(row['MICRO_NRG_WIN_DAYS']),
        "MICRO_NRG_MIN_12": float(row['MICRO_NRG_MIN_12']),
        "MICRO_NRG_MIN_23": float(row['MICRO_NRG_MIN_23']),
    }


def hydrateResultsCache(
    csvPath: str,
    primerDays: int,
    holdoutDays: int,
    getKlinesFn: Callable[[str, str, int, int | None, int], list],
    tickers: Iterable[str] | None = None,
    maxRows: int | None = None,
) -> None:
    if not os.path.exists(csvPath):
        return
    df = pd.read_csv(csvPath)
    if df.empty:
        return
    tickerSet = None
    if tickers is not None:
        tickerSet = {str(t).upper() for t in tickers}
        df = df[df['ticker'].str.upper().isin(tickerSet)]
        if df.empty:
            return
    if maxRows is not None and maxRows > 0:
        if 'scoreMetric' in df.columns:
            df = df.sort_values('scoreMetric', ascending=False).head(maxRows)
        else:
            df = df.head(maxRows)
    rows = df.to_dict(orient="records")
    for row in rows:
        spec = _specFromRow(
            row,
            primerDays,
            holdoutDays,
            getKlinesFn,
        )
        rowClean = {
            k: row[k]
            for k in RESULT_FIELD_NAMES
            if k in row
        }
        saveResultRow(spec, rowClean)


def mergeGlobalResults(
    csvPath: str,
    primerDays: int,
    holdoutDays: int,
    getKlinesFn: Callable[[str, str, int, int | None, int], list],
) -> None:
    if not os.path.exists(csvPath):
        return
    df = pd.read_csv(csvPath)
    if df.empty:
        return

    globalPath = os.path.join(
        cacheRoot(),
        "results",
        "global_results.csv",
    )
    existing: List[dict] = []
    seen: set[str] = set()
    if os.path.exists(globalPath):
        with open(globalPath) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                specHash = row.get('specHash')
                if specHash:
                    seen.add(str(specHash))
                rowClean = {
                    k: row[k]
                    for k in RESULT_FIELD_NAMES
                    if k in row
                }
                if specHash:
                    rowClean['specHash'] = specHash
                existing.append(rowClean)

    rows = df.to_dict(orient="records")
    for row in rows:
        spec = _specFromRow(
            row,
            primerDays,
            holdoutDays,
            getKlinesFn,
        )
        specLocal = dict(spec)
        specLocal["version"] = RESULT_CACHE_VERSION
        specHash = _specHash(specLocal)
        if specHash in seen:
            continue
        seen.add(specHash)
        newRow = {
            k: row[k]
            for k in RESULT_FIELD_NAMES
            if k in row
        }
        newRow['specHash'] = specHash
        existing.append(newRow)

    fieldNames = list(RESULT_FIELD_NAMES) + ['specHash']
    os.makedirs(os.path.dirname(globalPath), exist_ok=True)
    tmpPath = f"{globalPath}.tmp"
    with open(tmpPath, 'w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fieldNames)
        writer.writeheader()
        for row in existing:
            writer.writerow(row)
    os.replace(tmpPath, globalPath)
