#!/usr/bin/env python3
# daily_posture.py - daily cluster posture helpers for live execution.

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from engine import core
from repo_paths import livePath
from strategy.posture import (
    dailyPostureState,
    dailyPostureStep as strategyPostureStep,
    defaultDailyPosture,
    markDailyLockState,
)


########################################################################
# Fixed Daily Model
########################################################################

DAY_MS = 24 * 60 * 60 * 1000
FRAME_CACHE: Dict[str, Dict[str, np.ndarray]] = {}
MODEL_CACHE: Dict[str, dict[str, Any]] = {}


########################################################################
# Config Helpers
########################################################################

def dailyPostureEnabled(overrides: Dict[str, Any]) -> bool:
    modelPath = str(overrides.get('DAILY_CLUSTER_MODEL_PATH', '')).strip()
    labelPath = str(overrides.get('DAILY_CLUSTER_PATH', '')).strip()
    return bool(modelPath or labelPath)


def dailyPostureWarmupRows(overrides: Dict[str, Any]) -> int:
    # Use enough posture rows for model features and fixed rolling z-score.
    path = _modelPath(overrides)
    if path is None:
        return 168
    model = _loadModel(path)
    return max(int(model.get('windowBars', 60)), 168, 30)


def _livePath(rawPath: str) -> Path:
    path = Path(rawPath)
    if path.is_absolute():
        return path
    return livePath(path)


def _modelPath(overrides: Dict[str, Any]) -> Path | None:
    raw = str(overrides.get('DAILY_CLUSTER_MODEL_PATH', '')).strip()
    return _livePath(raw) if raw else None


def _clusterPath(overrides: Dict[str, Any]) -> Path | None:
    raw = str(overrides.get('DAILY_CLUSTER_PATH', '')).strip()
    return _livePath(raw) if raw else None


########################################################################
# Model Feature Helpers
########################################################################

def _loadModel(path: Path) -> dict[str, Any]:
    cacheKey = str(path.resolve())
    if cacheKey not in MODEL_CACHE:
        with path.open() as fh:
            MODEL_CACHE[cacheKey] = json.load(fh)
    return MODEL_CACHE[cacheKey]


def _rowsCol(rows: list, idx: int, dtype=float) -> np.ndarray:
    return np.asarray([dtype(r[idx]) for r in rows], dtype=dtype)


def _safePct(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan, dtype=float)
    valid = np.asarray(den, dtype=float) != 0.0
    np.divide(num, den, out=out, where=valid)
    return out * 100.0


def _retPct(values: np.ndarray, bars: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    lag = int(bars)
    if values.size <= lag:
        return out
    prev = values[:-lag]
    cur = values[lag:]
    valid = prev != 0.0
    ret = np.full(cur.shape, np.nan, dtype=float)
    np.divide(cur, prev, out=ret, where=valid)
    out[lag:] = (ret - 1.0) * 100.0
    return out


def _rollingHighLow(
    highVals: np.ndarray,
    lowVals: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    high, _highMin = core.rollingMaxMin(highVals, window)
    _lowMax, low = core.rollingMaxMin(lowVals, window)
    warm = max(int(window), 1) - 1
    high[:warm] = np.nan
    low[:warm] = np.nan
    return high, low


def _priorRollingZ(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    win = max(int(window), 1)
    mean, std = core.rollingMeanAndStd(values, win)
    meanPrior = np.concatenate(([np.nan], mean[:-1]))
    stdPrior = np.concatenate(([np.nan], std[:-1]))
    valid = (
        np.isfinite(meanPrior)
        & np.isfinite(stdPrior)
        & (stdPrior > 1e-12)
    )
    np.divide(values - meanPrior, stdPrior, out=out, where=valid)
    return np.clip(out, -10.0, 10.0)


def _gradPct(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    if values.size <= 1:
        return out
    num = values[1:] - values[:-1]
    den = np.where(values[1:] != 0.0, values[1:], np.nan)
    out[1:] = (num / den) * 100.0
    return out


def _dailyFeatures(rows: list, model: dict[str, Any]) -> dict[str, np.ndarray]:
    openPx = _rowsCol(rows, 1, float)
    high = _rowsCol(rows, 2, float)
    low = _rowsCol(rows, 3, float)
    close = _rowsCol(rows, 4, float)
    volume = _rowsCol(rows, 5, float)
    periods = dict(model['periods'])
    p1 = int(periods['fast'])
    p2 = int(periods['mid'])
    p3 = int(periods['slow'])
    window = int(model.get('windowBars', 60))
    ema1 = core.emaLpf(close, p1)
    ema2 = core.emaLpf(close, p2)
    ema3 = core.emaLpf(close, p3)
    highRoll, lowRoll = _rollingHighLow(high, low, window)
    logRet = np.log(close / np.roll(close, 1))
    logRet[0] = 0.0
    ret1 = _retPct(close, 1)
    ret24 = _retPct(close, 24)
    absRet1Mean, _absRet1Std = core.rollingMeanAndStd(
        np.nan_to_num(np.abs(ret1), nan=0.0),
        window,
    )
    absRet1Mean[:window] = np.nan
    sumAbs = absRet1Mean * float(window)
    body = close - openPx
    candleRange = high - low
    upper = high - np.maximum(openPx, close)
    lower = np.minimum(openPx, close) - low
    bodyAbsPct = _safePct(np.abs(body), close)
    rangePct = _safePct(candleRange, close)
    logVolume = np.log1p(volume)

    realVol = core.rollingMeanAndStd(logRet, window)[1]
    realVol[:window] = np.nan
    bodyAbsMean = core.rollingMeanAndStd(bodyAbsPct, window)[0]
    rangeMean = core.rollingMeanAndStd(rangePct, window)[0]

    features = {
        'emaGapFastPct': _safePct(close - ema1, close),
        'emaGapMidPct': _safePct(close - ema2, close),
        'emaGapSlowPct': _safePct(close - ema3, close),
        'emaSpreadFastMidPct': _safePct(ema1 - ema2, close),
        'emaSpreadMidSlowPct': _safePct(ema2 - ema3, close),
        'emaSpreadFastSlowPct': _safePct(ema1 - ema3, close),
        'gradFastPct': _gradPct(ema1),
        'gradMidPct': _gradPct(ema2),
        'gradSlowPct': _gradPct(ema3),
        'distHigh24Pct': _safePct(highRoll - close, close),
        'distLow24Pct': _safePct(close - lowRoll, close),
        'range24Pct': _safePct(highRoll - lowRoll, close),
        'realVol24': realVol,
        'ret1h': ret1,
        'ret3h': _retPct(close, 3),
        'ret6h': _retPct(close, 6),
        'ret12h': _retPct(close, 12),
        'ret24h': ret24,
        'trendEfficiency24': np.abs(ret24) / np.where(
            sumAbs > 1e-12,
            sumAbs,
            np.nan,
        ),
        'bodyPct': _safePct(body, close),
        'bodyAbsPct': bodyAbsPct,
        'upperWickPct': _safePct(upper, close),
        'lowerWickPct': _safePct(lower, close),
        'bodyAbsMean24': bodyAbsMean,
        'rangeMean24': rangeMean,
        'logVolumeZ168': _priorRollingZ(logVolume, 168),
    }
    return features


def _inferClusters(rows: list, model: dict[str, Any]) -> np.ndarray:
    names = list(model['features'])
    features = _dailyFeatures(rows, model)
    cols = [np.asarray(features[name], dtype=float) for name in names]
    xRaw = np.column_stack(cols)
    labels = np.full(xRaw.shape[0], -1, dtype=int)
    valid = np.isfinite(xRaw).all(axis=1)
    if not bool(np.any(valid)):
        return labels

    center = np.asarray(model['center'], dtype=float)
    scale = np.asarray(model['scale'], dtype=float)
    scale = np.where(scale != 0.0, scale, 1.0)
    scaled = (xRaw[valid] - center) / scale
    pcaCount = int(model.get('pcaCount', 0))
    if pcaCount > 0:
        pcaMean = np.asarray(model['pcaMean'], dtype=float)
        comps = np.asarray(model['pcaComponents'], dtype=float)
        modelX = (scaled - pcaMean) @ comps.T
    else:
        modelX = scaled

    centroids = np.asarray(model['centroids'], dtype=float)
    diff = modelX[:, None, :] - centroids[None, :, :]
    dist = np.sum(diff * diff, axis=2)
    pred = np.argmin(dist, axis=1).astype(int)
    remap = model.get('clusterRemap', [])
    if remap:
        mapArr = np.asarray(remap, dtype=int)
        pred = mapArr[pred]
    labels[np.flatnonzero(valid)] = pred
    return labels


########################################################################
# Cluster Alignment
########################################################################

def _loadClusterFrame(path: Path) -> Dict[str, np.ndarray]:
    cacheKey = str(path.resolve())
    if cacheKey in FRAME_CACHE:
        return FRAME_CACHE[cacheKey]

    closeMsVals: list[int] = []
    closeVals: list[float] = []
    clusterVals: list[int] = []

    with path.open(newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            cluster = int(float(r['cluster']))
            if cluster < 0:
                continue
            closeMsVals.append(int(float(r['closeMs'])))
            closeVals.append(float(r['close']))
            clusterVals.append(cluster)

    frame = {
        'closeMs': np.asarray(closeMsVals, dtype=np.int64),
        'close': np.asarray(closeVals, dtype=float),
        'cluster': np.asarray(clusterVals, dtype=int),
    }
    FRAME_CACHE[cacheKey] = frame
    return frame


def _dailyGraceMs(closeMs: np.ndarray) -> int:
    # Permit latest static label until the next daily close is due.
    if closeMs.size <= 1:
        return DAY_MS
    diffs = np.diff(closeMs).astype(float)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return DAY_MS
    return int(round(float(np.median(diffs))))


def _dailyRet30(closeVals: np.ndarray) -> np.ndarray:
    out = np.zeros(closeVals.size, dtype=float)
    if closeVals.size > 30:
        prev = closeVals[:-30]
        cur = closeVals[30:]
        np.divide(
            cur,
            np.where(prev > 0.0, prev, 1.0),
            out=out[30:],
        )
        out[30:] = (out[30:] - 1.0) * 100.0
    return out


def _dailyNearHigh(closeVals: np.ndarray) -> np.ndarray:
    high, _low = core.rollingMaxMin(closeVals, 60)
    out = np.zeros(closeVals.size, dtype=float)
    np.divide(
        high,
        closeVals,
        out=out,
        where=closeVals > 0.0,
    )
    return (out - 1.0) * 100.0


def _alignDaily(
    ctx: Dict[str, Any],
    dayClose: np.ndarray,
    closeVals: np.ndarray,
    clusterVals: np.ndarray,
    staleLimitMs: int | None,
) -> Dict[str, np.ndarray]:
    kOpen = np.asarray([int(k[0]) for k in ctx['klines']], dtype=np.int64)
    posRaw = np.searchsorted(dayClose, kOpen, side='right') - 1
    valid = posRaw >= 0
    if staleLimitMs is not None and dayClose.size > 0:
        valid &= kOpen <= (int(dayClose[-1]) + int(staleLimitMs))
    pos = np.clip(posRaw, 0, dayClose.size - 1)
    clusters = np.full(kOpen.size, -1, dtype=int)
    ret30 = np.zeros(kOpen.size, dtype=float)
    nearHigh = np.zeros(kOpen.size, dtype=float)
    dailyRet = _dailyRet30(closeVals)
    dailyNear = _dailyNearHigh(closeVals)
    clusters[valid] = clusterVals[pos[valid]]
    ret30[valid] = dailyRet[pos[valid]]
    nearHigh[valid] = dailyNear[pos[valid]]
    return {
        'cluster': clusters,
        'ret30': ret30,
        'nearHigh': nearHigh,
    }


def _modelDailyArrays(
    ctx: Dict[str, Any],
    overrides: Dict[str, Any],
    dailyRows: list,
) -> Dict[str, np.ndarray]:
    path = _modelPath(overrides)
    model = _loadModel(path) if path is not None else {}
    dayClose = _rowsCol(dailyRows, 6, int).astype(np.int64)
    closeVals = _rowsCol(dailyRows, 4, float)
    clusters = _inferClusters(dailyRows, model)
    return _alignDaily(ctx, dayClose, closeVals, clusters, None)


def _staticDailyArrays(
    ctx: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, np.ndarray] | None:
    path = _clusterPath(overrides)
    if path is None:
        return None
    frame = _loadClusterFrame(path)
    return _alignDaily(
        ctx,
        frame['closeMs'],
        frame['close'],
        frame['cluster'],
        _dailyGraceMs(frame['closeMs']),
    )


def dailyPostureArrays(
    ctx: Dict[str, Any],
    overrides: Dict[str, Any],
    dailyRows: list | None = None,
) -> Dict[str, np.ndarray] | None:
    if not dailyPostureEnabled(overrides):
        return None
    if _modelPath(overrides) is not None and dailyRows:
        return _modelDailyArrays(ctx, overrides, dailyRows)
    return _staticDailyArrays(ctx, overrides)


########################################################################
# Runtime Step
########################################################################

def dailyPostureStep(
    state: Dict[str, Any],
    daily: Dict[str, np.ndarray] | None,
    index: int,
    price: float,
    barsDay: float,
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    if daily is None:
        return defaultDailyPosture()
    cluster = int(daily['cluster'][index])
    return strategyPostureStep(
        state,
        cluster,
        price,
        index,
        barsDay,
        overrides,
    )


def markLockState(state: Dict[str, Any], index: int) -> None:
    markDailyLockState(state, index)


def dailyPostureForIndex(
    ctx: Dict[str, Any],
    overrides: Dict[str, Any],
    index: int,
    barsDay: float,
    dailyRows: list | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    daily = dailyPostureArrays(ctx, overrides, dailyRows)
    state = dailyPostureState()
    posture = {
        'cluster': -1,
        'strong': False,
        'down': False,
        'late': False,
        'forceLock': False,
        'exitTarget': 1.0,
        'lockTarget': 1.0,
    }
    if daily is None:
        return posture, state

    for i in range(0, int(index) + 1):
        price = float(ctx['closes'][i])
        posture = dailyPostureStep(state, daily, i, price, barsDay, overrides)
        if i < int(index) and bool(posture['forceLock']):
            markLockState(state, i)

    return posture, state
