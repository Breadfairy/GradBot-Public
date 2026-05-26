#!/usr/bin/env python3
# engine_shared.py – shared analytics/signals/spacing/time helpers.

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from engine import core

# ======================================================================
# Time helpers
# ======================================================================


def bars_per_day(ctx) -> float:
    if isinstance(ctx, dict):
        meta = ctx.get("_cache")
        interval = (
            ctx.get("intervalStr")
            or ctx.get("interval")
            or ctx.get("interval_str")
        )
        if not interval and isinstance(meta, dict):
            interval = meta.get("interval")
        if interval:
            val = core.barsPerDayFromInterval(str(interval))
            ctx["barsPerDay"] = val
            return float(val)
        cached = ctx.get("barsPerDay")
        if isinstance(cached, (int, float)) and cached > 0.0:
            return float(cached)
    kts = np.asarray([int(k[0]) for k in ctx["klines"]], dtype=np.int64)
    if kts.size <= 1:
        val = 96.0
        if isinstance(ctx, dict):
            ctx["barsPerDay"] = val
        return val
    diffs = np.diff(kts).astype(float)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        val = 96.0
        if isinstance(ctx, dict):
            ctx["barsPerDay"] = val
        return val
    med = float(np.median(diffs))
    day_ms = float(24 * 60 * 60 * 1000)
    val = day_ms / med if med > 0 else 96.0
    if isinstance(ctx, dict):
        ctx["barsPerDay"] = val
    return val


def periods_per_year(interval_str: str) -> float:
    key = str(interval_str).lower().strip()
    if key.endswith('m') and key[:-1].isdigit():
        mins = max(int(key[:-1]), 1)
        per_day = int((24 * 60) / mins)
        return per_day * 365.0
    if key.endswith('h') and key[:-1].isdigit():
        hours = max(int(key[:-1]), 1)
        per_day = int(24 / hours)
        return per_day * 365.0
    if key.endswith('d'):
        return 365.0
    return 365.0


# ======================================================================
# Context + gradients
# ======================================================================


def emaLpf(values, period: int):
    return core.emaLpf(values, period)


def buildContext(klines, periods):
    return core.buildContext(klines, periods)


def calcSpacing(ctx, i):
    m1, m2, m3 = (ctx["mas"][j][i] for j in range(3))
    if np.isnan(m1) or np.isnan(m2) or np.isnan(m3):
        return np.nan, np.nan, np.nan
    e2 = max(abs(m2), 1e-12)
    e3 = max(abs(m3), 1e-12)
    return abs(m1 - m2) / e2, abs(m2 - m3) / e3, abs(m1 - m3) / e3


def liveGradsAt(ctx, period, i, target=100.0):
    pi = ctx["periods"].index(period)
    sm = ctx["smoothMas"][pi]
    if i <= 0:
        return 0.0, 0.0, 0.0, 0.0

    g1n = sm[i] - sm[i - 1]
    g1d = sm[i] if sm[i] != 0 else 1e-12
    s1 = (g1n / g1d) * target

    if i >= 2:
        g1pn = sm[i - 1] - sm[i - 2]
        g1pd = sm[i - 1] if sm[i - 1] != 0 else 1e-12
        s1p = (g1pn / g1pd) * target
    else:
        s1p = 0.0
    g2 = s1 - s1p

    if i >= 3:
        g1ppn = sm[i - 2] - sm[i - 3]
        g1ppd = sm[i - 2] if sm[i - 2] != 0 else 1e-12
        s1pp = (g1ppn / g1ppd) * target
        g2p = s1p - s1pp
    else:
        g2p = 0.0
    g3 = g2 - g2p

    if i >= 4:
        g1pppn = sm[i - 3] - sm[i - 4]
        g1pppd = sm[i - 3] if sm[i - 3] != 0 else 1e-12
        s1ppp = (g1pppn / g1pppd) * target
        g2pp = s1pp - s1ppp
        g3p = g2p - g2pp
    else:
        g3p = 0.0
    g4 = g3 - g3p

    g3 *= -1.0
    return s1, g2 * 10.0, g3 * 10.0, g4 * 10.0


def trend(ctx, i):
    m1, m2, m3 = (ctx["mas"][j][i] for j in range(3))
    if m1 > m2 > m3:
        return "BULL"
    if m1 < m2 < m3:
        return "BEAR"
    if m1 > m3 > m2:
        return "HALF_BULL"
    if m1 < m3 < m2:
        return "HALF_BEAR"
    return "NEUTRAL"


# ======================================================================
# Signals
# ======================================================================


def _grad1Series(sm: np.ndarray, target: float = 100.0) -> np.ndarray:
    return core.grad1Series(sm, target=target)


def trendCodes(m1: np.ndarray, m2: np.ndarray, m3: np.ndarray) -> np.ndarray:
    return core.trendCodes(m1, m2, m3)


def macroDynCarry(
    macroDyn: np.ndarray,
    trendCode: np.ndarray,
) -> np.ndarray:
    return core.macroDynCarry(macroDyn, trendCode)


def _rollingPctArrays(
    closes: np.ndarray,
    lookbacks: List[int],
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    return core.rollingPctArrays(closes, lookbacks)


def buildSignals(
    ctx,
    lookbacks: List[int],
) -> Dict[str, object]:
    return core.buildSignals(ctx, lookbacks)


def rollingMeanAndStd(
    series: np.ndarray, window: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Rolling mean/std (ddof=0)."""
    return core.rollingMeanAndStd(series, window)


def zscoreFromStats(
    series: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    clip: float = 10.0,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    return core.zscoreFromStats(
        np.asarray(series, dtype=float),
        np.asarray(mean, dtype=float),
        np.asarray(std, dtype=float),
        clip=clip,
        eps=eps,
    )


def zscoreSeries(
    ctx,
    series: np.ndarray,
    windowBars: int,
    seriesId: str,
    clip: float = 10.0,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    winBars = int(windowBars)
    arr = np.asarray(series, dtype=float)
    mean, std = rollingMeanAndStd(arr, winBars)
    return zscoreFromStats(arr, mean, std, clip=clip, eps=eps)
