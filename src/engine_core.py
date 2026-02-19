#!/usr/bin/env python3
# engine_core.py – causal, cache-free numeric kernels (SoA-friendly).

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class CtxSoa:
    """SoA-style context for causal signal and flag generation."""

    opens: np.ndarray
    closes: np.ndarray
    periods: List[int]
    mas: List[np.ndarray]
    ath: np.ndarray


@dataclass(frozen=True)
class SigSoa:
    """SoA-style signals for causal flag generation."""

    g1P1: np.ndarray
    g1P3: np.ndarray
    s12: np.ndarray
    s23: np.ndarray
    trendCode: np.ndarray
    pctBelow: Dict[int, np.ndarray]
    pctAbove: Dict[int, np.ndarray]


def barsPerDayFromInterval(intervalStr: str) -> float:
    key = str(intervalStr).strip().lower()
    if key.endswith("m") and key[:-1].isdigit():
        mins = max(int(key[:-1]), 1)
        return float((24 * 60) / mins)
    if key.endswith("h") and key[:-1].isdigit():
        hours = max(int(key[:-1]), 1)
        return float(24 / hours)
    if key.endswith("d"):
        return 1.0
    return 96.0


def emaLpf(values, period: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return np.zeros(0, dtype=float)
    p = max(int(period), 1)
    alpha = 2.0 / (float(p) + 1.0)
    out = np.empty(n, dtype=float)
    out[0] = float(arr[0])
    a = float(alpha)
    b = 1.0 - a
    for i in range(1, n):
        out[i] = (a * float(arr[i])) + (b * float(out[i - 1]))
    return out


def buildContext(klines: list, periods: Iterable[int]) -> dict:
    periodsList = [int(p) for p in periods]
    closes = np.asarray([float(k[4]) for k in klines], dtype=float)
    opens = np.asarray([float(k[1]) for k in klines], dtype=float)
    mas = [emaLpf(closes, p) for p in periodsList]
    ath = np.maximum.accumulate(closes)
    return {
        "klines": klines,
        "closes": closes,
        "opens": opens,
        "periods": periodsList,
        "mas": mas,
        "smoothMas": mas,
        "ath": ath,
    }


def grad1Series(sm: np.ndarray, target: float = 100.0) -> np.ndarray:
    arr = np.asarray(sm, dtype=float)
    n = int(arr.size)
    out = np.zeros(n, dtype=float)
    if n <= 1:
        return out
    num = arr[1:] - arr[:-1]
    den = np.where(arr[1:] != 0.0, arr[1:], 1e-12)
    out[1:] = (num / den) * float(target)
    return out


def trendCodes(m1: np.ndarray, m2: np.ndarray, m3: np.ndarray) -> np.ndarray:
    m1a = np.asarray(m1, dtype=float)
    m2a = np.asarray(m2, dtype=float)
    m3a = np.asarray(m3, dtype=float)
    bull = (m1a > m2a) & (m2a > m3a)
    bear = (m1a < m2a) & (m2a < m3a)
    halfBull = (m1a > m3a) & (m3a > m2a)
    halfBear = (m1a < m3a) & (m3a < m2a)
    code = np.zeros_like(m1a, dtype=int)
    code[bull] = 1
    code[bear] = -1
    code[halfBear] = -2
    code[halfBull] = 2
    return code


def rollingMeanAndStd(
    series: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(series, dtype=float)
    n = int(arr.size)
    win = max(int(window), 1)
    mean = np.full(n, np.nan, dtype=float)
    std = np.full(n, np.nan, dtype=float)
    if n < win:
        return mean, std

    csum = np.concatenate(([0.0], np.cumsum(arr, dtype=float)))
    csum2 = np.concatenate(([0.0], np.cumsum(arr * arr, dtype=float)))
    sumWin = csum[win:] - csum[:-win]
    sum2Win = csum2[win:] - csum2[:-win]
    meanWin = sumWin / float(win)
    mean[win - 1:] = meanWin
    meanSq = sum2Win / float(win)
    var = meanSq - np.square(meanWin)
    var = np.where(var < 0.0, 0.0, var)
    std[win - 1:] = np.sqrt(var)
    return mean, std


def zscoreFromStats(
    series: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    clip: float = 10.0,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute z-scores from precomputed rolling mean/std.

    Cache-free and deterministic; callers handle any caching externally.
    """
    valid = np.isfinite(mean) & np.isfinite(std) & (std > float(eps))
    z = np.zeros_like(series, dtype=float)
    np.divide(series - mean, std, out=z, where=valid)
    z = np.clip(z, -float(clip), float(clip))
    return z, valid


def zscoreRolling(
    series: np.ndarray,
    window: int,
    clip: float = 10.0,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rolling z-score using mean/std (ddof=0)."""
    arr = np.asarray(series, dtype=float)
    mean, std = rollingMeanAndStd(arr, int(window))
    return zscoreFromStats(arr, mean, std, clip=clip, eps=eps)


def rollingMaxMin(
    values: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    win = max(int(window), 1)
    rmax = np.empty(n, dtype=float)
    rmin = np.empty(n, dtype=float)
    dqMax: deque[int] = deque()
    dqMin: deque[int] = deque()
    for i in range(n):
        start = i - win + 1
        while dqMax and dqMax[0] < start:
            dqMax.popleft()
        while dqMin and dqMin[0] < start:
            dqMin.popleft()

        val = float(arr[i])
        while dqMax and float(arr[dqMax[-1]]) <= val:
            dqMax.pop()
        dqMax.append(i)
        while dqMin and float(arr[dqMin[-1]]) >= val:
            dqMin.pop()
        dqMin.append(i)
        rmax[i] = float(arr[dqMax[0]])
        rmin[i] = float(arr[dqMin[0]])
    return rmax, rmin


def rollingPctArrays(
    closes: np.ndarray,
    lookbacks: List[int],
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    closesArr = np.asarray(closes, dtype=float)
    below: Dict[int, np.ndarray] = {}
    above: Dict[int, np.ndarray] = {}
    for lb in sorted(set(int(x) for x in lookbacks)):
        win = max(lb + 1, 1)
        rmax, rmin = rollingMaxMin(closesArr, win)
        with np.errstate(divide="ignore", invalid="ignore"):
            pb = (
                (rmax - closesArr)
                / np.where(rmax > 0.0, rmax, 1.0)
                * 100.0
            )
            pa = (
                (closesArr - rmin)
                / np.where(rmin > 0.0, rmin, 1.0)
                * 100.0
            )
        below[lb] = pb.astype(float)
        above[lb] = pa.astype(float)
    return below, above


def buildSignals(ctx, lookbacks: List[int]) -> Dict[str, object]:
    m1, m2, m3 = (ctx["mas"][i] for i in range(3))
    sm1 = ctx["smoothMas"][0]
    sm3 = ctx["smoothMas"][2]
    g1p1 = grad1Series(np.asarray(sm1, dtype=float))
    g1p3 = grad1Series(np.asarray(sm3, dtype=float))

    den2 = np.maximum(np.abs(m2), 1e-12)
    den3 = np.maximum(np.abs(m3), 1e-12)
    s12 = np.abs(m1 - m2) / den2
    s23 = np.abs(m2 - m3) / den3

    code = trendCodes(m1, m2, m3)

    closes = np.asarray(ctx["closes"], dtype=float)
    pctBelow, pctAbove = rollingPctArrays(closes, lookbacks)

    return {
        "g1P1": g1p1,
        "g1P3": g1p3,
        "s12": s12,
        "s23": s23,
        "trendCode": code,
        "pctBelow": pctBelow,
        "pctAbove": pctAbove,
    }


def energyCsumFromMas(
    m1: np.ndarray,
    m2: np.ndarray,
    m3: np.ndarray,
    trendCode: np.ndarray,
    leg: str,
) -> np.ndarray:
    code = np.asarray(trendCode, dtype=int)
    valid = np.abs(code) == 1
    regime = np.where(valid, code, 0)
    n = int(code.size)
    out = np.zeros(n, dtype=float)
    if n == 0:
        return out

    m1a = np.asarray(m1, dtype=float)
    m2a = np.asarray(m2, dtype=float)
    m3a = np.asarray(m3, dtype=float)
    delta = np.abs(m1a - m2a) if leg == "12" else np.abs(m2a - m3a)

    running = 0.0
    prevReg = 0
    prevValid = False
    for i in range(n):
        if not bool(valid[i]):
            running = 0.0
            out[i] = 0.0
            prevReg = 0
            prevValid = False
            continue
        reg = int(regime[i])
        if (not prevValid) or (reg != prevReg):
            running = 0.0
        running += float(delta[i])
        out[i] = running
        prevReg = reg
        prevValid = True
    return out


def energyCsum(ctx, trendCode: np.ndarray, leg: str) -> np.ndarray:
    m1, m2, m3 = (ctx["mas"][i] for i in range(3))
    return energyCsumFromMas(m1, m2, m3, trendCode, leg)


def spreadPeakRatioFromMas(
    mA: np.ndarray,
    mB: np.ndarray,
    trendCode: np.ndarray,
) -> np.ndarray:
    """Per-regime (bull/bear) spread / peak(spread) ratio in [0..1]."""
    code = np.asarray(trendCode, dtype=int)
    valid = np.abs(code) == 1
    regime = np.where(valid, code, 0)
    n = int(code.size)
    out = np.ones(n, dtype=float)
    if n == 0:
        return out

    mAa = np.asarray(mA, dtype=float)
    mBa = np.asarray(mB, dtype=float)
    spread = np.abs(mAa - mBa)

    peak = 0.0
    prevReg = 0
    prevValid = False
    for i in range(n):
        if not bool(valid[i]):
            peak = 0.0
            out[i] = 1.0
            prevReg = 0
            prevValid = False
            continue
        reg = int(regime[i])
        val = float(spread[i])
        if (not prevValid) or (reg != prevReg):
            peak = val
        elif val > peak:
            peak = val
        out[i] = (val / peak) if peak > 0.0 else 1.0
        prevReg = reg
        prevValid = True
    return out


def spreadPeakRatio(ctx, trendCode: np.ndarray) -> np.ndarray:
    m2, m3 = (ctx["mas"][i] for i in (1, 2))
    return spreadPeakRatioFromMas(m2, m3, trendCode)


def _zFracPositive(
    z: np.ndarray,
    zmin: float,
    zmax: float,
) -> np.ndarray:
    zp = np.maximum(np.asarray(z, dtype=float), 0.0)
    if float(zmax) == float(zmin):
        return np.zeros_like(zp, dtype=float)
    frac = (zp - float(zmin)) / (float(zmax) - float(zmin))
    return np.clip(frac, 0.0, 1.0)


def macroDynFromMas(
    m1: np.ndarray,
    m2: np.ndarray,
    m3: np.ndarray,
    barsPerDay: float,
    winDays: float,
    zmin: float,
    zmax: float,
    pctMax: float,
    pctMin: float,
    gradWinDays: float = 0.0,
    gradZMin: float = 0.0,
    gradZMax: float = 0.0,
    gradMultMin: float = 1.0,
    gradMultMax: float = 1.0,
) -> np.ndarray:
    """Compute macro dyn% from macro EMA1/EMA3 arrays.

    All computations are causal and cache-free (suitable for C port).
    """
    m1a = np.asarray(m1, dtype=float)
    m2a = np.asarray(m2, dtype=float)
    m3a = np.asarray(m3, dtype=float)
    spread13 = m1a - m3a
    spreadSign = np.sign(spread13).astype(int)

    den3 = np.maximum(np.abs(m3a), 1e-12)
    spacing13Pct = (np.abs(spread13) / den3) * 100.0

    winBars = max(int(round(float(winDays) * float(barsPerDay))), 1)
    zSpacing, _valid = zscoreRolling(spacing13Pct, winBars, clip=10.0)
    baseMag = _zFracPositive(zSpacing, float(zmin), float(zmax)) * float(pctMax)

    ratio = spreadPeakRatioFromMas(m1a, m3a, spreadSign)
    # Quadratic damping as EMA1-EMA3 spread collapses.
    mult = ratio * ratio
    pctMinVal = float(pctMin)

    regValid = np.abs(spreadSign) == 1
    idx = np.arange(spreadSign.size, dtype=int)
    start = regValid.copy()
    if start.size > 1:
        start[1:] &= (~regValid[:-1]) | (spreadSign[1:] != spreadSign[:-1])
    startIdx = np.where(start, idx, 0)
    lastStart = np.maximum.accumulate(startIdx)
    ageBars = np.where(regValid, idx - lastStart, 0).astype(float)
    warmFrac = np.clip(ageBars / float(winBars), 0.0, 1.0)

    if pctMinVal > 0.0:
        baseMag = np.maximum(baseMag, pctMinVal)
        # Warmup ramp within the regime to avoid z-score jump artifacts.
        baseMag = pctMinVal + (baseMag - pctMinVal) * warmFrac
        dynMag = pctMinVal + (baseMag - pctMinVal) * mult
    else:
        dynMag = (baseMag * warmFrac) * mult

    gradBars = max(
        int(round(float(gradWinDays) * float(barsPerDay))),
        1,
    )
    g1p2Abs = np.zeros_like(m2a, dtype=float)
    if g1p2Abs.size > 1:
        num = m2a[1:] - m2a[:-1]
        den = np.where(m2a[1:] != 0.0, m2a[1:], 1e-12)
        g1p2Abs[1:] = np.abs((num / den) * 100.0)

    zGrad, _gValid = zscoreRolling(g1p2Abs, gradBars, clip=10.0)
    gradFrac = _zFracPositive(zGrad, float(gradZMin), float(gradZMax))
    gradMultVal = float(gradMultMin) + gradFrac * (
        float(gradMultMax) - float(gradMultMin)
    )
    # Gradient multiplier tightens steep regimes and relaxes near flats.
    dynMag = dynMag * gradMultVal

    pctMaxVal = float(pctMax)
    if pctMinVal > 0.0:
        dynMag = np.clip(dynMag, pctMinVal, pctMaxVal)
    else:
        dynMag = np.clip(dynMag, 0.0, pctMaxVal)
    return np.sign(spread13) * dynMag
