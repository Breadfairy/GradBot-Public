#!/usr/bin/env python3
# metrics.py – portfolio and risk metrics helpers.

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import math
import numpy as np

# ======================================================================
# Equity curves and returns
# ======================================================================


def equityCurveFromTrades(
    closes: Sequence[float],
    trades: Iterable[Any],
    startIndex: int,
    seedQuote: float,
) -> np.ndarray:
    n = len(closes)
    byIdx: Dict[int, List[Any]] = {}
    for tr in trades:
        idx = int(getattr(tr, "index", -1))
        if idx < startIndex:
            continue
        byIdx.setdefault(idx, []).append(tr)
    quote = float(seedQuote)
    base = 0.0
    length = max(n - startIndex, 0)
    out = np.zeros(length, dtype=float)
    pos = 0
    for i in range(startIndex, n):
        for tr in byIdx.get(i, ()):
            quote += float(getattr(tr, "cashDelta", 0.0))
            base += float(getattr(tr, "baseDelta", 0.0))
        price = float(closes[i])
        out[pos] = quote + base * price
        pos += 1
    return out


def allocationCurveFromTrades(
    closes: Sequence[float],
    trades: Iterable[Any],
    startIndex: int,
    seedQuote: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(closes)
    byIdx: Dict[int, List[Any]] = {}
    for tr in trades:
        idx = int(getattr(tr, "index", -1))
        if idx < startIndex:
            continue
        byIdx.setdefault(idx, []).append(tr)
    quote = float(seedQuote)
    base = 0.0
    length = max(n - startIndex, 0)
    assetFrac = np.zeros(length, dtype=float)
    quoteFrac = np.zeros(length, dtype=float)
    pos = 0
    for i in range(startIndex, n):
        for tr in byIdx.get(i, ()):
            quote += float(getattr(tr, "cashDelta", 0.0))
            base += float(getattr(tr, "baseDelta", 0.0))
        price = float(closes[i])
        assetValue = base * price
        totalValue = quote + assetValue
        if totalValue > 1e-12:
            assetFrac[pos] = assetValue / totalValue
            quoteFrac[pos] = quote / totalValue
        else:
            assetFrac[pos] = 0.0
            quoteFrac[pos] = 1.0
        pos += 1
    return assetFrac, quoteFrac


def stepReturns(curve: Sequence[float]) -> np.ndarray:
    arr = np.asarray(curve, dtype=float)
    prev = np.where(arr[:-1] != 0.0, arr[:-1], 1e-12)
    return (arr[1:] / prev) - 1.0


def lifecycleEdgeStats(
    curveSim: Sequence[float],
    curveBench: Sequence[float],
    trackPct: float = 5.0,
) -> Dict[str, float]:
    sim = np.asarray(curveSim, dtype=float)
    bench = np.asarray(curveBench, dtype=float)
    n = min(int(sim.size), int(bench.size))
    if n <= 0:
        nan = float("nan")
        return {
            "lifecycleEdgeMean": nan,
            "lifecycleEdgeMedian": nan,
            "lifecycleEdgeP25": nan,
            "lifecycleEdgeMin": nan,
            "lifecycleUnderwaterPct": nan,
            "lifecycleUnderwaterMean": nan,
            "lifecycleTrackingPct": nan,
            "lifecycleEdgeMdd": nan,
            "lifecycleEdgeScore": nan,
        }

    sim = sim[:n]
    bench = np.where(bench[:n] > 0.0, bench[:n], 1e-12)
    edge = ((sim / bench) - 1.0) * 100.0
    finite = edge[np.isfinite(edge)]
    if finite.size == 0:
        return lifecycleEdgeStats([], [])

    sortedEdge = np.sort(finite)
    p25Idx = int(np.floor(0.25 * float(sortedEdge.size - 1)))
    meanVal = float(np.mean(finite))
    medianVal = float(np.median(finite))
    p25Val = float(sortedEdge[p25Idx])
    minVal = float(np.min(finite))
    under = np.maximum(-finite, 0.0)
    underPct = float(np.mean(finite < 0.0) * 100.0)
    underMean = float(np.mean(under))
    track = float(np.mean(np.abs(finite) < float(trackPct)) * 100.0)
    peak = np.maximum.accumulate(finite)
    edgeMdd = float(np.max(peak - finite))
    terminal = float(finite[-1])
    score = (
        terminal
        + (0.75 * medianVal)
        + p25Val
        + (1.25 * minVal)
        - (0.75 * underMean)
        - (0.50 * underPct)
        - (0.35 * track)
        - (0.75 * edgeMdd)
    )
    return {
        "lifecycleEdgeMean": meanVal,
        "lifecycleEdgeMedian": medianVal,
        "lifecycleEdgeP25": p25Val,
        "lifecycleEdgeMin": minVal,
        "lifecycleUnderwaterPct": underPct,
        "lifecycleUnderwaterMean": underMean,
        "lifecycleTrackingPct": track,
        "lifecycleEdgeMdd": edgeMdd,
        "lifecycleEdgeScore": float(score),
    }


def timeRegionStats(
    curveSim: Sequence[float],
    curveBench: Sequence[float],
) -> Dict[str, float]:
    lifecycle = lifecycleEdgeStats(curveSim, curveBench)
    sim = np.asarray(curveSim, dtype=float)
    bench = np.asarray(curveBench, dtype=float)
    n = min(int(sim.size), int(bench.size))
    if n <= 0:
        out = {
            "timeGoodPct": float("nan"),
            "timeEdge10Pct": float("nan"),
            "timeEdge25Pct": float("nan"),
            "timeBadPct": float("nan"),
            "timeNearHodlPct": float("nan"),
            "timeRegionScore": float("nan"),
        }
        out.update(lifecycle)
        return out
    edge = ((sim[:n] / np.maximum(bench[:n], 1e-12)) - 1.0) * 100.0
    finite = edge[np.isfinite(edge)]
    if finite.size == 0:
        return timeRegionStats([], [])
    timeGoodPct = float(np.mean(finite > 0.0) * 100.0)
    timeEdge10Pct = float(np.mean(finite > 10.0) * 100.0)
    timeEdge25Pct = float(np.mean(finite > 25.0) * 100.0)
    timeBadPct = float(np.mean(finite < 0.0) * 100.0)
    timeNearHodlPct = float(np.mean(np.abs(finite) < 5.0) * 100.0)
    timeRegionScore = (
        float(lifecycle["lifecycleEdgeScore"])
        + (0.35 * timeGoodPct)
        + (0.25 * timeEdge10Pct)
        + (0.15 * timeEdge25Pct)
        - (0.50 * timeBadPct)
        - (0.35 * timeNearHodlPct)
    )
    out = {
        "timeGoodPct": timeGoodPct,
        "timeEdge10Pct": timeEdge10Pct,
        "timeEdge25Pct": timeEdge25Pct,
        "timeBadPct": timeBadPct,
        "timeNearHodlPct": timeNearHodlPct,
        "timeRegionScore": timeRegionScore,
    }
    out.update(lifecycle)
    return out


# ======================================================================
# Risk ratios and drawdowns
# ======================================================================


def sharpeRatio(returns: Sequence[float], periodsPerYear: float) -> float:
    r = np.asarray(returns, dtype=float)
    mu = np.nanmean(r)
    sigma = np.nanstd(r, ddof=1)
    if sigma <= 1e-12:
        return float("nan")
    return (mu / sigma) * np.sqrt(float(periodsPerYear))


def _downsideStd(returns: np.ndarray, threshold: float = 0.0) -> float:
    downs = returns[returns < threshold]
    return np.nanstd(downs, ddof=1)


def sortinoRatio(returns: Sequence[float], periodsPerYear: float) -> float:
    r = np.asarray(returns, dtype=float)
    mu = np.nanmean(r)
    dstd = _downsideStd(r, 0.0)
    if not np.isfinite(dstd) or dstd <= 1e-12:
        return float("nan")
    return (mu / dstd) * np.sqrt(float(periodsPerYear))


def maxDrawdown(curve: Sequence[float]) -> float:
    arr = np.asarray(curve, dtype=float)
    if arr.size == 0:
        return float("nan")
    peak = -np.inf
    mdd = 0.0
    for v in arr:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > mdd:
                mdd = dd
    return mdd


def cagr(curve: Sequence[float], years: float) -> float:
    arr = np.asarray(curve, dtype=float)
    if arr.size == 0 or years <= 0.0:
        return float("nan")
    start = arr[0]
    end = arr[-1]
    if start <= 0.0:
        return float("nan")
    return (end / start) ** (1.0 / years) - 1.0


# ======================================================================
# Edge and scoring helpers
# ======================================================================


def grossPctVsBench(simValue: float, benchValue: float) -> float:
    return ((simValue / max(benchValue, 1e-12)) - 1.0) * 100.0


def edgeVsBench(
    simPostTaxValue: float,
    benchPostTaxValue: float,
    grossEdge: float,
    netEdge: float,
    taxMode: str,
) -> float:
    return grossEdge if str(taxMode).lower() == "income" else netEdge


def scoreFromEdge(edgeValue: float) -> float:
    return (
        round(edgeValue, 6)
        if isinstance(edgeValue, (int, float))
        and math.isfinite(edgeValue)
        else float('nan')
    )


# ======================================================================
# Aggregate risk summaries
# ======================================================================


def summarizeRisk(
    sharpe: float,
    cagr_v: float,
    mdd: float,
) -> Tuple[float, float, float]:
    sh = (
        float(sharpe)
        if isinstance(sharpe, (int, float)) and math.isfinite(sharpe)
        else float('nan')
    )
    cg = (
        float(cagr_v)
        if isinstance(cagr_v, (int, float)) and math.isfinite(cagr_v)
        else float('nan')
    )
    md = (
        float(mdd)
        if isinstance(mdd, (int, float)) and math.isfinite(mdd)
        else float('nan')
    )
    return sh, cg, md


def marRatio(cagr_v: float, mdd: float) -> float:
    if not isinstance(cagr_v, (int, float)) or not isinstance(mdd, (int, float)):
        return float('nan')
    if not math.isfinite(cagr_v) or not math.isfinite(mdd):
        return float('nan')
    if mdd <= 1e-12:
        return float('nan')
    return cagr_v / mdd


def summarizeRiskFull(
    sharpe: float,
    sortino: float,
    cagr_v: float,
    mdd: float,
) -> Tuple[float, float, float, float, float]:
    sh, cg, md = summarizeRisk(sharpe, cagr_v, mdd)
    so = (
        float(sortino)
        if isinstance(sortino, (int, float)) and math.isfinite(sortino)
        else float('nan')
    )
    mar = marRatio(cg, md)
    return sh, so, cg, md, mar


# ======================================================================
# Rolling medians for Sharpe/Sortino
# ======================================================================


def rollingSharpeSortinoMedian(
    returns: Sequence[float],
    periodsPerYear: float,
    window: int,
) -> Tuple[float, float]:
    arr = np.asarray(returns, dtype=float)
    n = int(arr.size)
    win = int(window)
    if n <= 1 or win <= 1 or n < win:
        return float('nan'), float('nan')

    ppy = float(periodsPerYear)

    cs = np.cumsum(arr)
    cs2 = np.cumsum(arr * arr)
    pad = np.concatenate(([0.0], cs[:-win]))
    pad2 = np.concatenate(([0.0], cs2[:-win]))
    sumWin = cs[win - 1 :] - pad
    sum2Win = cs2[win - 1 :] - pad2
    w = float(win)
    meanWin = sumWin / w
    varNum = sum2Win - (sumWin * sumWin) / w
    denom = max(win - 1, 1)
    varWin = varNum / float(denom)
    varWin = np.where(varWin < 0.0, 0.0, varWin)
    stdWin = np.sqrt(varWin)
    sharpeArr = np.full_like(meanWin, float('nan'))
    maskStd = stdWin > 1e-12
    sharpeArr[maskStd] = (meanWin[maskStd] / stdWin[maskStd]) * np.sqrt(ppy)

    neg = np.where(arr < 0.0, arr, 0.0)
    maskNeg = (arr < 0.0).astype(float)
    csNeg = np.cumsum(neg)
    cs2Neg = np.cumsum(neg * neg)
    csCntNeg = np.cumsum(maskNeg)
    padNeg = np.concatenate(([0.0], csNeg[:-win]))
    pad2Neg = np.concatenate(([0.0], cs2Neg[:-win]))
    padCntNeg = np.concatenate(([0.0], csCntNeg[:-win]))
    sumNegWin = csNeg[win - 1 :] - padNeg
    sum2NegWin = cs2Neg[win - 1 :] - pad2Neg
    cntNegWin = csCntNeg[win - 1 :] - padCntNeg

    sortinoArr = np.full_like(meanWin, float('nan'))
    cntMask = cntNegWin > 1.5
    validIdx = np.where(cntMask)[0]
    if validIdx.size > 0:
        cntUse = cntNegWin[validIdx]
        sumNegUse = sumNegWin[validIdx]
        sum2NegUse = sum2NegWin[validIdx]
        meanNeg = sumNegUse / cntUse
        varNumNeg = sum2NegUse - cntUse * meanNeg * meanNeg
        denomNeg = np.where(cntUse > 1.5, cntUse - 1.0, 1.0)
        varNeg = varNumNeg / denomNeg
        varNeg = np.where(varNeg < 0.0, 0.0, varNeg)
        dstd = np.sqrt(varNeg)
        maskDstd = dstd > 1e-12
        idxValid = validIdx[maskDstd]
        if idxValid.size > 0:
            muUse = meanWin[idxValid]
            dstdUse = dstd[maskDstd]
            sortinoArr[idxValid] = (muUse / dstdUse) * np.sqrt(ppy)

    sharpeFinite = np.isfinite(sharpeArr)
    sortinoFinite = np.isfinite(sortinoArr)
    sharpeMedian = (
        float(np.nanmedian(sharpeArr))
        if sharpeFinite.any()
        else float('nan')
    )
    sortinoMedian = (
        float(np.nanmedian(sortinoArr))
        if sortinoFinite.any()
        else float('nan')
    )
    return sharpeMedian, sortinoMedian
