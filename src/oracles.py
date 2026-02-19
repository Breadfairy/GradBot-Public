#!/usr/bin/env python3
# oracles.py – non-causal automatic oracle generation.

from __future__ import annotations

from typing import Any, Iterable, List, Tuple
import argparse
import json
import os
import types
import tempfile

from engine_shared import (
    buildContext,
    buildSignals,
    bars_per_day,
    rollingMeanAndStd,
    spacingState,
    energyCsum,
    trendCodes,
)
import numpy as np
import pandas as pd
from flags import Params
from cache import getKlinesCached
import profile
import cache
from dynamics import g1p3Series, alignMacroDyn, macroDynFromContext
from charting import (
    Chart,
    plotBellCurve,
    BG_COLOR,
    TEXT_COLOR,
    GRID_COLOR,
)
from params import overridesFromDict
from flags import _grad1ZscoreMask


# Local oracle settings for backtest-time oracles. The driver passes
# JSON-tunable settings via params.oracle_settings when available.
# For driver-provided settings we use a half-window in bars directly.
# Backtest-time defaults still use a days→bars conversion.
ORACLE_PEAK_WINDOW_DAYS_DEFAULT = 30.0
ORACLE_MIN_PCT_SINCE_FLAG_DEFAULT = 20.0


class OracleEngine:
    def __init__(self, ctx, params: Params):
        self.ctx = ctx
        self.params = params

    def _settings(self) -> Tuple[int, float]:
        settings = getattr(self.params, "oracle_settings", None)
        if settings is not None:
            halfBars = int(settings["peak_window"])
            minPct = float(settings["min_pct_since_oracle"])
            return max(halfBars, 1), minPct
        bpd = max(bars_per_day(self.ctx), 1.0)
        halfBars = max(
            int(round(float(ORACLE_PEAK_WINDOW_DAYS_DEFAULT) * bpd)),
            1,
        )
        return halfBars, float(ORACLE_MIN_PCT_SINCE_FLAG_DEFAULT)

    def generate(self) -> List[Tuple[int, str]]:
        ctx = self.ctx
        closes = np.asarray(ctx["closes"], dtype=float)
        n = closes.size
        if n == 0:
            return []

        m1 = np.asarray(ctx["mas"][0], dtype=float)
        m2 = np.asarray(ctx["mas"][1], dtype=float)
        m3 = np.asarray(ctx["mas"][2], dtype=float)
        trend = trendCodes(m1, m2, m3)

        halfWinBars, minPct = self._settings()
        winBars = (halfWinBars * 2) + 1

        s = pd.Series(closes)
        rollMax = (
            s.rolling(winBars, center=True, min_periods=1)
            .max()
            .to_numpy()
        )
        rollMin = (
            s.rolling(winBars, center=True, min_periods=1)
            .min()
            .to_numpy()
        )

        g1 = np.zeros(n, dtype=float)
        if n > 1:
            prev = np.where(closes[:-1] != 0.0, closes[:-1], 1e-12)
            g1[1:] = ((closes[1:] / prev) - 1.0) * 100.0
        prefix = np.empty(n + 1, dtype=float)
        prefix[0] = 0.0
        prefix[1:] = np.cumsum(g1)

        def segMean(start: int, end: int) -> float:
            if end <= start:
                return 0.0
            span = end - start
            return float(prefix[end] - prefix[start]) / float(span)

        flags: List[Tuple[int, str]] = []
        lastSell = None
        lastBuy = None
        prevCode = 0

        for idx in range(n):
            code = int(trend[idx])
            price = float(closes[idx])

            if code == 1 and prevCode != 1:
                lastSell = price
            elif code == -1 and prevCode != -1:
                lastBuy = price
            isBull = code == 1
            isBear = code == -1

            isPeak = isBull and price >= float(rollMax[idx])
            isTrough = isBear and price <= float(rollMin[idx])

            startL = max(0, idx - halfWinBars)
            endL = idx
            startR = idx + 1
            endR = min(n, idx + 1 + halfWinBars)
            slopeLeft = segMean(startL, endL)
            slopeRight = segMean(startR, endR)

            if isPeak:
                if not (slopeLeft > 0.0 and slopeRight < 0.0):
                    continue
                prev = float(lastSell)
                if price > 0.0 and prev > 0.0:
                    deltaPct = ((price / prev) - 1.0) * 100.0
                else:
                    deltaPct = 0.0
                keep = deltaPct >= float(minPct)
                if keep:
                    flags.append((idx, "O_SELL"))
                    lastSell = price
            elif isTrough:
                if not (slopeLeft < 0.0 and slopeRight > 0.0):
                    continue
                prev = float(lastBuy)
                if price > 0.0 and prev > 0.0:
                    deltaPct = ((prev / price) - 1.0) * 100.0
                else:
                    deltaPct = 0.0
                keep = deltaPct >= float(minPct)
                if keep:
                    flags.append((idx, "O_BUY"))
                    lastBuy = price

            prevCode = code

        return flags


# ======================================================================
# Oracle profile builder (merged from oracle_driver.py)
# ======================================================================


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ORACLES_BASE = os.path.join(ROOT_DIR, "outputs", "oracles")
DEFAULT_CONFIG = os.path.join(
    ROOT_DIR,
    "inputs",
    "profiles",
    "oracle-config.json",
)

GRAD_WINDOW_DAYS = list(range(7, 366, 2))
SPACING_WINDOW_DAYS = list(range(7, 366, 2))
ENERGY_WINDOW_DAYS = list(range(7, 366, 2))
EPS = 1e-6
_WIN_CAP_LOGGED: set = set()


def _flagsBySide(flags: List[Tuple[int, str]]) -> Tuple[List[int], List[int]]:
    buys: List[int] = []
    sells: List[int] = []
    for idx, label in flags:
        if label == "O_BUY":
            buys.append(int(idx))
        elif label == "O_SELL":
            sells.append(int(idx))
    return buys, sells


def _oracleGateRecall(
    ctx: dict,
    signals: dict,
    periods: List[int],
    startIdx: int,
    oracleBuys: List[int],
    oracleSells: List[int],
    overrides: dict,
    holdoutDays: int,
) -> Tuple[dict, dict, List[int], List[int]]:
    n = int(len(ctx.get("closes", [])))
    idxBuy = np.asarray([i for i in oracleBuys], dtype=int)
    idxSell = np.asarray([i for i in oracleSells], dtype=int)
    if n <= 0:
        empty = {
            "n": 0,
            "grad": 0.0,
            "spacing": 0.0,
            "energy": 0.0,
            "leg": 0.0,
            "all": 0.0,
        }
        return dict(empty), dict(empty), [], []
    idxBuy = idxBuy[(idxBuy >= startIdx) & (idxBuy < n)]
    idxSell = idxSell[(idxSell >= startIdx) & (idxSell < n)]
    idxBuy = np.sort(idxBuy)
    idxSell = np.sort(idxSell)

    trendCode = np.asarray(signals["trendCode"], dtype=int)
    allowBuy = (trendCode == -1) | (trendCode == -2)
    allowSell = (trendCode == 1)
    idxs = np.arange(n, dtype=int)
    validStart = idxs >= int(startIdx)
    bars = max(float(bars_per_day(ctx)), 1.0)

    def _enforceCooldownIndices(indices: np.ndarray, cooldown: int) -> List[int]:
        if indices.size == 0:
            return []
        keep: List[int] = []
        last = int(indices[0]) - int(cooldown)
        for idx in indices.tolist():
            val = int(idx)
            if val - last >= int(cooldown):
                keep.append(val)
                last = val
        return keep

    g1 = np.asarray(signals["g1P1"], dtype=float)
    gradMaskBuy = _grad1ZscoreMask(ctx, allowBuy, g1, overrides, 'BUY')
    gradMaskSell = _grad1ZscoreMask(ctx, allowSell, g1, overrides, 'SELL')

    s12 = np.asarray(signals["s12"], dtype=float)
    s23 = np.asarray(signals["s23"], dtype=float)
    spacingBuy = spacingState(ctx, trendCode, allowBuy, s12, s23, overrides)
    spacingSell = spacingState(ctx, trendCode, allowSell, s12, s23, overrides)
    spacingMaskBuy = spacingBuy.mask
    spacingMaskSell = spacingSell.mask
    energyMaskBuy = spacingBuy.energyMask
    energyMaskSell = spacingSell.energyMask

    intervalMacro = str(overrides["MACRO_INTERVAL"]).strip()
    macroDyn = None
    if intervalMacro:
        meta = ctx.get("_cache") if isinstance(ctx, dict) else None
        baseDays = int(meta.get("days", 0)) if isinstance(meta, dict) else 0
        baseTicker = (
            str(meta.get("ticker", "")).strip()
            if isinstance(meta, dict)
            else ""
        )
        if baseDays > 0 and baseTicker:
            winDays = overrides["MACRO_NRG_WIN_DAYS"]
            zmin = overrides["MACRO_NRG_Z_MIN"]
            zmax = overrides["MACRO_NRG_Z_MAX"]
            pctMin = overrides["MACRO_DYN_PCT_MIN"]
            pctMax = overrides["MACRO_DYN_PCT_MAX"]
            minCandles = (max(periods) * 2) + 1
            klMacro = getKlinesCached(
                baseTicker,
                intervalMacro,
                baseDays,
                minCandles,
                holdoutDays=holdoutDays,
            )
            periodsMacro = list(periods)
            macroP1 = int(overrides["MACRO_P1"])
            macroP2 = int(overrides["MACRO_P2"])
            macroP3 = int(overrides["MACRO_P3"])
            if macroP1 > 0 and len(periodsMacro) >= 1:
                periodsMacro[0] = macroP1
            if macroP2 > 0 and len(periodsMacro) >= 2:
                periodsMacro[1] = macroP2
            if macroP3 > 0:
                if len(periodsMacro) >= 3:
                    periodsMacro[2] = macroP3
                else:
                    periodsMacro.append(macroP3)
            ctxMacro = buildContext(klMacro, periodsMacro)
            ctxMacro["intervalStr"] = str(intervalMacro)
            gradWinDays = float(overrides["MACRO_GRAD_WIN_DAYS"])
            gradZMin = float(overrides["MACRO_GRAD_Z_MIN"])
            gradZMax = float(overrides["MACRO_GRAD_Z_MAX"])
            gradMultMin = float(overrides["MACRO_MULT_GRAD_MIN"])
            gradMultMax = float(overrides["MACRO_MULT_GRAD_MAX"])
            dynMacro = macroDynFromContext(
                ctxMacro,
                float(winDays),
                float(zmin),
                float(zmax),
                float(pctMax),
                float(pctMin),
                gradWinDays=gradWinDays,
                gradZMin=gradZMin,
                gradZMax=gradZMax,
                gradMultMin=gradMultMin,
                gradMultMax=gradMultMax,
            )
            tsMacro = pd.to_datetime(
                [k[0] for k in klMacro],
                unit="ms",
                utc=True,
            ).tz_convert(None).to_pydatetime().tolist()
            tsMicro = pd.to_datetime(
                [k[0] for k in ctx["klines"]],
                unit="ms",
                utc=True,
            ).tz_convert(None).to_pydatetime().tolist()
            if tsMacro and tsMicro:
                macroDyn = alignMacroDyn(tsMacro, dynMacro, tsMicro)

    dyn = macroDyn if macroDyn is not None else np.zeros(n, dtype=float)
    if dyn.shape[0] != n:
        m = min(n, int(dyn.shape[0]))
        dynAligned = np.zeros(n, dtype=float)
        if m > 0:
            dynAligned[:m] = dyn[:m]
        dyn = dynAligned

    buyAnchorIdx = np.full(n, -1, dtype=int)
    sellAnchorIdx = np.full(n, -1, dtype=int)
    lastBuyAnchor = -1
    lastSellAnchor = -1
    prevAllowBuy = False
    prevAllowSell = False
    for i in range(n):
        if allowBuy[i] and not prevAllowBuy:
            lastBuyAnchor = i
        if allowSell[i] and not prevAllowSell:
            lastSellAnchor = i
        buyAnchorIdx[i] = lastBuyAnchor
        sellAnchorIdx[i] = lastSellAnchor
        prevAllowBuy = bool(allowBuy[i])
        prevAllowSell = bool(allowSell[i])

    closesArr = np.asarray(ctx.get("closes"), dtype=float)

    def _dynFilter(
        indices: List[int],
        anchorIdx: np.ndarray,
        side: str,
    ) -> List[int]:
        kept: List[int] = []
        lastIdx = None
        lastPhase = None
        for idx in indices:
            i = int(idx)
            anchor = int(anchorIdx[i]) if 0 <= i < n else -1
            phaseId = anchor
            if phaseId != lastPhase:
                lastIdx = None
                lastPhase = phaseId
            if lastIdx is not None:
                refIdx = int(lastIdx)
            elif anchor >= 0:
                refIdx = anchor
            else:
                refIdx = -1
            if refIdx < 0:
                keep = True
            else:
                priceNow = float(closesArr[i])
                priceRef = float(closesArr[refIdx])
                if priceNow > 0.0 and priceRef > 0.0:
                    if side == "BUY":
                        deltaPct = ((priceRef / priceNow) - 1.0) * 100.0
                    else:
                        deltaPct = ((priceNow / priceRef) - 1.0) * 100.0
                else:
                    deltaPct = 0.0
                dynVal = float(dyn[i])
                req = 0.0 if not np.isfinite(dynVal) else max(0.0, abs(dynVal))
                keep = deltaPct >= req
            if keep:
                kept.append(i)
                lastIdx = i
        return kept

    cd = max(int(overrides["COOLDOWN"]), 0)

    def _filterIdx(idx: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if idx.size == 0:
            return idx
        return idx[mask[idx]]

    buyT = _filterIdx(idxBuy, allowBuy & validStart)
    buyG = _filterIdx(buyT, gradMaskBuy)
    buyGS = _filterIdx(buyG, spacingMaskBuy)
    buyGSE = _filterIdx(buyGS, energyMaskBuy)
    buyCd = np.asarray(_enforceCooldownIndices(buyGSE, cd), dtype=int)
    buyAll = np.asarray(
        _dynFilter(buyCd.tolist(), buyAnchorIdx, "BUY"),
        dtype=int,
    )

    sellT = _filterIdx(idxSell, allowSell & validStart)
    sellG = _filterIdx(sellT, gradMaskSell)
    sellGS = _filterIdx(sellG, spacingMaskSell)
    sellGSE = _filterIdx(sellGS, energyMaskSell)
    sellCd = np.asarray(_enforceCooldownIndices(sellGSE, cd), dtype=int)
    sellAll = np.asarray(
        _dynFilter(sellCd.tolist(), sellAnchorIdx, "SELL"),
        dtype=int,
    )

    buyRec = {
        "n": int(idxBuy.size),
        "t": int(buyT.size),
        "g": int(buyG.size),
        "gs": int(buyGS.size),
        "gse": int(buyGSE.size),
        "cd": int(buyCd.size),
        "dynamics": int(buyAll.size),
    }
    sellRec = {
        "n": int(idxSell.size),
        "t": int(sellT.size),
        "g": int(sellG.size),
        "gs": int(sellGS.size),
        "gse": int(sellGSE.size),
        "cd": int(sellCd.size),
        "dynamics": int(sellAll.size),
    }
    return buyRec, sellRec, buyAll.tolist(), sellAll.tolist()


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="oracle_driver",
        description="Build per-ticker configs using oracle flags.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to oracle-config.json (default inputs/profiles/oracle-config.json)",
    )
    parser.add_argument(
        "--template-profile",
        default=None,
        help="Optional path to base tuner profile (default template-config)",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Run label for outputs (under outputs/oracles/<label>/...)",
    )
    parser.add_argument(
        "--add",
        action="store_true",
        help="Add new tickers only; skip existing outputs",
    )
    return parser.parse_args()


def _ctxForTicker(
    ticker: str,
    interval: str,
    periods: List[int],
    totalDays: int,
    holdoutDays: int,
) -> Tuple[dict, list]:
    minCandles = (max(periods) * 2) + 1
    klines = getKlinesCached(
        ticker,
        interval,
        totalDays,
        minCandles,
        holdoutDays=holdoutDays,
    )
    ctx = cache.getContext(
        ticker,
        interval,
        totalDays,
        periods,
        klines,
        lambda: buildContext(klines, periods),
    )
    return ctx, klines


def _oracleWindows(spec: dict) -> Tuple[int, int, int, int]:
    return profile.windowParts(spec)


def _zForWindow(
    ctx: dict,
    series: np.ndarray,
    winDays: int,
    barsPerDayVal: float,
    key: str,
) -> np.ndarray:
    winBarsRaw = max(int(round(winDays * barsPerDayVal)), 1)
    winBars = min(winBarsRaw, series.size if series.size > 0 else winBarsRaw)
    if winBarsRaw != winBars:
        sig = (key, winBarsRaw, series.size)
        if sig not in _WIN_CAP_LOGGED:
            _WIN_CAP_LOGGED.add(sig)
            print(
                f"[oracles] window cap for {key}: "
                f"{winBarsRaw}→{winBars} bars "
                f"(series={series.size})"
            )
    mean, std = cache.getZStatsForSeries(
        ctx,
        key,
        winBars,
        lambda: rollingMeanAndStd(series, winBars),
    )
    valid = np.isfinite(mean) & np.isfinite(std) & (std > EPS)
    z = np.zeros_like(series, dtype=float)
    np.divide(series - mean, std, out=z, where=valid)
    return z


def _effectiveWindows(
    ctx: dict,
    series: np.ndarray,
    indices: Iterable[int],
    candidates: List[int],
    barsPerDayVal: float,
    zRef: float,
    cacheKey: str,
) -> List[int]:
    wins: List[int] = []
    candSorted = sorted(set(int(x) for x in candidates))
    zByWin: dict[int, np.ndarray] = {}
    for daysVal in candSorted:
        zByWin[int(daysVal)] = _zForWindow(
            ctx,
            series,
            int(daysVal),
            barsPerDayVal,
            cacheKey,
        )
    for idx in indices:
        if idx < 0:
            continue
        best = None
        bestDiff = float("inf")
        for daysVal, zArr in zByWin.items():
            if idx >= zArr.size:
                continue
            zVal = abs(float(zArr[int(idx)]))
            diff = abs(zVal - float(zRef))
            if diff < bestDiff:
                bestDiff = diff
                best = int(daysVal)
        if best is not None:
            wins.append(int(best))
    return wins


def _trimmedTriple(values: List[float]) -> List[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return []
    if arr.size == 1:
        val = float(arr[0])
        return [val, val, val]
    midVal = float(np.quantile(arr, 0.50))
    q75Val = float(np.quantile(arr, 0.75))
    q90Val = float(np.quantile(arr, 0.90))
    return [midVal, q75Val, q90Val]


def _summaryStats(values: List[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {"count": 0}
    q1 = float(np.quantile(arr, 0.25))
    mid = float(np.quantile(arr, 0.50))
    q3 = float(np.quantile(arr, 0.75))
    return {
        "count": n,
        "min": float(arr.min()),
        "q1": q1,
        "median": mid,
        "q3": q3,
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
    }


def _windowCandidates(totalDays: int) -> List[int]:
    maxDays = max(int(totalDays), 1)
    return list(range(1, maxDays + 1))


def _compressWindows(values: List[int]) -> List[int]:
    if not values:
        return []
    triple = _trimmedTriple([float(v) for v in values])
    if not triple:
        return []
    lo = max(1, int(round(triple[0])))
    mid = max(1, int(round(triple[1])))
    hi = max(1, int(round(triple[2])))
    if mid < lo:
        mid = lo
    if hi < mid:
        hi = mid
    outVals: List[int] = []
    for v in (lo, mid, hi):
        if not outVals or v != outVals[-1]:
            outVals.append(v)
    return outVals


def _windowList(val, default: List[int]) -> List[int]:
    if val is None:
        return list(default)
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    return [int(val)]


def _capWindows(
    windowDays: List[int],
    barsPerDayVal: float,
    nBars: int,
) -> List[int]:
    maxDays = max(int(round(nBars / max(barsPerDayVal, 1e-9))), 1)
    capped: List[int] = []
    for d in windowDays:
        val = min(max(d, 1), maxDays)
        if val not in capped:
            capped.append(val)
    return capped


def _zSamplesForIndices(
    ctx: dict,
    series: np.ndarray,
    indices: Iterable[int],
    candidates: List[int],
    barsPerDayVal: float,
    zRef: float,
    cacheKey: str,
) -> List[float]:
    wins = _effectiveWindows(
        ctx,
        series,
        indices,
        candidates,
        barsPerDayVal,
        zRef,
        cacheKey,
    )
    zVals: List[float] = []
    for win in wins:
        zArr = _zForWindow(
            ctx,
            series,
            win,
            barsPerDayVal,
            cacheKey,
        )
        for idx in indices:
            if 0 <= idx < zArr.size:
                zVals.append(float(zArr[int(idx)]))
    return zVals


def _gradSpacingEnergy(
    ctx: dict,
    signals: dict,
    barsPerDayVal: float,
    peakIndices: List[int],
    troughIndices: List[int],
    gradWindowDays: List[int],
    spacingWindowDays: List[int],
    energyWindowDays: List[int],
) -> dict:
    g1p3 = np.asarray(signals["g1P3"], dtype=float)
    s12 = np.asarray(signals["s12"], dtype=float)
    s23 = np.asarray(signals["s23"], dtype=float)
    trendCodesArr = np.asarray(signals["trendCode"], dtype=int)
    energy12 = energyCsum(ctx, trendCodesArr, '12')
    energy23 = energyCsum(ctx, trendCodesArr, '23')

    g1Peak = _zSamplesForIndices(
        ctx,
        g1p3,
        peakIndices,
        gradWindowDays,
        barsPerDayVal,
        -1.0,
        "g1p3",
    )
    g1Trough = _zSamplesForIndices(
        ctx,
        g1p3,
        troughIndices,
        gradWindowDays,
        barsPerDayVal,
        1.0,
        "g1p3",
    )
    spacing12Peak = _zSamplesForIndices(
        ctx,
        s12,
        peakIndices,
        spacingWindowDays,
        barsPerDayVal,
        1.0,
        "s12",
    )
    spacing12Trough = _zSamplesForIndices(
        ctx,
        s12,
        troughIndices,
        spacingWindowDays,
        barsPerDayVal,
        1.0,
        "s12",
    )
    spacing23Peak = _zSamplesForIndices(
        ctx,
        s23,
        peakIndices,
        spacingWindowDays,
        barsPerDayVal,
        1.0,
        "s23",
    )
    spacing23Trough = _zSamplesForIndices(
        ctx,
        s23,
        troughIndices,
        spacingWindowDays,
        barsPerDayVal,
        1.0,
        "s23",
    )
    energy12Peak = _zSamplesForIndices(
        ctx,
        energy12,
        peakIndices,
        energyWindowDays,
        barsPerDayVal,
        1.0,
        "e12",
    )
    energy12Trough = _zSamplesForIndices(
        ctx,
        energy12,
        troughIndices,
        energyWindowDays,
        barsPerDayVal,
        1.0,
        "e12",
    )
    energy23Peak = _zSamplesForIndices(
        ctx,
        energy23,
        peakIndices,
        energyWindowDays,
        barsPerDayVal,
        1.0,
        "e23",
    )
    energy23Trough = _zSamplesForIndices(
        ctx,
        energy23,
        troughIndices,
        energyWindowDays,
        barsPerDayVal,
        1.0,
        "e23",
    )

    return {
        "gradPeakZ": g1Peak,
        "gradTroughZ": g1Trough,
        "spacing12PeakZ": spacing12Peak,
        "spacing12TroughZ": spacing12Trough,
        "spacing23PeakZ": spacing23Peak,
        "spacing23TroughZ": spacing23Trough,
        "energy12PeakZ": energy12Peak,
        "energy12TroughZ": energy12Trough,
        "energy23PeakZ": energy23Peak,
        "energy23TroughZ": energy23Trough,
    }


def _signal_windows(
    ctx: dict,
    signals: dict,
    indices: Iterable[int],
    fixed: float,
    candidates: List[int],
    barsPerDayVal: float,
    cacheKey: str,
) -> List[int]:
    arr = np.asarray(signals[cacheKey], dtype=float)
    wins = _effectiveWindows(
        ctx,
        arr,
        indices,
        candidates,
        barsPerDayVal,
        fixed,
        cacheKey,
    )
    return _compressWindows(wins)


def _settingsFromWindows(
    ctx: dict,
    signals: dict,
    indices: List[int],
    barsPerDayVal: float,
    zRef: float,
    cacheKey: str,
    candidates: List[int],
) -> List[float]:
    arr = np.asarray(signals[cacheKey], dtype=float)
    zVals = _zSamplesForIndices(
        ctx,
        arr,
        indices,
        candidates,
        barsPerDayVal,
        zRef,
        cacheKey,
    )
    return _trimmedTriple(zVals)


def _energyStats(
    gradSpacingEnergy: dict,
) -> dict:
    return {
        "grad_peak_zscores": _summaryStats(gradSpacingEnergy["gradPeakZ"]),
        "grad_trough_zscores": _summaryStats(gradSpacingEnergy["gradTroughZ"]),
        "spacing_12_peak_zscores": _summaryStats(
            gradSpacingEnergy["spacing12PeakZ"]
        ),
        "spacing_12_trough_zscores": _summaryStats(
            gradSpacingEnergy["spacing12TroughZ"]
        ),
        "spacing_23_peak_zscores": _summaryStats(
            gradSpacingEnergy["spacing23PeakZ"]
        ),
        "spacing_23_trough_zscores": _summaryStats(
            gradSpacingEnergy["spacing23TroughZ"]
        ),
        "energy_12_peak_zscores": _summaryStats(
            gradSpacingEnergy["energy12PeakZ"]
        ),
        "energy_12_trough_zscores": _summaryStats(
            gradSpacingEnergy["energy12TroughZ"]
        ),
        "energy_23_peak_zscores": _summaryStats(
            gradSpacingEnergy["energy23PeakZ"]
        ),
        "energy_23_trough_zscores": _summaryStats(
            gradSpacingEnergy["energy23TroughZ"]
        ),
    }


def _zSamplesFixedWindow(
    ctx: dict,
    series: np.ndarray,
    indices: Iterable[int],
    winDays: int,
    barsPerDayVal: float,
    cacheKey: str,
    sign: float = 1.0,
) -> List[float]:
    idxArr = np.asarray([int(i) for i in indices], dtype=int)
    if idxArr.size == 0:
        return []
    n = int(series.size)
    validIdx = (idxArr >= 0) & (idxArr < n)
    if not np.any(validIdx):
        return []
    idxArr = idxArr[validIdx]
    winBars = max(int(round(float(winDays) * barsPerDayVal)), 1)
    mean, std = cache.getZStatsForSeries(
        ctx,
        str(cacheKey),
        winBars,
        lambda: rollingMeanAndStd(series, winBars),
    )
    valid = np.isfinite(mean) & np.isfinite(std) & (std > EPS)
    z = np.zeros_like(series, dtype=float)
    np.divide(series - mean, std, out=z, where=valid)
    z = np.clip(z, -10.0, 10.0)
    signed = z * float(sign)
    vals = np.asarray(signed[idxArr], dtype=float)
    vals = vals[valid[idxArr]]
    vals = vals[np.isfinite(vals)]
    return [float(v) for v in vals]


def _zTriplet(
    values: List[float],
    quantiles: Tuple[float, float, float] = (0.25, 0.50, 0.75),
) -> List[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return []
    if n == 1:
        val = max(0.0, float(arr[0]))
        val = round(val, 2)
        return [val, val, val]
    q1, q2, q3 = quantiles
    q1 = float(np.quantile(arr, float(q1)))
    q2 = float(np.quantile(arr, float(q2)))
    q3 = float(np.quantile(arr, float(q3)))
    lo = max(0.0, q1)
    mid = max(lo, q2)
    hi = max(mid, q3)
    return [round(lo, 2), round(mid, 2), round(hi, 2)]


def _macroDynPctTriples(
    ticker: str,
    macroInterval: str,
    totalDays: int,
    holdoutDays: int,
    macroWinDays: float,
    periods: List[int],
    template: dict,
    minPctConfig: float,
    pctMinQuantiles: Tuple[float, float] | None,
    pctMaxQuantiles: Tuple[float, float, float],
    klinesMicro: list,
    buys: List[int],
    sells: List[int],
    trendCode: np.ndarray,
) -> Tuple[
    List[float],
    List[float],
    dict,
    np.ndarray,
    np.ndarray,
]:
    if not (buys or sells):
        empty = np.zeros(0, dtype=float)
        return [], [], {"count": 0}, empty, empty

    closes = np.asarray(
        [float(row[4]) for row in klinesMicro],
        dtype=float,
    )
    nMicro = closes.size

    zAligned = np.zeros(nMicro, dtype=float)
    hasZ = False

    if macroInterval:
        periodsMacro = list(periods)
        macroP1Val = int(profile.scalarValue(template["MACRO_P1"], 0) or 0)
        macroP2Val = int(profile.scalarValue(template["MACRO_P2"], 0) or 0)
        macroP3Val = int(profile.scalarValue(template["MACRO_P3"], 0) or 0)
        if macroP1Val > 0 and len(periodsMacro) >= 1:
            periodsMacro[0] = macroP1Val
        if macroP2Val > 0 and len(periodsMacro) >= 2:
            periodsMacro[1] = macroP2Val
        if macroP3Val > 0:
            if len(periodsMacro) >= 3:
                periodsMacro[2] = macroP3Val
            else:
                periodsMacro.append(macroP3Val)

        if periodsMacro:
            minCandles = (max(periodsMacro) * 2) + 1
            klMacro = getKlinesCached(
                ticker,
                macroInterval,
                totalDays,
                minCandles,
                holdoutDays=holdoutDays,
            )
            ctxMacro = buildContext(klMacro, periodsMacro)
            gMacro = g1p3Series(ctxMacro)
            arrMacro = np.asarray(gMacro, dtype=float)
            arrMacro = arrMacro[np.isfinite(arrMacro)]
            if arrMacro.size < 2:
                empty = np.zeros(0, dtype=float)
                return [], [], {"count": 0}, empty, empty
            meanMacro = float(np.mean(arrMacro))
            stdMacro = float(np.std(arrMacro, ddof=0))
            if not np.isfinite(stdMacro) or abs(stdMacro) <= EPS:
                empty = np.zeros(0, dtype=float)
                return [], [], {"count": 0}, empty, empty
            zMacro = np.zeros_like(gMacro, dtype=float)
            np.divide(
                gMacro - meanMacro,
                stdMacro,
                out=zMacro,
                where=np.isfinite(gMacro),
            )
            tsMacro = np.array([k[0] for k in klMacro], dtype=float)
            tsMicro = np.array([k[0] for k in klinesMicro], dtype=float)
            zAligned = alignMacroDyn(tsMacro, zMacro, tsMicro)
            hasZ = True

    pctVals: List[float] = []
    zVals: List[float] = []

    sellsSorted = sorted(int(i) for i in sells)
    lastSellIdx = None
    lastSellReg = None
    for idx in sellsSorted:
        i = int(idx)
        if i < 0 or i >= nMicro:
            continue
        reg = int(np.sign(float(trendCode[i]))) if trendCode is not None else 0
        if lastSellIdx is None or reg != lastSellReg:
            lastSellIdx = i
            lastSellReg = reg
            continue
        if lastSellIdx is not None:
            ref = int(lastSellIdx)
            if 0 <= ref < nMicro:
                priceNow = float(closes[i])
                priceRef = float(closes[ref])
                if priceNow > 0.0 and priceRef > 0.0:
                    deltaPct = ((priceNow / priceRef) - 1.0) * 100.0
                else:
                    deltaPct = 0.0
                zVal = float(abs(zAligned[i])) if hasZ else 0.0
                if np.isfinite(deltaPct) and np.isfinite(zVal):
                    pctVals.append(deltaPct)
                    zVals.append(zVal)
        lastSellIdx = i

    buysSorted = sorted(int(i) for i in buys)
    lastBuyIdx = None
    lastBuyReg = None
    for idx in buysSorted:
        i = int(idx)
        if i < 0 or i >= nMicro:
            continue
        reg = int(np.sign(float(trendCode[i]))) if trendCode is not None else 0
        if lastBuyIdx is None or reg != lastBuyReg:
            lastBuyIdx = i
            lastBuyReg = reg
            continue
        if lastBuyIdx is not None:
            ref = int(lastBuyIdx)
            if 0 <= ref < nMicro:
                priceNow = float(closes[i])
                priceRef = float(closes[ref])
                if priceNow > 0.0 and priceRef > 0.0:
                    deltaPct = ((priceRef / priceNow) - 1.0) * 100.0
                else:
                    deltaPct = 0.0
                zVal = float(abs(zAligned[i])) if hasZ else 0.0
                if np.isfinite(deltaPct) and np.isfinite(zVal):
                    pctVals.append(deltaPct)
                    zVals.append(zVal)
        lastBuyIdx = i

    if not pctVals:
        empty = np.zeros(0, dtype=float)
        return [], [], {"count": 0}, empty, empty

    arrPct = np.abs(np.asarray(pctVals, dtype=float))
    nVals = int(arrPct.size)
    if nVals == 0 or len(zVals) != nVals:
        empty = np.zeros(0, dtype=float)
        return [], [], {"count": 0}, empty, empty

    order = np.argsort(arrPct)
    pctSorted = arrPct[order]

    vmin = float(pctSorted[0])
    vmed = float(pctSorted[nVals // 2])
    vmax = float(pctSorted[-1])

    q0 = float(np.quantile(arrPct, float(pctMaxQuantiles[0])))
    q1 = float(np.quantile(arrPct, float(pctMaxQuantiles[1])))
    q2 = float(np.quantile(arrPct, float(pctMaxQuantiles[2])))

    zAbs = [float(abs(z)) for z in zVals]
    zAbsSorted = sorted(zAbs)
    zmin = float(zAbsSorted[0])
    zmed = float(zAbsSorted[nVals // 2])
    zmax = float(zAbsSorted[-1])
    zMean = float(sum(zAbsSorted) / float(nVals))
    zVar = float(
        sum((v - zMean) ** 2 for v in zAbsSorted) / float(nVals)
    )
    zStd = zVar ** 0.5

    lo = max(0.0, q0)
    mid = max(lo, q1)
    hi = max(mid, q2)

    pctMaxTriple = [
        round(lo, 2),
        round(mid, 2),
        round(hi, 2),
    ]
    baseMin = float(minPctConfig)
    minLo = max(0.0, baseMin)
    if pctMinQuantiles is None:
        minMid = max(minLo, vmed)
        minHiRaw = (2.0 * minMid) - minLo
        minHi = max(minMid, minHiRaw)
    else:
        midQ = float(pctMinQuantiles[0])
        hiQ = float(pctMinQuantiles[1])
        qMid = float(np.quantile(arrPct, midQ))
        qHi = float(np.quantile(arrPct, hiQ))
        minMid = max(minLo, qMid)
        minHi = max(minMid, qHi)
    pctMinTriple = [
        round(minLo, 2),
        round(minMid, 2),
        round(minHi, 2),
    ]
    stats = {
        "count": nVals,
        "pct_min": vmin,
        "pct_median": vmed,
        "pct_max": vmax,
        "pct_mean": float(arrPct.mean()),
        "pct_std": float(arrPct.std(ddof=0)),
        "z_min": zmin,
        "z_median": zmed,
        "z_max": zmax,
        "z_mean": zMean,
        "z_std": zStd,
    }
    zArr = np.asarray(zAbsSorted, dtype=float)
    return pctMinTriple, pctMaxTriple, stats, arrPct, zArr


def _macroDyn(klines: list, interval: str, days: int) -> np.ndarray:
    periods = [1, 2, 3]
    ctx = buildContext(klines, periods)
    return g1p3Series(ctx)


def _alignedMacroDyn(
    baseTicker: str,
    microInterval: str,
    macroInterval: str,
    days: int,
    macroDays: int,
    macroDynWin: int,
    macroDynZMin: float,
    macroDynZMax: float,
    macroDynPctMax: float,
    macroDynPctMin: float,
    holdoutDays: int = 0,
) -> np.ndarray | None:
    minCandlesMicro = (3 * 2) + 1
    klMicro = getKlinesCached(
        baseTicker,
        microInterval,
        days + holdoutDays,
        minCandlesMicro,
        holdoutDays=holdoutDays,
    )
    minCandlesMacro = (3 * 2) + 1
    klMacro = getKlinesCached(
        baseTicker,
        macroInterval,
        macroDays + holdoutDays,
        minCandlesMacro,
        holdoutDays=holdoutDays,
    )
    if not klMacro or not klMicro:
        return None
    dynMacroRaw = _macroDyn(klMacro, macroInterval, macroDays)
    tsMacro = np.array([k[0] for k in klMacro], dtype=float)
    tsMicro = np.array([k[0] for k in klMicro], dtype=float)
    macroDynNorm = (
        (dynMacroRaw - np.median(dynMacroRaw))
        / np.maximum(np.subtract(*np.percentile(dynMacroRaw, [75, 25])), 1e-6)
    )
    dyn = alignMacroDyn(tsMacro, macroDynNorm, tsMicro)
    zmin = float(macroDynZMin)
    zmax = float(macroDynZMax)
    pctMax = float(macroDynPctMax)
    pctMin = float(macroDynPctMin)
    if not np.isfinite(zmin):
        zmin = 0.0
    if not np.isfinite(zmax):
        zmax = 0.0
    if not np.isfinite(pctMax):
        pctMax = 0.0
    if not np.isfinite(pctMin):
        pctMin = 0.0
    clipped = np.clip(dyn, -50.0, 50.0)
    scaled = np.where(
        clipped >= zmax,
        pctMax,
        np.where(clipped <= zmin, pctMin, clipped),
    )
    return scaled


def _oracleSignals(
    ctx: dict,
    barsPerDayVal: float,
    indices: List[int],
    windowDays: List[int],
    key: str,
) -> List[float]:
    arr = np.asarray(ctx[key], dtype=float)
    zVals = _zSamplesForIndices(
        ctx,
        arr,
        indices,
        windowDays,
        barsPerDayVal,
        1.0,
        key,
    )
    return _trimmedTriple(zVals)


def _oracleSeries(ctx: dict, klines: list, barsPerDayVal: float) -> dict:
    closes = np.asarray(ctx["closes"], dtype=float)
    rollMax = pd.Series(closes).rolling(
        window=3, center=True, min_periods=1
    ).max()
    rollMin = pd.Series(closes).rolling(
        window=3, center=True, min_periods=1
    ).min()
    peaks = rollMax.index[closes == rollMax.to_numpy()].tolist()
    troughs = rollMin.index[closes == rollMin.to_numpy()].tolist()
    return {
        "closes": closes,
        "peaks": peaks,
        "troughs": troughs,
        "barsPerDay": barsPerDayVal,
    }


def _buildProfileForTicker(
    ticker: str,
    baseCfg: dict,
    template: dict,
    dataDays: int,
    holdoutDays: int,
    outDir: str,
) -> dict | None:
    microInterval = str(baseCfg["interval"])
    p1 = profile.scalarValue(template["p1"])
    p2 = profile.scalarValue(template["p2"])
    p3 = profile.scalarValue(template["p3"])
    periods = [int(p1), int(p2), int(p3)]
    macroInterval = str(baseCfg["macro_interval"])
    macroDays = int(baseCfg["macro_window_days"])
    primerDays, tunerDays, holdoutLocal, totalDaysLocal = _oracleWindows(
        baseCfg
    )

    templateLocal = dict(template)
    macroP1Override = baseCfg.get("macro_p1")
    if macroP1Override is not None:
        macroP1Val = int(profile.scalarValue(macroP1Override))
        templateLocal["MACRO_P1"] = [macroP1Val]
    macroP2Override = baseCfg.get("macro_p2")
    if macroP2Override is not None:
        macroP2Val = int(profile.scalarValue(macroP2Override))
        templateLocal["MACRO_P2"] = [macroP2Val]
    macroP3Override = baseCfg.get("macro_p3")
    if macroP3Override is not None:
        macroP3Val = int(profile.scalarValue(macroP3Override))
        templateLocal["MACRO_P3"] = [macroP3Val]
    gradWindowDays = _windowList(
        baseCfg["grad_buy_window_days"],
        GRAD_WINDOW_DAYS,
    )
    spacingWindowDays12 = _windowList(
        baseCfg["spacing_window_days_12"],
        SPACING_WINDOW_DAYS,
    )
    spacingWindowDays23 = _windowList(
        baseCfg["spacing_window_days_23"],
        SPACING_WINDOW_DAYS,
    )
    spacingWindowDaysAll = sorted(
        set(spacingWindowDays12 + spacingWindowDays23)
    )
    energyWindowDays = _windowList(
        baseCfg["spacing_energy_window_days"],
        ENERGY_WINDOW_DAYS,
    )

    ctxMicro, klMicro = _ctxForTicker(
        ticker,
        microInterval,
        periods,
        totalDaysLocal,
        holdoutLocal,
    )
    oracleSettings = {
        "peak_window": int(baseCfg["peak_window"]),
        "min_pct_since_oracle": float(baseCfg["pct_increase"]),
    }
    paramsNs = types.SimpleNamespace(
        oracle_settings=oracleSettings,
    )
    engine = OracleEngine(ctxMicro, paramsNs)
    flags = engine.generate()
    barsPerDayVal = bars_per_day(ctxMicro)
    warmupBars = int(round(float(primerDays) * max(barsPerDayVal, 1.0)))
    if warmupBars > 0:
        flags = [(i, lab) for i, lab in flags if i >= warmupBars]
    if not flags:
        print(
            f"[oracles] no oracle flags for {ticker}; "
            "adjust oracle settings and rerun"
        )
        return None
    oracleBuys, oracleSells = _flagsBySide(flags)
    snapBuys = [i - 1 for i in oracleBuys if i > warmupBars]
    snapSells = [i - 1 for i in oracleSells if i > warmupBars]
    nBars = len(ctxMicro.get("closes", []))
    gradWindowDays = _capWindows(gradWindowDays, barsPerDayVal, nBars)
    spacingWindowDays12 = _capWindows(
        spacingWindowDays12, barsPerDayVal, nBars
    )
    spacingWindowDays23 = _capWindows(
        spacingWindowDays23, barsPerDayVal, nBars
    )
    spacingWindowDaysAll = sorted(
        set(spacingWindowDays12 + spacingWindowDays23)
    )
    energyWindowDays = _capWindows(
        energyWindowDays, barsPerDayVal, nBars
    )
    print(
        f"[oracles] {ticker} {microInterval} bars={nBars} "
        f"grad_win={gradWindowDays} "
        f"spacing12={spacingWindowDays12} "
        f"spacing23={spacingWindowDays23} "
        f"energy={energyWindowDays}"
    )
    signals = buildSignals(ctxMicro, [])
    oracleSeries = _oracleSeries(ctxMicro, klMicro, barsPerDayVal)

    peakIndices = [int(i) for i in snapSells]
    troughIndices = [int(i) for i in snapBuys]
    gradSpacing = _gradSpacingEnergy(
        ctxMicro,
        signals,
        barsPerDayVal,
        peakIndices,
        troughIndices,
        gradWindowDays,
        spacingWindowDaysAll,
        energyWindowDays,
    )

    energyStats = _energyStats(gradSpacing)

    g1p1 = np.asarray(signals["g1P1"], dtype=float)
    s12 = np.asarray(signals["s12"], dtype=float)
    s23 = np.asarray(signals["s23"], dtype=float)
    trendArr = np.asarray(signals["trendCode"], dtype=int)
    energy12 = energyCsum(ctxMicro, trendArr, "12")
    energy23 = energyCsum(ctxMicro, trendArr, "23")

    gradBuyWinDays = int(baseCfg["grad_buy_window_days"])
    gradSellWinDays = int(baseCfg["grad_sell_window_days"])
    spacingWinDays12 = int(baseCfg["spacing_window_days_12"])
    spacingWinDays23 = int(baseCfg["spacing_window_days_23"])
    spacingEnergyWinDays = int(baseCfg["spacing_energy_window_days"])

    spacingIdx = snapBuys + snapSells

    gradBuySamples = _zSamplesFixedWindow(
        ctxMicro,
        g1p1,
        snapBuys,
        gradBuyWinDays,
        barsPerDayVal,
        "g1p1",
        sign=-1.0,
    )
    gradSellSamples = _zSamplesFixedWindow(
        ctxMicro,
        g1p1,
        snapSells,
        gradSellWinDays,
        barsPerDayVal,
        "g1p1",
        sign=1.0,
    )
    spacing12Samples = _zSamplesFixedWindow(
        ctxMicro,
        s12,
        spacingIdx,
        spacingWinDays12,
        barsPerDayVal,
        "s12",
    )
    spacing23Samples = _zSamplesFixedWindow(
        ctxMicro,
        s23,
        spacingIdx,
        spacingWinDays23,
        barsPerDayVal,
        "s23",
    )
    energy12Samples = _zSamplesFixedWindow(
        ctxMicro,
        energy12,
        spacingIdx,
        spacingEnergyWinDays,
        barsPerDayVal,
        "e12",
    )
    energy23Samples = _zSamplesFixedWindow(
        ctxMicro,
        energy23,
        spacingIdx,
        spacingEnergyWinDays,
        barsPerDayVal,
        "e23",
    )

    def _qTriple(key: str) -> Tuple[float, float, float]:
        vals = baseCfg[key]
        return (float(vals[0]), float(vals[1]), float(vals[2]))

    gradQs = _qTriple("grad_zscore_quantiles")
    spacingQs = _qTriple("spacing_zscore_quantiles")
    energyQs = _qTriple("energy_zscore_quantiles")

    gradBuyTrip = _zTriplet(
        gradBuySamples,
        quantiles=gradQs,
    )
    gradSellTrip = _zTriplet(
        gradSellSamples,
        quantiles=gradQs,
    )
    spacing12Trip = _zTriplet(
        spacing12Samples,
        quantiles=spacingQs,
    )
    spacing23Trip = _zTriplet(
        spacing23Samples,
        quantiles=spacingQs,
    )
    energy12Trip = _zTriplet(
        energy12Samples,
        quantiles=energyQs,
    )
    energy23Trip = _zTriplet(
        energy23Samples,
        quantiles=energyQs,
    )

    minPctCfg = float(baseCfg["pct_increase"])
    pctMinQsRaw = baseCfg["macro_pctmin_quantiles"]
    macroPctMinQs = (float(pctMinQsRaw[0]), float(pctMinQsRaw[1]))
    (
        macroPctMin,
        macroPctMax,
        macroPctStats,
        macroPctArr,
        macroZArr,
    ) = _macroDynPctTriples(
        ticker,
        macroInterval,
        totalDaysLocal,
        holdoutLocal,
        float(macroDays),
        periods,
        templateLocal,
        minPctCfg,
        macroPctMinQs,
        _qTriple("macro_pctmax_quantiles"),
        klMicro,
        snapBuys,
        snapSells,
        trendArr,
    )
    macroDyn = None

    tpl = dict(templateLocal)
    tpl["_comment"] = f"{ticker}-config"
    tpl.update({
        "tickers": [ticker],
        "intervals": [microInterval],
        "primer_days": primerDays,
        "tuner_days": tunerDays,
        "holdout_days": holdoutLocal,
        "p1": periods[0],
        "p2": periods[1],
        "p3": periods[2],
        "GRAD1_BUY_Z_MIN": gradBuyTrip,
        "GRAD1_SELL_Z_MIN": gradSellTrip,
        "GRAD1_BUY_WIN_DAYS": gradBuyWinDays,
        "GRAD1_SELL_WIN_DAYS": gradSellWinDays,
        "SPACING_Z_MIN_12": spacing12Trip,
        "SPACING_Z_MIN_23": spacing23Trip,
        "SPACING_WIN_DAYS_12": spacingWindowDays12,
        "SPACING_WIN_DAYS_23": spacingWindowDays23,
        "MICRO_NRG_MODEL": templateLocal["MICRO_NRG_MODEL"],
        "MICRO_NRG_WIN_DAYS": energyWindowDays,
        "MICRO_NRG_MIN_12": energy12Trip,
        "MICRO_NRG_MIN_23": energy23Trip,
        "MACRO_INTERVAL": macroInterval,
        "MACRO_P1": templateLocal["MACRO_P1"],
        "MACRO_P2": templateLocal["MACRO_P2"],
        "MACRO_P3": templateLocal["MACRO_P3"],
        "MACRO_GRAD_WIN_DAYS": templateLocal["MACRO_GRAD_WIN_DAYS"],
        "MACRO_GRAD_Z_MIN": templateLocal["MACRO_GRAD_Z_MIN"],
        "MACRO_GRAD_Z_MAX": templateLocal["MACRO_GRAD_Z_MAX"],
        "MACRO_MULT_GRAD_MIN": templateLocal["MACRO_MULT_GRAD_MIN"],
        "MACRO_MULT_GRAD_MAX": templateLocal["MACRO_MULT_GRAD_MAX"],
    })

    if macroDays > 0:
        tpl["MACRO_NRG_WIN_DAYS"] = [float(macroDays)]
    if macroPctMin:
        tpl["MACRO_DYN_PCT_MIN"] = macroPctMin
    if macroPctMax:
        tpl["MACRO_DYN_PCT_MAX"] = macroPctMax
    if macroPctStats and macroPctStats.get("count", 0) > 0:
        if isinstance(macroZArr, np.ndarray) and macroZArr.size > 0:
            macroZMinQs = _qTriple("macro_zmin_quantiles")
            macroZMaxQs = _qTriple("macro_zmax_quantiles")
            macroZTrip = _zTriplet(
                [min(float(abs(v)), 10.0) for v in macroZArr],
                quantiles=macroZMinQs,
            )
            if macroZTrip:
                arrAbs = np.asarray(
                    [
                        min(abs(float(v)), 10.0)
                        for v in macroZArr
                    ],
                    dtype=float,
                )
                arrAbs = arrAbs[np.isfinite(arrAbs)]
                maxTrip: List[float] = []
                if arrAbs.size >= 2:
                    q0 = float(np.quantile(arrAbs, macroZMaxQs[0]))
                    q1 = float(np.quantile(arrAbs, macroZMaxQs[1]))
                    q2 = float(np.quantile(arrAbs, macroZMaxQs[2]))
                    loMax = max(macroZTrip[0] + 0.1, q0)
                    midMax = max(loMax, macroZTrip[1] + 0.1, q1)
                    hiMax = max(midMax, macroZTrip[2] + 0.1, q2)
                    maxTrip = [
                        round(loMax, 1),
                        round(midMax, 1),
                        round(hiMax, 1),
                    ]
                if not maxTrip:
                    zMaxStat = float(
                        macroPctStats.get("z_max", 0.0)
                    )
                    if (
                        not np.isfinite(zMaxStat)
                        or zMaxStat <= macroZTrip[-1]
                    ):
                        zMaxStat = macroZTrip[-1] + 0.1
                    zMaxStat = min(zMaxStat, 10.0)
                    zMaxStat = round(zMaxStat, 1)
                    maxTrip = [
                        zMaxStat for _ in macroZTrip
                    ]
                tpl["MACRO_NRG_Z_MIN"] = macroZTrip
                tpl["MACRO_NRG_Z_MAX"] = maxTrip

    statsSummary = {
        "GRAD1_BUY_ZSCORE": _summaryStats(gradBuySamples),
        "GRAD1_SELL_ZSCORE": _summaryStats(gradSellSamples),
        "SPACING_ZSCORE_12": _summaryStats(spacing12Samples),
        "SPACING_ZSCORE_23": _summaryStats(spacing23Samples),
        "SPACING_ENERGY_ZSCORE_12": _summaryStats(energy12Samples),
        "SPACING_ENERGY_ZSCORE_23": _summaryStats(energy23Samples),
        "MACRO_DYN_PCT_MAX": macroPctStats,
    }
    statsDir = os.path.join(outDir, ticker, "stats")
    os.makedirs(statsDir, exist_ok=True)
    statsPath = os.path.join(statsDir, "oracle_stats.json")
    _writeProfile(statsPath, statsSummary)

    bellSpecs = [
        ("GRAD1_BUY_ZSCORE", gradBuySamples),
        ("GRAD1_SELL_ZSCORE", gradSellSamples),
        ("SPACING_ZSCORE_12", spacing12Samples),
        ("SPACING_ZSCORE_23", spacing23Samples),
        ("SPACING_ENERGY_ZSCORE_12", energy12Samples),
        ("SPACING_ENERGY_ZSCORE_23", energy23Samples),
    ]
    for key, vals in bellSpecs:
        if not vals:
            continue
        arrVals = np.asarray(vals, dtype=float)
        arrVals = arrVals[np.isfinite(arrVals)]
        if arrVals.size < 2:
            continue
        bellName = key.lower()
        bellPath = os.path.join(
            statsDir,
            f"{bellName}_bell.png",
        )
        bellTitle = f"{ticker} {key}"
        plotBellCurve(arrVals, bellTitle, bellPath)

    if isinstance(macroPctArr, np.ndarray) and macroPctArr.size >= 2:
        macroPctPath = os.path.join(
            statsDir,
            "macro_dyn_pct_bell.png",
        )
        macroPctTitle = f"{ticker} MACRO_DYN_PCT"
        plotBellCurve(macroPctArr, macroPctTitle, macroPctPath)

    if isinstance(macroZArr, np.ndarray) and macroZArr.size >= 2:
        macroZPath = os.path.join(
            statsDir,
            "macro_dyn_z_bell.png",
        )
        macroZTitle = f"{ticker} MACRO_DYN_Z"
        plotBellCurve(macroZArr, macroZTitle, macroZPath)

    startIdx = (max(periods) * 2) + warmupBars
    scalarOverrides = overridesFromDict(tpl)
    for key, raw in tpl.items():
        if (
            isinstance(raw, list)
            and len(raw) == 3
            and all(isinstance(v, (int, float)) for v in raw)
        ):
            scalarOverrides[key] = float(raw[1])
    buyRec, sellRec, keptBuys, keptSells = _oracleGateRecall(
        ctxMicro,
        signals,
        periods,
        startIdx,
        snapBuys,
        snapSells,
        scalarOverrides,
        holdoutLocal,
    )

    chunkDays = float(baseCfg["chart_days"])
    if chunkDays > 0.0:
        barsVal = max(barsPerDayVal, 1.0)
        chunkBars = int(round(chunkDays * barsVal))
        if chunkBars <= 0:
            chunkBars = int(max(barsVal, 1.0))
        chartsDir = os.path.join(outDir, ticker, "charts")
        os.makedirs(chartsDir, exist_ok=True)
        nBars = len(klMicro)
        tsList = pd.to_datetime(
            [k[0] for k in klMicro],
            unit="ms",
            utc=True,
        )
        tsList = tsList.tz_convert(None).to_pydatetime().tolist()
        mas = ctxMicro["mas"]
        g1p3 = np.asarray(signals["g1P3"], dtype=float)
        trend = np.asarray(signals["trendCode"], dtype=int)
        allowBuy = (trend == -1)
        allowSell = (trend == 1)
        buyEsp = spacingState(
            ctxMicro,
            trend,
            allowBuy,
            np.asarray(signals["s12"], dtype=float),
            np.asarray(signals["s23"], dtype=float),
            scalarOverrides,
        )
        sellEsp = spacingState(
            ctxMicro,
            trend,
            allowSell,
            np.asarray(signals["s12"], dtype=float),
            np.asarray(signals["s23"], dtype=float),
            scalarOverrides,
        )
        energyActive = (allowBuy & ~buyEsp.energyMask) | (
            allowSell & ~sellEsp.energyMask
        )

        macroInterval = str(scalarOverrides["MACRO_INTERVAL"]).strip()
        macroDynFull: np.ndarray | None = None
        macroCloseFull: np.ndarray | None = None
        macroMasFull: list[np.ndarray] | None = None
        macroPeriodsUsed: list[int] | None = None
        tsMicro = np.array([k[0] for k in klMicro], dtype=float)
        if macroInterval:
            meta = (
                ctxMicro.get("_cache") if isinstance(ctxMicro, dict) else None
            )
            baseDays = (
                meta.get("days") if isinstance(meta, dict) else 0
            )
            baseTicker = (
                meta.get("ticker") if isinstance(meta, dict) else ticker
            )
            periodsMacro = list(ctxMicro.get("periods", []))
            macroP1 = int(scalarOverrides["MACRO_P1"])
            macroP2 = int(scalarOverrides["MACRO_P2"])
            macroP3 = int(scalarOverrides["MACRO_P3"])
            if macroP1 > 0 and len(periodsMacro) >= 1:
                periodsMacro[0] = macroP1
            if macroP2 > 0 and len(periodsMacro) >= 2:
                periodsMacro[1] = macroP2
            if macroP3 > 0:
                if len(periodsMacro) >= 3:
                    periodsMacro[2] = macroP3
                else:
                    periodsMacro.append(macroP3)
            if not periodsMacro:
                periodsMacro = [1]
            minCandles = max(int(max(periodsMacro) * 2 + 1), 1)
            macroPeriodsUsed = list(periodsMacro)
            klMacro = getKlinesCached(
                str(baseTicker),
                str(macroInterval),
                int(baseDays),
                minCandles,
                holdoutDays=0,
            )
            ctxMacroFull = buildContext(klMacro, periodsMacro)
            ctxMacroFull["intervalStr"] = str(macroInterval)
            tsMacro = np.array([k[0] for k in klMacro], dtype=float)
            macroCloseFull = alignMacroDyn(
                tsMacro, np.asarray(ctxMacroFull["closes"], dtype=float), tsMicro
            )
            macroMasFull = [
                alignMacroDyn(tsMacro, np.asarray(ma, dtype=float), tsMicro)
                for ma in ctxMacroFull["mas"][:3]
            ]
            winDays = scalarOverrides["MACRO_NRG_WIN_DAYS"]
            zmin = scalarOverrides["MACRO_NRG_Z_MIN"]
            zmax = scalarOverrides["MACRO_NRG_Z_MAX"]
            pctMin = scalarOverrides["MACRO_DYN_PCT_MIN"]
            pctMax = scalarOverrides["MACRO_DYN_PCT_MAX"]
            gradWinDays = float(scalarOverrides["MACRO_GRAD_WIN_DAYS"])
            gradZMin = float(scalarOverrides["MACRO_GRAD_Z_MIN"])
            gradZMax = float(scalarOverrides["MACRO_GRAD_Z_MAX"])
            gradMultMin = float(scalarOverrides["MACRO_MULT_GRAD_MIN"])
            gradMultMax = float(scalarOverrides["MACRO_MULT_GRAD_MAX"])
            dynMacro = macroDynFromContext(
                ctxMacroFull,
                float(winDays),
                float(zmin),
                float(zmax),
                float(pctMax),
                float(pctMin),
                gradWinDays=gradWinDays,
                gradZMin=gradZMin,
                gradZMax=gradZMax,
                gradMultMin=gradMultMin,
                gradMultMax=gradMultMax,
            )
            macroDynFull = alignMacroDyn(tsMacro, dynMacro, tsMicro)

        blockedGradMask = np.zeros(nBars, dtype=bool)
        eps = 1e-6
        winBuyDays = float(scalarOverrides["GRAD1_BUY_WIN_DAYS"])
        threshBuy = float(scalarOverrides["GRAD1_BUY_Z_MIN"])
        winBuyBars = max(int(round(winBuyDays * barsVal)), 1)
        meanBuy, stdBuy = cache.getZStatsForSeries(
            ctxMicro,
            "g1p1",
            winBuyBars,
            lambda: rollingMeanAndStd(g1p1, winBuyBars),
        )
        validBuy = (
            np.isfinite(meanBuy)
            & np.isfinite(stdBuy)
            & (stdBuy > eps)
        )
        zBuy = np.zeros_like(g1p1, dtype=float)
        np.divide(g1p1 - meanBuy, stdBuy, out=zBuy, where=validBuy)
        zBuy = np.clip(zBuy, -10.0, 10.0)
        signedBuy = -zBuy
        readyBuy = allowBuy & validBuy
        gateBuyBlocked = readyBuy & (signedBuy < threshBuy)

        winSellDays = float(scalarOverrides["GRAD1_SELL_WIN_DAYS"])
        threshSell = float(scalarOverrides["GRAD1_SELL_Z_MIN"])
        winSellBars = max(int(round(winSellDays * barsVal)), 1)
        meanSell, stdSell = cache.getZStatsForSeries(
            ctxMicro,
            "g1p1",
            winSellBars,
            lambda: rollingMeanAndStd(g1p1, winSellBars),
        )
        validSell = (
            np.isfinite(meanSell)
            & np.isfinite(stdSell)
            & (stdSell > eps)
        )
        zSell = np.zeros_like(g1p1, dtype=float)
        np.divide(g1p1 - meanSell, stdSell, out=zSell, where=validSell)
        zSell = np.clip(zSell, -10.0, 10.0)
        signedSell = zSell
        readySell = allowSell & validSell
        gateSellBlocked = readySell & (signedSell < threshSell)

        blockedGradMask = gateBuyBlocked | gateSellBlocked
        chartSwitch = dict(scalarOverrides)
        if macroDynFull is not None:
            p1 = periods[0]
            chartSwitch["GRAD_FAST_LABEL"] = f"g1(p{p1}) micro"
            chartSwitch["GRAD_SLOW_LABEL"] = "macro dyn%"

        seq = 1
        startBar = max(0, warmupBars)
        for start in range(startBar, nBars, chunkBars):
            end = min(nBars, start + chunkBars)
            segFlags = [
                (idx, lab)
                for idx, lab in flags
                if start <= idx < end
            ]
            if not segFlags:
                continue
            segment = slice(start, end)
            title = (
                f"{ticker} {microInterval} – "
                f"{tsList[start].date()} → "
                f"{tsList[end - 1].date()} (UTC)"
            )
            markers: List[Tuple[Any, str]] = [
                (tsList[idx], lab) for idx, lab in segFlags
            ]
            for idxVal in keptBuys:
                idx = int(idxVal)
                if start <= idx < end:
                    markers.append((tsList[idx], "M_BUY"))
            for idxVal in keptSells:
                idx = int(idxVal)
                if start <= idx < end:
                    markers.append((tsList[idx], "M_SELL"))

            segMas = [ma[segment] for ma in mas]
            gradsSeg = {
                periods[0]: {"grad1": g1p1[segment]},
            }
            if macroDynFull is not None:
                gradsSeg[periods[-1]] = {"grad1": macroDynFull[segment]}
            else:
                gradsSeg[periods[-1]] = {"grad1": g1p3[segment]}

            spans = []
            blk = energyActive[segment].astype(int)
            if blk.size > 0:
                pad = np.pad(blk, (1, 1))
                diff = np.diff(pad)
                starts = np.flatnonzero(diff == 1)
                ends = np.flatnonzero(diff == -1)
                for i0, i1 in zip(starts, ends):
                    a = start + max(0, int(i0))
                    b = start + max(0, int(i1) - 1)
                    if (
                        a < 0
                        or b < 0
                        or a >= len(tsList)
                        or b >= len(tsList)
                    ):
                        continue
                    if a > b:
                        continue
                    idxEnd = b + 1 if (b + 1) < len(tsList) else b
                    spans.append((tsList[a], tsList[idxEnd]))

            gradFastSpans: list[tuple[Any, Any]] = []
            blkSegment = blockedGradMask[segment].astype(int)
            if blkSegment.size > 0:
                padM = np.pad(blkSegment, (1, 1))
                diffM = np.diff(padM)
                startsM = np.flatnonzero(diffM == 1)
                endsM = np.flatnonzero(diffM == -1)
                for i0, i1 in zip(startsM, endsM):
                    a = start + max(0, int(i0))
                    b = start + max(0, int(i1) - 1)
                    if (
                        a < 0
                        or b < 0
                        or a >= len(tsList)
                        or b >= len(tsList)
                    ):
                        continue
                    if a > b:
                        continue
                    idxEndM = b + 1 if (b + 1) < len(tsList) else b
                    gradFastSpans.append((tsList[a], tsList[idxEndM]))

            gradSlowHeat: list[
                tuple[Any, Any, tuple[float, float, float]]
            ] = []
            if macroDynFull is not None:
                pctMax = float(chartSwitch["MACRO_DYN_PCT_MAX"])
                if pctMax > 0.0:
                    dynSeg = macroDynFull[segment]
                    mag = np.abs(dynSeg) / pctMax
                    mag = np.clip(mag, 0.0, 1.0)
                    for iLocal, mval in enumerate(mag):
                        iAbs = start + iLocal
                        t0 = tsList[iAbs]
                        t1 = (
                            tsList[iAbs + 1]
                            if (iAbs + 1) < len(tsList)
                            else tsList[iAbs]
                        )
                        if mval <= 0.5:
                            t = mval / 0.5 if 0.5 > 0 else 0.0
                            r = 0.0 + t * (1.0 - 0.0)
                            g = 1.0 + t * (0.65 - 1.0)
                            b = 0.0
                        else:
                            t = (mval - 0.5) / 0.5 if 0.5 > 0 else 0.0
                            r = 1.0
                            g = 0.65 + t * (0.0 - 0.65)
                            b = 0.0
                        gradSlowHeat.append((t0, t1, (r, g, b)))

            macroCloseSeg = None
            macroMasSeg = None
            if macroCloseFull is not None and macroMasFull is not None:
                macroCloseSeg = macroCloseFull[segment]
                macroMasSeg = [m[segment] for m in macroMasFull]
            chart = Chart(
                klines=klMicro[segment],
                ticker=ticker,
                markers=markers,
                mas=segMas,
                grads=gradsSeg,
                switchInfo=chartSwitch,
                energySpans=spans,
                gradFastSpans=gradFastSpans,
                gradSlowHeat=gradSlowHeat,
                macroClose=macroCloseSeg,
                macroMas=macroMasSeg,
                macroPeriods=macroPeriodsUsed,
                macroInterval=macroInterval,
            )
            outPath = os.path.join(
                chartsDir,
                f"{ticker}_{microInterval}_oracles_{seq}.png",
            )
            chart.plot(title=title, savePath=outPath)
            seq += 1

    stageKeys = ["t", "g", "gs", "gse", "cd", "dynamics"]

    def _stageParts(rec: dict) -> List[str]:
        parts: List[str] = []
        prevKey = stageKeys[0]
        prevVal = int(rec.get(prevKey, 0))
        parts.append(f"{prevKey}={prevVal}")
        for key in stageKeys[1:]:
            val = int(rec.get(key, 0))
            if prevVal > 0:
                dropPct = ((prevVal - val) / float(prevVal)) * 100.0
            else:
                dropPct = 0.0
            if dropPct < 0.0:
                dropPct = 0.0
            parts.append(f"{key}={val}(-{dropPct:.0f}%)")
            prevVal = val
        return parts

    buyParts = _stageParts(buyRec)
    sellParts = _stageParts(sellRec)

    print(" " * 2 + "=" * 54)
    print(f"[oracle] {ticker}")
    print(
        f"[oracle] - BUYS: {buyRec['n']}, SELLS: {sellRec['n']}"
    )
    print(
        "[oracle] - STAGES BUY "
        f"(n={buyRec['n']}): "
        f"{' '.join(buyParts[:4])}"
    )
    print(
        f"[oracle]   {' '.join(buyParts[4:])}"
    )
    print(
        "[oracle] - STAGES SELL "
        f"(n={sellRec['n']}): "
        f"{' '.join(sellParts[:4])}"
    )
    print(
        f"[oracle]   {' '.join(sellParts[4:])}"
    )
    print(" " * 2 + "=" * 54)
    print("[oracle] finished.")

    return {
        "profile": tpl,
        "series": oracleSeries,
        "energy_stats": energyStats,
        "macro_dyn": macroDyn.tolist() if macroDyn is not None else [],
        "gate_stages": {"buy": buyRec, "sell": sellRec},
    }


def _writeProfile(path: str, data: dict) -> None:
    dirName = os.path.dirname(path)
    if dirName:
        os.makedirs(dirName, exist_ok=True)

    def _fmt(obj: Any, level: int = 0) -> str:
        indent = "  " * level
        if isinstance(obj, dict):
            parts: List[str] = []
            for key, val in obj.items():
                valStr = _fmt(val, level + 1)
                parts.append(f'{indent}  "{key}": {valStr}')
            inner = ",\n".join(parts)
            return "{\n" + inner + "\n" + indent + "}"
        if isinstance(obj, list):
            return json.dumps(obj, separators=(", ", ": "))
        return json.dumps(obj)

    payload = _fmt(data, 0)

    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=dirName if dirName else None,
    ) as tmp:
        tmp.write(payload)
        tmpPath = tmp.name
    os.replace(tmpPath, path)


def _midConfig(cfg: dict) -> dict:
    def _convert(val: Any) -> Any:
        if isinstance(val, dict):
            return {k: _convert(v) for k, v in val.items()}
        if isinstance(val, list):
            if (
                len(val) == 3
                and all(isinstance(x, (int, float)) for x in val)
            ):
                return _convert(val[1])
            return [_convert(v) for v in val]
        return val

    return _convert(cfg)


def _plotOracleGateStages(
    stagesByTicker: dict,
    outDir: str,
    label: str,
) -> None:
    if not stagesByTicker:
        return

    import matplotlib.pyplot as plt

    stageKeys = ["t", "g", "gs", "gse", "cd", "dynamics"]
    x = np.arange(len(stageKeys), dtype=float)

    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(12, 6.5),
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )
    axBuy, axSell = axes
    fig.patch.set_facecolor(BG_COLOR)
    for ax in axes:
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(
            color=GRID_COLOR,
            linestyle=":",
            linewidth=0.6,
            alpha=0.7,
        )
        ax.set_ylim(0.0, 120.0)

    axBuy.set_title(
        f"{label} oracle gate stages (percent kept)",
        color=TEXT_COLOR,
        pad=10,
    )
    axBuy.set_ylabel("BUY %", color=TEXT_COLOR)
    axSell.set_ylabel("SELL %", color=TEXT_COLOR)
    axSell.set_xlabel("Stage", color=TEXT_COLOR)

    tickers = sorted(stagesByTicker.keys())
    for ticker in tickers:
        rec = stagesByTicker.get(ticker) or {}
        buyRec = rec.get("buy") or {}
        sellRec = rec.get("sell") or {}

        nBuy = int(buyRec.get("n", 0))
        nSell = int(sellRec.get("n", 0))
        buyPct = [
            (float(buyRec.get(k, 0)) / float(nBuy)) * 100.0
            if nBuy > 0
            else 0.0
            for k in stageKeys
        ]
        sellPct = [
            (float(sellRec.get(k, 0)) / float(nSell)) * 100.0
            if nSell > 0
            else 0.0
            for k in stageKeys
        ]

        axBuy.plot(
            x,
            buyPct,
            marker="o",
            linewidth=1.2,
            label=ticker,
        )
        axSell.plot(
            x,
            sellPct,
            marker="o",
            linewidth=1.2,
            label=ticker,
        )

    axSell.set_xticks(x)
    axSell.set_xticklabels(stageKeys)

    leg = axBuy.legend(
        loc="upper right",
        fontsize=8,
        frameon=True,
        ncol=1,
    )
    if leg is not None and leg.get_frame() is not None:
        frame = leg.get_frame()
        frame.set_facecolor(BG_COLOR)
        frame.set_edgecolor(GRID_COLOR)

    fig.tight_layout(pad=0.8)
    statsDir = os.path.join(outDir, "stats")
    os.makedirs(statsDir, exist_ok=True)
    outPath = os.path.join(statsDir, "gate_stages.png")
    fig.savefig(outPath, facecolor=BG_COLOR)
    plt.close(fig)


def buildOracleProfiles(
    configPath: str | None = None,
    templateProfilePath: str | None = None,
    addMode: bool = False,
    label: str | None = None,
) -> dict:
    cfgPath = (
        configPath if configPath is not None else DEFAULT_CONFIG
    )
    baseCfg = profile.loadJson(cfgPath)
    templatePath = (
        templateProfilePath
        if templateProfilePath is not None
        else os.path.join(
            ROOT_DIR,
            "inputs",
            "profiles",
            "template-config.json",
        )
    )
    template = profile.loadJson(templatePath)
    primerDays, tunerDays, holdoutDays, totalDays = _oracleWindows(baseCfg)
    oracleDays = primerDays + tunerDays

    tickers = profile._requireTickers(baseCfg)
    outLabel = str(label) if label is not None else "default"
    outDir = os.path.join(ORACLES_BASE, outLabel)
    os.makedirs(outDir, exist_ok=True)
    _writeProfile(os.path.join(outDir, "oracle-config.json"), baseCfg)
    outputs: dict = {}
    for ticker in tickers:
        if addMode:
            tickerDir = os.path.join(outDir, ticker)
            if os.path.isdir(tickerDir):
                continue
        print(
            f"[oracles] building profile for {ticker} "
            f"(oracle window {oracleDays}d, holdout {holdoutDays}d)"
        )
        result = _buildProfileForTicker(
            ticker,
            baseCfg,
            template,
            oracleDays,
            holdoutDays,
            outDir,
        )
        if result is None:
            continue
        fullCfg = result["profile"]
        midCfg = _midConfig(fullCfg)
        profilePathFull = os.path.join(
            ROOT_DIR,
            "inputs",
            "profiles",
            f"{ticker}-full-config.json",
        )
        _writeProfile(profilePathFull, fullCfg)
        profilePathMid = os.path.join(
            ROOT_DIR,
            "inputs",
            "profiles",
            f"{ticker}-mid-config.json",
        )
        _writeProfile(profilePathMid, midCfg)
        profilePathRunFull = os.path.join(
            outDir,
            ticker,
            f"{ticker}-full-config.json",
        )
        _writeProfile(profilePathRunFull, fullCfg)
        profilePathRunMid = os.path.join(
            outDir,
            ticker,
            f"{ticker}-mid-config.json",
        )
        _writeProfile(profilePathRunMid, midCfg)
        outputs[ticker] = result
        print(
            "[oracles] wrote profiles: "
            f"{os.path.basename(profilePathFull)}, "
            f"{os.path.basename(profilePathMid)}"
        )

    stagesByTicker: dict = {}
    for ticker, result in outputs.items():
        gateStages = result.get("gate_stages") if isinstance(result, dict) else None
        if gateStages:
            stagesByTicker[str(ticker)] = gateStages

    if stagesByTicker:
        statsDir = os.path.join(outDir, "stats")
        os.makedirs(statsDir, exist_ok=True)
        stagesPath = os.path.join(statsDir, "gate_stages.json")
        _writeProfile(
            stagesPath,
            {
                "stages": ["t", "g", "gs", "gse", "cd", "dynamics"],
                "tickers": stagesByTicker,
            },
        )
        _plotOracleGateStages(stagesByTicker, outDir, outLabel)
    return outputs


def main() -> None:
    args = parseArgs()
    buildOracleProfiles(
        configPath=args.config,
        templateProfilePath=args.template_profile,
        addMode=bool(getattr(args, "add", False)),
        label=str(getattr(args, "label", "") or "default"),
    )


if __name__ == "__main__":
    main()
