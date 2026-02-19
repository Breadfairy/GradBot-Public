#!/usr/bin/env python3
# tune.py – Parameter tuner producing CSV results and post-run backtest.

import csv
import json
import time
import os
from typing import Any, Dict, List, Tuple, Callable
import math
import io
import contextlib
import numpy as np

from backtest import Backtest
from charting import plotTimVal
from params import TuneParams
from engine_shared import (
    buildContext,
    buildSignals,
    rollingMeanAndStd,
    bars_per_day,
    energyCsum,
)
from wallet import PHASE_BUY_PORTIONS_DEFAULT, PHASE_SELL_PORTIONS_DEFAULT
from cache import RESULT_FIELD_NAMES, klinesMeta
from metrics import (
    edgeVsBench as edgeVsBenchMetric,
    scoreFromEdge,
    grossPctVsBench,
    summarizeRiskFull,
)
from cache import getKlinesCached, profile_windows as profileWindows
from genScatter import generate_scatter
import cache
import profile
from config_compare import configsEqual
from tune_artifacts import bestConfigFromRow, writeBestArtifacts
from tune_axes import (
    axesFromConfig,
    buildIntervalGroups,
    buildParamProduct,
    gradientVariants,
    spacingVariants,
)
from tune_select import riskScoreFromRow


def showProgress(
    doneCount: int,
    totalCount: int,
    startTime: float,
    width: int = 50,
) -> None:
    import sys as _sys
    if totalCount <= 0:
        return
    fraction = doneCount / totalCount
    filled = int(width * fraction)
    bar = '#' * filled + '-' * (width - filled)
    import time as _t
    elapsed = max(_t.time() - startTime, 1e-6)
    rate = doneCount / elapsed
    remaining = (totalCount - doneCount) / rate if rate > 0 else 0.0

    def _fmt(seconds: float) -> str:
        seconds = int(seconds)
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    endChar = '\n' if doneCount >= totalCount else ''
    print(
        f"\r[{bar}] {fraction*100:6.2f}% "
        f"({doneCount}/{totalCount}) ETA {_fmt(remaining)}",
        end=endChar,
        file=_sys.stderr,
        flush=True,
    )


def buildIntervalKlines(
    getKlinesCachedFn: Callable[[str, str, int, int], list],
    ticker: str,
    intervals: List[str],
    days: int,
    minCandles: int,
) -> Dict[str, list]:
    return {
        intervalValue: getKlinesCachedFn(
            ticker, intervalValue, days, minCandles
        )
        for intervalValue in intervals
    }


def _fingerprintPayload(
    config: dict,
    intervals: list[str],
    axes: dict,
    klineMetas: Dict[str, dict],
    primerDays: int,
    tunerDays: int,
    holdoutDays: int,
    totalDays: int,
) -> dict:
    axesCopy = {k: list(v) for k, v in axes.items()}
    return {
        "version": 1,
        "engineHash": cache.engineFingerprint(),
        "tickers": list(config["tickers"]),
        "baseTicker": config["tickers"][0],
        "intervals": list(intervals),
        "primerDays": int(primerDays),
        "tunerDays": int(tunerDays),
        "holdoutDays": int(holdoutDays),
        "totalDays": int(totalDays),
        "axes": axesCopy,
        "klines": klineMetas,
    }


def buildFingerprint(config: dict) -> dict:
    cfg = dict(config)
    profile.ensureFinalPortionPct(cfg)
    profile.validate(cfg, kind="tuner")
    intervals = profile.intervalsFromConfig(cfg)
    primerDays, tunerDays, holdoutDays, totalDays = profileWindows(cfg)
    axes = _buildAxes(cfg)
    maxPeriod = max(
        max(axes['p1Values']),
        max(axes['p2Values']),
        max(axes['p3Values']),
    )
    minCandles = (maxPeriod * 2) + 1
    klineMetas: Dict[str, dict] = {}
    ticker = cfg['tickers'][0]
    for iv in intervals:
        kl = getKlinesCached(
            ticker,
            iv,
            totalDays,
            minCandles,
            holdoutDays=holdoutDays,
        )
        klineMetas[iv] = klinesMeta(kl)
    return _fingerprintPayload(
        cfg,
        intervals,
        axes,
        klineMetas,
        primerDays,
        tunerDays,
        holdoutDays,
        totalDays,
    )


def _writeFingerprint(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, 'w') as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


def runOnce(
    ticker: str,
    interval: str,
    days: int,
    periods: List[int],
    overrides: Dict[str, Any] | None = None,
    showSummary: bool = False,
    minCandles: int | None = None,
    holdoutDays: int = 0,
    primerDays: int = 0,
):
    requiredCandles = (
        minCandles if minCandles is not None else (max(periods) * 2 + 1)
    )
    klines = getKlinesCached(
        ticker,
        interval,
        days,
        requiredCandles,
        holdoutDays=holdoutDays,
    )
    ctx = cache.getContext(
        ticker,
        interval,
        days,
        periods,
        klines,
        lambda: buildContext(klines, periods),
    )

    bt = Backtest(
        ticker,
        klines,
        interval,
        periods,
        days=days,
        doOracles=False,
        showCharts=False,
        showPrints=False,
        showSummary=bool(showSummary),
        overrides=overrides or {},
        ctx=ctx,
        primerDays=primerDays,
        holdoutDays=holdoutDays,
    )
    result = bt.run()
    return result


def _rowForParams(
    ticker: str,
    days: int,
    interval: str,
    klines: list,
    periods: list[int],
    params: TuneParams,
    baseOverrides: dict,
    ctx,
    signals=None,
    primerDays: int = 0,
    cacheOnly: bool = False,
    holdoutDays: int = 0,
) -> tuple[dict, 'BacktestResult']:
    grad1BuyZMin = float(params.grad1BuyZscoreMin)
    grad1SellZMin = float(params.grad1SellZscoreMin)
    grad1BuyWinDays = int(params.grad1BuyWindowDays)
    grad1SellWinDays = int(params.grad1SellWindowDays)
    spacingZscoreMin12 = float(params.spacingZscoreMin12)
    spacingZscoreMin23 = float(params.spacingZscoreMin23)
    spacingWindowDays12 = int(params.spacingWindowDays12)
    spacingWindowDays23 = int(params.spacingWindowDays23)
    spacingEnergyModel = str(params.spacingEnergyModel)
    spacingEnergyWinDays = int(params.spacingEnergyWinDays)
    spacingEnergyMin12 = float(params.spacingEnergyMin12)
    spacingEnergyMin23 = float(params.spacingEnergyMin23)
    phaseBuy = int(params.phaseBuy)
    phaseSell = int(params.phaseSell)
    finalPortionPctRaw = float(params.finalPortionPct)
    if finalPortionPctRaw < 0.0:
        finalPortionPct = 0.0
    elif finalPortionPctRaw > 1.0:
        finalPortionPct = 1.0
    else:
        finalPortionPct = finalPortionPctRaw
    cooldown = int(params.cooldown)
    taxModeStr = str(params.taxMode).lower()
    incomeBase = float(params.annualIncomeBase)
    sweepIntervalRaw = str(params.profitSweepInterval).lower().strip()
    sweepIntervalOverride = (
        None if sweepIntervalRaw in ("", "none") else sweepIntervalRaw
    )
    sweepIntervalRow = (
        "" if sweepIntervalOverride is None else sweepIntervalOverride
    )
    sweepShare = float(params.profitSweepShare)
    macroDynWin = int(getattr(params, 'macroDynWindowDays', 0) or 0)
    macroDynZMin = float(getattr(params, 'macroDynZMin', 0.0) or 0.0)
    macroDynZMax = float(getattr(params, 'macroDynZMax', 0.0) or 0.0)
    macroDynPctMin = float(getattr(params, 'macroDynPctMin', 0.0) or 0.0)
    macroDynPctMax = float(getattr(params, 'macroDynPctMax', 0.0) or 0.0)
    macroP1 = int(getattr(params, 'macroP1', 0) or 0)
    macroP2 = int(getattr(params, 'macroP2', 0) or 0)
    macroP3 = int(getattr(params, 'macroP3', 0) or 0)
    macroGradWinDays = int(getattr(params, 'macroGradWinDays', 0) or 0)
    macroGradZMin = float(getattr(params, 'macroGradZMin', 0.0) or 0.0)
    macroGradZMax = float(getattr(params, 'macroGradZMax', 0.0) or 0.0)
    macroGradMultMin = float(
        getattr(params, 'macroGradMultMin', 1.0) or 1.0
    )
    macroGradMultMax = float(
        getattr(params, 'macroGradMultMax', 1.0) or 1.0
    )
    macroIntervalStr = str(
        getattr(params, 'macroInterval', '') or ''
    ).strip()
    overrides = dict(baseOverrides)
    overrides.update({
        'GRAD1_BUY_Z_MIN': grad1BuyZMin,
        'GRAD1_SELL_Z_MIN': grad1SellZMin,
        'GRAD1_BUY_WIN_DAYS': grad1BuyWinDays,
        'GRAD1_SELL_WIN_DAYS': grad1SellWinDays,
        'PHASE_BUY_PORTIONS': phaseBuy,
        'PHASE_SELL_PORTIONS': phaseSell,
        'FINAL_PORTION_PCT': finalPortionPct,
        'COOLDOWN': cooldown,
        'TAX_MODE': taxModeStr,
        'ANNUAL_INCOME_BASE': incomeBase,
        'PROFIT_SWEEP_INTERVAL': sweepIntervalRow,
        'PROFIT_SWEEP_SHARE': sweepShare,
        'SPACING_Z_MIN_12': spacingZscoreMin12,
        'SPACING_Z_MIN_23': spacingZscoreMin23,
        'SPACING_WIN_DAYS_12': spacingWindowDays12,
        'SPACING_WIN_DAYS_23': spacingWindowDays23,
        'MICRO_NRG_MODEL': spacingEnergyModel,
        'MICRO_NRG_WIN_DAYS': spacingEnergyWinDays,
        'MICRO_NRG_MIN_12': spacingEnergyMin12,
        'MICRO_NRG_MIN_23': spacingEnergyMin23,
    })
    overrides['MACRO_INTERVAL'] = macroIntervalStr
    overrides['MACRO_NRG_WIN_DAYS'] = macroDynWin
    overrides['MACRO_NRG_Z_MIN'] = macroDynZMin
    overrides['MACRO_NRG_Z_MAX'] = macroDynZMax
    overrides['MACRO_DYN_PCT_MIN'] = macroDynPctMin
    overrides['MACRO_DYN_PCT_MAX'] = macroDynPctMax
    overrides['MACRO_GRAD_WIN_DAYS'] = macroGradWinDays
    overrides['MACRO_GRAD_Z_MIN'] = macroGradZMin
    overrides['MACRO_GRAD_Z_MAX'] = macroGradZMax
    overrides['MACRO_MULT_GRAD_MIN'] = macroGradMultMin
    overrides['MACRO_MULT_GRAD_MAX'] = macroGradMultMax
    overrides['MACRO_P1'] = macroP1
    overrides['MACRO_P2'] = macroP2
    overrides['MACRO_P3'] = macroP3
    ctxMeta = ctx["_cache"]
    spec = {
        "ticker": ticker,
        "interval": interval,
        "days": int(days),
        "primerDays": int(primerDays),
        "ctxSpecHash": ctxMeta["specHash"],
        "p1": int(periods[0]),
        "p2": int(periods[1]),
        "p3": int(periods[2]),
        "SUMMARY_LABEL": overrides["SUMMARY_LABEL"],
        "CHART_CHUNK_SIZE": overrides["CHART_CHUNK_SIZE"],
        "WALLET_SEED_QUOTE": overrides["WALLET_SEED_QUOTE"],
        "WALLET_FEE_RATE": overrides["WALLET_FEE_RATE"],
        "QUOTE_TO_AUD_RATE": overrides["QUOTE_TO_AUD_RATE"],
        "MACRO_INTERVAL": macroIntervalStr,
        "MACRO_NRG_WIN_DAYS": macroDynWin,
        "MACRO_NRG_Z_MIN": macroDynZMin,
        "MACRO_NRG_Z_MAX": macroDynZMax,
        "MACRO_DYN_PCT_MIN": macroDynPctMin,
        "MACRO_DYN_PCT_MAX": macroDynPctMax,
        "MACRO_P1": macroP1,
        "MACRO_P2": macroP2,
        "MACRO_P3": macroP3,
        "MACRO_GRAD_WIN_DAYS": macroGradWinDays,
        "MACRO_GRAD_Z_MIN": macroGradZMin,
        "MACRO_GRAD_Z_MAX": macroGradZMax,
        "MACRO_MULT_GRAD_MIN": macroGradMultMin,
        "MACRO_MULT_GRAD_MAX": macroGradMultMax,
        "MACRO_BUY_MULT_BULL": float(overrides["MACRO_BUY_MULT_BULL"]),
        "MACRO_BUY_MULT_BEAR": float(overrides["MACRO_BUY_MULT_BEAR"]),
        "MACRO_BUY_MULT_REV": float(overrides["MACRO_BUY_MULT_REV"]),
        "MACRO_BUY_MULT_ROLL": float(overrides["MACRO_BUY_MULT_ROLL"]),
        "MACRO_SELL_MULT_BULL": float(overrides["MACRO_SELL_MULT_BULL"]),
        "MACRO_SELL_MULT_BEAR": float(overrides["MACRO_SELL_MULT_BEAR"]),
        "MACRO_SELL_MULT_REV": float(overrides["MACRO_SELL_MULT_REV"]),
        "MACRO_SELL_MULT_ROLL": float(overrides["MACRO_SELL_MULT_ROLL"]),
        "GRAD1_BUY_Z_MIN": grad1BuyZMin,
        "GRAD1_SELL_Z_MIN": grad1SellZMin,
        "GRAD1_BUY_WIN_DAYS": grad1BuyWinDays,
        "GRAD1_SELL_WIN_DAYS": grad1SellWinDays,
        "PHASE_BUY_PORTIONS": phaseBuy,
        "PHASE_SELL_PORTIONS": phaseSell,
        "FINAL_PORTION_PCT": finalPortionPct,
        "COOLDOWN": cooldown,
        "TAX_MODE": taxModeStr,
        "ANNUAL_INCOME_BASE": incomeBase,
        "PROFIT_SWEEP_INTERVAL": sweepIntervalRow,
        "PROFIT_SWEEP_SHARE": sweepShare,
        "SPACING_Z_MIN_12": spacingZscoreMin12,
        "SPACING_Z_MIN_23": spacingZscoreMin23,
        "SPACING_WIN_DAYS_12": spacingWindowDays12,
        "SPACING_WIN_DAYS_23": spacingWindowDays23,
        "MICRO_NRG_MODEL": spacingEnergyModel,
        "MICRO_NRG_WIN_DAYS": spacingEnergyWinDays,
        "MICRO_NRG_MIN_12": spacingEnergyMin12,
        "MICRO_NRG_MIN_23": spacingEnergyMin23,
    }
    cachedRow = cache.loadResultRow(spec)
    if cachedRow is not None:
        filteredRow = {
            k: cachedRow[k]
            for k in RESULT_FIELD_NAMES
            if k in cachedRow
        }
        return dict(filteredRow), None
    if cacheOnly:
        return None, None
    # No moon/crab overlays; overrides are single-config only
    bt = Backtest(
        ticker,
        klines,
        interval,
        periods,
        days=days,
        doOracles=False,
        showCharts=False,
        showPrints=False,
        showSummary=False,
        overrides=overrides,
        ctx=ctx,
        signals=signals,
        computeRisk=True,
        primerDays=primerDays,
        holdoutDays=holdoutDays,
    )
    result = bt.run()
    simMetrics = result.sim
    benchMetrics = result.bench
    accountingSim = result.accountingSim
    accountingBench = result.accountingBench
    simNetRaw = (
        simMetrics['portfolio_value'] - simMetrics['tax_liability']
    )
    benchNetRaw = (
        benchMetrics['portfolio_value'] - benchMetrics['tax_liability']
    )
    if benchNetRaw > 0:
        netPctRaw = ((simNetRaw / benchNetRaw) - 1.0) * 100.0
    else:
        netPctRaw = float('nan')
    netPct = (
        round(netPctRaw, 6)
        if math.isfinite(netPctRaw)
        else float('nan')
    )
    simPotential = accountingSim.get('potential_profit', result.potentialProfit)
    benchPotential = accountingBench.get(
        'potential_profit', result.potentialProfitBench
    )
    simNetAfterTaxProfit = accountingSim.get(
        'net_after_tax', result.netAfterTaxProfit
    )
    benchNetAfterTaxProfit = accountingBench.get(
        'net_after_tax', result.netAfterTaxProfitBench
    )
    simValueQuote = float(simMetrics['portfolio_value'])
    benchValueQuote = float(benchMetrics['portfolio_value'])
    simPostTaxValueRaw = result.simPostTaxValue
    benchPostTaxValueRaw = result.benchPostTaxValue
    simPostTaxValue = (
        float(simPostTaxValueRaw)
        if isinstance(simPostTaxValueRaw, (int, float))
        else simNetRaw
    )
    benchPostTaxValue = (
        float(benchPostTaxValueRaw)
        if isinstance(benchPostTaxValueRaw, (int, float))
        else benchNetRaw
    )
    grossEdgeVsBench = simValueQuote - benchValueQuote
    netEdgeVsBench = simPostTaxValue - benchPostTaxValue
    edgeVsBench = edgeVsBenchMetric(
        simPostTaxValue,
        benchPostTaxValue,
        grossEdgeVsBench,
        netEdgeVsBench,
        taxModeStr,
    )
    scoreMetric = edgeVsBench
    # Risk metrics: recorded to CSV and used for Sharpe-based ranking.
    sharpe = result.sharpe
    sortino = result.sortino
    mdd = result.mdd
    cagr_v = result.cagr
    sharpe4w = result.sharpe4w
    sortino4w = result.sortino4w
    sharpe13w = result.sharpe13w
    sortino13w = result.sortino13w
    sharpe4wAbs = result.sharpe4wAbs
    sortino4wAbs = result.sortino4wAbs
    sharpe13wAbs = result.sharpe13wAbs
    sortino13wAbs = result.sortino13wAbs

    row = {
        'ticker': ticker,
        'interval': interval,
        'days': days,
        'p1': periods[0],
        'p2': periods[1],
        'p3': periods[2],
        'MACRO_INTERVAL': macroIntervalStr,
        'MACRO_NRG_WIN_DAYS': macroDynWin,
        'MACRO_NRG_Z_MIN': macroDynZMin,
        'MACRO_NRG_Z_MAX': macroDynZMax,
        'MACRO_DYN_PCT_MIN': macroDynPctMin,
        'MACRO_DYN_PCT_MAX': macroDynPctMax,
        'MACRO_P1': macroP1,
        'MACRO_P2': macroP2,
        'MACRO_P3': macroP3,
        'MACRO_GRAD_WIN_DAYS': macroGradWinDays,
        'MACRO_GRAD_Z_MIN': macroGradZMin,
        'MACRO_GRAD_Z_MAX': macroGradZMax,
        'MACRO_MULT_GRAD_MIN': macroGradMultMin,
        'MACRO_MULT_GRAD_MAX': macroGradMultMax,
        'MACRO_BUY_MULT_BULL': float(overrides['MACRO_BUY_MULT_BULL']),
        'MACRO_BUY_MULT_BEAR': float(overrides['MACRO_BUY_MULT_BEAR']),
        'MACRO_BUY_MULT_REV': float(overrides['MACRO_BUY_MULT_REV']),
        'MACRO_BUY_MULT_ROLL': float(overrides['MACRO_BUY_MULT_ROLL']),
        'MACRO_SELL_MULT_BULL': float(overrides['MACRO_SELL_MULT_BULL']),
        'MACRO_SELL_MULT_BEAR': float(overrides['MACRO_SELL_MULT_BEAR']),
        'MACRO_SELL_MULT_REV': float(overrides['MACRO_SELL_MULT_REV']),
        'MACRO_SELL_MULT_ROLL': float(overrides['MACRO_SELL_MULT_ROLL']),
        'GRAD1_BUY_Z_MIN': grad1BuyZMin,
        'GRAD1_SELL_Z_MIN': grad1SellZMin,
        'GRAD1_BUY_WIN_DAYS': grad1BuyWinDays,
        'GRAD1_SELL_WIN_DAYS': grad1SellWinDays,
        'PHASE_BUY_PORTIONS': phaseBuy,
        'PHASE_SELL_PORTIONS': phaseSell,
        'FINAL_PORTION_PCT': finalPortionPct,
        'COOLDOWN': cooldown,
        'TAX_MODE': taxModeStr,
        'ANNUAL_INCOME_BASE': incomeBase,
        'PROFIT_SWEEP_INTERVAL': sweepIntervalRow,
        'PROFIT_SWEEP_SHARE': sweepShare,
        'SPACING_Z_MIN_12': spacingZscoreMin12,
        'SPACING_Z_MIN_23': spacingZscoreMin23,
        'SPACING_WIN_DAYS_12': spacingWindowDays12,
        'SPACING_WIN_DAYS_23': spacingWindowDays23,
        'MICRO_NRG_MODEL': spacingEnergyModel,
        'MICRO_NRG_WIN_DAYS': spacingEnergyWinDays,
        'MICRO_NRG_MIN_12': spacingEnergyMin12,
        'MICRO_NRG_MIN_23': spacingEnergyMin23,
        'preTaxEdge': round(result.preTaxEdge, 6),
        'postTaxEdge': round(result.postTaxEdge, 6),
        'netPctVsHodl': netPct,
        'simValue': round(simMetrics['portfolio_value'], 6),
        'simPostTax': round(simNetRaw, 6),
        'benchValue': round(benchMetrics['portfolio_value'], 6),
        'benchPostTax': round(benchNetRaw, 6),
        'trades': simMetrics['trades'],
        'fees': round(simMetrics['fees_paid_quote'], 6),
        'tax': round(simMetrics['tax_liability'], 6),
        'lockedProfit': round(simMetrics.get('locked_profit', 0.0), 6),
        'potentialProfit': (
            round(simPotential, 6)
            if isinstance(simPotential, (int, float))
            else float('nan')
        ),
        'potentialProfitBench': (
            round(benchPotential, 6)
            if isinstance(benchPotential, (int, float))
            else float('nan')
        ),
        'netAfterTaxProfit': (
            round(simNetAfterTaxProfit, 6)
            if isinstance(simNetAfterTaxProfit, (int, float))
            else float('nan')
        ),
        'netAfterTaxProfitBench': (
            round(benchNetAfterTaxProfit, 6)
            if isinstance(benchNetAfterTaxProfit, (int, float))
            else float('nan')
        ),
        'grossEdgeVsBench': round(grossEdgeVsBench, 6),
        'netEdgeVsBench': round(netEdgeVsBench, 6),
        'edgeVsBench': round(edgeVsBench, 6),
        'sharpe': (
            round(sharpe, 6)
            if isinstance(sharpe, (int, float))
            else float('nan')
        ),
        'sortino': (
            round(sortino, 6)
            if isinstance(sortino, (int, float))
            else float('nan')
        ),
        'mdd': (
            round(mdd, 6)
            if isinstance(mdd, (int, float))
            else float('nan')
        ),
        'cagr': (
            round(cagr_v, 6)
            if isinstance(cagr_v, (int, float))
            else float('nan')
        ),
        'sharpe4w': (
            round(sharpe4w, 6)
            if isinstance(sharpe4w, (int, float))
            else float('nan')
        ),
        'sortino4w': (
            round(sortino4w, 6)
            if isinstance(sortino4w, (int, float))
            else float('nan')
        ),
        'sharpe13w': (
            round(sharpe13w, 6)
            if isinstance(sharpe13w, (int, float))
            else float('nan')
        ),
        'sortino13w': (
            round(sortino13w, 6)
            if isinstance(sortino13w, (int, float))
            else float('nan')
        ),
        'sharpe4wAbs': (
            round(sharpe4wAbs, 6)
            if isinstance(sharpe4wAbs, (int, float))
            else float('nan')
        ),
        'sortino4wAbs': (
            round(sortino4wAbs, 6)
            if isinstance(sortino4wAbs, (int, float))
            else float('nan')
        ),
        'sharpe13wAbs': (
            round(sharpe13wAbs, 6)
            if isinstance(sharpe13wAbs, (int, float))
            else float('nan')
        ),
        'sortino13wAbs': (
            round(sortino13wAbs, 6)
            if isinstance(sortino13wAbs, (int, float))
            else float('nan')
        ),
        'scoreMetric': scoreFromEdge(edgeVsBench),
    }
    cache.saveResultRow(spec, row)
    return row, result


def _buildAxes(config: Dict[str, Any]) -> Dict[str, List[Any]]:
    return axesFromConfig(config)


def _baseFromTicker(sym: str) -> str:
    u = str(sym).upper()
    for q in ('USDT','BUSD','USDC','TUSD','FDUSD','USD'):
        if u.endswith(q):
            return sym[: -len(q)]
    return sym


def _dir_size_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            total += os.path.getsize(fp)
    return total


def _format_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    i = 0
    while size >= 1024.0 and i < len(units) - 1:
        size /= 1024.0
        i += 1
    return f"{size:.2f} {units[i]}"


def warmZstats(ctx, signals: dict, axes: dict) -> None:
    """Precompute z-stats and energy csums for windows in axes."""
    bpd = bars_per_day(ctx)
    gradWins = set(
        int(x) for x in (
            axes.get('grad1BuyWindowValues', [])
            + axes.get('grad1SellWindowValues', [])
        )
    )
    sp12Wins = set(int(x) for x in axes.get('spacingWindow12Values', []))
    sp23Wins = set(int(x) for x in axes.get('spacingWindow23Values', []))
    enWins = set(int(x) for x in axes.get('spacingEnergyWinValues', []))

    g1p1 = np.asarray(signals['g1P1'], dtype=float)
    g1p3 = np.asarray(signals['g1P3'], dtype=float)
    s12 = np.asarray(signals['s12'], dtype=float)
    s23 = np.asarray(signals['s23'], dtype=float)

    for d in sorted(gradWins):
        win = max(int(round(d * bpd)), 1)
        cache.getZStatsForSeries(
            ctx, 'g1p1', win, lambda: rollingMeanAndStd(g1p1, win)
        )
    for d in sorted(sp12Wins):
        win = max(int(round(d * bpd)), 1)
        cache.getZStatsForSeries(
            ctx, 's12', win, lambda: rollingMeanAndStd(s12, win)
        )
    for d in sorted(sp23Wins):
        win = max(int(round(d * bpd)), 1)
        cache.getZStatsForSeries(
            ctx, 's23', win, lambda: rollingMeanAndStd(s23, win)
        )

    trend = np.asarray(signals['trendCode'], dtype=int)
    csum12 = energyCsum(ctx, trend, '12')
    csum23 = energyCsum(ctx, trend, '23')
    for d in sorted(enWins):
        win = max(int(round(d * bpd)), 1)
        cache.getZStatsForSeries(
            ctx, 'e12', win, lambda: rollingMeanAndStd(csum12, win)
        )
        cache.getZStatsForSeries(
            ctx, 'e23', win, lambda: rollingMeanAndStd(csum23, win)
        )


def _render_summary_to_path(
    rowObj: dict,
    cfgObj: dict,
    dest_path: str,
    ticker: str,
    days: int,
    klinesByInterval: dict,
    holdoutDays: int = 0,
):
    primerDays, tunerDays, holdoutLocal, totalDays = profileWindows(cfgObj)
    intervalValue = rowObj['interval']
    periodsLocal = [int(rowObj['p1']), int(rowObj['p2']), int(rowObj['p3'])]
    kl = klinesByInterval.get(intervalValue)
    minCandlesLocal = (max(periodsLocal) * 2) + 1
    if kl is None or len(kl) < minCandlesLocal:
        kl = getKlinesCached(
            ticker,
            intervalValue,
            totalDays,
            minCandlesLocal,
            holdoutDays=holdoutLocal,
        )
        klinesByInterval[intervalValue] = kl
    ctxLocal = cache.getContext(
        ticker,
        intervalValue,
        totalDays,
        periodsLocal,
        kl,
        lambda: buildContext(kl, periodsLocal),
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        Backtest(
            ticker,
            kl,
            intervalValue,
            periodsLocal,
            days=days,
            doOracles=False,
            showCharts=False,
            showPrints=False,
            showSummary=True,
            overrides=cfgObj,
            ctx=ctxLocal,
            primerDays=primerDays,
            holdoutDays=holdoutLocal,
        ).run()
    tmp = f"{dest_path}.tmp"
    with open(tmp, 'w') as outf:
        outf.write(buf.getvalue())
    os.replace(tmp, dest_path)


def _generate_scatter(
    csv_path: str,
    png_path: str,
    tickers: list[str] | None = None,
) -> None:
    if not os.path.exists(csv_path):
        return
    generate_scatter(csv_path, png_path, tickers=tickers)


# ======================================================================
# Tuner orchestrator (merged from pipelines/tune_runner.py)
# ======================================================================


BAR = "=" * 55


def runTuner(config: dict, out_dir: str) -> str:
    tickers = config['tickers']
    ticker = tickers[0]
    primerDays, tunerDays, holdoutDays, totalDays = profileWindows(config)
    intervals = profile.intervalsFromConfig(config)
    outDir = os.path.abspath(out_dir)
    os.makedirs(outDir, exist_ok=True)
    resultsCsvPath = os.path.join(outDir, "results.csv")
    chartsRoot = os.path.join(outDir, "charts")
    tuneChartsDir = os.path.join(chartsRoot, "tune")

    axes = _buildAxes(config)
    p1Values = axes['p1Values']
    p2Values = axes['p2Values']
    p3Values = axes['p3Values']

    fieldNames = list(RESULT_FIELD_NAMES)

    maxPeriod = max(max(p1Values), max(p2Values), max(p3Values))
    globalMinCandles = (maxPeriod * 2) + 1

    klinesByInterval = buildIntervalKlines(
        lambda tkr, iv, d, mc: getKlinesCached(
            tkr,
            iv,
            totalDays,
            mc,
            holdoutDays=holdoutDays,
        ),
        ticker,
        intervals,
        totalDays,
        globalMinCandles,
    )
    intervalGroups = buildIntervalGroups(
        intervals, p1Values, p2Values, p3Values
    )
    gradVariants = gradientVariants(axes)
    spacingVariantsList = spacingVariants(axes)
    gradCombos = len(gradVariants)
    spacingCombos = len(spacingVariantsList)
    paramCombos = (
        len(axes['macroIntervalValues'])
        * len(axes['macroDynWinValues'])
        * len(axes['macroDynZMinValues'])
        * len(axes['macroDynZMaxValues'])
        * len(axes['macroDynPctMinValues'])
        * len(axes['macroDynPctMaxValues'])
        * len(axes['macroP1Values'])
        * len(axes['macroP2Values'])
        * len(axes['macroP3Values'])
        * len(axes['macroGradWinValues'])
        * len(axes['macroGradZMinValues'])
        * len(axes['macroGradZMaxValues'])
        * len(axes['macroGradMultMinValues'])
        * len(axes['macroGradMultMaxValues'])
        * len(axes['phaseBuyValues'])
        * len(axes['phaseSellValues'])
        * len(axes['finalPortionValues'])
        * len(axes['cooldownValues'])
        * len(axes['taxModeValues'])
        * len(axes['annualIncomeBaseValues'])
        * len(axes['profitSweepIntervalValues'])
        * len(axes['profitSweepShareValues'])
        * len(axes['spacingEnergyModelValues'])
        * len(axes['spacingEnergyWinValues'])
        * len(axes['spacingEnergyMin12Values'])
        * len(axes['spacingEnergyMin23Values'])
        * gradCombos
        * spacingCombos
    )
    totalCombos = len(intervalGroups) * max(paramCombos, 1)

    cache_dir = cache.cacheRoot()
    if os.path.isdir(cache_dir):
        sz = _dir_size_bytes(cache_dir)
        print(f"[warning] Cache size: {_format_size(sz)}")
    else:
        print("[warning] Cache size: 0 B")

    startTime = time.time()
    bestRowLocal = None
    bestScoreLocal = float('-inf')
    cacheHits = 0
    bestRiskRow: dict | None = None
    bestRiskScore = float('-inf')
    completed = 0
    baseOverrides = profile.overrides(
        {
            "SUMMARY_LABEL": config["SUMMARY_LABEL"],
            "CHART_CHUNK_SIZE": config["CHART_CHUNK_SIZE"],
            "WALLET_SEED_QUOTE": config["WALLET_SEED_QUOTE"],
            "WALLET_FEE_RATE": config["WALLET_FEE_RATE"],
            "QUOTE_TO_AUD_RATE": config["QUOTE_TO_AUD_RATE"],
            "MACRO_BUY_MULT_BULL": config["MACRO_BUY_MULT_BULL"],
            "MACRO_BUY_MULT_BEAR": config["MACRO_BUY_MULT_BEAR"],
            "MACRO_BUY_MULT_REV": config["MACRO_BUY_MULT_REV"],
            "MACRO_BUY_MULT_ROLL": config["MACRO_BUY_MULT_ROLL"],
            "MACRO_SELL_MULT_BULL": config["MACRO_SELL_MULT_BULL"],
            "MACRO_SELL_MULT_BEAR": config["MACRO_SELL_MULT_BEAR"],
            "MACRO_SELL_MULT_REV": config["MACRO_SELL_MULT_REV"],
            "MACRO_SELL_MULT_ROLL": config["MACRO_SELL_MULT_ROLL"],
        }
    )
    with open(resultsCsvPath, 'w', newline='') as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=fieldNames)
        writer.writeheader()
        for intervalValue, p1Value, p2Value, p3Value in intervalGroups:
            intervalKlines = klinesByInterval[intervalValue]
            periods = [p1Value, p2Value, p3Value]
            context = cache.getContext(
                ticker,
                intervalValue,
                totalDays,
                periods,
                intervalKlines,
                lambda: buildContext(intervalKlines, periods),
            )
            signals = cache.getSignals(
                ticker,
                intervalValue,
                totalDays,
                periods,
                [],
                intervalKlines,
                lambda: buildSignals(context, []),
            )
            warmZstats(context, signals, axes)
            for params in buildParamProduct(
                axes, gradVariants, spacingVariantsList
            ):
                row, result = _rowForParams(
                    ticker,
                    totalDays,
                    intervalValue,
                    intervalKlines,
                    periods,
                    params,
                    baseOverrides,
                    context,
                    signals=signals,
                    primerDays=primerDays,
                    cacheOnly=True,
                    holdoutDays=holdoutDays,
                )
                if row is None:
                    row, result = _rowForParams(
                        ticker,
                        totalDays,
                        intervalValue,
                        intervalKlines,
                        periods,
                        params,
                        baseOverrides,
                        context,
                        signals=signals,
                        primerDays=primerDays,
                        holdoutDays=holdoutDays,
                    )
                else:
                    cacheHits += 1
                writer.writerow(row)
                simVal = float(row['simValue'])
                benchVal = float(row['benchValue'])
                grossPct = grossPctVsBench(simVal, benchVal)
                if grossPct > bestScoreLocal:
                    bestScoreLocal = grossPct
                    bestRowLocal = row
                riskScore = riskScoreFromRow(row)
                if riskScore > bestRiskScore:
                    bestRiskScore = riskScore
                    bestRiskRow = row
                tr = int(float(row['trades']))
                completed += 1
                showProgress(completed, totalCombos, startTime)

    if bestRowLocal is None:
        raise RuntimeError("No valid tuning combination produced a score.")
    baseBestRow = bestRowLocal
    chosenRow = baseBestRow

    statsRow = bestRiskRow or chosenRow
    statsConfigPreview = bestConfigFromRow(statsRow, config)

    ensureFn = lambda tkr, iv, d, mc: getKlinesCached(
        tkr,
        iv,
        totalDays,
        mc,
        holdoutDays=holdoutDays,
    )
    bestConfig, bestConfigPath, _, _bestDirPath = writeBestArtifacts(
        chosenRow,
        outDir,
        config,
        ticker,
        totalDays,
        klinesByInterval,
        ensureFn,
    )

    statsSameAsBest = configsEqual(bestConfig, statsConfigPreview)
    if statsSameAsBest:
        statsConfig = bestConfig
        statsConfigPath = bestConfigPath
        print("[tune] stats == best; skipping duplicate artifacts...")
    else:
        statsConfig, statsConfigPath, _, _ = writeBestArtifacts(
            statsRow,
            outDir,
            config,
            ticker,
            totalDays,
            klinesByInterval,
            ensureFn,
            "stats",
        )

    bestChartsDir = os.path.join(tuneChartsDir, "best")
    statsChartsDir = os.path.join(tuneChartsDir, "stats")
    os.makedirs(bestChartsDir, exist_ok=True)
    if not statsSameAsBest:
        os.makedirs(statsChartsDir, exist_ok=True)
    prevChartsDir = os.environ.get("CHARTS_OUT_DIR")
    print("[tune] preparing charts...")
    if statsSameAsBest:
        print("[tune] stats == best; skipping duplicate charts...")

    ivBest = str(chosenRow.get('interval', ''))
    periodsBest = [
        int(chosenRow['p1']),
        int(chosenRow['p2']),
        int(chosenRow['p3']),
    ]
    klTuneBest = getKlinesCached(
        ticker,
        ivBest,
        totalDays,
        globalMinCandles,
        holdoutDays=holdoutDays,
    )
    os.environ["CHARTS_OUT_DIR"] = bestChartsDir
    resTuneBest = Backtest(
        ticker,
        klTuneBest,
        ivBest,
        periodsBest,
        days=totalDays,
        doOracles=False,
        showCharts=True,
        showPrints=False,
        showSummary=False,
        overrides=bestConfig,
        ctx=None,
        signals=None,
        computeRisk=True,
        primerDays=primerDays,
        holdoutDays=holdoutDays,
    ).run()
    if (
        resTuneBest.curveTs is not None
        and resTuneBest.curveSim is not None
        and resTuneBest.curveBench is not None
    ):
        timValPath = os.path.join(tuneChartsDir, "best-timVal.png")
        title = f"{ticker} {ivBest} - tune best timVal"
        plotTimVal(
            resTuneBest.curveTs,
            resTuneBest.curveSim,
            resTuneBest.curveBench,
            title,
            timValPath,
        )
    simBest = resTuneBest.sim
    benchBest = resTuneBest.bench
    seedBest = float(resTuneBest.seedQuote or 0.0)
    simValBest = float(simBest['portfolio_value'])
    benchValBest = float(benchBest['portfolio_value'])
    grossBest = grossPctVsBench(simValBest, benchValBest)
    if seedBest > 0.0:
        edgePctBest = ((simValBest / seedBest) - 1.0) * 100.0
        hodlPctBest = ((benchValBest / seedBest) - 1.0) * 100.0
    else:
        edgePctBest = float('nan')
        hodlPctBest = float('nan')
    tradesBest = int(simBest.get('trades', 0))
    buysBest = int(resTuneBest.buyTrades)
    sellsBest = int(resTuneBest.sellTrades)
    (
        sharpeBestTuner,
        sortinoBestTuner,
        cagrBestTuner,
        mddBestTuner,
        marBestTuner,
    ) = summarizeRiskFull(
        resTuneBest.sharpe,
        resTuneBest.sortino,
        resTuneBest.cagr,
        resTuneBest.mdd,
    )

    resTuneStats = resTuneBest
    ivStats = ivBest
    periodsStats = periodsBest
    if not statsSameAsBest:
        ivStats = str(statsRow.get('interval', ''))
        periodsStats = [
            int(statsRow['p1']),
            int(statsRow['p2']),
            int(statsRow['p3']),
        ]
        klTuneStats = getKlinesCached(
            ticker,
            ivStats,
            totalDays,
            globalMinCandles,
            holdoutDays=holdoutDays,
        )
        os.environ["CHARTS_OUT_DIR"] = statsChartsDir
        resTuneStats = Backtest(
            ticker,
            klTuneStats,
            ivStats,
            periodsStats,
            days=totalDays,
            doOracles=False,
            showCharts=True,
            showPrints=False,
            showSummary=False,
            overrides=statsConfig,
            ctx=None,
            signals=None,
            computeRisk=True,
            primerDays=primerDays,
            holdoutDays=holdoutDays,
        ).run()
        if (
            resTuneStats.curveTs is not None
            and resTuneStats.curveSim is not None
            and resTuneStats.curveBench is not None
        ):
            timValPath = os.path.join(tuneChartsDir, "stats-timVal.png")
            title = f"{ticker} {ivStats} - tune stats timVal"
            plotTimVal(
                resTuneStats.curveTs,
                resTuneStats.curveSim,
                resTuneStats.curveBench,
                title,
                timValPath,
            )
    if prevChartsDir is None:
        os.environ.pop("CHARTS_OUT_DIR", None)
    else:
        os.environ["CHARTS_OUT_DIR"] = prevChartsDir
    simStats = resTuneStats.sim
    benchStats = resTuneStats.bench
    seedStats = float(resTuneStats.seedQuote or 0.0)
    simValStats = float(simStats['portfolio_value'])
    benchValStats = float(benchStats['portfolio_value'])
    grossStats = grossPctVsBench(simValStats, benchValStats)
    if seedStats > 0.0:
        edgePctStats = ((simValStats / seedStats) - 1.0) * 100.0
        hodlPctStats = ((benchValStats / seedStats) - 1.0) * 100.0
    else:
        edgePctStats = float('nan')
        hodlPctStats = float('nan')
    tradesStats = int(simStats.get('trades', 0))
    buysStats = int(resTuneStats.buyTrades)
    sellsStats = int(resTuneStats.sellTrades)
    (
        _shStatsDummy,
        _soStatsDummy,
        cagrStatsTuner,
        mddStatsTuner,
        marStatsTuner,
    ) = summarizeRiskFull(
        resTuneStats.sharpe,
        resTuneStats.sortino,
        resTuneStats.cagr,
        resTuneStats.mdd,
    )

    print(BAR)
    print("[tune] GrossVhodl: best")
    print(
        "[tune] - {sym}: {pct:+6.2f}%  TRADES: {tr:d}".format(
            sym=ticker,
            pct=grossBest,
            tr=tradesBest,
        )
    )
    print(
        "[tune] - EDGE%: {edge:+6.2f}%, HODL%: {hodl:+6.2f}%".format(
            edge=edgePctBest,
            hodl=hodlPctBest,
        )
    )
    print(
        "[tune] - BUYS: {buys:d}, SELLS: {sells:d}".format(
            buys=buysBest,
            sells=sellsBest,
        )
    )
    print("[tune] RiskStats")
    print(
        "[tune] - SHARPE 1w  (abs/rel): "
        "{a1:.2f} / {r1:.2f}".format(
            a1=resTuneBest.sharpe1wAbs,
            r1=resTuneBest.sharpe1w,
        )
    )
    print(
        "[tune] - SHARPE 4w  (abs/rel): "
        "{a4:.2f} / {r4:.2f}".format(
            a4=resTuneBest.sharpe4wAbs,
            r4=resTuneBest.sharpe4w,
        )
    )
    print(
        "[tune] - SHARPE 13w (abs/rel): "
        "{a13:.2f} / {r13:.2f}".format(
            a13=resTuneBest.sharpe13wAbs,
            r13=resTuneBest.sharpe13w,
        )
    )
    print(
        "[tune] - SORTINO 1w  (abs/rel): "
        "{a1:.2f} / {r1:.2f}".format(
            a1=resTuneBest.sortino1wAbs,
            r1=resTuneBest.sortino1w,
        )
    )
    print(
        "[tune] - SORTINO 4w  (abs/rel): "
        "{a4:.2f} / {r4:.2f}".format(
            a4=resTuneBest.sortino4wAbs,
            r4=resTuneBest.sortino4w,
        )
    )
    print(
        "[tune] - SORTINO 13w (abs/rel): "
        "{a13:.2f} / {r13:.2f}".format(
            a13=resTuneBest.sortino13wAbs,
            r13=resTuneBest.sortino13w,
        )
    )
    print(f"[tune] - MAR (CAGR/MDD): {marBestTuner:.2f}")
    print(
        "[tune] - MDD: {mdd:.2f}%".format(
            mdd=resTuneBest.mdd * 100.0,
        )
    )
    print(
        "[tune] - CAGR: {cagr:.2f}%".format(
            cagr=resTuneBest.cagr * 100.0,
        )
    )
    print(BAR)
    print(BAR)
    print("[tune] GrossVhodl: stats")
    print(
        "[tune] - {sym}: {pct:+6.2f}%  TRADES: {tr:d}".format(
            sym=ticker,
            pct=grossStats,
            tr=tradesStats,
        )
    )
    print(
        "[tune] - EDGE%: {edge:+6.2f}%, HODL%: {hodl:+6.2f}%".format(
            edge=edgePctStats,
            hodl=hodlPctStats,
        )
    )
    print(
        "[tune] - BUYS: {buys:d}, SELLS: {sells:d}".format(
            buys=buysStats,
            sells=sellsStats,
        )
    )
    print("[tune] RiskStats")
    print(
        "[tune] - SHARPE 1w  (abs/rel): "
        "{a1:.2f} / {r1:.2f}".format(
            a1=resTuneStats.sharpe1wAbs,
            r1=resTuneStats.sharpe1w,
        )
    )
    print(
        "[tune] - SHARPE 4w  (abs/rel): "
        "{a4:.2f} / {r4:.2f}".format(
            a4=resTuneStats.sharpe4wAbs,
            r4=resTuneStats.sharpe4w,
        )
    )
    print(
        "[tune] - SHARPE 13w (abs/rel): "
        "{a13:.2f} / {r13:.2f}".format(
            a13=resTuneStats.sharpe13wAbs,
            r13=resTuneStats.sharpe13w,
        )
    )
    print(
        "[tune] - SORTINO 1w  (abs/rel): "
        "{a1:.2f} / {r1:.2f}".format(
            a1=resTuneStats.sortino1wAbs,
            r1=resTuneStats.sortino1w,
        )
    )
    print(
        "[tune] - SORTINO 4w  (abs/rel): "
        "{a4:.2f} / {r4:.2f}".format(
            a4=resTuneStats.sortino4wAbs,
            r4=resTuneStats.sortino4w,
        )
    )
    print(
        "[tune] - SORTINO 13w (abs/rel): "
        "{a13:.2f} / {r13:.2f}".format(
            a13=resTuneStats.sortino13wAbs,
            r13=resTuneStats.sortino13w,
        )
    )
    print(f"[tune] - MAR (CAGR/MDD): {marStatsTuner:.2f}")
    print(
        "[tune] - MDD: {mdd:.2f}%".format(
            mdd=resTuneStats.mdd * 100.0,
        )
    )
    print(
        "[tune] - CAGR: {cagr:.2f}%".format(
            cagr=resTuneStats.cagr * 100.0,
        )
    )

    summaryPath = os.path.join(outDir, "run.log")
    _render_summary_to_path(
        chosenRow,
        bestConfig,
        summaryPath,
        ticker,
        totalDays,
        klinesByInterval,
        holdoutDays=holdoutDays,
    )

    scatterPath = os.path.join(tuneChartsDir, "scatter.png")
    _generate_scatter(resultsCsvPath, scatterPath, tickers=tickers)

    fingerprint = buildFingerprint(config)
    fpPath = os.path.join(outDir, "fingerprint.json")
    _writeFingerprint(fpPath, fingerprint)

    print(BAR)
    duration = time.time() - startTime
    print(f"[tune] duration: {duration/60.0:.1f} minutes")
    print(BAR)
    return resultsCsvPath
