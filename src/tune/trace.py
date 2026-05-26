#!/usr/bin/env python3
"""Replay and selected-config trace runtime for tuning."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.shared import (
    buildContext,
    buildSignals,
    bars_per_day,
    periods_per_year,
)
from engine.macro_view import buildMacroView
from analysis.metrics import (
    allocationCurveFromTrades,
    equityCurveFromTrades,
    stepReturns,
    lifecycleEdgeStats,
    sharpeRatio,
    sortinoRatio,
    maxDrawdown,
    cagr,
    rollingSharpeSortinoMedian,
)
from portfolio.accounting import (
    FIXED_INCOME_BASE,
    buildStatement,
    marginalIncomeTaxRate,
)
from data.klines_io import loadWindowedKlines
from portfolio.wallet import (
    simulateFromFlags,
)
from analysis.summary import printTraceSummary
from analysis.reporting import holdoutTableText, resultMetrics
from runtime.diag import generateFlags
from runtime.gates import paramsFromSettings
from runtime.posture_feed import dailyPostureArrays
from data.time_bounds import resolveAnchorMs
from tune.artifacts import configsEqual
from tune.paths import chartsHoldoutDir, fingerprintPath, holdoutLogPath
from tune.selection import selectedConfigPaths
from config import profile
from config.params import overridesFromDict


###############################################################################
# Replay Runtime
###############################################################################

def _walletLabel(side: str, note: str = "") -> str:
    if side == "BUY" and note == "seed_buy":
        return "W_SEED_BUY"
    if side == "BUY" and note == "daily_strong_target_buy":
        return "W_TARGET_BUY"
    if side == "SELL" and note == "daily_posture_lock":
        return "W_LOCK_SELL"
    if side == "SELL" and note == "daily_crab_cap":
        return "W_CRAB_CAP"
    if side == "SELL" and note == "peak_lock":
        return "W_PEAK_LOCK"
    if side == "BUY" and note == "peak_lock_capped_buy":
        return "W_PEAK_BUY"
    if side == "BUY":
        return "W_BUY"
    return "W_SELL"


def _executionTradeHealth(trades: List[Any], barsDay: float) -> Dict[str, int]:
    rows = [
        (
            int(getattr(i, "index", -1)),
            str(getattr(i, "side", "")),
        )
        for i in trades
        if str(getattr(i, "note", "")) != "seed_buy"
    ]
    maxBars = max(int(round(float(barsDay))), 1)
    sameBar = 0
    dayFlip = 0
    for i in range(1, len(rows)):
        prevIdx, prevSide = rows[i - 1]
        idx, side = rows[i]
        opposite = prevSide and side and prevSide != side
        if opposite and idx == prevIdx:
            sameBar += 1
        if opposite and 0 <= idx - prevIdx <= maxBars:
            dayFlip += 1
    return {
        "same_bar_opposite_flips": int(sameBar),
        "day_opposite_flips": int(dayFlip),
    }


def _neutralExposureHealth(
    ctx: Dict[str, Any],
    overrides: Dict[str, Any],
    startIdx: int,
    assetFrac: np.ndarray | None,
) -> Dict[str, float | int]:
    if assetFrac is None:
        return {
            "neutral_low_exposure_pct": float("nan"),
            "neutral_half_exposure_pct": float("nan"),
            "neutral_low_exposure_bars": 0,
            "neutral_half_exposure_bars": 0,
        }
    daily = dailyPostureArrays(ctx, overrides)
    if daily is None:
        return {
            "neutral_low_exposure_pct": float("nan"),
            "neutral_half_exposure_pct": float("nan"),
            "neutral_low_exposure_bars": 0,
            "neutral_half_exposure_bars": 0,
        }
    cluster = np.asarray(daily["cluster"][startIdx:], dtype=int)
    asset = np.asarray(assetFrac, dtype=float)
    n = min(int(cluster.size), int(asset.size))
    if n <= 0:
        return {
            "neutral_low_exposure_pct": float("nan"),
            "neutral_half_exposure_pct": float("nan"),
            "neutral_low_exposure_bars": 0,
            "neutral_half_exposure_bars": 0,
        }
    neutral = cluster[:n] == 1
    low = neutral & (asset[:n] < 0.10)
    half = neutral & (asset[:n] >= 0.45) & (asset[:n] <= 0.55)
    return {
        "neutral_low_exposure_pct": float(np.mean(low) * 100.0),
        "neutral_half_exposure_pct": float(np.mean(half) * 100.0),
        "neutral_low_exposure_bars": int(np.sum(low)),
        "neutral_half_exposure_bars": int(np.sum(half)),
    }


@dataclass
class TraceResult:
    sim: Dict[str, Any]
    bench: Dict[str, Any]
    accountingSim: Dict[str, Any]
    accountingBench: Dict[str, Any]
    preTaxEdge: float
    postTaxEdge: float
    netPctVsHodl: float
    simPostTaxValue: float
    benchPostTaxValue: float
    lastPrice: float
    potentialProfit: float
    potentialProfitBench: float
    netAfterTaxProfit: float
    netAfterTaxProfitBench: float
    grossProfit: float
    grossProfitBench: float
    seedQuote: float
    buyTrades: int
    sellTrades: int
    sharpe: float
    sortino: float
    mdd: float
    cagr: float
    sharpe1w: float
    sortino1w: float
    sharpe4w: float
    sortino4w: float
    sharpe13w: float
    sortino13w: float
    sharpe1wAbs: float
    sortino1wAbs: float
    sharpe4wAbs: float
    sortino4wAbs: float
    sharpe13wAbs: float
    sortino13wAbs: float
    lifecycleEdgeMean: float
    lifecycleEdgeMedian: float
    lifecycleEdgeP25: float
    lifecycleEdgeMin: float
    lifecycleUnderwaterPct: float
    lifecycleUnderwaterMean: float
    lifecycleTrackingPct: float
    lifecycleEdgeMdd: float
    lifecycleEdgeScore: float
    rawStartTs: Any | None = None
    rawEndTs: Any | None = None
    visibleStartTs: Any | None = None
    visibleEndTs: Any | None = None
    curveTs: List[Any] | None = None
    curveSim: np.ndarray | None = None
    curveBench: np.ndarray | None = None
    curveAssetFrac: np.ndarray | None = None
    curveQuoteFrac: np.ndarray | None = None
    tradeNotes: Dict[str, int] | None = None
    postureStats: Dict[str, Any] | None = None
    executionHealth: Dict[str, float | int] | None = None

    def toDict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "sim": self.sim,
            "bench": self.bench,
            "accounting": {
                "sim": self.accountingSim,
                "bench": self.accountingBench,
            },
        }
        scalarMap = {
            "pre_tax_edge": "preTaxEdge",
            "post_tax_edge": "postTaxEdge",
            "net_pct_vs_hodl": "netPctVsHodl",
            "sim_post_tax_value": "simPostTaxValue",
            "bench_post_tax_value": "benchPostTaxValue",
            "last_price": "lastPrice",
            "potential_profit": "potentialProfit",
            "potential_profit_bench": "potentialProfitBench",
            "net_after_tax_profit": "netAfterTaxProfit",
            "net_after_tax_profit_bench": "netAfterTaxProfitBench",
            "gross_profit": "grossProfit",
            "gross_profit_bench": "grossProfitBench",
            "sharpe": "sharpe",
            "sortino": "sortino",
            "mdd": "mdd",
            "cagr": "cagr",
            "sharpe1w": "sharpe1w",
            "sortino1w": "sortino1w",
            "sharpe4w": "sharpe4w",
            "sortino4w": "sortino4w",
            "sharpe13w": "sharpe13w",
            "sortino13w": "sortino13w",
            "sharpe1wAbs": "sharpe1wAbs",
            "sortino1wAbs": "sortino1wAbs",
            "sharpe4wAbs": "sharpe4wAbs",
            "sortino4wAbs": "sortino4wAbs",
            "sharpe13wAbs": "sharpe13wAbs",
            "sortino13wAbs": "sortino13wAbs",
            "lifecycle_edge_mean": "lifecycleEdgeMean",
            "lifecycle_edge_median": "lifecycleEdgeMedian",
            "lifecycle_edge_p25": "lifecycleEdgeP25",
            "lifecycle_edge_min": "lifecycleEdgeMin",
            "lifecycle_underwater_pct": "lifecycleUnderwaterPct",
            "lifecycle_underwater_mean": "lifecycleUnderwaterMean",
            "lifecycle_tracking_pct": "lifecycleTrackingPct",
            "lifecycle_edge_mdd": "lifecycleEdgeMdd",
            "lifecycle_edge_score": "lifecycleEdgeScore",
        }
        for key, attr in scalarMap.items():
            out[key] = getattr(self, attr)
        out["trade_notes"] = self.tradeNotes or {}
        out["posture_stats"] = self.postureStats or {}
        out["execution_health"] = self.executionHealth or {}
        return out


class Trace:
    """Run historical trace and optionally chart + print summaries.

    Returns a metrics dict from run() so programmatic callers (e.g., tuner)
    can collect results without parsing prints.
    """
    def __init__(
        self,
        ticker,
        klines,
        intervalStr,
        periods,
        days=None,
        showCharts=False,
        showPrints=False,
        showSummary=True,
        overrides=None,
        ctx=None,
        signals=None,
        computeRisk=True,
        primerDays: int | None = None,
        holdoutDays: int = 0,
        anchorMs: int | None = None,
    ):
        self.ticker = ticker
        self.klines = klines
        self.intervalStr = intervalStr
        self.periods = periods
        self.days = days
        self.showCharts = showCharts
        self.showPrints = showPrints
        self.showSummary = showSummary
        self.overrides = overrides or {}
        self.ctx = ctx
        self.signals = signals
        self.computeRisk = bool(computeRisk)
        self.primerDays = int(primerDays) if primerDays is not None else 0
        self.holdoutDays = int(holdoutDays) if holdoutDays is not None else 0
        self.anchorMs = int(anchorMs) if anchorMs is not None else None
        self.result: Optional[TraceResult] = None

    def _minCandles(self) -> int:
        return max(self.periods) * 2 + 1

    def _buildContext(self):
        ctx = buildContext(self.klines, self.periods)
        ctx["ticker"] = self.ticker
        ctx["days"] = int(self.days) if self.days is not None else 0
        ctx["intervalStr"] = self.intervalStr
        ctx["_cache"] = {
            "ticker": self.ticker,
            "interval": self.intervalStr,
            "days": int(self.days) if self.days is not None else 0,
            "anchorMs": self.anchorMs,
        }
        return ctx

    def _ensureContext(self):
        if self.ctx is not None:
            self.ctx["klines"] = self.klines
            self.ctx["ticker"] = self.ticker
            self.ctx["days"] = int(self.days) if self.days is not None else 0
            self.ctx["intervalStr"] = self.intervalStr
            self.ctx.setdefault(
                "_cache",
                {
                    "ticker": self.ticker,
                    "interval": self.intervalStr,
                    "days": int(self.days) if self.days is not None else 0,
                    "anchorMs": self.anchorMs,
                },
            )
            if self.anchorMs is not None:
                self.ctx["_cache"]["anchorMs"] = self.anchorMs
            return self.ctx
        self.ctx = self._buildContext()
        return self.ctx

    def _timestamps(self) -> List[pd.Timestamp]:
        if self.ctx is not None:
            cached = self.ctx.get("_ts")
            if cached is not None:
                return cached
        ts = pd.to_datetime([k[0] for k in self.klines], unit="ms", utc=True)
        ts = ts.tz_convert(None)
        out = ts.to_pydatetime().tolist()
        if self.ctx is not None:
            self.ctx["_ts"] = out
        return out

    def _buildSignals(self, ctx, lookbacks):
        return buildSignals(ctx, lookbacks)

    def _ensureSignals(self, ctx, lookbacks):
        if self.signals is not None:
            return self.signals
        return self._buildSignals(ctx, lookbacks)

    def _flagParamsAndSignals(self, ctx):
        # Normalize tuner-style specs (lists/ranges) to scalars and
        # build signals without pct-band lookbacks.
        p = paramsFromSettings(overridesFromDict(self.overrides))
        sig = self._ensureSignals(ctx, [])
        return p, sig

    def _macroDynArray(
        self,
        ts: List[Any],
        overrides: Dict[str, Any],
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self.days is None:
            return None
        macro = buildMacroView(
            self.ticker,
            int(self.days),
            self.holdoutDays,
            list(self.periods),
            overrides,
            ts,
            anchorMs=self.anchorMs,
        )
        if macro is None:
            return None
        return macro.dyn, macro.dir, macro.mom

    def _generateFlags(self, ctx, ts, startIdx, params, signals):
        ov = overridesFromDict(self.overrides)
        macroDyn = None
        macroDir = None
        macroMom = None
        macro = self._macroDynArray(ts, ov)
        if macro is not None:
            macroDyn, macroDir, macroMom = macro
        idxFlags = generateFlags(
            ctx,
            signals,
            params,
            startIdx,
            overrides=ov,
            macroDyn=macroDyn,
            macroDir=macroDir,
            macroMom=macroMom,
        )
        return idxFlags, [(ts[i], str(lab)) for i, lab in idxFlags]

    def _walletOverrides(
        self,
    ) -> Tuple[float, float, str, float, float, float]:
        seed = float(self.overrides['WALLET_SEED_QUOTE'])
        seedAssetPct = float(
            self.overrides.get('WALLET_SEED_ASSET_PCT', 1.0)
        )
        taxMode = str(self.overrides['TAX_MODE']).lower()
        incomeBase = FIXED_INCOME_BASE
        audRate = float(self.overrides['QUOTE_TO_AUD_RATE'])
        finalPortionPct = float(self.overrides['FINAL_PORTION_PCT'])
        return (
            seed,
            seedAssetPct,
            taxMode,
            incomeBase,
            audRate,
            finalPortionPct,
        )

    def _simulateWallets(
        self,
        ctx,
        flagsIdx: List[Tuple[int, str]],
        startIdx: int,
        taxMode: str,
        incomeBase: float,
        seed: float,
        seedAssetPct: float,
        finalPortionPct: float = 1.0,
        trendCodeArr=None,
    ):
        baseSymbol = self.ticker
        taxRate = marginalIncomeTaxRate(incomeBase)
        walletTrading = simulateFromFlags(
            ctx,
            flagsIdx,
            baseSymbol=baseSymbol,
            startingCash=0.0,
            feeRate=float(self.overrides['WALLET_FEE_RATE']),
            taxRate=taxRate,
            discountDays=365,
            discountRate=0.50,
            seedInvestQuote=seed,
            seedAssetPct=seedAssetPct,
            seedIndex=startIdx,
            doPrints=self.showPrints,
            phaseBuyPortions=int(self.overrides['PHASE_BUY_PORTIONS']),
            phaseSellPortions=int(self.overrides['PHASE_SELL_PORTIONS']),
            taxMode=taxMode,
            annualIncomeBase=incomeBase,
            finalPortionPct=finalPortionPct,
            trendCodes=trendCodeArr,
            overrides=self.overrides,
        )
        from portfolio.wallet import Wallet
        return walletTrading, Wallet(
            baseSymbol=baseSymbol,
            startingCash=seed,
            feeRate=float(self.overrides['WALLET_FEE_RATE']),
            taxRate=taxRate,
            taxMode=taxMode,
            annualIncomeBase=incomeBase,
        )

    def _walletMarkers(
        self,
        walletTrading,
    ) -> List[Tuple[Any, str]]:
        out = []
        for tr in walletTrading.trades:
            out.append((tr.ts, _walletLabel(tr.side, tr.note)))
        return out

    def _printSummary(
        self,
        startIdx: int,
        ts: List[Any],
        endDt,
        lastPrice: float,
        sim,
        ben,
        simStatement,
        benchStatement,
        seed: float,
        taxMode: str,
        incomeBase: float,
        quoteToAudRate: float,
        simPostTaxValue: float,
        benchPostTaxValue: float,
        netPctVsHodl: float,
        walletTrading,
        walletBench,
    ) -> None:
        if not self.showSummary:
            return
        printTraceSummary(
            ticker=self.ticker,
            ts=ts,
            startIdx=startIdx,
            endDt=endDt,
            lastPrice=lastPrice,
            sim=sim,
            ben=ben,
            simStatement=simStatement,
            benchStatement=benchStatement,
            labelStr="",
            seed=seed,
            incomeBase=incomeBase,
            audRate=quoteToAudRate,
            simPostTaxValue=simPostTaxValue,
            benchPostTaxValue=benchPostTaxValue,
            netPctVsHodl=netPctVsHodl,
            walletTrading=walletTrading,
            ctx=self._ensureContext(),
            intervalStr=self.intervalStr,
            taxMode=taxMode,
            walletBench=walletBench,
        )

    def _charts(
        self,
        ctx,
        ts: List[Any],
        startIdx: int,
        flagsTs: List[Tuple[Any, str]],
        walletMarkers: List[Tuple[Any, str]],
        signals: dict | None,
    ) -> None:
        from analysis.charting import plotTraceCharts

        plotTraceCharts(
            self.showCharts,
            ctx,
            ts,
            startIdx,
            flagsTs,
            walletMarkers,
            signals,
            overridesFromDict(self.overrides),
            self.klines,
            self.ticker,
            self.intervalStr,
        )

    def run(self) -> TraceResult:
        minLen = self._minCandles()
        if len(self.klines) < minLen:
            raise ValueError(f"Need {minLen} candles, have {len(self.klines)}")

        ctx = self._ensureContext()
        ts = self._timestamps()
        startIdx = max(self.periods) * 2
        if self.primerDays > 0:
            primerBars = int(round(self.primerDays * bars_per_day(ctx)))
            if primerBars > 0:
                startIdx += primerBars
        params, signals = self._flagParamsAndSignals(ctx)
        flagsIdx, flagsTs = self._generateFlags(
            ctx,
            ts,
            startIdx,
            params,
            signals,
        )

        (seed, seedAssetPct, taxMode, incomeBase, audRate,
         finalPortionPct) = self._walletOverrides()
        trendArr = np.asarray(signals["trendCode"], dtype=int)
        walletTrading, walletBench = self._simulateWallets(
            ctx,
            flagsIdx,
            startIdx,
            taxMode,
            incomeBase,
            seed,
            seedAssetPct,
            finalPortionPct=finalPortionPct,
            trendCodeArr=trendArr,
        )

        walletMarkers = self._walletMarkers(walletTrading)

        price0 = float(ctx["closes"][startIdx])
        ts0 = ts[startIdx]
        walletBench.buyAll(startIdx, ts0, price0)

        lastIdx = len(ts) - 1
        lastPrice = float(ctx["closes"][lastIdx])
        endDt = ts[lastIdx]

        sim = walletTrading.summary(currentPrice=lastPrice, currentTs=endDt)
        ben = walletBench.summary(currentPrice=lastPrice, currentTs=endDt)
        buyTrades = sum(
            1 for tr in walletTrading.trades if tr.side == "BUY"
        )
        sellTrades = sum(
            1 for tr in walletTrading.trades if tr.side == "SELL"
        )
        tradeNotes = Counter(
            str(tr.note or "signal_trade") for tr in walletTrading.trades
        )
        executionHealth = _executionTradeHealth(
            walletTrading.trades,
            bars_per_day(ctx),
        )
        simStatement = buildStatement(sim, seed, taxMode, incomeBase)
        benchStatement = buildStatement(ben, seed, taxMode, incomeBase)

        deltaPreTax = sim["portfolio_value"] - ben["portfolio_value"]

        simNetAfterTaxProfit = simStatement.get(
            "net_after_tax", simStatement["potential_profit"]
        )
        benchNetAfterTaxProfit = benchStatement.get(
            "net_after_tax", benchStatement["potential_profit"]
        )
        simPostTaxValue = simStatement.get(
            "net_after_tax_value", simNetAfterTaxProfit + seed
        )
        benchPostTaxValue = benchStatement.get(
            "net_after_tax_value", benchNetAfterTaxProfit + seed
        )
        deltaPostTax = simNetAfterTaxProfit - benchNetAfterTaxProfit
        benchValueForPct = benchStatement["potential_profit"] + seed
        simValueForPct = simStatement["potential_profit"] + seed
        netPctVsHodl = (
            ((simValueForPct / benchValueForPct) - 1.0) * 100.0
            if benchValueForPct > 0 else float('nan')
        )

        curveSim: np.ndarray | None = None
        curveBench: np.ndarray | None = None
        curveTs: List[Any] | None = None
        curveAssetFrac: np.ndarray | None = None
        curveQuoteFrac: np.ndarray | None = None

        # Compute risk metrics for programmatic consumers (e.g., tuner)
        if self.computeRisk:
            durationSeconds = max(
                (endDt - ts[startIdx]).total_seconds(), 1.0
            )
            durationYears = max(durationSeconds / 86400.0, 1.0) / 365.0
            closes = ctx["closes"]
            curveSim = equityCurveFromTrades(
                closes=closes,
                trades=walletTrading.trades,
                startIndex=startIdx,
                seedQuote=seed,
            )
            curveBench = equityCurveFromTrades(
                closes=closes,
                trades=walletBench.trades,
                startIndex=startIdx,
                seedQuote=seed,
            )
            curveAssetFrac, curveQuoteFrac = allocationCurveFromTrades(
                closes=closes,
                trades=walletTrading.trades,
                startIndex=startIdx,
                seedQuote=seed,
            )
            executionHealth.update(_neutralExposureHealth(
                ctx,
                self.overrides,
                startIdx,
                curveAssetFrac,
            ))
            curveTs = ts[startIdx:]
            rets = stepReturns(curveSim)
            retsBenchRisk = stepReturns(curveBench)
            nRisk = min(len(rets), len(retsBenchRisk))
            edgeRetsRisk = (
                rets[:nRisk] - retsBenchRisk[:nRisk]
                if nRisk > 0
                else rets
            )
            ppy = periods_per_year(self.intervalStr)
            sharpe_v = sharpeRatio(rets, ppy)
            sortino_v = sortinoRatio(rets, ppy)
            mdd_v = maxDrawdown(curveSim)
            cagr_v = cagr(curveSim, durationYears)

            bpdRisk = bars_per_day(ctx)
            win1 = max(int(round(7.0 * bpdRisk)), 2)
            win4 = max(int(round(28.0 * bpdRisk)), 2)
            win13 = max(int(round(91.0 * bpdRisk)), 2)
            sharpe1wAbs_v, sortino1wAbs_v = rollingSharpeSortinoMedian(
                rets,
                ppy,
                win1,
            )
            sharpe4wAbs_v, sortino4wAbs_v = rollingSharpeSortinoMedian(
                rets,
                ppy,
                win4,
            )
            sharpe13wAbs_v, sortino13wAbs_v = rollingSharpeSortinoMedian(
                rets,
                ppy,
                win13,
            )
            sharpe1w_v, sortino1w_v = rollingSharpeSortinoMedian(
                edgeRetsRisk,
                ppy,
                win1,
            )
            sharpe4w_v, sortino4w_v = rollingSharpeSortinoMedian(
                edgeRetsRisk,
                ppy,
                win4,
            )
            sharpe13w_v, sortino13w_v = rollingSharpeSortinoMedian(
                edgeRetsRisk,
                ppy,
                win13,
            )
            lifecycle = lifecycleEdgeStats(curveSim, curveBench)
        else:
            sharpe_v = float('nan')
            sortino_v = float('nan')
            mdd_v = float('nan')
            cagr_v = float('nan')
            sharpe1w_v = float('nan')
            sortino1w_v = float('nan')
            sharpe4w_v = float('nan')
            sortino4w_v = float('nan')
            sharpe13w_v = float('nan')
            sortino13w_v = float('nan')
            sharpe1wAbs_v = float('nan')
            sortino1wAbs_v = float('nan')
            sharpe4wAbs_v = float('nan')
            sortino4wAbs_v = float('nan')
            sharpe13wAbs_v = float('nan')
            sortino13wAbs_v = float('nan')
            lifecycle = lifecycleEdgeStats([], [])

        self._printSummary(
            startIdx,
            ts,
            endDt,
            lastPrice,
            sim,
            ben,
            simStatement,
            benchStatement,
            seed,
            taxMode,
            incomeBase,
            audRate,
            simPostTaxValue,
            benchPostTaxValue,
            netPctVsHodl,
            walletTrading,
            walletBench,
        )

        self._charts(
            ctx,
            ts,
            startIdx,
            flagsTs,
            walletMarkers,
            signals,
        )

        self.result = TraceResult(
            sim=sim,
            bench=ben,
            accountingSim=simStatement,
            accountingBench=benchStatement,
            preTaxEdge=deltaPreTax,
            postTaxEdge=deltaPostTax,
            netPctVsHodl=netPctVsHodl,
            simPostTaxValue=simPostTaxValue,
            benchPostTaxValue=benchPostTaxValue,
            lastPrice=lastPrice,
            potentialProfit=simStatement["potential_profit"],
            potentialProfitBench=benchStatement["potential_profit"],
            netAfterTaxProfit=simNetAfterTaxProfit,
            netAfterTaxProfitBench=benchNetAfterTaxProfit,
            grossProfit=simStatement["gross_profit"],
            grossProfitBench=benchStatement["gross_profit"],
            seedQuote=seed,
            buyTrades=buyTrades,
            sellTrades=sellTrades,
            sharpe=sharpe_v,
            sortino=sortino_v,
            mdd=mdd_v,
            cagr=cagr_v,
            sharpe4w=sharpe4w_v,
            sortino4w=sortino4w_v,
            sharpe13w=sharpe13w_v,
            sortino13w=sortino13w_v,
            sharpe1wAbs=sharpe1wAbs_v,
            sortino1wAbs=sortino1wAbs_v,
            sharpe4wAbs=sharpe4wAbs_v,
            sortino4wAbs=sortino4wAbs_v,
            sharpe13wAbs=sharpe13wAbs_v,
            sortino13wAbs=sortino13wAbs_v,
            lifecycleEdgeMean=float(lifecycle["lifecycleEdgeMean"]),
            lifecycleEdgeMedian=float(lifecycle["lifecycleEdgeMedian"]),
            lifecycleEdgeP25=float(lifecycle["lifecycleEdgeP25"]),
            lifecycleEdgeMin=float(lifecycle["lifecycleEdgeMin"]),
            lifecycleUnderwaterPct=float(
                lifecycle["lifecycleUnderwaterPct"]
            ),
            lifecycleUnderwaterMean=float(
                lifecycle["lifecycleUnderwaterMean"]
            ),
            lifecycleTrackingPct=float(lifecycle["lifecycleTrackingPct"]),
            lifecycleEdgeMdd=float(lifecycle["lifecycleEdgeMdd"]),
            lifecycleEdgeScore=float(lifecycle["lifecycleEdgeScore"]),
            rawStartTs=ts[0] if ts else None,
            rawEndTs=ts[-1] if ts else None,
            visibleStartTs=ts[startIdx] if ts else None,
            visibleEndTs=endDt,
            sharpe1w=sharpe1w_v,
            sortino1w=sortino1w_v,
            curveTs=curveTs,
            curveSim=curveSim,
            curveBench=curveBench,
            curveAssetFrac=curveAssetFrac,
            curveQuoteFrac=curveQuoteFrac,
            tradeNotes=dict(tradeNotes),
            postureStats=dict(getattr(walletTrading, "postureStats", {})),
            executionHealth=dict(executionHealth),
        )
        return self.result

###############################################################################
# Anchor Helpers
###############################################################################

def anchorMsFromRun(runDir: Path) -> int | None:
    path = fingerprintPath(runDir)
    if not path.is_file():
        return None
    with open(path) as fh:
        data = json.load(fh)
    anchor = data.get("anchorMs")
    return None if anchor is None else int(anchor)


###############################################################################
# Holdout Trace
###############################################################################

BAR = "=" * 54


def _start_pcts(
    startMinPct: int,
    startMaxPct: int,
    startStepPct: int,
) -> list[int]:
    out = []
    firstPct = int(startMinPct)
    lastPct = int(startMaxPct)
    stepPct = int(startStepPct)
    for i in range(firstPct, lastPct + 1, stepPct):
        out.append(i)
    return out


def _start_offsets(
    holdoutDays: int,
    startMinPct: int,
    startMaxPct: int,
    startStepPct: int,
) -> list[Tuple[int, int]]:
    out = []
    seen = set()
    startPcts = _start_pcts(startMinPct, startMaxPct, startStepPct)
    for pct in startPcts:
        offsetDays = int(round(float(holdoutDays) * float(pct) / 100.0))
        if offsetDays not in seen and offsetDays < holdoutDays:
            out.append((pct, offsetDays))
            seen.add(offsetDays)
    return out


def _start_label(baseLabel: str, pct: int) -> str:
    if pct == 0:
        return baseLabel
    return f"{baseLabel}-s{pct:02d}"


def _chart_enabled(cfg: dict, key: str) -> bool:
    val = profile.scalarValue(cfg.get(key), True)
    if isinstance(val, bool):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() not in {"0", "false", "no", "off"}
    return bool(int(val))


def _effective_holdout_days(cfgPath: Path, holdoutOverride: int) -> int:
    cfg = profile.loadJson(cfgPath)
    _primerDays, _trainingDays, _tunerDays, holdoutDays, _totalDays = (
        profile.windowParts(cfg)
    )
    if holdoutOverride > 0:
        holdoutDays = holdoutOverride
    return holdoutDays


def _ensureMplConfig() -> None:
    cacheDir = Path("/tmp/gradbot-cache")
    cacheDir.mkdir(parents=True, exist_ok=True)
    if "XDG_CACHE_HOME" not in os.environ:
        os.environ["XDG_CACHE_HOME"] = str(cacheDir)
    if "MPLCONFIGDIR" in os.environ:
        return
    mplDir = Path("/tmp/gradbot-mpl")
    mplDir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mplDir)


def _start_sweep_text(
    holdoutDays: int,
    startMinPct: int,
    startMaxPct: int,
    startStepPct: int,
) -> str:
    parts = [
        f"{pct}%={offsetDays}d"
        for pct, offsetDays in _start_offsets(
            holdoutDays,
            startMinPct,
            startMaxPct,
            startStepPct,
        )
    ]
    return (
        f"[holdout] start sweep over {startMinPct}-{startMaxPct}% "
        f"of holdout by {startStepPct}%: "
        + ", ".join(parts)
        + "\n"
    )


def _config_parts(
    cfg: dict,
) -> Tuple[str, str, list[int], dict, int, int, int, int, int]:
    ticker = cfg["tickers"][0]
    interval = profile.intervalsFromConfig(cfg)[0]
    periods = [int(cfg["p1"]), int(cfg["p2"]), int(cfg["p3"])]
    primerDays, trainingDays, tunerDays, holdoutDays, totalDays = (
        profile.windowParts(cfg)
    )
    baseFields = {
        "ticker",
        "tickers",
        "intervals",
        "p1",
        "p2",
        "p3",
        "primer_days",
        "training_days",
        "tuner_days",
        "holdout_days",
        *profile.HOLDOUT_START_KEYS,
        "CHARTS_TIMEVAL",
        "CHARTS_TRADES",
    }
    overrides = {k: v for k, v in cfg.items() if k not in baseFields}
    overridesNorm = profile.overrides(overrides)
    profile.validate(overridesNorm, kind="backtest")
    return (
        ticker,
        interval,
        periods,
        overridesNorm,
        primerDays,
        trainingDays,
        tunerDays,
        holdoutDays,
        totalDays,
    )


def _run_one(
    label: str,
    cfgPath: Path,
    holdoutOverride: int,
    chartsRoot: Path | None = None,
    anchorMs: int | None = None,
    startOffsetDays: int = 0,
    runCache: dict | None = None,
) -> dict[str, float | int | str]:
    _ensureMplConfig()
    cacheKey = (str(Path(cfgPath).resolve()), int(holdoutOverride), anchorMs)
    cached = runCache.get(cacheKey) if runCache is not None else None
    if cached is None:
        cfg = profile.loadJson(cfgPath)
        profile.ensureFinalPortionPct(cfg)
        (
            ticker,
            interval,
            periods,
            overrides,
            primerDays,
            trainingDays,
            tunerDays,
            holdoutDays,
            totalDays,
        ) = _config_parts(cfg)
        dataDays = totalDays
        warmupDays = primerDays + trainingDays + tunerDays
        if holdoutOverride > 0:
            holdoutDays = holdoutOverride
            dataDays = warmupDays + holdoutDays
        if dataDays <= 0 or holdoutDays <= 0:
            raise SystemExit(
                "holdout requires holdout_days in config or --days"
            )
        minCandles = (max(periods) * 2) + 1
        klines = loadWindowedKlines(
            ticker,
            interval,
            dataDays,
            minCandles,
            holdoutDays=0,
            anchorMs=anchorMs,
        )
        ctx = buildContext(klines, periods)
        ctx["ticker"] = ticker
        ctx["days"] = int(dataDays)
        ctx["intervalStr"] = interval
        ctx["_cache"] = {
            "ticker": ticker,
            "interval": interval,
            "days": int(dataDays),
            "anchorMs": anchorMs,
        }
        cached = {
            "ticker": ticker,
            "interval": interval,
            "periods": periods,
            "overrides": overrides,
            "holdoutDays": holdoutDays,
            "dataDays": dataDays,
            "warmupDays": warmupDays,
            "chartsTimeval": _chart_enabled(cfg, "CHARTS_TIMEVAL"),
            "chartsTrades": _chart_enabled(cfg, "CHARTS_TRADES"),
            "klines": klines,
            "ctx": ctx,
            "signals": buildSignals(ctx, []),
        }
        if runCache is not None:
            runCache[cacheKey] = cached

    ticker = cached["ticker"]
    interval = cached["interval"]
    periods = cached["periods"]
    overrides = cached["overrides"]
    holdoutDays = int(cached["holdoutDays"])
    dataDays = int(cached["dataDays"])
    activePrimerDays = int(cached["warmupDays"]) + int(startOffsetDays)
    activeDays = holdoutDays - int(startOffsetDays)
    chartsTimeval = bool(cached["chartsTimeval"])
    chartsTrades = bool(cached["chartsTrades"])
    klines = cached["klines"]
    if activeDays <= 0:
        raise SystemExit("holdout start offset leaves no active days")
    chartsDir = None
    if chartsRoot is not None and chartsTrades:
        chartsDir = Path(chartsRoot) / label
        os.makedirs(chartsDir, exist_ok=True)
    if chartsDir is not None:
        os.environ["CHARTS_OUT_DIR"] = str(chartsDir)

    bt = Trace(
        ticker,
        klines,
        interval,
        periods,
        days=dataDays,
        showCharts=chartsDir is not None and chartsTrades,
        showPrints=False,
        showSummary=False,
        overrides=overrides,
        ctx=cached["ctx"],
        signals=cached["signals"],
        computeRisk=True,
        primerDays=activePrimerDays,
        holdoutDays=0,
        anchorMs=anchorMs,
    )
    res = bt.run()
    if (
        chartsRoot is not None
        and chartsTimeval
        and res.curveTs is not None
        and res.curveSim is not None
        and res.curveBench is not None
        and res.curveAssetFrac is not None
        and res.curveQuoteFrac is not None
    ):
        from analysis.charting import plotTimVal

        outPath = Path(chartsRoot) / f"{label}-timVal.png"
        title = f"{ticker} {interval} - {label} timVal"
        plotTimVal(
            res.curveTs,
            res.curveSim,
            res.curveBench,
            res.curveAssetFrac,
            res.curveQuoteFrac,
            title,
            str(outPath),
            overrides,
        )
    return resultMetrics(label, ticker, res)


def runHoldout(
    bestCfg: Path,
    statsCfg: Path | None,
    extraCfgs: list[tuple[str, Path]] | None,
    holdoutDays: int,
    chartsRoot: Path | None,
    summaryPath: Path | None = None,
    anchorMs: int | None = None,
    startMinPct: int | None = None,
    startMaxPct: int | None = None,
    startStepPct: int | None = None,
) -> None:
    bestCfgObj = profile.loadJson(str(bestCfg))
    profile.ensureFinalPortionPct(bestCfgObj)
    startMinPct, startMaxPct, startStepPct = profile.holdoutStartParts(
        bestCfgObj,
        startMinPct,
        startMaxPct,
        startStepPct,
    )
    effectiveHoldoutDays = _effective_holdout_days(bestCfg, holdoutDays)
    if effectiveHoldoutDays <= 0:
        raise SystemExit("holdout start sweep requires holdout_days > 0")
    if startStepPct <= 0:
        raise SystemExit("holdout start step must be positive")
    if startMinPct < 0 or startMaxPct < startMinPct:
        raise SystemExit("holdout start pct range is invalid")
    startOffsets = _start_offsets(
        effectiveHoldoutDays,
        startMinPct,
        startMaxPct,
        startStepPct,
    )
    statsSameAsBest = False
    if statsCfg is not None and statsCfg.is_file():
        statsCfgObj = profile.loadJson(str(statsCfg))
        profile.ensureFinalPortionPct(statsCfgObj)
        statsSameAsBest = configsEqual(bestCfgObj, statsCfgObj)

    summaryParts: list[str] = []
    tableSeries = []
    runCache: dict = {}
    if statsSameAsBest:
        msg = "[holdout] stats == best; skipping duplicate run..."
        print(msg)
        summaryParts.append(msg + "\n")
    sweepText = _start_sweep_text(
        effectiveHoldoutDays,
        startMinPct,
        startMaxPct,
        startStepPct,
    )
    print(sweepText, end="")
    summaryParts.append(sweepText)
    bestRows = []
    for pct, offsetDays in startOffsets:
        label = _start_label("best", pct)
        best = _run_one(
            label,
            bestCfg,
            holdoutDays,
            chartsRoot,
            anchorMs=anchorMs,
            startOffsetDays=offsetDays,
            runCache=runCache,
        )
        bestRows.append(best)
    tableSeries.append(("best", bestRows))
    if (
        statsCfg is not None
        and statsCfg.is_file()
        and not statsSameAsBest
    ):
        statsRows = []
        for pct, offsetDays in startOffsets:
            label = _start_label("stats", pct)
            stats = _run_one(
                label,
                statsCfg,
                holdoutDays,
                chartsRoot,
                anchorMs=anchorMs,
                startOffsetDays=offsetDays,
                runCache=runCache,
            )
            statsRows.append(stats)
        tableSeries.append(("stats", statsRows))
    for extraLabel, extraPath in extraCfgs or []:
        if not extraPath.is_file():
            continue
        extraRows = []
        for pct, offsetDays in startOffsets:
            label = _start_label(extraLabel, pct)
            extra = _run_one(
                label,
                extraPath,
                holdoutDays,
                chartsRoot,
                anchorMs=anchorMs,
                startOffsetDays=offsetDays,
                runCache=runCache,
            )
            extraRows.append(extra)
        tableSeries.append((extraLabel, extraRows))
    tableText = holdoutTableText("holdout", tableSeries, BAR)
    print(tableText, end="")
    summaryParts.append(tableText)
    if summaryPath is not None:
        tmp = Path(f"{summaryPath}.tmp")
        tmp.write_text("".join(summaryParts))
        os.replace(tmp, summaryPath)


def parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="holdout",
        description="Run holdout traces for best and stats configs.",
    )
    parser.add_argument(
        "--best",
        required=True,
        help="Path to best-config.json",
    )
    parser.add_argument(
        "--stats",
        help="Path to beststats-config.json",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Extra holdout config as label=path",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=0,
        help="Override holdout window in days (optional)",
    )
    parser.add_argument(
        "--charts-root",
        default=None,
        help="Directory for holdout charts (subdirs per label)",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional path to write the holdout summary text",
    )
    parser.add_argument(
        "--anchor-ms",
        type=int,
        default=None,
        help="Optional UTC millisecond anchor for historical runs",
    )
    parser.add_argument(
        "--anchor-date",
        default=None,
        help="Optional UTC anchor date for historical runs (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--start-min-pct",
        type=int,
        default=None,
        help="First holdout start offset percent",
    )
    parser.add_argument(
        "--start-max-pct",
        type=int,
        default=None,
        help="Last holdout start offset percent",
    )
    parser.add_argument(
        "--start-step-pct",
        type=int,
        default=None,
        help="Holdout start offset percent step",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parseArgs(argv)
    bestPath = Path(args.best)
    statsPath = Path(args.stats) if args.stats else None
    extraCfgs = []
    for item in args.extra:
        label, rawPath = item.split("=", 1)
        extraCfgs.append((label, Path(rawPath)))
    chartsRoot = Path(args.charts_root) if args.charts_root else None
    summaryPath = Path(args.summary_path) if args.summary_path else None
    runHoldout(
        bestPath,
        statsPath,
        extraCfgs,
        args.days,
        chartsRoot,
        summaryPath=summaryPath,
        anchorMs=resolveAnchorMs(
            anchorMs=args.anchor_ms,
            anchorDate=args.anchor_date,
        ),
        startMinPct=args.start_min_pct,
        startMaxPct=args.start_max_pct,
        startStepPct=args.start_step_pct,
    )
    return 0


def traceHoldoutRun(runDir: Path) -> None:
    bestCfg, statsCfg, extraCfgs = selectedConfigPaths(runDir)
    chartsRoot = chartsHoldoutDir(runDir)
    chartsRoot.mkdir(parents=True, exist_ok=True)
    runHoldout(
        bestCfg,
        statsCfg,
        extraCfgs,
        0,
        chartsRoot,
        summaryPath=holdoutLogPath(runDir),
        anchorMs=anchorMsFromRun(runDir),
    )


if __name__ == "__main__":
    raise SystemExit(main())
