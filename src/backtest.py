#!/usr/bin/env python3
# backtest.py – backtest orchestration and CLI

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import json

from charting import Chart
from oracles import OracleEngine
from engine_shared import (
    buildContext,
    spacingState,
    bars_per_day,
)
import cache
from metrics import (
    equityCurveFromTrades,
    stepReturns,
    sharpeRatio,
    sortinoRatio,
    maxDrawdown,
    cagr,
    grossPctVsBench,
    rollingSharpeSortinoMedian,
)
from engine_shared import periods_per_year
from flags import (
    paramsFromSettings,
    generateFlags,
)
from engine_shared import buildSignals
from accounting import buildStatement, marginalIncomeTaxRate
from wallet import (
    simulateFromFlags,
    PHASE_BUY_PORTIONS_DEFAULT,
    PHASE_SELL_PORTIONS_DEFAULT,
)
from summary import printBacktestSummary
from charting import plotBacktestCharts
import profile
from params import overridesFromDict
from dynamics import macroDynFromContext, alignMacroDyn
from cache import getKlinesCached

# Local presentation and chart segmentation (non-JSON)
BAR = "=" * 50
# Approx. days per chart image; converted to bars at runtime.
CHART_CHUNK_SIZE = 30


@dataclass
class BacktestResult:
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
    curveTs: List[Any] | None = None
    curveSim: np.ndarray | None = None
    curveBench: np.ndarray | None = None

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
        }
        for key, attr in scalarMap.items():
            out[key] = getattr(self, attr)
        return out


class Backtest:
    """Run historical backtest and optionally chart + print summaries.

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
        doOracles=False,
        showCharts=False,
        showPrints=False,
        showSummary=True,
        overrides=None,
        ctx=None,
        signals=None,
        computeRisk=True,
        primerDays: int | None = None,
        holdoutDays: int = 0,
    ):
        self.ticker = ticker
        self.klines = klines
        self.intervalStr = intervalStr
        self.periods = periods
        self.days = days
        self.doOracles = doOracles
        self.showCharts = showCharts
        self.showPrints = showPrints
        self.showSummary = showSummary
        self.overrides = overrides or {}
        self.ctx = ctx
        self.signals = signals
        self.computeRisk = bool(computeRisk)
        self.primerDays = int(primerDays) if primerDays is not None else 0
        self.holdoutDays = int(holdoutDays) if holdoutDays is not None else 0
        self.result: Optional[BacktestResult] = None

    def _minCandles(self) -> int:
        return max(self.periods) * 2 + 1

    def _buildContext(self):
        return buildContext(self.klines, self.periods)

    def _ensureContext(self):
        if self.ctx is not None:
            self.ctx["klines"] = self.klines
            return self.ctx
        if self.days is None:
            self.ctx = self._buildContext()
        else:
            self.ctx = cache.getContext(
                self.ticker,
                self.intervalStr,
                self.days,
                self.periods,
                self.klines,
                self._buildContext,
            )
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

    def _maybeOracles(self, ctx, ts, params):
        if not self.doOracles:
            return []
        engine = OracleEngine(ctx, params)
        raw = engine.generate()
        return [(ts[i], lab) for i, lab in raw if i < len(ts)]

    def _buildSignals(self, ctx, lookbacks):
        return buildSignals(ctx, lookbacks)

    def _ensureSignals(self, ctx, lookbacks):
        if self.signals is not None:
            return self.signals
        if self.days is None:
            return self._buildSignals(ctx, lookbacks)
        return cache.getSignals(
            self.ticker,
            self.intervalStr,
            self.days,
            self.periods,
            lookbacks,
            self.klines,
            lambda: self._buildSignals(ctx, lookbacks),
        )

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
        intervalMacro = str(overrides['MACRO_INTERVAL']).strip()
        if not intervalMacro:
            return None
        winDays = float(overrides['MACRO_NRG_WIN_DAYS'])
        zmin = float(overrides['MACRO_NRG_Z_MIN'])
        zmax = float(overrides['MACRO_NRG_Z_MAX'])
        pctMax = float(overrides['MACRO_DYN_PCT_MAX'])
        macroP1 = int(overrides['MACRO_P1'])
        macroP2 = int(overrides['MACRO_P2'])
        macroP3 = int(overrides['MACRO_P3'])
        if self.days is None:
            return None
        daysInt = int(self.days)
        minCandles = self._minCandles()
        klinesMacro = getKlinesCached(
            self.ticker,
            str(intervalMacro),
            daysInt,
            minCandles,
            holdoutDays=self.holdoutDays,
        )
        periodsMacro = list(self.periods)
        if macroP1 is not None and int(macroP1) > 0 and len(periodsMacro) >= 1:
            periodsMacro[0] = int(macroP1)
        if macroP2 is not None and int(macroP2) > 0 and len(periodsMacro) >= 2:
            periodsMacro[1] = int(macroP2)
        if macroP3 is not None and int(macroP3) > 0:
            p3Macro = int(macroP3)
            if len(periodsMacro) >= 3:
                periodsMacro[2] = p3Macro
            else:
                periodsMacro.append(p3Macro)
        ctxMacro = cache.getContext(
            self.ticker,
            str(intervalMacro),
            daysInt,
            periodsMacro,
            klinesMacro,
            lambda: buildContext(klinesMacro, periodsMacro),
        )
        ctxMacro["intervalStr"] = str(intervalMacro)
        pctMin = float(overrides['MACRO_DYN_PCT_MIN'])
        gradWinDays = float(overrides['MACRO_GRAD_WIN_DAYS'])
        gradZMin = float(overrides['MACRO_GRAD_Z_MIN'])
        gradZMax = float(overrides['MACRO_GRAD_Z_MAX'])
        gradMultMin = float(overrides['MACRO_MULT_GRAD_MIN'])
        gradMultMax = float(overrides['MACRO_MULT_GRAD_MAX'])
        arrMacro = macroDynFromContext(
            ctxMacro,
            winDays,
            zmin,
            zmax,
            pctMax,
            pctMin,
            gradWinDays=gradWinDays,
            gradZMin=gradZMin,
            gradZMax=gradZMax,
            gradMultMin=gradMultMin,
            gradMultMax=gradMultMax,
        )
        mas = ctxMacro.get("mas", [])
        m1 = np.asarray(mas[0], dtype=float)
        m2 = np.asarray(mas[1], dtype=float)
        m3 = np.asarray(mas[2], dtype=float)
        macroDir = np.zeros_like(m1, dtype=int)
        macroDir[m1 > m3] = 1
        macroDir[m1 < m3] = -1
        macroMom = np.zeros_like(m1, dtype=int)
        macroMom[m1 > m2] = 1
        macroMom[m1 < m2] = -1
        tsMacro = pd.to_datetime(
            [k[0] for k in klinesMacro],
            unit="ms",
            utc=True,
        )
        tsMacro = tsMacro.tz_convert(None).to_pydatetime().tolist()
        if not tsMacro:
            return None
        dyn = alignMacroDyn(tsMacro, arrMacro, ts)
        dirAligned = alignMacroDyn(
            tsMacro, macroDir.astype(float), ts
        ).astype(int)
        momAligned = alignMacroDyn(
            tsMacro, macroMom.astype(float), ts
        ).astype(int)
        return dyn, dirAligned, momAligned

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
        return idxFlags, [(ts[i], lab) for i, lab in idxFlags]

    def _walletOverrides(
        self,
    ) -> Tuple[float, str, float, Optional[str], float, float, float]:
        seed = float(self.overrides['WALLET_SEED_QUOTE'])
        taxMode = str(self.overrides['TAX_MODE']).lower()
        incomeBase = float(self.overrides['ANNUAL_INCOME_BASE'])
        rawInterval = str(self.overrides['PROFIT_SWEEP_INTERVAL']).lower()
        sweepInterval = (
            None if rawInterval in ('', 'none') else rawInterval
        )
        sweepShare = float(self.overrides['PROFIT_SWEEP_SHARE'])
        audRate = float(self.overrides['QUOTE_TO_AUD_RATE'])
        finalPortionPct = float(self.overrides['FINAL_PORTION_PCT'])
        return (
            seed,
            taxMode,
            incomeBase,
            sweepInterval,
            sweepShare,
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
        sweepInterval: Optional[str],
        sweepShare: float,
        seed: float,
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
            seedIndex=startIdx,
            doPrints=self.showPrints,
            phaseBuyPortions=int(self.overrides['PHASE_BUY_PORTIONS']),
            phaseSellPortions=int(self.overrides['PHASE_SELL_PORTIONS']),
            taxMode=taxMode,
            annualIncomeBase=incomeBase,
            profitSweepInterval=sweepInterval,
            profitSweepShare=sweepShare,
            finalPortionPct=finalPortionPct,
            trendCodes=trendCodeArr,
        )
        from wallet import Wallet
        return walletTrading, Wallet(
            baseSymbol=baseSymbol,
            startingCash=seed,
            feeRate=float(self.overrides['WALLET_FEE_RATE']),
            taxRate=taxRate,
            taxMode=taxMode,
            annualIncomeBase=incomeBase,
            profitSweepInterval=sweepInterval,
            profitSweepShare=sweepShare,
        )

    def _walletMarkers(self, walletTrading) -> List[Tuple[Any, str]]:
        out = []
        for tr in walletTrading.trades:
            if tr.side == "BUY":
                out.append((tr.ts, "W_BUY"))
            elif tr.side == "SELL":
                out.append((tr.ts, "W_SELL"))
        return out

    def _printBest(self, sim, ben, label: str) -> None:
        hdr = f"{label}:"
        indent = ' ' * 11
        grossPct = grossPctVsBench(
            float(sim['portfolio_value']),
            float(ben['portfolio_value']),
        )
        print(f"{hdr:<11}GrossVsHODL  : {grossPct:+.2f}% ")
        print(f"{indent}Trades       : {sim['trades']}")

    def _printStats(self, sim, ben, sharpe: float, sortino: float, mdd: float, cagr_v: float) -> None:
        hdr = "BESTSTATS:"
        indent = ' ' * 11
        grossPct = grossPctVsBench(
            float(sim['portfolio_value']),
            float(ben['portfolio_value']),
        )
        print(f"{hdr:<11}GrossVsHODL  : {grossPct:+.2f}% ")
        print(f"{indent}Trades       : {sim['trades']}")
        print(f"{indent}Sharpe       : {sharpe:.3f} ")
        print(f"{indent}Sortino      : {sortino:.3f} ")
        print(f"{indent}MDD          : {mdd*100.0:.2f}% ")
        print(f"{indent}CAGR         : {cagr_v*100.0:.2f}%")

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
        sharpe: float,
        sortino: float,
        mdd: float,
        cagr_v: float,
        sharpe1w: float,
        sortino1w: float,
        sharpe4w: float,
        sortino4w: float,
        sharpe13w: float,
        sortino13w: float,
        sharpe1wAbs: float,
        sortino1wAbs: float,
        sharpe4wAbs: float,
        sortino4wAbs: float,
        sharpe13wAbs: float,
        sortino13wAbs: float,
    ) -> None:
        if not self.showSummary:
            return
        # Optional label (e.g., A/B/C) from overrides
        labelStr = str(self.overrides['SUMMARY_LABEL']).strip()

        # Compact modes for BEST/STATS
        if labelStr.upper() == 'BEST':
            self._printBest(sim, ben, 'BEST')
            return
        if len(labelStr) == 1 and labelStr.isalpha():
            self._printBest(sim, ben, f"BEST{labelStr.upper()}")
            return
        if labelStr.upper() == 'STATS':
            self._printStats(sim, ben, sharpe, sortino, mdd, cagr_v)
            return

        printBacktestSummary(
            ticker=self.ticker,
            ts=ts,
            startIdx=startIdx,
            endDt=endDt,
            lastPrice=lastPrice,
            sim=sim,
            ben=ben,
            simStatement=simStatement,
            benchStatement=benchStatement,
            labelStr=labelStr,
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
        oracles: List[Tuple[Any, str]],
        signals: dict | None,
    ) -> None:
        plotBacktestCharts(
            self.showCharts,
            ctx,
            ts,
            startIdx,
            flagsTs,
            walletMarkers,
            oracles,
            signals,
            overridesFromDict(self.overrides),
            self.klines,
            self.ticker,
            self.intervalStr,
        )

    def run(self) -> BacktestResult:
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
        oracles = self._maybeOracles(ctx, ts, params)
        flagsIdx, flagsTs = self._generateFlags(
            ctx,
            ts,
            startIdx,
            params,
            signals,
        )

        (seed, taxMode, incomeBase, sweepInterval,
         sweepShare, audRate, finalPortionPct) = self._walletOverrides()
        trendArr = np.asarray(signals["trendCode"], dtype=int)
        walletTrading, walletBench = self._simulateWallets(
            ctx,
            flagsIdx,
            startIdx,
            taxMode,
            incomeBase,
            sweepInterval,
            sweepShare,
            seed,
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
            sharpe_v,
            sortino_v,
            mdd_v,
            cagr_v,
            sharpe1w_v,
            sortino1w_v,
            sharpe4w_v,
            sortino4w_v,
            sharpe13w_v,
            sortino13w_v,
            sharpe1wAbs_v,
            sortino1wAbs_v,
            sharpe4wAbs_v,
            sortino4wAbs_v,
            sharpe13wAbs_v,
            sortino13wAbs_v,
        )

        self._charts(
            ctx,
            ts,
            startIdx,
            flagsTs,
            walletMarkers,
            oracles,
            signals,
        )

        self.result = BacktestResult(
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
            sharpe1w=sharpe1w_v,
            sortino1w=sortino1w_v,
            curveTs=curveTs,
            curveSim=curveSim,
            curveBench=curveBench,
        )
        return self.result
