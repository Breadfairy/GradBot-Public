#!/usr/bin/env python3
# summary.py - trace summary printing helpers

from __future__ import annotations

from typing import Any, Dict, List, Optional

from portfolio.accounting import calcIncomeTax, bracketForIncome
from analysis.metrics import (
    equityCurveFromTrades,
    stepReturns,
    maxDrawdown,
    cagr,
    grossPctVsBench,
    rollingSharpeSortinoMedian,
)
from engine.shared import periods_per_year, bars_per_day


BAR = "=" * 50

def _durations(ts, startIdx, endDt):
    startDt = ts[startIdx]
    durationSeconds = max((endDt - startDt).total_seconds(), 1.0)
    durationDaysFloat = max(durationSeconds / 86400.0, 1.0)
    durationDays = max(1, int(durationSeconds // 86400))
    rawSeconds = max((ts[-1] - ts[0]).total_seconds(), 1.0)
    rawDays = max(1, int(rawSeconds // 86400))
    durationYears = durationDaysFloat / 365.0
    return durationDays, rawDays, durationYears

def _printHeader(
    labelStr: str,
    seed: float,
    quote: str,
    durationDays: int,
    rawDays: int,
    startDt,
    rawStartDt,
    endDt,
    lastPrice: float,
) -> None:
    title = f"Portfolio {labelStr} Summary" if labelStr else "Portfolio Summary"
    print(f"\n{BAR}")
    print(f"=== {title} – {seed:.0f} {quote}, {durationDays} days (raw {rawDays}) ===")
    print(BAR)
    print(
        f"Active bounds: {startDt:%Y-%m-%d %H:%M} UTC"
        f" -> {endDt:%Y-%m-%d %H:%M} UTC"
    )
    print(
        f"Raw bounds   : {rawStartDt:%Y-%m-%d %H:%M} UTC"
        f" -> {endDt:%Y-%m-%d %H:%M} UTC"
    )
    print(f"Snapshot at end: {endDt:%Y-%m-%d %H:%M} UTC @ {lastPrice:.4f}")


def printTaxAssessmentAnnual(
    baseAnnualAud: float,
    tradingAnnualAud: float,
    audSymbol: str = "AUD",
) -> None:
    combinedAnnual = baseAnnualAud + tradingAnnualAud
    combinedTax = calcIncomeTax(combinedAnnual)
    netIncome = combinedAnnual - combinedTax
    threshold, bracketBase, rate = bracketForIncome(combinedAnnual)
    ratePct = rate * 100.0

    print("\n" + BAR)
    print("=== Tax Assessment – Per Year Estimate ====")
    print(BAR)
    print(f"  Base Income       : {baseAnnualAud:.2f} {audSymbol}")
    print(f"  Trading Income    : {tradingAnnualAud:.2f} {audSymbol}")
    print(f"  TOTAL             : {combinedAnnual:.2f} {audSymbol}")
    print()
    print(f"  Tax Base          : {bracketBase:.2f} {audSymbol}")
    if rate <= 0.0:
        print("  Tax Rate          : within tax-free threshold")
    else:
        print(
            "  Tax Rate          : "
            f"{ratePct:.2f}% per dollar over "
            f"{threshold:,.0f} {audSymbol}"
        )
    print(f"  Income Tax        : {combinedTax:.2f} {audSymbol}")
    print("  ------------------------------------------------")
    print(f"  Net Income        : {netIncome:.2f} {audSymbol}")
    print("  ================================================")


# ======================================================================
# Trace summary wrapper
# ======================================================================


def printTraceSummary(
    ticker: str,
    ts,
    startIdx: int,
    endDt,
    lastPrice: float,
    sim: Dict[str, Any],
    ben: Dict[str, Any],
    simStatement: Dict[str, Any],
    benchStatement: Dict[str, Any],
    labelStr: str,
    seed: float,
    incomeBase: float,
    audRate: float,
    simPostTaxValue: float,
    benchPostTaxValue: float,
    netPctVsHodl: float,
    walletTrading,
    ctx,
    intervalStr: str,
    taxMode: str,
    walletBench,
) -> None:
    if str(taxMode).lower() == "income":
        printIncomeSummary(
            ticker=ticker,
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
            audRate=audRate,
            netPctVsHodl=netPctVsHodl,
            ctx=ctx,
            walletTrading=walletTrading,
            walletBench=walletBench,
            intervalStr=intervalStr,
        )
    else:
        printCgtSummary(
            ticker=ticker,
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
            audRate=audRate,
            simPostTaxValue=simPostTaxValue,
            benchPostTaxValue=benchPostTaxValue,
            netPctVsHodl=netPctVsHodl,
            walletTrading=walletTrading,
            ctx=ctx,
            walletBench=walletBench,
            intervalStr=intervalStr,
        )


def _printPortfolioBlock(
    ticker: str,
    sim: Dict[str, Any],
    asset: str,
    quote: str,
    lockedLine: Optional[str],
    taxOwedLine: str,
    taxModeLine: str,
) -> None:
    print("Trading Portfolio:")
    print(f"  Ticker   : {ticker}")
    print(f"  Cash     : {sim['cash']:.2f} {quote}")
    print(
        f"  Asset    : {sim['asset_qty']:.6f} {asset} "
        f"@ {sim['asset_price']:.4f}"
    )
    print(f"  Realized : {sim['realized_gain']:.2f} {quote}")
    if lockedLine:
        print(lockedLine)
    print(
        f"  Gross Val: {sim['portfolio_value']:.2f} "
        f"{quote}"
    )
    print(f"  Trades   : {sim['trades']}")
    print(taxOwedLine)
    print(f"  Fees Paid: {sim['fees_paid_quote']:.2f} {quote}")
    print(f"  Tax Mode : {taxModeLine}")


def _printBenchBlock(ben: Dict[str, Any], ticker: str, asset: str, quote: str) -> None:
    print("\nBenchmark (Buy-and-Hold):")
    print(f"  Ticker  : {ticker}")
    print(f"  Cash    : {ben['cash']:.2f} {quote}")
    print(
        f"  Asset   : {ben['asset_qty']:.6f} {asset} "
        f"@ {ben['asset_price']:.4f}"
    )
    print(
        f"  Gross Val: {ben['portfolio_value']:.2f} "
        f"{quote}"
    )


def _formatLocked(summaryDict: Dict[str, Any], quoteSymbol: str) -> Optional[str]:
    locked = summaryDict.get("locked_profit", 0.0)
    if locked <= 1e-6:
        return None
    share = summaryDict.get("profit_sweep_share", 0.0) * 100.0
    interval = summaryDict.get("profit_sweep_interval", "")
    suffix = " (pre-tax)"
    if interval:
        cadence = {
            "month": "monthly",
            "week": "weekly",
            "day": "daily",
        }.get(interval, f"per {interval}")
        if share > 0.0:
            suffix = f" (pre-tax, {share:.0f}% {cadence})"
        else:
            suffix = f" (pre-tax, {cadence})"
    return f"  Locked   : {locked:.2f} {quoteSymbol}{suffix}"


def _printRiskMetrics(
    ctx,
    walletTrading,
    walletBench,
    startIdx: int,
    durationYears: float,
    intervalStr: str,
    seedQuote: float,
) -> None:
    closes = ctx["closes"]
    curveSim = equityCurveFromTrades(
        closes=closes,
        trades=walletTrading.trades,
        startIndex=startIdx,
        seedQuote=seedQuote,
    )
    curveBench = equityCurveFromTrades(
        closes=closes,
        trades=walletBench.trades,
        startIndex=startIdx,
        seedQuote=seedQuote,
    )
    retsSim = stepReturns(curveSim)
    retsBench = stepReturns(curveBench)
    n = min(len(retsSim), len(retsBench))
    edgeRets = retsSim[:n] - retsBench[:n] if n > 0 else retsSim

    ppy = periods_per_year(intervalStr)
    mdd = maxDrawdown(curveSim)
    growth = cagr(curveSim, durationYears)
    mar = (
        growth / mdd
        if isinstance(growth, (int, float))
        and isinstance(mdd, (int, float))
        and mdd > 0.0
        else float('nan')
    )

    bpd = bars_per_day(ctx)
    win1 = max(int(round(7.0 * bpd)), 2)
    win4 = max(int(round(28.0 * bpd)), 2)
    win13 = max(int(round(91.0 * bpd)), 2)
    sharpe1wAbs, sortino1wAbs = rollingSharpeSortinoMedian(
        retsSim,
        ppy,
        win1,
    )
    sharpe4wAbs, sortino4wAbs = rollingSharpeSortinoMedian(
        retsSim,
        ppy,
        win4,
    )
    sharpe13wAbs, sortino13wAbs = rollingSharpeSortinoMedian(
        retsSim,
        ppy,
        win13,
    )
    sharpe1wRel, sortino1wRel = rollingSharpeSortinoMedian(
        edgeRets,
        ppy,
        win1,
    )
    sharpe4wRel, sortino4wRel = rollingSharpeSortinoMedian(
        edgeRets,
        ppy,
        win4,
    )
    sharpe13wRel, sortino13wRel = rollingSharpeSortinoMedian(
        edgeRets,
        ppy,
        win13,
    )

    print("\n" + BAR)
    print("=== Risk Assessment ===")
    print(BAR)
    print(
        f"  Sharpe 1w   (abs/rel): "
        f"{sharpe1wAbs:.3f} / {sharpe1wRel:.3f}"
    )
    print(
        f"  Sharpe 4w   (abs/rel): "
        f"{sharpe4wAbs:.3f} / {sharpe4wRel:.3f}"
    )
    print(
        f"  Sharpe 13w  (abs/rel): "
        f"{sharpe13wAbs:.3f} / {sharpe13wRel:.3f}"
    )
    print(
        f"  Sortino 4w  (abs/rel): "
        f"{sortino4wAbs:.3f} / {sortino4wRel:.3f}"
    )
    print(
        f"  Sortino 13w (abs/rel): "
        f"{sortino13wAbs:.3f} / {sortino13wRel:.3f}"
    )
    print(f"  MDD    : {mdd*100.0:.2f}%")
    print(f"  MAR    : {mar:.3f}")
    print(f"  CAGR   : {growth*100.0:.2f}%")


def printIncomeSummary(
    ticker: str,
    ts,
    startIdx: int,
    endDt,
    lastPrice: float,
    sim: Dict[str, Any],
    ben: Dict[str, Any],
    simStatement: Dict[str, Any],
    benchStatement: Dict[str, Any],
    labelStr: str,
    seed: float,
    incomeBase: float,
    audRate: float,
    netPctVsHodl: float,
    ctx,
    walletTrading,
    walletBench,
    intervalStr: str,
) -> None:
    durationDays, rawDays, durationYears = _durations(ts, startIdx, endDt)
    annualFactor = 1.0 / durationYears if durationYears > 1e-9 else 1.0

    quote = sim.get("quote", "USD")
    asset = sim.get("base", ticker)
    lockedLine = _formatLocked(sim, quote)

    grossPctVsHodl = grossPctVsBench(
        float(sim["portfolio_value"]),
        float(ben["portfolio_value"]),
    )

    grossPnL = simStatement["gross_profit"]
    # Income base is already an annual figure
    baseAnnual = incomeBase
    tradingTotal = simStatement["potential_profit"]
    tradingAnnualGrossQuote = tradingTotal * annualFactor
    tradingAnnualGrossAud = tradingAnnualGrossQuote * audRate
    _printHeader(
        labelStr,
        seed,
        quote,
        durationDays,
        rawDays,
        ts[startIdx],
        ts[0],
        endDt,
        lastPrice,
    )
    _printPortfolioBlock(
        ticker=ticker,
        sim=sim,
        asset=asset,
        quote=quote,
        lockedLine=lockedLine,
        taxOwedLine=f"  Tax Owed : n/a (income mode)",
        taxModeLine=f"income (base {incomeBase:.2f})",
    )
    print(f"  P&L (gross)    : {grossPnL:.2f} {quote}")
    _printBenchBlock(ben, ticker, asset, quote)
    print("\nGross Values Comparison:")
    print(f"  Trading : {sim['portfolio_value']:.2f} {quote}")
    print(f"  HODL    : {ben['portfolio_value']:.2f} {quote}")
    print(f"  Gross vs HODL: {grossPctVsHodl:+.2f}%")
    _printRiskMetrics(
        ctx,
        walletTrading,
        walletBench,
        startIdx,
        durationYears,
        intervalStr,
        seed,
    )
    printTaxAssessmentAnnual(baseAnnual, tradingAnnualGrossAud, "AUD")
    print(BAR)


def printCgtSummary(
    ticker: str,
    ts,
    startIdx: int,
    endDt,
    lastPrice: float,
    sim: Dict[str, Any],
    ben: Dict[str, Any],
    simStatement: Dict[str, Any],
    benchStatement: Dict[str, Any],
    labelStr: str,
    seed: float,
    incomeBase: float,
    audRate: float,
    simPostTaxValue: float,
    benchPostTaxValue: float,
    netPctVsHodl: float,
    walletTrading,
    ctx,
    walletBench,
    intervalStr: str,
) -> None:
    durationDays, rawDays, durationYears = _durations(ts, startIdx, endDt)
    annualFactor = 1.0 / durationYears if durationYears > 1e-9 else 1.0

    quote = sim.get("quote", "USD")
    asset = sim.get("base", ticker)
    lockedLine = _formatLocked(sim, quote)

    grossPctVsHodl = grossPctVsBench(
        float(sim["portfolio_value"]),
        float(ben["portfolio_value"]),
    )

    grossPnL = simStatement["gross_profit"]
    tradingTax = simStatement.get("trading_tax", 0.0)
    netProfit = simStatement.get(
        "net_after_tax", simStatement["potential_profit"]
    )
    effRate = simStatement.get("effective_tax_rate", 0.0)
    tradingAnnualNetQuote = netProfit * annualFactor
    tradingAnnualNetAud = tradingAnnualNetQuote * audRate

    _printHeader(
        labelStr,
        seed,
        quote,
        durationDays,
        rawDays,
        ts[startIdx],
        ts[0],
        endDt,
        lastPrice,
    )
    _printPortfolioBlock(
        ticker=ticker,
        sim=sim,
        asset=asset,
        quote=quote,
        lockedLine=lockedLine,
        taxOwedLine=f"  Tax Owed : {tradingTax:.2f} {quote}",
        taxModeLine=(
            f"cgt (rate {walletTrading.taxRate*100.0:.1f}%, "
            f"discount {walletTrading.discountRate*100.0:.1f}%)"
        ),
    )
    print(f"  P&L (gross)    : {grossPnL:.2f} {quote}")
    print(f"  P&L (after tax): {netProfit:.2f} {quote}")
    print(f"  Eff. CGT rate  : {effRate*100.0:.2f}%")
    _printBenchBlock(ben, ticker, asset, quote)
    print("\nNet Values Comparison:")
    print(f"  Trading : {simPostTaxValue:.2f} {quote}")
    print(f"  HODL    : {benchPostTaxValue:.2f} {quote}")
    print(f"  Net vs HODL: {netPctVsHodl:+.2f}%")
    print(f"  Gross vs HODL: {grossPctVsHodl:+.2f}%")
    _printRiskMetrics(
        ctx,
        walletTrading,
        walletBench,
        startIdx,
        durationYears,
        intervalStr,
        seed,
    )
    printTaxAssessmentAnnual(
        baseAnnualAud=incomeBase,
        tradingAnnualAud=tradingAnnualNetAud,
        audSymbol="AUD",
    )
    print(BAR)
