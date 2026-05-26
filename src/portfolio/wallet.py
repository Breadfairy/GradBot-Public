#!/usr/bin/env python3
"""
wallet.py - Simple causal wallet simulator for traces and live parity.

Goals:
- Mirror a Binance-style spot wallet for one pair (e.g., LINK/USDT).
- Consume BUY/SELL flags in order with causal sizing decisions.
- Track per-trade telemetry, fees, and CGT-style tax estimates.

Notes:
- Fees are deducted from the quote balance on both BUY and SELL.
- CGT uses FIFO lots; gains held >= `discountDays` receive the configured
  discount before the marginal tax rate is applied.
- Tax accrues as a liability and is not removed from cash balances.
- Phase sizing:
  - BEAR: base buy value = cashAtPhaseStart / PHASE_BUY_PORTIONS.
  - BULL: base sell value = holdingsAtPhaseStart / PHASE_SELL_PORTIONS.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

from engine.shared import bars_per_day, trend
from portfolio.accounting import FIXED_INCOME_BASE, calcIncomeTax
from runtime.posture_feed import (
    dailyPostureArrays,
    dailyPostureState,
    dailyPostureStats,
    dailyPostureStep,
)
from strategy.execution import (
    buySpend as strategyBuySpend,
    buySpendToTargetCap as strategyBuySpendToTargetCap,
    calcBuyScale,
    calcSellScale,
    floorSellValueCap as strategyFloorSellValueCap,
    phaseBuyValue,
    phaseSellValue,
    sellQty as strategySellQty,
)
from strategy.supervisor import (
    armPeakLock,
    evaluatePeakLock,
    peakLockConfigFromOverrides,
    peakLockState,
    peakLockStats,
    recordPeakCappedBuy,
    recordPeakLock,
    stepActivePeakLock,
    stepPeakPid,
    stepPeakStrong,
    warmPeakLockState,
)
from strategy.signals import trendLabel as strategyTrendLabel

# Shared phase sizing defaults
PHASE_BUY_PORTIONS_DEFAULT = 10
PHASE_SELL_PORTIONS_DEFAULT = 10


@dataclass
class Lot:
    qty: float
    costPerUnit: float  # includes buy fee in cost base per unit
    ts: datetime          # acquisition time for CGT holding period
    index: int            # candle index (for traceability)


@dataclass
class Trade:
    index: int
    ts: datetime
    side: str            # "BUY" or "SELL"
    price: float
    qty: float
    notional: float      # qty * price
    feeRate: float
    feeAmount: float    # charged in quote currency
    cashDelta: float    # + for sell net proceeds, - for buy spend (incl. fee)
    baseDelta: float    # +qty for buy, -qty for sell
    realizedGain: float # only for SELL; pre-tax
    taxableGain: float  # only for SELL; after discount; >= 0
    taxApplied: float   # only for SELL; taxableGain * taxRate
    taxYear: Optional[int] = None
    note: str = ""


class Wallet:
    def __init__(
        self,
        baseSymbol: str,
        quoteSymbol: str = "USDT",
        startingCash: float = 10000.0,
        feeRate: float = 0.001,  # 0.1%
        taxRate: float = 0.30,    # marginal tax rate (0..1)
        discountDays: int = 365,
        discountRate: float = 0.50,  # 50% CGT discount (individuals)
        taxMode: str = "cgt",
        annualIncomeBase: float = FIXED_INCOME_BASE,
        taxYearStartMonth: int = 7,
    ):
        self.baseSymbol = baseSymbol
        self.quoteSymbol = quoteSymbol
        self.quoteBalance = float(startingCash)
        self.baseBalance = 0.0
        self.feeRate = float(feeRate)
        self.taxRate = float(taxRate)
        self.discountDays = int(discountDays)
        self.discountRate = float(discountRate)
        self.taxMode = str(taxMode).lower()
        self.annualIncomeBase = float(annualIncomeBase)
        month = int(taxYearStartMonth)
        if month < 1:
            month = 1
        elif month > 12:
            month = 12
        self.taxYearStartMonth = month

        self.lots: List[Lot] = []
        self.trades: List[Trade] = []

        self.feesPaidQuote = 0.0
        self.realizedGain = 0.0
        self.taxLiability = 0.0
        self.baseIncomeTax = (
            calcIncomeTax(self.annualIncomeBase)
            if self.taxMode == "income" else 0.0
        )
        self.incomeRealized = defaultdict(float)
        self.incomeTaxAccrued = defaultdict(float)
        self.lastTradeTs: Optional[datetime] = None

    # ---------- helpers -------------------------------------------------
    def _toDt(self, ms: int) -> datetime:
        return datetime.utcfromtimestamp(ms / 1000.0)

    def _taxYearForDate(self, dt: datetime) -> int:
        startMonth = self.taxYearStartMonth
        return dt.year + 1 if dt.month >= startMonth else dt.year

    def _recordIncomeGain(self, ts: datetime, gain: float) -> Tuple[float, int]:
        year = self._taxYearForDate(ts)
        self.incomeRealized[year] += gain
        totalIncome = self.annualIncomeBase + self.incomeRealized[year]
        totalTax = calcIncomeTax(totalIncome)
        tradingTax = max(0.0, totalTax - self.baseIncomeTax)
        previous = self.incomeTaxAccrued[year]
        delta = tradingTax - previous
        self.incomeTaxAccrued[year] = tradingTax
        return delta, year

    def _applyBuy(
        self,
        index: int,
        ts: datetime,
        price: float,
        spendQuote: Optional[float] = None,
    ) -> Optional[Trade]:
        spend = (
            self.quoteBalance
            if spendQuote is None
            else min(spendQuote, self.quoteBalance)
        )
        if spend <= 0:
            return None
        fee = spend * self.feeRate
        netQuote = spend - fee
        if netQuote <= 0:
            return None
        qty = netQuote / price
        if qty <= 0:
            return None

        # Update balances
        self.quoteBalance -= spend
        self.baseBalance += qty
        self.feesPaidQuote += fee

        # Cost base includes the buy fee (incidental costs)
        costPerUnit = spend / qty
        self.lots.append(
            Lot(qty=qty, costPerUnit=costPerUnit, ts=ts, index=index)
        )

        tr = Trade(
            index=index, ts=ts, side="BUY", price=price, qty=qty,
            notional=qty * price, feeRate=self.feeRate, feeAmount=fee,
            cashDelta=-spend, baseDelta=qty, realizedGain=0.0,
            taxableGain=0.0, taxApplied=0.0,
        )
        self.trades.append(tr)
        self.lastTradeTs = ts
        return tr

    def _applySell(
        self,
        index: int,
        ts: datetime,
        price: float,
        qty: Optional[float] = None,
    ) -> Optional[Trade]:
        sellQty = (
            self.baseBalance if qty is None
            else min(qty, self.baseBalance)
        )
        if sellQty <= 0:
            return None

        gross = sellQty * price
        fee = gross * self.feeRate
        netProceeds = gross - fee

        # Compute FIFO cost base and CGT per-lot
        remaining = sellQty
        lotIdx = 0
        totalCost = 0.0
        realizedGain = 0.0
        taxableGain = 0.0
        while remaining > 1e-12 and lotIdx < len(self.lots):
            lot = self.lots[lotIdx]
            use = min(remaining, lot.qty)
            if use <= 0:
                lotIdx += 1
                continue
            # Allocate a proportional share of the sell fee to this lot
            feeAlloc = fee * (use / sellQty)
            proceedsLot = use * price - feeAlloc
            costLot = use * lot.costPerUnit
            gainLot = proceedsLot - costLot
            realizedGain += gainLot
            totalCost += costLot
            if self.taxMode == "income":
                taxableGain += gainLot
            else:
                holdDays = (ts - lot.ts).days
                if gainLot > 0 and holdDays >= self.discountDays:
                    taxableGain += gainLot * (1.0 - self.discountRate)
                elif gainLot > 0:
                    taxableGain += gainLot
                else:
                    taxableGain += gainLot

            lot.qty -= use
            remaining -= use
            if lot.qty <= 1e-12:
                lotIdx += 1

        # Remove depleted lots
        self.lots = [l for l in self.lots if l.qty > 1e-12]

        tax = taxableGain * self.taxRate if taxableGain > 0 else 0.0

        # Update balances
        self.baseBalance -= sellQty
        self.quoteBalance += netProceeds
        self.feesPaidQuote += fee
        self.realizedGain += realizedGain

        taxYear: Optional[int] = None
        if self.taxMode == "income":
            taxAmount, taxYear = self._recordIncomeGain(ts, realizedGain)
            tax = taxAmount
            taxableGain = realizedGain
        else:
            tax = taxableGain * self.taxRate if taxableGain > 0 else 0.0
        self.taxLiability += tax

        tr = Trade(
            index=index, ts=ts, side="SELL", price=price, qty=sellQty,
            notional=sellQty * price, feeRate=self.feeRate, feeAmount=fee,
            cashDelta=netProceeds, baseDelta=-sellQty,
            realizedGain=realizedGain, taxableGain=taxableGain,
            taxApplied=tax, taxYear=taxYear,
        )
        self.trades.append(tr)
        self.lastTradeTs = ts
        return tr

    # ---------- public API ---------------------------------------------
    def buyAll(
        self,
        index: int,
        ts: datetime,
        price: float,
    ) -> Optional[Trade]:
        return self._applyBuy(index, ts, price, spendQuote=None)

    def sellAll(
        self,
        index: int,
        ts: datetime,
        price: float,
    ) -> Optional[Trade]:
        return self._applySell(index, ts, price, qty=None)

    def portfolioValue(self, price: float) -> float:
        return self.quoteBalance + self.baseBalance * price

    def _estimateLiquidation(
        self,
        price: float,
        ts: Optional[datetime],
    ) -> Dict[str, float]:
        qty = self.baseBalance
        existingTax = self.taxLiability
        if qty <= 1e-12 or price <= 0.0:
            netCash = self.quoteBalance
            return {
                "sell_qty": 0.0,
                "gross_proceeds": 0.0,
                "fee": 0.0,
                "net_proceeds": 0.0,
                "realized_gain": 0.0,
                "taxable_gain": 0.0,
                "tax": 0.0,
                "net_cash": netCash,
                "total_tax": existingTax,
            }
        fee = qty * price * self.feeRate
        netProceeds = qty * price - fee
        lotsCopy = [
            Lot(qty=lot.qty, costPerUnit=lot.costPerUnit,
                ts=lot.ts, index=lot.index)
            for lot in self.lots
        ]
        remaining = qty
        realizedGain = 0.0
        taxableGain = 0.0
        refTs = ts or self.lastTradeTs or (
            lotsCopy[-1].ts if lotsCopy else datetime.utcfromtimestamp(0)
        )
        for lot in lotsCopy:
            if remaining <= 1e-12:
                break
            use = min(remaining, lot.qty)
            if use <= 0:
                continue
            feeAlloc = fee * (use / qty)
            proceedsLot = use * price - feeAlloc
            costLot = use * lot.costPerUnit
            gainLot = proceedsLot - costLot
            realizedGain += gainLot
            if self.taxMode == "income":
                taxableGain += gainLot
            else:
                holdDays = (refTs - lot.ts).days if refTs and lot.ts else 0
                if gainLot > 0 and holdDays >= self.discountDays:
                    taxableGain += gainLot * (1.0 - self.discountRate)
                elif gainLot > 0:
                    taxableGain += gainLot
                else:
                    taxableGain += gainLot
            remaining -= use
        if self.taxMode == "income":
            tax = 0.0
        else:
            tax = taxableGain * self.taxRate if taxableGain > 0 else 0.0
        return {
            "sell_qty": qty,
            "gross_proceeds": qty * price,
            "fee": fee,
            "net_proceeds": netProceeds,
            "realized_gain": realizedGain,
            "taxable_gain": taxableGain,
            "tax": tax,
            "net_cash": self.quoteBalance + netProceeds,
            "total_tax": existingTax + tax,
            "additional_tax": tax,
        }

    def summary(
        self,
        currentPrice: float,
        currentTs: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        incomeTaxYears = {
            str(year): round(value, 6)
            for year, value in self.incomeTaxAccrued.items()
            if value or self.taxMode == "income"
        }
        summary = {
            "base": self.baseSymbol,
            "quote": self.quoteSymbol,
            "cash": round(self.quoteBalance, 6),
            "asset_qty": round(self.baseBalance, 6),
            "asset_price": round(currentPrice, 6),
            "portfolio_value": round(self.portfolioValue(currentPrice), 6),
            "fees_paid_quote": round(self.feesPaidQuote, 6),
            "realized_gain": round(self.realizedGain, 6),
            "tax_liability": round(self.taxLiability, 6),
            "trades": len(self.trades),
            "tax_mode": self.taxMode,
        }
        if self.taxMode == "income":
            summary["annual_income_base"] = round(self.annualIncomeBase, 6)
            if incomeTaxYears:
                summary["income_tax_years"] = incomeTaxYears
        liquidation = self._estimateLiquidation(
            currentPrice,
            currentTs,
        )
        summary["liquidation"] = {
            "sell_qty": round(liquidation["sell_qty"], 6),
            "gross_proceeds": round(liquidation["gross_proceeds"], 6),
            "fee": round(liquidation["fee"], 6),
            "net_proceeds": round(liquidation["net_proceeds"], 6),
            "realized_gain": round(liquidation["realized_gain"], 6),
            "taxable_gain": round(liquidation["taxable_gain"], 6),
            "tax": round(liquidation["tax"], 6),
        }
        summary["liquidation_net_cash"] = round(
            liquidation["net_cash"], 6
        )
        summary["liquidation_total_tax"] = round(
            liquidation["total_tax"], 6
        )
        summary["liquidation_additional_tax"] = round(
            liquidation.get("additional_tax", 0.0), 6
        )
        summary["liquidation_net_after_tax"] = round(
            liquidation["net_cash"] - liquidation["total_tax"],
            6,
        )
        return summary

    def tradesAsDicts(self) -> List[Dict[str, Any]]:
        return [asdict(t) for t in self.trades]


# ---------------- Phase + sizing helpers (extracted) --------------------
def enterBuyPhase(wallet: "Wallet", phaseBuyPortions: int) -> float:
    return phaseBuyValue(wallet.quoteBalance, phaseBuyPortions)


def enterSellPhase(
    wallet: "Wallet", price: float, phaseSellPortions: int
) -> float:
    return phaseSellValue(wallet.baseBalance, price, phaseSellPortions)


def applyScaledBuy(
    wallet: "Wallet",
    i: int,
    ts: datetime,
    price: float,
    phaseBaseValue: float,
    scale: float,
    portionsRemaining: float | None = None,
    finalPortionPct: float = 1.0,
    maxSpendQuote: float | None = None,
):
    spend, portionUsed = strategyBuySpend(
        wallet.quoteBalance,
        phaseBaseValue,
        scale,
        portionsRemaining,
        finalPortionPct,
        maxSpendQuote,
    )
    if spend <= 0:
        return None, 0.0
    trade = wallet._applyBuy(i, ts, price, spendQuote=spend)
    return trade, portionUsed


def applyScaledSell(
    wallet: "Wallet",
    i: int,
    ts: datetime,
    price: float,
    phaseBaseValue: float,
    scale: float,
    portionsRemaining: float | None = None,
    finalPortionPct: float = 1.0,
    maxSellValue: float | None = None,
):
    qty, portionUsed = strategySellQty(
        wallet.baseBalance,
        price,
        phaseBaseValue,
        scale,
        portionsRemaining,
        finalPortionPct,
        maxSellValue,
    )
    if qty <= 0:
        return None, 0.0
    trade = wallet._applySell(i, ts, price, qty=qty)
    return trade, portionUsed


def floorSellValueCap(
    wallet: "Wallet",
    price: float,
    floorPct: float,
) -> float:
    return strategyFloorSellValueCap(
        wallet.quoteBalance,
        wallet.baseBalance,
        price,
        floorPct,
        wallet.feeRate,
    )


def buySpendToTargetCap(
    wallet: "Wallet",
    price: float,
    targetPct: float,
) -> float:
    return strategyBuySpendToTargetCap(
        wallet.quoteBalance,
        wallet.baseBalance,
        price,
        targetPct,
        wallet.feeRate,
    )


def walletValueAt(wallet: "Wallet", price: float) -> float:
    return wallet.quoteBalance + (wallet.baseBalance * price)


def _trendName(
    ctx: Dict[str, Any],
    index: int,
    trendCodes: Optional[Any],
) -> str:
    if trendCodes is not None:
        return strategyTrendLabel(int(trendCodes[index]))
    return trend(ctx, index)


def _sellToFloor(
    wallet: "Wallet",
    index: int,
    ts: datetime,
    price: float,
    targetPct: float,
    note: str,
) -> bool:
    maxValue = floorSellValueCap(wallet, price, targetPct)
    if maxValue <= 0.0 or price <= 0.0:
        return False
    trade = wallet._applySell(index, ts, price, qty=maxValue / price)
    if trade is not None:
        trade.note = note
        return True
    return False


def _buyToTarget(
    wallet: "Wallet",
    index: int,
    ts: datetime,
    price: float,
    targetPct: float,
    note: str,
) -> bool:
    spend = buySpendToTargetCap(wallet, price, targetPct)
    if spend <= 0.0:
        return False
    trade = wallet._applyBuy(index, ts, price, spendQuote=spend)
    if trade is not None:
        trade.note = note
        return True
    return False


def simulateWithDailyPosture(
    ctx: Dict[str, Any],
    flags: List[Tuple[int, str]],
    baseSymbol: str,
    quoteSymbol: str = "USDT",
    startingCash: float = 10000.0,
    feeRate: float = 0.001,
    taxRate: float = 0.30,
    discountDays: int = 365,
    discountRate: float = 0.50,
    seedInvestQuote: float = 0.0,
    seedAssetPct: float = 1.0,
    seedIndex: Optional[int] = None,
    doPrints: bool = False,
    phaseBuyPortions: int = PHASE_BUY_PORTIONS_DEFAULT,
    phaseSellPortions: int = PHASE_SELL_PORTIONS_DEFAULT,
    taxMode: str = "cgt",
    annualIncomeBase: float = FIXED_INCOME_BASE,
    taxYearStartMonth: int = 7,
    finalPortionPct: float = 1.0,
    trendCodes: Optional[Any] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Wallet:
    wallet = Wallet(
        baseSymbol=baseSymbol,
        quoteSymbol=quoteSymbol,
        startingCash=startingCash,
        feeRate=feeRate,
        taxRate=taxRate,
        discountDays=discountDays,
        discountRate=discountRate,
        taxMode=taxMode,
        annualIncomeBase=annualIncomeBase,
        taxYearStartMonth=taxYearStartMonth,
    )
    daily = dailyPostureArrays(ctx, overrides)
    state = dailyPostureState()
    flagsByIndex: Dict[int, List[str]] = defaultdict(list)
    for i, label in flags:
        flagsByIndex[int(i)].append(str(label))

    startIndex = seedIndex if seedIndex is not None else (
        flags[0][0] if flags else 0
    )
    seedAssetPct = max(0.0, min(1.0, float(seedAssetPct)))
    price0 = float(ctx["closes"][startIndex])
    ts0 = datetime.utcfromtimestamp(ctx["klines"][startIndex][0] / 1000.0)
    if seedInvestQuote and seedInvestQuote > 0.0:
        wallet.quoteBalance += seedInvestQuote
        seedSpend = float(seedInvestQuote) * seedAssetPct
        if seedSpend > 0.0:
            trade = wallet._applyBuy(
                startIndex,
                ts0,
                price0,
                spendQuote=seedSpend,
            )
            if trade is not None:
                trade.note = "seed_buy"

    phaseSide: Optional[str] = None
    phaseBaseValue: float = 0.0
    phaseLastPrice: Optional[float] = None
    phasePortionsRemaining: float | None = 0.0
    finalPortionPct = max(0.0, min(1.0, float(finalPortionPct)))
    lastTrendLabel: Optional[str] = None
    barsDay = max(bars_per_day(ctx), 1.0)
    peakCfg = peakLockConfigFromOverrides(overrides, barsDay)
    peakState = peakLockState(
        float(ctx["closes"][0]),
        price0,
        seedInvestQuote,
        feeRate,
    )
    warmPeakLockState(peakState, peakCfg, ctx["closes"], int(startIndex))
    targetBlockBars = max(
        int(round(barsDay)),
        int((overrides or {}).get("COOLDOWN", 0)),
    )
    lastSignalSellIndex = -targetBlockBars - 1

    for i in range(int(startIndex), len(ctx["closes"])):
        price = float(ctx["closes"][i])
        ts = datetime.utcfromtimestamp(ctx["klines"][i][0] / 1000.0)
        labels = flagsByIndex.get(i, [])
        targetBuyBlocked = (
            "SELL" in labels
            or (int(i) - int(lastSignalSellIndex)) <= targetBlockBars
        )
        posture = dailyPostureStep(state, daily, i, price, barsDay, overrides)
        cloudActive = bool(posture.get("cloudActive", False))
        clusterEnabled = bool(posture.get("clusterEnabled", True))
        pidEnabled = bool(posture.get("pidEnabled", True))
        strongNow = bool(posture["strong"])
        strongEntry = False
        peakGraceActive = False
        if peakCfg.enabled and pidEnabled and not cloudActive:
            stepPeakPid(peakState, peakCfg, price)
            strongEntry, peakGraceActive = stepPeakStrong(
                peakState,
                peakCfg,
                strongNow,
            )
        entryPrice = float(state.get("ultraEntryPrice", 0.0))
        peakPrice = float(state.get("ultraPeakPrice", 0.0))
        ultraGainPct = (
            ((peakPrice / entryPrice) - 1.0) * 100.0
            if entryPrice > 0.0 else 0.0
        )
        givebackPct = (
            ((peakPrice / price) - 1.0) * 100.0
            if peakPrice > 0.0 else 0.0
        )
        if peakCfg.enabled and pidEnabled and not cloudActive:
            armPeakLock(peakState, peakCfg, strongNow, ultraGainPct)
        if bool(posture["forceLock"]):
            didLock = _sellToFloor(
                wallet,
                i,
                ts,
                price,
                float(posture["exitTarget"]),
                "daily_posture_lock",
            )
            if didLock:
                state["forcedLocks"] = int(state["forcedLocks"]) + 1
                phaseSide = None
                phaseLastPrice = None
                phasePortionsRemaining = 0.0
            state["episodeLocked"] = True
            state["lockActive"] = True
            state["lockStart"] = int(i)
            state["armHigh"] = 0.0

        if peakCfg.enabled and pidEnabled and not cloudActive:
            peakDecision = evaluatePeakLock(
                peakState,
                peakCfg,
                price,
                walletValueAt(wallet, price),
                givebackPct,
                strongEntry,
                peakGraceActive,
            )
            if peakDecision.canLock:
                didPeakLock = _sellToFloor(
                    wallet,
                    i,
                    ts,
                    price,
                    peakCfg.capPct,
                    "peak_lock",
                )
                if didPeakLock:
                    recordPeakLock(peakState, peakCfg, i)
                    phaseSide = None
                    phaseLastPrice = None
                    phasePortionsRemaining = 0.0

        if peakCfg.enabled and pidEnabled and not cloudActive:
            stepActivePeakLock(peakState, peakCfg, i)

        crabCap = max(0.0, min(1.0, float((overrides or {}).get(
            "CRAB_ASSET_CAP_PCT", 1.0
        ))))
        if (
            bool(posture["downEntry"])
            and not cloudActive
            and crabCap < 1.0 - 1e-9
        ):
            didCap = _sellToFloor(
                wallet,
                i,
                ts,
                price,
                crabCap,
                "daily_crab_cap",
            )
            if didCap:
                state["crabCapSells"] = int(state["crabCapSells"]) + 1
                phaseSide = None
                phaseLastPrice = None
                phasePortionsRemaining = 0.0
        if (
            bool(posture["rawStrong"])
            and not bool(state["lockActive"])
            and clusterEnabled
            and not targetBuyBlocked
        ):
            targetPct = float((overrides or {}).get(
                "ULTRA_EXPOSURE_TARGET",
                0.0,
            ))
            if (
                peakCfg.enabled
                and peakState.active
                and peakState.cap < 1.0 - 1e-9
            ):
                if peakState.cap <= peakCfg.capPct + 1e-9:
                    targetPct = 0.0
                else:
                    targetPct = min(targetPct, float(peakState.cap))
            didTarget = _buyToTarget(
                wallet,
                i,
                ts,
                price,
                targetPct,
                "daily_strong_target_buy",
            )
            if didTarget:
                state["targetBuys"] = int(state["targetBuys"]) + 1
                phaseSide = None
                phaseLastPrice = None
                phasePortionsRemaining = 0.0

        for label in labels:
            currentTrend = _trendName(ctx, i, trendCodes)
            buyTrendActive = currentTrend == "BEAR"
            sellTrendActive = currentTrend == "BULL"
            newBuyRegime = buyTrendActive and lastTrendLabel != "BEAR"
            newSellRegime = sellTrendActive and lastTrendLabel != "BULL"
            lastTrendLabel = currentTrend

            if label == "BUY" and buyTrendActive and wallet.quoteBalance > 0:
                if phaseSide != "BUY" or newBuyRegime:
                    phaseSide = "BUY"
                    phaseLastPrice = None
                    phaseBaseValue = enterBuyPhase(
                        wallet, phaseBuyPortions
                    )
                    if finalPortionPct >= 1.0 - 1e-9:
                        phasePortionsRemaining = None
                    else:
                        phasePortionsRemaining = (
                            float(phaseBuyPortions)
                            if phaseBaseValue > 0 else 0.0
                        )
                scale, pctChange = calcBuyScale(phaseLastPrice, price)
                maxSpendQuote = None
                if bool(posture["down"]):
                    scale *= float((overrides or {}).get(
                        "DAILY_DOWN_BUY_MULT", 0.4
                    ))
                    state["buyShrinks"] = int(state["buyShrinks"]) + 1
                    if crabCap < 1.0 - 1e-9:
                        maxSpendQuote = buySpendToTargetCap(
                            wallet,
                            price,
                            crabCap,
                        )
                if bool(posture.get("coastActive", False)):
                    capPct = float(posture.get(
                        "cloudMaxAssetPct",
                        posture.get("coastTarget", 1.0),
                    ))
                    capSpendQuote = buySpendToTargetCap(
                        wallet,
                        price,
                        capPct,
                    )
                    if maxSpendQuote is None:
                        maxSpendQuote = capSpendQuote
                    else:
                        maxSpendQuote = min(maxSpendQuote, capSpendQuote)
                    if maxSpendQuote <= 0.0:
                        continue
                if (
                    peakCfg.enabled
                    and not cloudActive
                    and peakState.active
                    and peakState.cap < 1.0 - 1e-9
                ):
                    capAllowsBuy = peakState.cap > peakCfg.capPct + 1e-9
                    capTarget = min(
                        float(peakState.cap),
                        crabCap if crabCap < 1.0 - 1e-9 else 1.0,
                    )
                    capSpendQuote = buySpendToTargetCap(
                        wallet,
                        price,
                        capTarget,
                    )
                    if not capAllowsBuy:
                        continue
                    if maxSpendQuote is None:
                        maxSpendQuote = capSpendQuote
                    else:
                        maxSpendQuote = min(maxSpendQuote, capSpendQuote)
                    if maxSpendQuote <= 0.0:
                        continue
                if scale > 0.0:
                    prevRemaining = phasePortionsRemaining
                    trade, portionUsed = applyScaledBuy(
                        wallet,
                        i,
                        ts,
                        price,
                        phaseBaseValue,
                        scale,
                        portionsRemaining=phasePortionsRemaining,
                        finalPortionPct=finalPortionPct,
                        maxSpendQuote=maxSpendQuote,
                    )
                    if trade:
                        if (
                            peakCfg.enabled
                            and not cloudActive
                            and peakState.active
                        ):
                            trade.note = "peak_lock_capped_buy"
                            recordPeakCappedBuy(peakState)
                        phaseLastPrice = price
                        if (
                            portionUsed > 0
                            and prevRemaining is not None
                            and finalPortionPct < 1.0 - 1e-9
                        ):
                            newRem = float(prevRemaining) - float(portionUsed)
                            phasePortionsRemaining = max(0.0, newRem)

            elif label == "SELL" and sellTrendActive and wallet.baseBalance > 0:
                if phaseSide != "SELL" or newSellRegime:
                    phaseSide = "SELL"
                    phaseLastPrice = None
                    phaseBaseValue = enterSellPhase(
                        wallet, price, phaseSellPortions
                    )
                    if finalPortionPct >= 1.0 - 1e-9:
                        phasePortionsRemaining = None
                    else:
                        phasePortionsRemaining = (
                            float(phaseSellPortions)
                            if phaseBaseValue > 0 else 0.0
                        )
                scale, pctChange = calcSellScale(phaseLastPrice, price)
                if scale > 0.0:
                    prevRemaining = phasePortionsRemaining
                    lockTargetPct = 0.0
                    if bool(posture["strong"]) and not cloudActive:
                        strongFloor = float((overrides or {}).get(
                            "ULTRA_EXPOSURE_TARGET",
                            0.0,
                        ))
                        if (
                            peakCfg.enabled
                            and peakState.active
                            and peakState.cap < 1.0 - 1e-9
                        ):
                            strongFloor = min(strongFloor, peakState.cap)
                        lockTargetPct = max(lockTargetPct, strongFloor)
                        scale *= float((overrides or {}).get(
                            "ULTRA_SELL_MULT", 0.0
                        ))
                        state["sellShrinks"] = int(state["sellShrinks"]) + 1
                    if bool(state["lockActive"]):
                        lockTargetPct = max(
                            lockTargetPct,
                            float(posture["lockTarget"]),
                        )
                    trade, portionUsed = applyScaledSell(
                        wallet,
                        i,
                        ts,
                        price,
                        phaseBaseValue,
                        scale,
                        portionsRemaining=phasePortionsRemaining,
                        finalPortionPct=finalPortionPct,
                        maxSellValue=(
                            floorSellValueCap(wallet, price, lockTargetPct)
                            if lockTargetPct > 0.0 else None
                        ),
                    )
                    if trade:
                        lastSignalSellIndex = int(i)
                        phaseLastPrice = price
                        if (
                            portionUsed > 0
                            and prevRemaining is not None
                            and finalPortionPct < 1.0 - 1e-9
                        ):
                            newRem = float(prevRemaining) - float(portionUsed)
                            phasePortionsRemaining = max(0.0, newRem)

    wallet.postureStats = dailyPostureStats(state)
    if peakCfg.enabled:
        wallet.postureStats.update(peakLockStats(peakState))
    return wallet


# ---------------- Convenience runner for traces --------------------------
def simulateFromFlags(
    ctx: Dict[str, Any],
    flags: List[Tuple[int, str]],
    baseSymbol: str,
    quoteSymbol: str = "USDT",
    startingCash: float = 10000.0,
    feeRate: float = 0.001,
    taxRate: float = 0.30,
    discountDays: int = 365,
    discountRate: float = 0.50,
    seedInvestQuote: float = 0.0,
    seedAssetPct: float = 1.0,
    seedIndex: Optional[int] = None,
    doPrints: bool = False,
    phaseBuyPortions: int = PHASE_BUY_PORTIONS_DEFAULT,
    phaseSellPortions: int = PHASE_SELL_PORTIONS_DEFAULT,
    taxMode: str = "cgt",
    annualIncomeBase: float = FIXED_INCOME_BASE,
    taxYearStartMonth: int = 7,
    finalPortionPct: float = 1.0,
    trendCodes: Optional[np.ndarray] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Wallet:
    """Process BUY/SELL flags in order and return a populated Wallet.

    flags: list of (index, label) with label in {"BUY","SELL"}.
    ctx: analysis context with keys: "closes", "klines" (for timestamps).
    finalPortionPct: 0..1 fraction applied repeatedly when one portion remains.
    """
    return simulateWithDailyPosture(
        ctx,
        flags,
        baseSymbol,
        quoteSymbol=quoteSymbol,
        startingCash=startingCash,
        feeRate=feeRate,
        taxRate=taxRate,
        discountDays=discountDays,
        discountRate=discountRate,
        seedInvestQuote=seedInvestQuote,
        seedAssetPct=seedAssetPct,
        seedIndex=seedIndex,
        doPrints=doPrints,
        phaseBuyPortions=phaseBuyPortions,
        phaseSellPortions=phaseSellPortions,
        taxMode=taxMode,
        annualIncomeBase=annualIncomeBase,
        taxYearStartMonth=taxYearStartMonth,
        finalPortionPct=finalPortionPct,
        trendCodes=trendCodes,
        overrides=overrides,
    )
