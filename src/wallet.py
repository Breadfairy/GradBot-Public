#!/usr/bin/env python3
"""
wallet.py – Simple causal wallet simulator for backtests.

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

from engine_shared import trend
from accounting import calcIncomeTax

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
        annualIncomeBase: float = 0.0,
        taxYearStartMonth: int = 7,
        profitSweepInterval: Optional[str] = None,
        profitSweepShare: float = 0.0,
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
        self.profitSweepInterval = (
            None if profitSweepInterval is None
            else str(profitSweepInterval).lower()
        )
        share = float(profitSweepShare)
        if share < 0.0:
            share = 0.0
        elif share > 1.0:
            share = 1.0
        self.profitSweepShare = share

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
        self.lockedByYear = defaultdict(float)
        self.lockedProfit = 0.0
        self.realizedGainSinceSweep = 0.0
        self.lastSweepMonth: Optional[Tuple[int, int]] = None
        self.lastSweepDate: Optional[datetime] = None
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

    def _monthKey(self, dt: datetime) -> Tuple[int, int]:
        return (dt.year, dt.month)

    def _performSweep(self, anchor: datetime) -> None:
        if self.profitSweepShare <= 0.0:
            self.realizedGainSinceSweep = 0.0
            return
        gain = max(self.realizedGainSinceSweep, 0.0)
        lockAmount = gain * self.profitSweepShare
        if lockAmount <= 0.0:
            self.realizedGainSinceSweep = 0.0
            return
        usable = min(lockAmount, self.quoteBalance)
        if usable <= 0.0:
            self.realizedGainSinceSweep = 0.0
            return
        self.quoteBalance -= usable
        self.lockedProfit += usable
        year = self._taxYearForDate(anchor)
        self.lockedByYear[year] += usable
        self.realizedGainSinceSweep = 0.0

    def maybeSweep(self, ts: datetime) -> None:
        if self.profitSweepInterval != "month":
            return
        key = self._monthKey(ts)
        if self.lastSweepMonth is None:
            self.lastSweepMonth = key
            self.lastSweepDate = ts
            return
        if key == self.lastSweepMonth:
            self.lastSweepDate = ts
            return
        anchorYear, anchorMonth = self.lastSweepMonth
        anchor = datetime(anchorYear, anchorMonth, 1)
        self._performSweep(anchor)
        self.lastSweepMonth = key
        self.lastSweepDate = ts

    def finalizeSweeps(self, ts: datetime) -> None:
        if self.profitSweepInterval != "month":
            return
        if self.lastSweepMonth is None:
            return
        anchorYear, anchorMonth = self.lastSweepMonth
        anchor = datetime(anchorYear, anchorMonth, 1)
        self._performSweep(anchor)
        self.lastSweepDate = ts

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
        self.realizedGainSinceSweep += realizedGain

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
        lockedByYear = {
            str(year): round(value, 6)
            for year, value in self.lockedByYear.items()
            if value
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
            "locked_profit": round(self.lockedProfit, 6),
            "tax_mode": self.taxMode,
            "profit_sweep_interval": self.profitSweepInterval or "",
            "profit_sweep_share": round(self.profitSweepShare, 6),
        }
        if self.taxMode == "income":
            summary["annual_income_base"] = round(self.annualIncomeBase, 6)
            if incomeTaxYears:
                summary["income_tax_years"] = incomeTaxYears
        if lockedByYear:
            summary["locked_profit_years"] = lockedByYear
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
    baseCash = wallet.quoteBalance
    if phaseBuyPortions > 0:
        return baseCash / float(phaseBuyPortions)
    return 0.0


def enterSellPhase(
    wallet: "Wallet", price: float, phaseSellPortions: int
) -> float:
    holdingsValue = wallet.baseBalance * price
    if phaseSellPortions > 0:
        return holdingsValue / float(phaseSellPortions)
    return 0.0


def calcBuyScale(
    phaseLastPrice: float | None, price: float
) -> tuple[float, float]:
    if phaseLastPrice is None or phaseLastPrice <= 0:
        return 1.0, 0.0
    pctChange = (price - phaseLastPrice) / phaseLastPrice
    scale = 1.0 - pctChange
    if scale < 0.0:
        scale = 0.0
    return scale, pctChange


def calcSellScale(
    phaseLastPrice: float | None, price: float
) -> tuple[float, float]:
    if phaseLastPrice is None or phaseLastPrice <= 0:
        return 1.0, 0.0
    pctChange = (price - phaseLastPrice) / phaseLastPrice
    scale = 1.0 + pctChange
    if scale < 0.0:
        scale = 0.0
    return scale, pctChange


def applyScaledBuy(
    wallet: "Wallet",
    i: int,
    ts: datetime,
    price: float,
    phaseBaseValue: float,
    scale: float,
    portionsRemaining: float | None = None,
    finalPortionPct: float = 1.0,
):
    if phaseBaseValue <= 0 or scale <= 0:
        return None, 0.0
    remaining = (
        float('inf') if portionsRemaining is None else max(portionsRemaining, 0.0)
    )
    requestedValue = phaseBaseValue * scale
    if requestedValue <= 0:
        return None, 0.0
    # Convert requested value into requested portions
    reqPortions = requestedValue / phaseBaseValue
    usePortions = reqPortions
    # Never exceed total remaining budget in portions
    if remaining != float('inf'):
        usePortions = min(usePortions, remaining)
    # Apply final-portion gate with infinite halving of last portion
    fpct = max(0.0, min(1.0, finalPortionPct))
    eps = 1e-9
    if remaining != float('inf') and fpct < 1.0:
        if remaining <= 1.0 + eps:
            # Already in final window: cap to fraction of remaining
            capped = remaining * fpct
            usePortions = min(usePortions, capped)
        else:
            # Trade crosses into final window: limit final slice
            head = max(0.0, remaining - 1.0)
            if usePortions > head + eps:
                finalRem = max(remaining - head, 0.0)
                finalReq = usePortions - head
                maxFinalUse = finalRem * fpct
                finalUse = min(finalReq, maxFinalUse)
                usePortions = head + finalUse
    spend = phaseBaseValue * usePortions
    spend = min(spend, wallet.quoteBalance)
    if spend <= 0:
        return None, 0.0
    trade = wallet._applyBuy(i, ts, price, spendQuote=spend)
    portionUsed = usePortions if phaseBaseValue > 0 else 0.0
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
):
    if phaseBaseValue <= 0 or price <= 0 or scale <= 0:
        return None, 0.0
    remaining = (
        float('inf') if portionsRemaining is None else max(portionsRemaining, 0.0)
    )
    targetValue = phaseBaseValue * scale
    if targetValue <= 0:
        return None, 0.0
    # Convert target value into requested portions
    reqPortions = targetValue / phaseBaseValue
    usePortions = reqPortions
    if remaining != float('inf'):
        usePortions = min(usePortions, remaining)
    # Apply final-portion gate with infinite halving of last portion
    fpct = max(0.0, min(1.0, finalPortionPct))
    eps = 1e-9
    if remaining != float('inf') and fpct < 1.0:
        if remaining <= 1.0 + eps:
            capped = remaining * fpct
            usePortions = min(usePortions, capped)
        else:
            head = max(0.0, remaining - 1.0)
            if usePortions > head + eps:
                finalRem = max(remaining - head, 0.0)
                finalReq = usePortions - head
                maxFinalUse = finalRem * fpct
                finalUse = min(finalReq, maxFinalUse)
                usePortions = head + finalUse
    maxValue = phaseBaseValue * usePortions
    maxValue = min(maxValue, wallet.baseBalance * price)
    if maxValue <= 0:
        return None, 0.0
    sellQty = maxValue / price
    trade = wallet._applySell(i, ts, price, qty=sellQty)
    portionUsed = usePortions if phaseBaseValue > 0 else 0.0
    return trade, portionUsed


# ---------------- Convenience runner for backtests -----------------------
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
    seedIndex: Optional[int] = None,
    doPrints: bool = False,
    phaseBuyPortions: int = PHASE_BUY_PORTIONS_DEFAULT,
    phaseSellPortions: int = PHASE_SELL_PORTIONS_DEFAULT,
    taxMode: str = "cgt",
    annualIncomeBase: float = 0.0,
    profitSweepInterval: Optional[str] = None,
    profitSweepShare: float = 0.0,
    taxYearStartMonth: int = 7,
    finalPortionPct: float = 1.0,
    trendCodes: Optional[np.ndarray] = None,
) -> Wallet:
    """Process BUY/SELL flags in order and return a populated Wallet.

    flags: list of (index, label) with label in {"BUY","SELL"}.
    ctx: analysis context with keys: "closes", "klines" (for timestamps).
    finalPortionPct: 0..1 fraction applied repeatedly when one portion remains.
    """
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
        profitSweepInterval=profitSweepInterval,
        profitSweepShare=profitSweepShare,
    )

    # Optional initial investment (buy-and-hold seed) at a given index
    if seedInvestQuote and seedInvestQuote > 0.0:
        si = (
            seedIndex if seedIndex is not None
            else (flags[0][0] if flags else 0)
        )
        price0 = float(ctx["closes"][si])
        ts0 = datetime.utcfromtimestamp(ctx["klines"][si][0] / 1000.0)
        wallet.maybeSweep(ts0)
        # Fund wallet with seed quote and buy all
        wallet.quoteBalance += seedInvestQuote
        wallet._applyBuy(si, ts0, price0, spendQuote=None)

    # Phase-based sizing state
    phaseSide: Optional[str] = None  # "BUY" or "SELL"
    phaseBaseValue: float = 0.0
    phaseLastPrice: Optional[float] = None
    phasePortionsRemaining: float | None = 0.0
    finalPortionPct = max(0.0, min(1.0, float(finalPortionPct)))
    lastTrendLabel: Optional[str] = None

    # Flags from generateFlags are already in causal order.
    for i, label in flags:
        price = float(ctx["closes"][i])
        ts = datetime.utcfromtimestamp(ctx["klines"][i][0] / 1000.0)
        wallet.maybeSweep(ts)
        if trendCodes is not None:
            code = int(trendCodes[i])
            if code == 1:
                currentTrend = "BULL"
            elif code == -1:
                currentTrend = "BEAR"
            elif code == 2:
                currentTrend = "HALF_BULL"
            elif code == -2:
                currentTrend = "HALF_BEAR"
            else:
                currentTrend = "NEUTRAL"
        else:
            currentTrend = trend(ctx, i)

        newBearRegime = (
            currentTrend == "BEAR" and lastTrendLabel != "BEAR"
        )
        newBullRegime = (
            currentTrend == "BULL" and lastTrendLabel != "BULL"
        )
        lastTrendLabel = currentTrend

        if label == "BUY":
            if currentTrend == "BEAR" and wallet.quoteBalance > 0:
                if phaseSide != "BUY" or newBearRegime:
                    phaseSide = "BUY"
                    phaseLastPrice = None
                    phaseBaseValue = enterBuyPhase(
                        wallet, phaseBuyPortions
                    )
                    if finalPortionPct >= 1.0 - 1e-9:
                        phasePortionsRemaining = None
                    else:
                        phasePortionsRemaining = (
                            float(phaseBuyPortions) if phaseBaseValue > 0 else 0.0
                        )
                scale, pctChange = calcBuyScale(phaseLastPrice, price)
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
                    )
                    if trade:
                        phaseLastPrice = price
                        if (
                            portionUsed > 0
                            and prevRemaining is not None
                            and finalPortionPct < 1.0 - 1e-9
                        ):
                            newRem = float(prevRemaining) - float(portionUsed)
                            phasePortionsRemaining = max(0.0, newRem)

        elif label == "SELL":
            if currentTrend == "BULL" and wallet.baseBalance > 0:
                if phaseSide != "SELL" or newBullRegime:
                    phaseSide = "SELL"
                    phaseLastPrice = None
                    phaseBaseValue = enterSellPhase(
                        wallet, price, phaseSellPortions
                    )
                    if finalPortionPct >= 1.0 - 1e-9:
                        phasePortionsRemaining = None
                    else:
                        phasePortionsRemaining = (
                            float(phaseSellPortions) if phaseBaseValue > 0 else 0.0
                        )
                scale, pctChange = calcSellScale(phaseLastPrice, price)
                if scale > 0.0:
                    prevRemaining = phasePortionsRemaining
                    trade, portionUsed = applyScaledSell(
                        wallet,
                        i,
                        ts,
                        price,
                        phaseBaseValue,
                        scale,
                        portionsRemaining=phasePortionsRemaining,
                        finalPortionPct=finalPortionPct,
                    )
                    if trade:
                        phaseLastPrice = price
                        if (
                            portionUsed > 0
                            and prevRemaining is not None
                            and finalPortionPct < 1.0 - 1e-9
                        ):
                            newRem = float(prevRemaining) - float(portionUsed)
                            phasePortionsRemaining = max(0.0, newRem)
            else:
                # Outside allowed phase, ignore
                pass
        else:
            # ignore any non-trading markers (e.g., oracles)
            pass

    if ctx["klines"]:
        endTs = datetime.utcfromtimestamp(ctx["klines"][-1][0] / 1000.0)
        wallet.finalizeSweeps(endTs)

    return wallet
