#!/usr/bin/env python3
"""
accounting.py – Portfolio profit and tax assessment helpers.

Provides:
    - Australian progressive income tax calculation (shared with wallet).
    - Potential profit computation for both CGT and income tax modes.
    - High-level accounting statements blending trading results with
      external base income so backtests can estimate tax implications.
"""

from __future__ import annotations

from typing import Dict


# Australian progressive income tax brackets
# (threshold, base tax, marginal rate above threshold)
AU_INCOME_BRACKETS = (
    (0.0,       0.0,    0.0),
    (18200.0,   0.0,    0.16),
    (45000.0,   4288.0, 0.30),
    (135000.0, 31288.0, 0.37),
    (190000.0, 51638.0, 0.45),
)

FIXED_INCOME_BASE = 36000.0


def calcIncomeTax(income: float) -> float:
    """Return tax owed for a given taxable income under AU brackets."""
    taxable = max(float(income), 0.0)
    tax = 0.0
    for idx, (threshold, base, rate) in enumerate(AU_INCOME_BRACKETS):
        nextThreshold = (
            AU_INCOME_BRACKETS[idx + 1][0]
            if idx + 1 < len(AU_INCOME_BRACKETS)
            else None
        )
        if taxable < threshold:
            continue
        if nextThreshold is None or taxable < nextThreshold:
            tax = base + (taxable - threshold) * rate
            break
    return max(tax, 0.0)


def bracketForIncome(income: float) -> tuple[float, float, float]:
    """Return the active bracket (threshold, base tax, rate) for income."""
    taxable = max(float(income), 0.0)
    threshold = AU_INCOME_BRACKETS[0][0]
    baseTax = AU_INCOME_BRACKETS[0][1]
    rate = AU_INCOME_BRACKETS[0][2]
    for idx, (th, base, rt) in enumerate(AU_INCOME_BRACKETS):
        nextTh = (
            AU_INCOME_BRACKETS[idx + 1][0]
            if idx + 1 < len(AU_INCOME_BRACKETS)
            else None
        )
        if taxable >= th and (nextTh is None or taxable < nextTh):
            threshold, baseTax, rate = th, base, rt
            break
    return threshold, baseTax, rate


def marginalIncomeTaxRate(income: float) -> float:
    """Return marginal tax rate (0..1) for the given AU income."""
    _th, _baseTax, rate = bracketForIncome(income)
    return float(rate)


def potentialProfit(
    summary: Dict[str, float],
    seed: float,
    taxMode: str,
) -> float:
    """Return profit relevant for optimisation under the given tax mode."""
    seedCapital = float(seed)
    grossValue = float(summary.get("portfolio_value", 0.0))
    mode = str(taxMode).lower()
    if mode == "income":
        return grossValue - seedCapital
    netAfterTaxValue = float(
        summary.get(
            "liquidation_net_after_tax",
            grossValue - float(summary.get("tax_liability", 0.0)),
        )
    )
    return netAfterTaxValue - seedCapital


def buildStatement(
    summary: Dict[str, float],
    seed: float,
    taxMode: str,
    annualIncomeBase: float = FIXED_INCOME_BASE,
) -> Dict[str, float]:
    """Return a consolidated accounting statement for a wallet summary."""
    mode = str(taxMode).lower()
    seedCapital = float(seed)
    grossValue = float(summary.get("portfolio_value", 0.0))
    potential = potentialProfit(summary, seedCapital, mode)
    if mode == "income":
        grossProfit = grossValue - seedCapital
    else:
        netCash = float(
            summary.get("liquidation_net_cash", grossValue)
        )
        grossProfit = netCash - seedCapital

    statement: Dict[str, float] = {
        "tax_mode": mode,
        "portfolio_value": grossValue,
        "seed_capital": seedCapital,
        "gross_profit": grossProfit,
        "potential_profit": potential,
    }

    if mode == "income":
        baseIncome = float(annualIncomeBase)
        baseTax = calcIncomeTax(baseIncome)
        combinedIncome = baseIncome + max(potential, 0.0)
        combinedTax = calcIncomeTax(combinedIncome)
        tradingTax = max(0.0, combinedTax - baseTax)
        netAfterTax = potential - tradingTax
        effectiveRate = (
            tradingTax / potential if potential > 1e-12 else 0.0
        )
        statement.update({
            "annual_income_base": baseIncome,
            "base_tax": baseTax,
            "combined_income": combinedIncome,
            "combined_tax": combinedTax,
            "trading_tax": tradingTax,
            "net_after_tax": netAfterTax,
            "effective_tax_rate": effectiveRate,
        })
    else:
        totalTax = float(
            summary.get(
                "liquidation_total_tax",
                summary.get("tax_liability", 0.0),
            )
        )
        netAfterTaxValue = float(
            summary.get(
                "liquidation_net_after_tax",
                grossValue - float(summary.get("tax_liability", 0.0)),
            )
        )
        effectiveRate = (
            totalTax / grossProfit if grossProfit > 1e-12 else 0.0
        )
        statement.update({
            "trading_tax": totalTax,
            "additional_tax": float(
                summary.get("liquidation_additional_tax", 0.0)
            ),
            "net_after_tax": netAfterTaxValue - seedCapital,
            "net_after_tax_value": netAfterTaxValue,
            "effective_tax_rate": effectiveRate,
        })
    return statement
