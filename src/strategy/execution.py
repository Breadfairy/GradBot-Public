#!/usr/bin/env python3
"""Pure execution sizing helpers shared by trace and live trading."""

from __future__ import annotations


###############################################################################
# Helpers
###############################################################################

def clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


###############################################################################
# Phase Sizing
###############################################################################

def phaseBuyValue(quoteFree: float, phaseBuyPortions: int) -> float:
    portions = int(phaseBuyPortions)
    if portions > 0:
        return float(quoteFree) / float(portions)
    return 0.0


def phaseSellValue(
    baseFree: float,
    price: float,
    phaseSellPortions: int,
) -> float:
    portions = int(phaseSellPortions)
    holdingsValue = float(baseFree) * float(price)
    if portions > 0:
        return holdingsValue / float(portions)
    return 0.0


def calcBuyScale(
    lastPrice: float | None,
    nowPrice: float,
) -> tuple[float, float]:
    pctChange = 0.0
    scale = 1.0
    if lastPrice is None or float(lastPrice) <= 0.0:
        return scale, pctChange
    pctChange = (float(nowPrice) - float(lastPrice)) / float(lastPrice)
    scale = max(0.0, 1.0 - pctChange)
    return scale, pctChange


def calcSellScale(
    lastPrice: float | None,
    nowPrice: float,
) -> tuple[float, float]:
    pctChange = 0.0
    scale = 1.0
    if lastPrice is None or float(lastPrice) <= 0.0:
        return scale, pctChange
    pctChange = (float(nowPrice) - float(lastPrice)) / float(lastPrice)
    scale = max(0.0, 1.0 + pctChange)
    return scale, pctChange


###############################################################################
# Portion Gates
###############################################################################

def gateFinalPortion(
    reqPortions: float,
    portionsRemaining: float | None,
    finalPortionPct: float,
) -> float:
    remaining = float("inf")
    usePortions = float(reqPortions)
    fpct = clamp(float(finalPortionPct), 0.0, 1.0)
    eps = 1e-9
    head = 0.0
    finalRem = 0.0
    finalReq = 0.0
    finalUse = 0.0
    maxFinalUse = 0.0
    if portionsRemaining is None:
        return usePortions
    remaining = max(float(portionsRemaining), 0.0)
    usePortions = min(usePortions, remaining)
    if fpct >= 1.0 - eps:
        return usePortions
    if remaining <= 1.0 + eps:
        return min(usePortions, remaining * fpct)
    head = max(0.0, remaining - 1.0)
    if usePortions <= head + eps:
        return usePortions
    finalRem = max(remaining - head, 0.0)
    finalReq = usePortions - head
    maxFinalUse = finalRem * fpct
    finalUse = min(finalReq, maxFinalUse)
    return head + finalUse


###############################################################################
# Order Sizing
###############################################################################

def buySpend(
    quoteFree: float,
    phaseBaseValue: float,
    scale: float,
    portionsRemaining: float | None,
    finalPortionPct: float,
    maxSpendQuote: float | None = None,
) -> tuple[float, float]:
    requestedValue = 0.0
    reqPortions = 0.0
    usePortions = 0.0
    spend = 0.0
    if float(phaseBaseValue) <= 0.0 or float(scale) <= 0.0:
        return 0.0, 0.0
    requestedValue = float(phaseBaseValue) * float(scale)
    if requestedValue <= 0.0:
        return 0.0, 0.0
    reqPortions = requestedValue / float(phaseBaseValue)
    usePortions = gateFinalPortion(
        reqPortions,
        portionsRemaining,
        finalPortionPct,
    )
    spend = float(phaseBaseValue) * usePortions
    if maxSpendQuote is not None:
        spend = min(spend, max(0.0, float(maxSpendQuote)))
    spend = min(spend, max(0.0, float(quoteFree)))
    if spend <= 0.0:
        return 0.0, 0.0
    return spend, usePortions


def sellQty(
    baseFree: float,
    price: float,
    phaseBaseValue: float,
    scale: float,
    portionsRemaining: float | None,
    finalPortionPct: float,
    maxSellValue: float | None,
) -> tuple[float, float]:
    targetValue = 0.0
    reqPortions = 0.0
    usePortions = 0.0
    maxValue = 0.0
    qty = 0.0
    if (
        float(phaseBaseValue) <= 0.0
        or float(price) <= 0.0
        or float(scale) <= 0.0
    ):
        return 0.0, 0.0
    targetValue = float(phaseBaseValue) * float(scale)
    if targetValue <= 0.0:
        return 0.0, 0.0
    reqPortions = targetValue / float(phaseBaseValue)
    usePortions = gateFinalPortion(
        reqPortions,
        portionsRemaining,
        finalPortionPct,
    )
    maxValue = float(phaseBaseValue) * usePortions
    if maxSellValue is not None:
        maxValue = min(maxValue, max(0.0, float(maxSellValue)))
    maxValue = min(maxValue, max(0.0, float(baseFree)) * float(price))
    if maxValue <= 0.0:
        return 0.0, 0.0
    qty = maxValue / float(price)
    return qty, usePortions


def floorSellValueCap(
    quoteFree: float,
    baseFree: float,
    price: float,
    floorPct: float,
    feeRate: float,
) -> float:
    assetValue = float(baseFree) * float(price)
    quoteValue = float(quoteFree)
    totalValue = assetValue + quoteValue
    floor = clamp(float(floorPct), 0.0, 1.0)
    fee = max(float(feeRate), 0.0)
    denom = 1.0 - (floor * fee)
    cap = 0.0
    if floor <= 0.0:
        return assetValue
    if denom > 1e-12:
        cap = (assetValue - (floor * totalValue)) / denom
    return max(0.0, min(assetValue, cap))


def buySpendToTargetCap(
    quoteFree: float,
    baseFree: float,
    price: float,
    targetPct: float,
    feeRate: float,
) -> float:
    target = clamp(float(targetPct), 0.0, 1.0)
    quote = max(float(quoteFree), 0.0)
    fee = max(float(feeRate), 0.0)
    assetValue = max(float(baseFree), 0.0) * float(price)
    totalValue = assetValue + quote
    needValue = (target * totalValue) - assetValue
    denom = 1.0 - fee + (target * fee)
    if (
        target <= 0.0
        or float(price) <= 0.0
        or quote <= 0.0
        or totalValue <= 0.0
        or needValue <= 0.0
        or denom <= 1e-12
    ):
        return 0.0
    return min(quote, needValue / denom)


def dailyLockQty(
    quoteFree: float,
    baseFree: float,
    price: float,
    targetPct: float,
    feeRate: float,
) -> float:
    maxValue = floorSellValueCap(
        quoteFree,
        baseFree,
        price,
        targetPct,
        feeRate,
    )
    if float(price) <= 0.0 or maxValue <= 0.0:
        return 0.0
    return maxValue / float(price)
