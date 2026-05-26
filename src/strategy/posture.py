#!/usr/bin/env python3
"""Daily posture state machine shared by trace and live adapters."""

from __future__ import annotations

from typing import Any


###############################################################################
# Constants
###############################################################################

DAILY_STRONG_CLUSTER = 2
DAILY_DOWN_MASK = 9


###############################################################################
# Helpers
###############################################################################

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def ultraScore(gainPct: float, minPct: float, maxPct: float) -> float:
    span = max(float(maxPct) - float(minPct), 1e-12)
    return clamp((float(gainPct) - float(minPct)) / span, 0.0, 1.0)


def clearDailyLock(state: dict[str, Any]) -> None:
    state["lockActive"] = False
    state["lockStart"] = -1
    state["lockTargetPct"] = 1.0
    state["lockHoldDays"] = 0.0
    state["lockReleaseOnStrong"] = False


def clearDailyCoast(state: dict[str, Any]) -> None:
    state["coastActive"] = False
    state["coastStart"] = -1
    state["coastTargetPct"] = 1.0
    state["cloudMinAssetPct"] = 0.0
    state["cloudMaxAssetPct"] = 1.0


def dailyDownNow(cluster: int) -> bool:
    return 0 <= cluster < 30 and bool(DAILY_DOWN_MASK & (1 << cluster))


def defaultDailyPosture() -> dict[str, Any]:
    return {
        "cluster": -1,
        "strong": False,
        "rawStrong": False,
        "down": False,
        "downEntry": False,
        "late": False,
        "forceLock": False,
        "exitTarget": 1.0,
        "lockTarget": 1.0,
        "coastActive": False,
        "coastTarget": 1.0,
        "coastRelease": False,
        "doubleTopRelease": False,
        "cloudActive": False,
        "cloudMinAssetPct": 0.0,
        "cloudMaxAssetPct": 1.0,
        "cloudRelease": False,
        "cloudDoubleTopRelease": False,
        "clusterEnabled": True,
        "pidEnabled": True,
    }


###############################################################################
# State
###############################################################################

def dailyPostureState() -> dict[str, Any]:
    return {
        "strongDays": 0,
        "lockActive": False,
        "lockStart": -1,
        "lockTargetPct": 1.0,
        "lockHoldDays": 0.0,
        "coastActive": False,
        "coastStart": -1,
        "coastTargetPct": 1.0,
        "cloudMinAssetPct": 0.0,
        "cloudMaxAssetPct": 1.0,
        "ultraEntryPrice": 0.0,
        "ultraPeakPrice": 0.0,
        "prevStrong": False,
        "prevDown": False,
        "episodeLocked": False,
        "bridgeBars": 0,
        "lockReleaseOnStrong": False,
        "forcedLocks": 0,
        "strongBars": 0,
        "bridgeStrongBars": 0,
        "lateBars": 0,
        "downBars": 0,
        "buyShrinks": 0,
        "sellShrinks": 0,
        "targetBuys": 0,
        "crabCapSells": 0,
    }


def markDailyLockState(state: dict[str, Any], index: int) -> None:
    state["lockActive"] = True
    state["lockStart"] = int(index)
    state["episodeLocked"] = True


###############################################################################
# Runtime Step
###############################################################################

def dailyPostureStep(
    state: dict[str, Any],
    cluster: int,
    price: float,
    index: int,
    barsDay: float,
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    ov = overrides or {}
    rawStrong = int(cluster) == DAILY_STRONG_CLUSTER
    downNow = dailyDownNow(int(cluster))
    prevStrong = bool(state["prevStrong"])
    prevDown = bool(state.get("prevDown", False))
    bridgeBarsMax = int(round(
        max(float(ov.get("ULTRA_BRIDGE_DAYS", 0.0)), 0.0)
        * max(float(barsDay), 1.0)
    ))
    strongNow = False
    forceLock = False
    exitTarget = 1.0
    score = 0.0
    entryPrice = 0.0
    peakPrice = 0.0
    peakGainPct = 0.0
    depth = 0.0
    gainMaxPct = 0.0
    gainMinPct = 0.0
    givebackPct = 0.0
    exitTrigger = False
    lockDays = 0.0
    maxDays = 0.0
    releaseStrong = False
    releaseTimed = False
    coastTarget = 1.0
    coastGiveback = 0.0
    coastReaccum = 0.0
    coastDoubleTop = 0.0
    coastMaxDays = 0.0
    cloudMin = 0.0
    cloudMax = 1.0
    cloudGiveback = 0.0
    cloudReaccum = 0.0
    cloudDoubleTop = 0.0
    cloudMaxDays = 0.0
    cloudGivebackRaw = 0.0
    cloudEnabled = False
    cloudTriggered = False
    coastRelease = False
    doubleTopRelease = False
    oldPeakPrice = 0.0
    reaccumPrice = 0.0
    coastDays = 0.0

    coastTarget = clamp(
        float(ov.get("POST_ULTRA_COAST_TARGET_PCT", 1.0)),
        0.0,
        1.0,
    )
    coastGiveback = max(
        float(ov.get("POST_ULTRA_GIVEBACK_PCT", 0.0)),
        0.0,
    )
    coastReaccum = clamp(
        float(ov.get("POST_ULTRA_REACCUM_PCT", 0.0)),
        -95.0,
        1000.0,
    )
    coastDoubleTop = max(
        float(ov.get("POST_ULTRA_DOUBLE_TOP_PCT", 0.0)),
        0.0,
    )
    coastMaxDays = max(float(ov.get("POST_ULTRA_MAX_DAYS", 0.0)), 0.0)
    cloudMin = clamp(
        float(ov.get("POST_ULTRA_LOCK_MIN_ASSET_PCT", 0.0)),
        0.0,
        1.0,
    )
    cloudMax = clamp(
        float(ov.get("POST_ULTRA_LOCK_MAX_ASSET_PCT", 1.0)),
        0.0,
        1.0,
    )
    cloudGivebackRaw = float(ov.get("POST_ULTRA_LOCK_GIVEBACK_PCT", 0.0))
    if (
        cloudGivebackRaw <= 0.0
        and coastGiveback > 0.0
        and cloudMax >= 1.0 - 1e-9
    ):
        cloudMin = coastTarget
        cloudMax = coastTarget
    cloudMax = max(cloudMax, cloudMin)
    cloudGiveback = (
        max(cloudGivebackRaw, 0.0)
        if cloudGivebackRaw > 0.0 else coastGiveback
    )
    cloudReaccum = clamp(
        float(
            ov.get("POST_ULTRA_LOCK_REACCUM_PCT", coastReaccum)
            if cloudGivebackRaw > 0.0 else coastReaccum
        ),
        -95.0,
        1000.0,
    )
    cloudDoubleTop = max((
        float(ov.get("POST_ULTRA_LOCK_DOUBLE_TOP_PCT", coastDoubleTop))
        if cloudGivebackRaw > 0.0 else coastDoubleTop
    ),
        0.0
    )
    cloudMaxDays = max(
        float(
            ov.get("POST_ULTRA_LOCK_MAX_DAYS", coastMaxDays)
            if cloudGivebackRaw > 0.0 else coastMaxDays
        ),
        0.0,
    )
    cloudEnabled = cloudMax < 1.0 - 1e-9 and cloudGiveback > 0.0
    oldPeakPrice = float(state.get("ultraPeakPrice", 0.0))
    doubleTopRelease = (
        bool(state.get("coastActive", False))
        and rawStrong
        and cloudDoubleTop > 0.0
        and oldPeakPrice > 0.0
        and float(price) >= oldPeakPrice * (1.0 - (cloudDoubleTop / 100.0))
    )
    if doubleTopRelease:
        clearDailyCoast(state)
        clearDailyLock(state)
        coastRelease = True

    if rawStrong:
        if not prevStrong and not doubleTopRelease:
            state["episodeLocked"] = False
            state["ultraEntryPrice"] = float(price)
            state["ultraPeakPrice"] = float(price)
        elif doubleTopRelease:
            if float(state["ultraEntryPrice"]) <= 0.0:
                state["ultraEntryPrice"] = float(price)
            state["ultraPeakPrice"] = max(oldPeakPrice, float(price))
        state["bridgeBars"] = 0
        strongNow = True
        state["strongDays"] = (
            int(state["strongDays"]) + 1 if prevStrong else 1
        )
        state["strongBars"] = int(state.get("strongBars", 0)) + 1
    elif prevStrong:
        state["bridgeBars"] = int(state["bridgeBars"]) + 1
        strongNow = int(state["bridgeBars"]) <= bridgeBarsMax
        if strongNow:
            state["strongDays"] = int(state["strongDays"]) + 1
            state["bridgeStrongBars"] = (
                int(state.get("bridgeStrongBars", 0)) + 1
            )
        else:
            state["strongDays"] = 0
    else:
        state["strongDays"] = 0
        state["bridgeBars"] = 0

    if strongNow:
        state["ultraPeakPrice"] = max(
            float(state["ultraPeakPrice"]),
            float(price),
        )

    if downNow:
        state["downBars"] = int(state.get("downBars", 0)) + 1

    entryPrice = float(state["ultraEntryPrice"])
    peakPrice = float(state["ultraPeakPrice"])
    peakGainPct = (
        ((peakPrice / entryPrice) - 1.0) * 100.0
        if entryPrice > 0.0 else 0.0
    )
    givebackPct = (
        ((peakPrice / float(price)) - 1.0) * 100.0
        if peakPrice > 0.0 and float(price) > 0.0 else 0.0
    )
    depth = clamp(float(ov.get("ULTRA_EXIT_DEPTH", 0.0)), 0.0, 1.0)
    gainMinPct = float(ov.get("ULTRA_GAIN_MIN_PCT", 5.0))
    gainMaxPct = float(ov.get("ULTRA_GAIN_MAX_PCT", 35.0))
    if (
        strongNow
        and not bool(state["episodeLocked"])
        and depth > 0.0
        and peakGainPct >= gainMaxPct
    ):
        exitTarget = clamp(1.0 - depth, 0.0, 1.0)
        state["lockTargetPct"] = exitTarget
        state["lockHoldDays"] = float(ov.get("ULTRA_EXIT_HOLD_DAYS", 60.0))
        state["lockReleaseOnStrong"] = False
        forceLock = True
        state["lateBars"] = int(state.get("lateBars", 0)) + 1

    exitTrigger = prevStrong and not strongNow
    if exitTrigger and not bool(state["episodeLocked"]):
        score = ultraScore(
            peakGainPct,
            gainMinPct,
            float(ov.get("ULTRA_GAIN_MAX_PCT", 35.0)),
        )
        exitTarget = clamp(1.0 - (depth * score), 0.0, 1.0)
        state["lockTargetPct"] = exitTarget
        state["lockHoldDays"] = (
            float(ov.get("ULTRA_EXIT_HOLD_DAYS", 60.0)) * score
        )
        state["lockReleaseOnStrong"] = True
        forceLock = score > 0.0 and depth > 0.0
        if forceLock:
            state["lateBars"] = int(state.get("lateBars", 0)) + 1

    cloudTriggered = (
        cloudEnabled
        and not bool(state.get("coastActive", False))
        and peakGainPct >= gainMinPct
        and givebackPct >= cloudGiveback
    )
    if cloudTriggered:
        exitTarget = cloudMin
        state["coastActive"] = True
        state["coastStart"] = int(index)
        state["coastTargetPct"] = cloudMax
        state["cloudMinAssetPct"] = cloudMin
        state["cloudMaxAssetPct"] = cloudMax
        state["lockTargetPct"] = cloudMin
        state["lockHoldDays"] = cloudMaxDays
        state["lockReleaseOnStrong"] = False
        forceLock = True
        state["lateBars"] = int(state.get("lateBars", 0)) + 1

    if not strongNow:
        state["bridgeBars"] = 0

    lockDays = (
        (int(index) - int(state["lockStart"])) / max(float(barsDay), 1.0)
        if int(state["lockStart"]) >= 0 else 0.0
    )
    maxDays = float(state["lockHoldDays"])
    releaseStrong = strongNow and bool(state["lockReleaseOnStrong"])
    releaseTimed = lockDays >= maxDays
    if bool(state.get("coastActive", False)) and maxDays <= 0.0:
        releaseTimed = False

    if bool(state.get("coastActive", False)) and not cloudTriggered:
        reaccumPrice = entryPrice * (1.0 + (cloudReaccum / 100.0))
        coastDays = (
            (int(index) - int(state["coastStart"])) / max(float(barsDay), 1.0)
            if int(state["coastStart"]) >= 0 else 0.0
        )
        if entryPrice > 0.0 and float(price) <= reaccumPrice:
            coastRelease = True
        if cloudMaxDays > 0.0 and coastDays >= cloudMaxDays:
            coastRelease = True
        if coastRelease:
            clearDailyCoast(state)
            clearDailyLock(state)

    if bool(state["lockActive"]) and (releaseStrong or releaseTimed):
        clearDailyLock(state)
        if bool(state.get("coastActive", False)):
            clearDailyCoast(state)
    state["prevStrong"] = strongNow
    state["prevDown"] = downNow

    return {
        "cluster": int(cluster),
        "strong": strongNow,
        "rawStrong": rawStrong,
        "down": downNow,
        "downEntry": downNow and not prevDown,
        "late": forceLock,
        "forceLock": forceLock,
        "exitTarget": exitTarget,
        "lockTarget": float(state["lockTargetPct"]),
        "coastActive": bool(state.get("coastActive", False)),
        "coastTarget": float(state.get("coastTargetPct", 1.0)),
        "coastRelease": coastRelease,
        "doubleTopRelease": doubleTopRelease,
        "cloudActive": bool(state.get("coastActive", False)),
        "cloudMinAssetPct": float(state.get("cloudMinAssetPct", 0.0)),
        "cloudMaxAssetPct": float(state.get("cloudMaxAssetPct", 1.0)),
        "cloudRelease": coastRelease,
        "cloudDoubleTopRelease": doubleTopRelease,
        "clusterEnabled": not bool(state.get("coastActive", False)),
        "pidEnabled": not bool(state.get("coastActive", False)),
    }


###############################################################################
# Telemetry
###############################################################################

def dailyPostureStats(state: dict[str, Any]) -> dict[str, int]:
    keys = (
        "forcedLocks",
        "strongBars",
        "lateBars",
        "downBars",
        "buyShrinks",
        "sellShrinks",
        "targetBuys",
        "bridgeStrongBars",
        "crabCapSells",
    )
    return {k: int(state.get(k, 0)) for k in keys}
