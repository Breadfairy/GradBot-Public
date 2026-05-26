#!/usr/bin/env python3
"""Peak-lock supervisor state used by traces, tuning, and live."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


###############################################################################
# Constants
###############################################################################

EPS = 1e-12


###############################################################################
# Helpers
###############################################################################

def clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


###############################################################################
# Data
###############################################################################

@dataclass
class PeakLockConfig:
    enabled: bool
    capPct: float
    unlockGainPct: float
    reentryStepPct: float
    armGainPct: float
    givebackPct: float
    maxLockBars: int
    edgeDrawPct: float
    edgeSlopeBars: int
    requireEdgeRisk: bool
    maBars: int
    alpha: float
    kp: float
    ki: float
    kd: float
    integralDecay: float
    entryThreshold: float
    exitThreshold: float
    confirmBars: int
    releaseTargetPct: float
    ultraGraceBars: int


@dataclass
class PeakLockState:
    ma: float
    benchQty: float
    integral: float = 0.0
    prevErr: float = 0.0
    long: bool = False
    bearCount: int = 0
    strongGraceBars: int = 0
    strongReleases: int = 0
    prevStrong: bool = False
    active: bool = False
    start: int = -1
    cap: float = 1.0
    edgeStart: float = 0.0
    edgeNow: float = 0.0
    edgePeak: float = 0.0
    lockGain: float = 0.0
    lockGainMax: float = 0.0
    locks: int = 0
    cappedBuys: int = 0
    lockHours: int = 0
    unlockSteps: int = 0
    armed: bool = False
    edgeVals: list[float] = field(default_factory=list)


@dataclass
class PeakLockDecision:
    canLock: bool = False


@dataclass
class PeakPidSample:
    movingAverage: float
    priceError: float
    priceIntegral: float
    priceDerivative: float
    rawSignal: float
    long: bool
    bearCount: int


###############################################################################
# Construction
###############################################################################

def peakLockConfigFromOverrides(
    overrides: dict[str, Any] | None,
    barsDay: float,
) -> PeakLockConfig:
    ov = overrides or {}
    capPct = clamp(float(ov.get("PEAK_LOCK_CAP_PCT", 1.0)), 0.0, 1.0)
    maBars = max(2, int(round(
        float(ov.get("PEAK_LOCK_MA_DAYS", 30.0)) * barsDay
    )))
    alpha = 2.0 / float(maBars + 1)
    maxLockBars = int(round(
        float(ov.get("PEAK_LOCK_MAX_DAYS", 120.0)) * barsDay
    ))
    edgeSlopeBars = int(round(
        float(ov.get("PEAK_LOCK_EDGE_SLOPE_DAYS", 7.0)) * barsDay
    ))
    ultraGraceBars = int(round(
        max(float(ov.get("PEAK_LOCK_ULTRA_GRACE_DAYS", 0.0)), 0.0)
        * barsDay
    ))
    return PeakLockConfig(
        enabled=capPct < 1.0 - 1e-9,
        capPct=capPct,
        unlockGainPct=float(ov.get("PEAK_LOCK_UNLOCK_GAIN_PCT", 25.0)),
        reentryStepPct=float(ov.get("PEAK_LOCK_REENTRY_STEP_PCT", 0.15)),
        armGainPct=float(ov.get("PEAK_LOCK_ARM_GAIN_PCT", 15.0)),
        givebackPct=float(ov.get("PEAK_LOCK_GIVEBACK_PCT", 4.0)),
        maxLockBars=maxLockBars,
        edgeDrawPct=float(ov.get("PEAK_LOCK_EDGE_DRAW_PCT", 5.0)),
        edgeSlopeBars=edgeSlopeBars,
        requireEdgeRisk=bool(int(
            ov.get("PEAK_LOCK_REQUIRE_EDGE_RISK", 1)
        )),
        maBars=maBars,
        alpha=alpha,
        kp=float(ov.get("PEAK_LOCK_KP", 6.0)),
        ki=float(ov.get("PEAK_LOCK_KI", 0.0)),
        kd=float(ov.get("PEAK_LOCK_KD", 0.0)),
        integralDecay=float(
            ov.get("PEAK_LOCK_INTEGRAL_DECAY", 0.985)
        ),
        entryThreshold=float(
            ov.get("PEAK_LOCK_ENTRY_THRESHOLD", 0.25)
        ),
        exitThreshold=float(
            ov.get("PEAK_LOCK_EXIT_THRESHOLD", 0.05)
        ),
        confirmBars=int(ov.get("PEAK_LOCK_CONFIRM_BARS", 6)),
        releaseTargetPct=clamp(
            float(ov.get("PEAK_LOCK_RELEASE_TARGET_PCT", 0.0)),
            0.0,
            1.0,
        ),
        ultraGraceBars=ultraGraceBars,
    )


def peakLockState(
    firstPrice: float,
    seedPrice: float,
    seedInvestQuote: float,
    feeRate: float,
) -> PeakLockState:
    benchQty = (
        (float(seedInvestQuote) * (1.0 - float(feeRate)))
        / max(float(seedPrice), EPS)
        if float(seedInvestQuote) > 0.0 else 0.0
    )
    return PeakLockState(
        ma=float(firstPrice),
        benchQty=benchQty,
    )


def peakLockStateFromBenchQty(
    firstPrice: float,
    benchQty: float,
) -> PeakLockState:
    return PeakLockState(
        ma=float(firstPrice),
        benchQty=max(float(benchQty), 0.0),
    )


###############################################################################
# State Transitions
###############################################################################

def warmPeakLockState(
    state: PeakLockState,
    cfg: PeakLockConfig,
    closes: Sequence[float],
    startIndex: int,
) -> None:
    price = 0.0
    err = 0.0
    if not cfg.enabled:
        return
    for i in range(0, int(startIndex)):
        price = float(closes[i])
        state.ma = (cfg.alpha * price) + ((1.0 - cfg.alpha) * state.ma)
        err = (price - state.ma) / max(state.ma, EPS)
        state.integral = (cfg.integralDecay * state.integral) + err
        state.prevErr = err
    state.bearCount = 0


def stepPeakPid(
    state: PeakLockState,
    cfg: PeakLockConfig,
    price: float,
) -> PeakPidSample:
    err = 0.0
    deriv = 0.0
    raw = 0.0
    if not cfg.enabled:
        return PeakPidSample(
            movingAverage=float(state.ma),
            priceError=0.0,
            priceIntegral=float(state.integral),
            priceDerivative=0.0,
            rawSignal=0.0,
            long=bool(state.long),
            bearCount=int(state.bearCount),
        )
    state.ma = (cfg.alpha * price) + ((1.0 - cfg.alpha) * state.ma)
    err = (price - state.ma) / max(state.ma, EPS)
    deriv = err - state.prevErr
    state.integral = (cfg.integralDecay * state.integral) + err
    raw = (cfg.kp * err) + (cfg.ki * state.integral) + (cfg.kd * deriv)
    if raw > cfg.entryThreshold:
        state.long = True
    elif raw < cfg.exitThreshold:
        state.long = False
    if state.long:
        state.bearCount = 0
    else:
        state.bearCount += 1
    state.prevErr = err
    return PeakPidSample(
        movingAverage=float(state.ma),
        priceError=float(err),
        priceIntegral=float(state.integral),
        priceDerivative=float(deriv),
        rawSignal=float(raw),
        long=bool(state.long),
        bearCount=int(state.bearCount),
    )


def stepPeakStrong(
    state: PeakLockState,
    cfg: PeakLockConfig,
    strongNow: bool,
) -> tuple[bool, bool]:
    strongEntry = bool(strongNow) and not state.prevStrong
    graceActive = False
    if cfg.enabled and bool(strongNow):
        state.strongGraceBars = cfg.ultraGraceBars
    elif cfg.enabled and state.strongGraceBars > 0:
        state.strongGraceBars -= 1
    graceActive = (
        cfg.enabled
        and not bool(strongNow)
        and state.strongGraceBars > 0
    )
    state.prevStrong = bool(strongNow)
    return strongEntry, graceActive


def armPeakLock(
    state: PeakLockState,
    cfg: PeakLockConfig,
    strongNow: bool,
    ultraGainPct: float,
) -> None:
    if cfg.enabled and bool(strongNow) and ultraGainPct >= cfg.armGainPct:
        state.armed = True


def evaluatePeakLock(
    state: PeakLockState,
    cfg: PeakLockConfig,
    price: float,
    walletValue: float,
    givebackPct: float,
    strongEntry: bool,
    graceActive: bool,
) -> PeakLockDecision:
    benchValue = 0.0
    edgeDraw = 0.0
    slope = 0.0
    peakRisk = False
    edgeRisk = False
    oldCap = 0.0
    canLock = False
    if not cfg.enabled:
        return PeakLockDecision(canLock=False)
    benchValue = state.benchQty * price
    state.edgeNow = (
        ((walletValue / max(benchValue, EPS)) - 1.0) * 100.0
        if benchValue > 0.0 else 0.0
    )
    state.edgePeak = max(float(state.edgePeak), float(state.edgeNow))
    edgeDraw = state.edgePeak - state.edgeNow
    state.edgeVals.append(state.edgeNow)
    if len(state.edgeVals) > cfg.edgeSlopeBars:
        slope = state.edgeVals[-1] - state.edgeVals[-cfg.edgeSlopeBars - 1]
    else:
        slope = 0.0
    peakRisk = givebackPct >= cfg.givebackPct
    edgeRisk = edgeDraw >= cfg.edgeDrawPct and slope < 0.0
    if strongEntry and state.active and cfg.releaseTargetPct > 0.0:
        oldCap = float(state.cap)
        state.cap = max(float(state.cap), cfg.releaseTargetPct)
        if state.cap > oldCap + 1e-9:
            state.strongReleases += 1
            state.edgeStart = state.edgeNow
        if state.cap >= 1.0 - 1e-9:
            state.active = False
            state.start = -1
            state.cap = 1.0
        state.armed = False
        state.bearCount = 0
        state.long = True
    canLock = (
        state.armed
        and not state.active
        and state.bearCount >= cfg.confirmBars
        and not graceActive
        and peakRisk
        and (edgeRisk or not cfg.requireEdgeRisk)
    )
    return PeakLockDecision(canLock=canLock)


def recordPeakLock(
    state: PeakLockState,
    cfg: PeakLockConfig,
    index: int,
) -> None:
    state.locks += 1
    state.active = True
    state.start = int(index)
    state.cap = cfg.capPct
    state.edgeStart = state.edgeNow
    state.lockGain = 0.0
    state.armed = False


def stepActivePeakLock(
    state: PeakLockState,
    cfg: PeakLockConfig,
    index: int,
) -> None:
    age = 0
    canStep = False
    if not cfg.enabled or not state.active:
        return
    state.lockHours += 1
    state.lockGain = state.edgeNow - state.edgeStart
    state.lockGainMax = max(state.lockGainMax, state.lockGain)
    age = int(index) - int(state.start)
    canStep = (
        cfg.reentryStepPct > EPS
        and state.cap <= cfg.capPct + 1e-9
    )
    if state.lockGain >= cfg.unlockGainPct and state.cap < 1.0 and canStep:
        state.cap = clamp(state.cap + cfg.reentryStepPct, 0.0, 1.0)
        state.edgeStart = state.edgeNow
        state.unlockSteps += 1
    if state.cap >= 1.0 - 1e-9 or age >= cfg.maxLockBars:
        state.active = False
        state.start = -1
        state.cap = 1.0


def recordPeakCappedBuy(state: PeakLockState) -> None:
    state.cappedBuys += 1


###############################################################################
# Telemetry
###############################################################################

def peakLockStats(state: PeakLockState) -> dict[str, float | int]:
    return {
        "peakLocks": int(state.locks),
        "peakCappedBuys": int(state.cappedBuys),
        "peakUnlockSteps": int(state.unlockSteps),
        "peakLockHours": int(state.lockHours),
        "peakLockGainMax": float(state.lockGainMax),
        "peakStrongReleases": int(state.strongReleases),
    }
