#!/usr/bin/env python3
# strategy_golden_check.py - deterministic pure strategy replay checks.

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


########################################################################
# Imports
########################################################################

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from strategy.posture import (
    dailyPostureState,
    dailyPostureStats,
    dailyPostureStep,
    markDailyLockState,
)
from strategy.execution import (
    buySpend,
    buySpendToTargetCap,
    calcBuyScale,
    calcSellScale,
    dailyLockQty,
    floorSellValueCap,
    gateFinalPortion,
    phaseBuyValue,
    phaseSellValue,
    sellQty,
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
)


########################################################################
# Helpers
########################################################################

def _roundValue(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [_roundValue(i) for i in value]
    if isinstance(value, dict):
        return {k: _roundValue(v) for k, v in value.items()}
    return value


def _assertSame(name: str, actual: dict[str, Any], expected: dict[str, Any]):
    actualText = json.dumps(_roundValue(actual), sort_keys=True, indent=2)
    expectedText = json.dumps(expected, sort_keys=True, indent=2)
    if actualText != expectedText:
        print(f"[strategy-golden] {name} mismatch", file=sys.stderr)
        print("[strategy-golden] actual:", file=sys.stderr)
        print(actualText, file=sys.stderr)
        print("[strategy-golden] expected:", file=sys.stderr)
        print(expectedText, file=sys.stderr)
        raise SystemExit(1)


def _postureRow(index: int, posture: dict[str, Any], state: dict[str, Any]):
    return {
        "i": int(index),
        "cluster": int(posture["cluster"]),
        "strong": bool(posture["strong"]),
        "rawStrong": bool(posture["rawStrong"]),
        "down": bool(posture["down"]),
        "downEntry": bool(posture["downEntry"]),
        "forceLock": bool(posture["forceLock"]),
        "exitTarget": round(float(posture["exitTarget"]), 6),
        "lockTarget": round(float(posture["lockTarget"]), 6),
        "lockActive": bool(state["lockActive"]),
        "lockStart": int(state["lockStart"]),
    }


def _postureState(state: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "strongDays",
        "lockActive",
        "lockStart",
        "lockTargetPct",
        "lockHoldDays",
        "ultraEntryPrice",
        "ultraPeakPrice",
        "prevStrong",
        "prevDown",
        "episodeLocked",
        "bridgeBars",
        "lockReleaseOnStrong",
        "strongBars",
        "bridgeStrongBars",
        "lateBars",
        "downBars",
    )
    return {k: state[k] for k in keys}


########################################################################
# Posture Golden
########################################################################

def postureGolden() -> dict[str, Any]:
    state = dailyPostureState()
    rows = []
    overrides = {
        "ULTRA_BRIDGE_DAYS": 0.5,
        "ULTRA_EXIT_DEPTH": 0.5,
        "ULTRA_GAIN_MIN_PCT": 5.0,
        "ULTRA_GAIN_MAX_PCT": 20.0,
        "ULTRA_EXIT_HOLD_DAYS": 1.0,
    }
    samples = (
        (2, 100.0),
        (2, 125.0),
        (-1, 120.0),
        (-1, 115.0),
        (-1, 110.0),
        (3, 105.0),
    )
    posture = {}
    for i, sample in enumerate(samples):
        posture = dailyPostureStep(
            state,
            int(sample[0]),
            float(sample[1]),
            i,
            4.0,
            overrides,
        )
        if bool(posture["forceLock"]):
            markDailyLockState(state, i)
        rows.append(_postureRow(i, posture, state))
    return {
        "rows": rows,
        "state": _postureState(state),
        "stats": dailyPostureStats(state),
    }


def expectedPosture() -> dict[str, Any]:
    return {
        "rows": [
            {
                "i": 0,
                "cluster": 2,
                "strong": True,
                "rawStrong": True,
                "down": False,
                "downEntry": False,
                "forceLock": False,
                "exitTarget": 1.0,
                "lockTarget": 1.0,
                "lockActive": False,
                "lockStart": -1,
            },
            {
                "i": 1,
                "cluster": 2,
                "strong": True,
                "rawStrong": True,
                "down": False,
                "downEntry": False,
                "forceLock": True,
                "exitTarget": 0.5,
                "lockTarget": 0.5,
                "lockActive": True,
                "lockStart": 1,
            },
            {
                "i": 2,
                "cluster": -1,
                "strong": True,
                "rawStrong": False,
                "down": False,
                "downEntry": False,
                "forceLock": False,
                "exitTarget": 1.0,
                "lockTarget": 0.5,
                "lockActive": True,
                "lockStart": 1,
            },
            {
                "i": 3,
                "cluster": -1,
                "strong": True,
                "rawStrong": False,
                "down": False,
                "downEntry": False,
                "forceLock": False,
                "exitTarget": 1.0,
                "lockTarget": 0.5,
                "lockActive": True,
                "lockStart": 1,
            },
            {
                "i": 4,
                "cluster": -1,
                "strong": False,
                "rawStrong": False,
                "down": False,
                "downEntry": False,
                "forceLock": False,
                "exitTarget": 1.0,
                "lockTarget": 0.5,
                "lockActive": True,
                "lockStart": 1,
            },
            {
                "i": 5,
                "cluster": 3,
                "strong": False,
                "rawStrong": False,
                "down": True,
                "downEntry": True,
                "forceLock": False,
                "exitTarget": 1.0,
                "lockTarget": 1.0,
                "lockActive": False,
                "lockStart": -1,
            },
        ],
        "state": {
            "strongDays": 0,
            "lockActive": False,
            "lockStart": -1,
            "lockTargetPct": 1.0,
            "lockHoldDays": 0.0,
            "ultraEntryPrice": 100.0,
            "ultraPeakPrice": 125.0,
            "prevStrong": False,
            "prevDown": True,
            "episodeLocked": True,
            "bridgeBars": 0,
            "lockReleaseOnStrong": False,
            "strongBars": 2,
            "bridgeStrongBars": 2,
            "lateBars": 1,
            "downBars": 1,
        },
        "stats": {
            "forcedLocks": 0,
            "strongBars": 2,
            "lateBars": 1,
            "downBars": 1,
            "buyShrinks": 0,
            "sellShrinks": 0,
            "targetBuys": 0,
            "bridgeStrongBars": 2,
            "crabCapSells": 0,
        },
    }


########################################################################
# Supervisor Golden
########################################################################

def supervisorOverrides() -> dict[str, Any]:
    return {
        "PEAK_LOCK_CAP_PCT": 0.35,
        "PEAK_LOCK_UNLOCK_GAIN_PCT": 5.0,
        "PEAK_LOCK_REENTRY_STEP_PCT": 0.25,
        "PEAK_LOCK_ARM_GAIN_PCT": 10.0,
        "PEAK_LOCK_GIVEBACK_PCT": 4.0,
        "PEAK_LOCK_MAX_DAYS": 10.0,
        "PEAK_LOCK_EDGE_DRAW_PCT": 1.0,
        "PEAK_LOCK_EDGE_SLOPE_DAYS": 1.0,
        "PEAK_LOCK_REQUIRE_EDGE_RISK": 0,
        "PEAK_LOCK_MA_DAYS": 2.0,
        "PEAK_LOCK_KP": 12.0,
        "PEAK_LOCK_KI": 0.0,
        "PEAK_LOCK_KD": 0.0,
        "PEAK_LOCK_INTEGRAL_DECAY": 0.985,
        "PEAK_LOCK_ENTRY_THRESHOLD": 0.25,
        "PEAK_LOCK_EXIT_THRESHOLD": 0.05,
        "PEAK_LOCK_CONFIRM_BARS": 1,
        "PEAK_LOCK_RELEASE_TARGET_PCT": 0.8,
        "PEAK_LOCK_ULTRA_GRACE_DAYS": 0.0,
    }


def _supervisorRow(label: str, state, **values) -> dict[str, Any]:
    row = {"step": label}
    row.update(values)
    row.update({
        "active": bool(state.active),
        "cap": round(float(state.cap), 6),
        "edgeNow": round(float(state.edgeNow), 6),
    })
    return row


def supervisorGolden() -> dict[str, Any]:
    cfg = peakLockConfigFromOverrides(supervisorOverrides(), 1.0)
    state = peakLockState(100.0, 100.0, 1000.0, 0.0)
    rows = []
    strongEntry = False
    graceActive = False
    decision = None

    strongEntry, graceActive = stepPeakStrong(state, cfg, True)
    armPeakLock(state, cfg, True, 15.0)
    decision = evaluatePeakLock(
        state,
        cfg,
        100.0,
        1000.0,
        0.0,
        strongEntry,
        graceActive,
    )
    rows.append(_supervisorRow(
        "arm",
        state,
        strongEntry=strongEntry,
        grace=graceActive,
        armed=bool(state.armed),
        canLock=bool(decision.canLock),
        bearCount=int(state.bearCount),
    ))

    stepPeakPid(state, cfg, 80.0)
    strongEntry, graceActive = stepPeakStrong(state, cfg, False)
    decision = evaluatePeakLock(
        state,
        cfg,
        80.0,
        1200.0,
        5.0,
        strongEntry,
        graceActive,
    )
    rows.append(_supervisorRow(
        "lock_decision",
        state,
        strongEntry=strongEntry,
        grace=graceActive,
        armed=bool(state.armed),
        canLock=bool(decision.canLock),
        bearCount=int(state.bearCount),
    ))
    if bool(decision.canLock):
        recordPeakLock(state, cfg, 1)
    stepActivePeakLock(state, cfg, 1)
    rows.append(_supervisorRow(
        "locked",
        state,
        locks=int(state.locks),
        lockHours=int(state.lockHours),
        edgeStart=round(float(state.edgeStart), 6),
        lockGain=round(float(state.lockGain), 6),
    ))

    strongEntry, graceActive = stepPeakStrong(state, cfg, True)
    evaluatePeakLock(
        state,
        cfg,
        85.0,
        1275.0,
        0.0,
        strongEntry,
        graceActive,
    )
    stepActivePeakLock(state, cfg, 2)
    rows.append(_supervisorRow(
        "release",
        state,
        strongEntry=strongEntry,
        strongReleases=int(state.strongReleases),
        lockHours=int(state.lockHours),
        edgeStart=round(float(state.edgeStart), 6),
    ))

    strongEntry, graceActive = stepPeakStrong(state, cfg, False)
    evaluatePeakLock(
        state,
        cfg,
        100.0,
        1550.0,
        0.0,
        strongEntry,
        graceActive,
    )
    stepActivePeakLock(state, cfg, 3)
    recordPeakCappedBuy(state)
    rows.append(_supervisorRow(
        "unlock",
        state,
        unlockSteps=int(state.unlockSteps),
        lockHours=int(state.lockHours),
        lockGainMax=round(float(state.lockGainMax), 6),
        edgeStart=round(float(state.edgeStart), 6),
    ))

    return {
        "rows": rows,
        "stats": _roundValue(peakLockStats(state)),
        "state": {
            "ma": round(float(state.ma), 6),
            "prevErr": round(float(state.prevErr), 6),
            "long": bool(state.long),
            "bearCount": int(state.bearCount),
            "active": bool(state.active),
            "start": int(state.start),
            "cap": round(float(state.cap), 6),
            "edgePeak": round(float(state.edgePeak), 6),
            "edgeVals": [round(float(i), 6) for i in state.edgeVals],
        },
    }


def expectedSupervisor() -> dict[str, Any]:
    return {
        "rows": [
            {
                "step": "arm",
                "strongEntry": True,
                "grace": False,
                "armed": True,
                "canLock": False,
                "bearCount": 0,
                "active": False,
                "cap": 1.0,
                "edgeNow": 0.0,
            },
            {
                "step": "lock_decision",
                "strongEntry": False,
                "grace": False,
                "armed": True,
                "canLock": True,
                "bearCount": 1,
                "active": False,
                "cap": 1.0,
                "edgeNow": 50.0,
            },
            {
                "step": "locked",
                "locks": 1,
                "lockHours": 1,
                "edgeStart": 50.0,
                "lockGain": 0.0,
                "active": True,
                "cap": 0.35,
                "edgeNow": 50.0,
            },
            {
                "step": "release",
                "strongEntry": True,
                "strongReleases": 1,
                "lockHours": 2,
                "edgeStart": 50.0,
                "active": True,
                "cap": 0.8,
                "edgeNow": 50.0,
            },
            {
                "step": "unlock",
                "unlockSteps": 0,
                "lockHours": 3,
                "lockGainMax": 5.0,
                "edgeStart": 50.0,
                "active": True,
                "cap": 0.8,
                "edgeNow": 55.0,
            },
        ],
        "stats": {
            "peakLocks": 1,
            "peakCappedBuys": 1,
            "peakUnlockSteps": 0,
            "peakLockHours": 3,
            "peakLockGainMax": 5.0,
            "peakStrongReleases": 1,
        },
        "state": {
            "ma": 86.666667,
            "prevErr": -0.076923,
            "long": True,
            "bearCount": 0,
            "active": True,
            "start": 1,
            "cap": 0.8,
            "edgePeak": 55.0,
            "edgeVals": [0.0, 50.0, 50.0, 55.0],
        },
    }


########################################################################
# Execution Golden
########################################################################

def executionGolden() -> dict[str, Any]:
    return {
        "phaseBuy": phaseBuyValue(1000.0, 10),
        "phaseSell": phaseSellValue(5.0, 100.0, 10),
        "buyScale": list(calcBuyScale(100.0, 90.0)),
        "sellScale": list(calcSellScale(100.0, 110.0)),
        "finalOne": gateFinalPortion(1.5, 1.0, 0.5),
        "finalCross": gateFinalPortion(2.0, 3.0, 0.5),
        "buySpend": list(buySpend(1000.0, 100.0, 1.2, 3.0, 0.5)),
        "buySpendCap": list(buySpend(
            1000.0,
            100.0,
            1.2,
            3.0,
            0.5,
            75.0,
        )),
        "sellQty": list(sellQty(5.0, 100.0, 100.0, 1.2, 2.0, 0.5, None)),
        "sellQtyCap": list(sellQty(
            5.0,
            100.0,
            100.0,
            1.2,
            2.0,
            0.5,
            80.0,
        )),
        "floorSell": floorSellValueCap(500.0, 10.0, 100.0, 0.5, 0.001),
        "targetBuy": buySpendToTargetCap(500.0, 2.0, 100.0, 0.5, 0.001),
        "dailyLock": dailyLockQty(500.0, 10.0, 100.0, 0.5, 0.001),
    }


def expectedExecution() -> dict[str, Any]:
    return {
        "phaseBuy": 100.0,
        "phaseSell": 50.0,
        "buyScale": [1.1, -0.1],
        "sellScale": [1.1, 0.1],
        "finalOne": 0.5,
        "finalCross": 2.0,
        "buySpend": [120.0, 1.2],
        "buySpendCap": [75.0, 1.2],
        "sellQty": [1.2, 1.2],
        "sellQtyCap": [0.8, 1.2],
        "floorSell": 250.125063,
        "targetBuy": 150.075038,
        "dailyLock": 2.501251,
    }


########################################################################
# Main
########################################################################

def main() -> int:
    _assertSame("posture", postureGolden(), expectedPosture())
    _assertSame("execution", executionGolden(), expectedExecution())
    _assertSame("supervisor", supervisorGolden(), expectedSupervisor())
    print("[strategy-golden] posture, execution, and supervisor checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
