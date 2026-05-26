#!/usr/bin/env python3
# live_state_check.py - deterministic live restart-state roundtrip check.

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


###############################################################################
# Imports
###############################################################################

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from live.state_store import loadState, restoreDashboardState
from live.state_store import restorePhaseState, saveRuntimeState
from strategy.supervisor import PeakLockState


###############################################################################
# Helpers
###############################################################################

def _assertNear(name: str, actual: float, expected: float) -> None:
    if abs(float(actual) - float(expected)) > 1e-9:
        raise SystemExit(
            f"[live-state] {name} mismatch: {actual} != {expected}"
        )


def _assertSame(name: str, actual, expected) -> None:
    if actual != expected:
        raise SystemExit(
            f"[live-state] {name} mismatch: {actual} != {expected}"
        )


def _runCfg():
    return SimpleNamespace(
        paperTrading=True,
        dryRun=False,
        symbol='LINKUSDT',
        interval='1h',
        macroInterval='1d',
    )


def _dash():
    return SimpleNamespace(
        lastTrade={
            'timeMs': 100,
            'flag': 'BUY',
            'side': 'BUY',
            'orderId': 'paper-1',
        },
        lastClosedPrice=12.0,
        tradingEnabled=True,
        seeded=True,
        seedQuote=1000.0,
        hodlQty=100.0,
        hodlEntryPrice=10.0,
        quoteTotal=400.0,
        baseTotal=50.0,
        tradeCount=3,
        currentDailyCluster=2,
        currentPosture='up',
    )


def _phase():
    peak = PeakLockState(ma=9.5, benchQty=100.0)
    peak.integral = 0.25
    peak.prevErr = -0.1
    peak.long = True
    peak.bearCount = 2
    peak.strongGraceBars = 3
    peak.strongReleases = 4
    peak.prevStrong = True
    peak.active = True
    peak.start = 77
    peak.cap = 0.35
    peak.edgeStart = 12.5
    peak.edgeNow = 14.5
    peak.edgePeak = 18.0
    peak.lockGain = 2.0
    peak.lockGainMax = 5.0
    peak.locks = 6
    peak.cappedBuys = 7
    peak.lockHours = 8
    peak.unlockSteps = 9
    peak.armed = True
    peak.edgeVals = [1.0, 2.5, 3.0]
    return SimpleNamespace(
        phaseSide='BUY',
        phaseBaseValue=25.0,
        phaseLastPrice=11.0,
        phasePortionsRemaining=2.0,
        lastTrendLabel='BEAR',
        peakState=peak,
    )


###############################################################################
# Main
###############################################################################

def main() -> int:
    runCfg = _runCfg()
    dash = _dash()
    phase = _phase()
    with tempfile.TemporaryDirectory() as tmpDir:
        path = Path(tmpDir) / 'state.csv'
        saveRuntimeState(path, runCfg, dash, phase, 10, 20)
        row = loadState(path)
    newDash = _dash()
    newPhase = SimpleNamespace()
    restoreDashboardState(newDash, runCfg, row)
    restorePhaseState(newPhase, row)
    peak = newPhase.peakState
    _assertSame('phaseSide', newPhase.phaseSide, 'BUY')
    _assertNear('phaseBaseValue', newPhase.phaseBaseValue, 25.0)
    _assertNear('phaseLastPrice', newPhase.phaseLastPrice, 11.0)
    _assertNear('phaseRemaining', newPhase.phasePortionsRemaining, 2.0)
    _assertSame('lastTrendLabel', newPhase.lastTrendLabel, 'BEAR')
    _assertSame('peak exists', peak is not None, True)
    _assertNear('peak ma', peak.ma, 9.5)
    _assertNear('peak benchQty', peak.benchQty, 100.0)
    _assertSame('peak active', peak.active, True)
    _assertSame('peak locks', peak.locks, 6)
    _assertSame('peak cappedBuys', peak.cappedBuys, 7)
    _assertSame('peak edgeVals', peak.edgeVals, [1.0, 2.5, 3.0])
    _assertNear('dash quote', newDash.quoteTotal, 400.0)
    print('[live-state] restart state checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
