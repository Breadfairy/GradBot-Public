#!/usr/bin/env python3
# trace_golden_check.py - fixed LINK trace regression checks.

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


###############################################################################
# Imports
###############################################################################

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from data.time_bounds import resolveAnchorMs
from tools.common import loadConfig, loadKlinesForWindow
from tools.common import runTraceForWindow


###############################################################################
# Constants
###############################################################################

PROFILE_PATH = (
    'outputs/codex/link-2021-archetype-transition-fix/rel80.json'
)
ANCHOR_DATE = '2022-03-31'


###############################################################################
# Helpers
###############################################################################

def _roundValue(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: _roundValue(v) for k, v in value.items()}
    return value


def _assertSame(name: str, actual: dict[str, Any], expected: dict[str, Any]):
    actualText = json.dumps(_roundValue(actual), sort_keys=True, indent=2)
    expectedText = json.dumps(expected, sort_keys=True, indent=2)
    if actualText != expectedText:
        print(f"[trace-golden] {name} mismatch", file=sys.stderr)
        print("[trace-golden] actual:", file=sys.stderr)
        print(actualText, file=sys.stderr)
        print("[trace-golden] expected:", file=sys.stderr)
        print(expectedText, file=sys.stderr)
        raise SystemExit(1)


def _windowRow(window: str) -> dict[str, Any]:
    cfg = loadConfig(PROFILE_PATH)
    anchorMs = resolveAnchorMs(anchorDate=ANCHOR_DATE)
    klines, parts = loadKlinesForWindow(cfg, window, anchorMs=anchorMs)
    result, _trace = runTraceForWindow(
        cfg,
        klines,
        parts,
        window,
        anchorMs=anchorMs,
    )
    return {
        'window': window,
        'portfolio': float(result.sim['portfolio_value']),
        'bench': float(result.bench['portfolio_value']),
        'grossVsHodl': (
            (float(result.sim['portfolio_value'])
             / float(result.bench['portfolio_value']))
            - 1.0
        ) * 100.0,
        'trades': int(result.sim['trades']),
        'buyTrades': int(result.buyTrades),
        'sellTrades': int(result.sellTrades),
        'mdd': float(result.mdd),
        'lifecycleEdgeScore': float(result.lifecycleEdgeScore),
        'lifecycleEdgeMean': float(result.lifecycleEdgeMean),
        'postureStats': dict(result.postureStats or {}),
        'tradeNotes': dict(result.tradeNotes or {}),
        'executionHealth': dict(result.executionHealth or {}),
    }


###############################################################################
# Expected
###############################################################################

def expectedTune() -> dict[str, Any]:
    return {
        'window': 'tune',
        'portfolio': 131759.52693,
        'bench': 125667.23669,
        'grossVsHodl': 4.847954,
        'trades': 251,
        'buyTrades': 172,
        'sellTrades': 79,
        'mdd': 0.436816,
        'lifecycleEdgeScore': -104.897129,
        'lifecycleEdgeMean': 12.50787,
        'postureStats': {
            'bridgeStrongBars': 0,
            'buyShrinks': 81,
            'crabCapSells': 0,
            'downBars': 4467,
            'forcedLocks': 8,
            'lateBars': 15,
            'peakCappedBuys': 59,
            'peakLockGainMax': 58.698013,
            'peakLockHours': 6621,
            'peakLocks': 3,
            'peakStrongReleases': 3,
            'peakUnlockSteps': 2,
            'sellShrinks': 115,
            'strongBars': 6427,
            'targetBuys': 46,
        },
        'tradeNotes': {
            'daily_posture_lock': 8,
            'daily_strong_target_buy': 46,
            'peak_lock': 3,
            'peak_lock_capped_buy': 59,
            'seed_buy': 1,
            'signal_trade': 134,
        },
        'executionHealth': {
            'day_opposite_flips': 7,
            'neutral_half_exposure_bars': 14,
            'neutral_half_exposure_pct': 0.118403,
            'neutral_low_exposure_bars': 0,
            'neutral_low_exposure_pct': 0.0,
            'same_bar_opposite_flips': 0,
        },
    }


def expectedHoldout() -> dict[str, Any]:
    return {
        'window': 'holdout',
        'portfolio': 6925.278481,
        'bench': 5099.123437,
        'grossVsHodl': 35.813117,
        'trades': 283,
        'buyTrades': 195,
        'sellTrades': 88,
        'mdd': 0.603739,
        'lifecycleEdgeScore': 6.003079,
        'lifecycleEdgeMean': 27.005035,
        'postureStats': {
            'bridgeStrongBars': 0,
            'buyShrinks': 107,
            'crabCapSells': 0,
            'downBars': 5263,
            'forcedLocks': 6,
            'lateBars': 8,
            'peakCappedBuys': 96,
            'peakLockGainMax': 71.120568,
            'peakLockHours': 7221,
            'peakLocks': 3,
            'peakStrongReleases': 3,
            'peakUnlockSteps': 2,
            'sellShrinks': 63,
            'strongBars': 3124,
            'targetBuys': 69,
        },
        'tradeNotes': {
            'daily_posture_lock': 6,
            'daily_strong_target_buy': 69,
            'peak_lock': 3,
            'peak_lock_capped_buy': 96,
            'seed_buy': 1,
            'signal_trade': 108,
        },
        'executionHealth': {
            'day_opposite_flips': 8,
            'neutral_half_exposure_bars': 0,
            'neutral_half_exposure_pct': 0.0,
            'neutral_low_exposure_bars': 0,
            'neutral_low_exposure_pct': 0.0,
            'same_bar_opposite_flips': 0,
        },
    }


###############################################################################
# Main
###############################################################################

def main() -> int:
    _assertSame('tune', _windowRow('tune'), expectedTune())
    _assertSame('holdout', _windowRow('holdout'), expectedHoldout())
    print('[trace-golden] fixed LINK checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
