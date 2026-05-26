#!/usr/bin/env python3
"""Deterministic check for the live signal/gate adapter boundary."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np


###############################################################################
# Import Path
###############################################################################

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from live import daily_posture
from live import live_engine
from repo_paths import LIVE_MODEL_DIR
from strategy.posture import DAILY_STRONG_CLUSTER


###############################################################################
# Fixtures
###############################################################################

def _syntheticRows(stepHours: int, count: int) -> list[list[float]]:
    startMs = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp())
    stepMs = int(stepHours) * 60 * 60 * 1000
    idx = np.arange(int(count), dtype=float)
    openMs = (startMs * 1000) + idx.astype(np.int64) * stepMs
    closeMs = openMs + stepMs - 1
    close = (
        100.0
        + (np.sin(idx / 12.0) * 6.0)
        + (np.sin(idx / 47.0) * 20.0)
        + (idx * 0.02)
    )
    openVal = np.concatenate((np.asarray([100.0]), close[:-1]))
    high = np.maximum(openVal, close) * 1.01
    low = np.minimum(openVal, close) * 0.99
    volume = np.full(int(count), 1000.0, dtype=float)
    return np.column_stack(
        (openMs, openVal, high, low, close, volume, closeMs),
    ).tolist()


def _overrides() -> dict[str, object]:
    return {
        'COOLDOWN': 5,
        'GRAD1_BUY_WIN_DAYS': 2,
        'GRAD1_SELL_WIN_DAYS': 2,
        'GRAD1_BUY_Z_MIN': 0.1,
        'GRAD1_SELL_Z_MIN': 0.1,
        'MACRO_INTERVAL': '4h',
        'MACRO_P1': 5,
        'MACRO_GRAD_PERIOD': 10,
        'MACRO_P3': 20,
        'MACRO_NRG_WIN_DAYS': 3,
        'MACRO_NRG_Z_MIN': 0.0,
        'MACRO_NRG_Z_MAX': 2.0,
        'MACRO_DYN_PCT_MIN': 0.2,
        'MACRO_DYN_PCT_MAX': 4.0,
        'MACRO_GRAD_WIN_DAYS': 3,
        'MACRO_GRAD_Z_MIN': 0.0,
        'MACRO_GRAD_Z_MAX': 2.0,
        'MACRO_MULT_GRAD_MIN': 0.6,
        'MACRO_MULT_GRAD_MAX': 1.4,
        'MACRO_SELL_RELAX_PCT': 15.0,
    }


###############################################################################
# Checks
###############################################################################

def _checkLiveSignals() -> None:
    rows = _syntheticRows(1, 360)
    macroRows = rows[::4]
    overrides = _overrides()
    pack = live_engine.evaluate(
        rows,
        macroRows,
        '1h',
        [5, 10, 20],
        1,
        overrides,
    )
    params = live_engine.paramsFromSettings(overrides)
    direct = live_engine.generateFlags(
        pack.ctx,
        pack.signals,
        params,
        pack.startIdx,
        overrides,
        pack.macroDyn,
        pack.macroDir,
        pack.macroMom,
    )
    assert pack.flags == direct
    assert len(pack.flags) == 27
    assert pack.flags[:5] == [
        (64, 'SELL'),
        (69, 'SELL'),
        (74, 'SELL'),
        (79, 'SELL'),
        (84, 'SELL'),
    ]
    assert pack.flags[-1] == (352, 'BUY')


def _checkPostureText() -> None:
    assert live_engine.postureText(DAILY_STRONG_CLUSTER) == 'up'
    assert live_engine.postureText(0) == 'down'
    assert live_engine.postureText(3) == 'down'
    assert live_engine.postureText(1) == 'neutral'
    assert live_engine.postureText(-1) == 'unknown'


def _checkLiveClusterFixture() -> None:
    path = LIVE_MODEL_DIR / (
        'linkusdt-6h-ema-fast-posture-k04-clustered_features.csv'
    )
    modelPath = LIVE_MODEL_DIR / 'cluster_model.json'
    model = daily_posture._loadModel(modelPath)
    rows = []
    expected = []
    with path.open(newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append([
                int(float(r['openMs'])),
                float(r['open']),
                float(r['high']),
                float(r['low']),
                float(r['close']),
                float(r['volume']),
                int(float(r['closeMs'])),
            ])
            expected.append(int(float(r['cluster'])))
    actual = daily_posture._inferClusters(rows, model)
    expectedArr = np.asarray(expected, dtype=int)
    valid = expectedArr >= 0
    assert int(np.sum(actual[valid] != expectedArr[valid])) == 0


###############################################################################
# Main
###############################################################################

def main() -> int:
    _checkLiveSignals()
    _checkPostureText()
    _checkLiveClusterFixture()
    print('[live-signal] shared signal boundary checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
