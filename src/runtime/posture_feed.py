#!/usr/bin/env python3
# daily_posture.py - 1d cluster posture helpers for wallet execution.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from strategy.posture import (
    DAILY_DOWN_MASK,
    DAILY_STRONG_CLUSTER,
    dailyPostureState,
    dailyPostureStats,
    dailyPostureStep as strategyPostureStep,
    defaultDailyPosture,
)


########################################################################
# Fixed Daily Model
########################################################################

ROOT_DIR = Path(__file__).resolve().parents[2]


########################################################################
# Config Helpers
########################################################################

def dailyPostureEnabled(overrides: Dict[str, Any] | None) -> bool:
    ov = overrides or {}
    return bool(str(ov.get("DAILY_CLUSTER_PATH", "")).strip())


def _clusterPath(rawPath: str) -> str:
    path = Path(rawPath)
    return str(path if path.is_absolute() else ROOT_DIR / path)


########################################################################
# Cluster Alignment
########################################################################

def dailyPostureArrays(
    ctx: Dict[str, Any],
    overrides: Dict[str, Any] | None,
) -> Dict[str, np.ndarray] | None:
    ov = overrides or {}
    if not dailyPostureEnabled(ov):
        return None
    path = _clusterPath(str(ov["DAILY_CLUSTER_PATH"]).strip())

    frame = pd.read_csv(path, usecols=["closeMs", "close", "cluster"])
    frame = frame[frame["cluster"] >= 0].copy()
    frame["ret30"] = (
        (frame["close"] / frame["close"].shift(30)) - 1.0
    ) * 100.0
    high = frame["close"].rolling(60, min_periods=1).max()
    frame["nearHigh"] = ((high / frame["close"]) - 1.0) * 100.0

    dayClose = frame["closeMs"].to_numpy(dtype=np.int64)
    kOpen = np.asarray([int(k[0]) for k in ctx["klines"]], dtype=np.int64)
    posRaw = np.searchsorted(dayClose, kOpen, side="right") - 1
    valid = posRaw >= 0
    pos = np.clip(posRaw, 0, len(frame) - 1)
    cluster = np.full(kOpen.size, -1, dtype=int)
    ret30 = np.zeros(kOpen.size, dtype=float)
    nearHigh = np.zeros(kOpen.size, dtype=float)
    cluster[valid] = frame["cluster"].to_numpy(dtype=int)[pos[valid]]
    ret30[valid] = frame["ret30"].fillna(0.0).to_numpy(dtype=float)[
        pos[valid]
    ]
    nearHigh[valid] = frame["nearHigh"].fillna(0.0).to_numpy(dtype=float)[
        pos[valid]
    ]
    return {
        "cluster": cluster,
        "ret30": ret30,
        "nearHigh": nearHigh,
    }


########################################################################
# Runtime Step
########################################################################

def dailyPostureStep(
    state: Dict[str, Any],
    daily: Dict[str, np.ndarray] | None,
    index: int,
    price: float,
    barsDay: float,
    overrides: Dict[str, Any] | None,
) -> Dict[str, Any]:
    if daily is None:
        return defaultDailyPosture()
    cluster = int(daily["cluster"][index])
    return strategyPostureStep(
        state,
        cluster,
        price,
        index,
        barsDay,
        overrides,
    )
