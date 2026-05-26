#!/usr/bin/env python3
"""Pure signal helper utilities shared by trace and live adapters."""

from __future__ import annotations

import numpy as np


###############################################################################
# Array Helpers
###############################################################################

def fitSignalArray(
    values: np.ndarray | None,
    size: int,
    dtype,
) -> np.ndarray:
    out = np.zeros(int(size), dtype=dtype)
    if values is None:
        return out
    arr = np.asarray(values, dtype=dtype)
    limit = min(int(size), int(arr.shape[0]))
    if limit > 0:
        out[:limit] = arr[:limit]
    return out


###############################################################################
# Regime Helpers
###############################################################################

def regimeAnchors(allowReg: np.ndarray) -> np.ndarray:
    n = int(allowReg.size)
    out = np.full(n, -1, dtype=int)
    lastAnchor = -1
    prevAllow = False
    allow = False
    for i in range(n):
        allow = bool(allowReg[i])
        if allow and not prevAllow:
            lastAnchor = i
        out[i] = lastAnchor
        prevAllow = allow
    return out


def trendLabel(code: int) -> str:
    if int(code) == 1:
        return 'BULL'
    if int(code) == -1:
        return 'BEAR'
    if int(code) == 2:
        return 'HALF_BULL'
    if int(code) == -2:
        return 'HALF_BEAR'
    return 'NEUTRAL'
