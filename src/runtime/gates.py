#!/usr/bin/env python3

########################################################################
# Imports
########################################################################

from dataclasses import dataclass
from typing import List

import numpy as np

from engine.shared import bars_per_day, zscoreSeries


########################################################################
# Parameters
########################################################################

@dataclass(frozen=True)
class Params:
    # Mechanics only
    COOLDOWN: int


def paramsFromSettings(overrides: dict | None = None) -> Params:
    overridesDict = overrides or {}

    def _require_int(name: str) -> int:
        if name not in overridesDict:
            raise KeyError(f"missing required param: {name}")
        return int(overridesDict[name])

    cooldown = _require_int('COOLDOWN')

    return Params(
        COOLDOWN=cooldown,
    )


def enforceCooldown(indices: np.ndarray, cooldown: int) -> List[int]:
    if indices.size == 0:
        return []
    keep: List[int] = []
    last = indices[0] - cooldown
    for idx in indices.tolist():
        if idx - last >= cooldown:
            keep.append(idx)
            last = idx
    return keep


def grad1ZscoreMask(
    ctx,
    allowReg: np.ndarray,
    g1: np.ndarray,
    overrides: dict,
    side: str,
) -> np.ndarray:
    winKey = f'GRAD1_{side}_WIN_DAYS'
    zKey = f'GRAD1_{side}_Z_MIN'
    winDays = max(int(overrides[winKey]), 1)
    winBars = max(int(round(winDays * bars_per_day(ctx))), 1)
    thresh = float(overrides[zKey])
    z, valid = zscoreSeries(ctx, g1, winBars, "g1p1")
    sign = -1.0 if side == 'BUY' else 1.0
    signed = z * sign
    ready = allowReg & valid
    return ready & (signed >= thresh)
