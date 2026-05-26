#!/usr/bin/env python3
# dynamics.py – dynamic threshold helpers.

from __future__ import annotations

import numpy as np

from engine.shared import bars_per_day
from engine import core


def macroDynFromContext(
    ctx,
    winDays: float,
    zmin: float,
    zmax: float,
    pctMax: float,
    pctMin: float,
    gradWinDays: float = 0.0,
    gradZMin: float = 0.0,
    gradZMax: float = 0.0,
    gradMultMin: float = 1.0,
    gradMultMax: float = 1.0,
) -> np.ndarray:
    """Compute macro dyn% series from a macro ctx.

    Delegates to cache-free numeric kernels in engine_core for easier
    eventual porting to C.
    """
    m1 = np.asarray(ctx["mas"][0], dtype=float)
    m2 = np.asarray(ctx["mas"][1], dtype=float)
    m3 = np.asarray(ctx["mas"][2], dtype=float)
    bpd = float(bars_per_day(ctx))
    return core.macroDynFromMas(
        m1,
        m2,
        m3,
        bpd,
        float(winDays),
        float(zmin),
        float(zmax),
        float(pctMax),
        float(pctMin),
        float(gradWinDays),
        float(gradZMin),
        float(gradZMax),
        float(gradMultMin),
        float(gradMultMax),
    )


def alignMacroDyn(
    tsMacro,
    dynMacro: np.ndarray,
    tsMicro,
) -> np.ndarray:
    """Align macro dyn% to micro timestamps using last-known sample."""
    tsMacroArr = np.asarray(tsMacro)
    tsMicroArr = np.asarray(tsMicro)
    out = np.zeros(tsMicroArr.shape, dtype=float)
    if tsMacroArr.size == 0:
        return out
    j = 0
    last = tsMacroArr.size - 1
    for i, tval in enumerate(tsMicroArr):
        while j < last and tsMacroArr[j + 1] <= tval:
            j += 1
        out[i] = dynMacro[j]
    return out


__all__ = [
    'macroDynFromContext',
    'alignMacroDyn',
]
