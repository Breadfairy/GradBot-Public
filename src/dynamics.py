#!/usr/bin/env python3
# dynamics.py – dynamic threshold helpers.

from __future__ import annotations

import numpy as np

from engine_shared import bars_per_day
import engine_core as core

def g1p3Series(ctx) -> np.ndarray:
    sm3 = np.asarray(ctx["smoothMas"][2], dtype=float)
    n = sm3.size
    out = np.zeros(n, dtype=float)
    if n <= 1:
        return out
    num = sm3[1:] - sm3[:-1]
    den = np.where(sm3[1:] != 0.0, sm3[1:], 1e-12)
    out[1:] = (num / den) * 100.0
    return out


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
    'g1p3Series',
    'macroDynFromContext',
    'alignMacroDyn',
]
