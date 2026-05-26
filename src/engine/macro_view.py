#!/usr/bin/env python3
# macro_view.py - shared macro alignment helpers for active runtime paths.

from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
from typing import Any, Iterable

import numpy as np

from engine.dynamics import alignMacroDyn, macroDynFromContext
from engine.shared import buildContext, trendCodes
from data.klines_io import loadWindowedKlines


@dataclass(frozen=True)
class MacroView:
    dyn: np.ndarray
    trend: np.ndarray
    dir: np.ndarray
    mom: np.ndarray
    close: np.ndarray
    mas: list[np.ndarray]
    periods: list[int]


def _targetMs(targetTs: Iterable[Any]) -> np.ndarray:
    arr = np.asarray(targetTs)
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(float)
    out: list[float] = []
    for item in targetTs:
        if getattr(item, "tzinfo", None) is None:
            out.append(
                item.replace(tzinfo=timezone.utc).timestamp() * 1000.0
            )
        else:
            out.append(item.timestamp() * 1000.0)
    return np.asarray(
        out,
        dtype=float,
    )


def _macroPeriods(basePeriods: list[int], overrides: dict) -> list[int]:
    periods = list(basePeriods)
    macroP1 = int(overrides["MACRO_P1"])
    macroP3 = int(overrides["MACRO_P3"])
    macroGradPeriod = int(overrides["MACRO_GRAD_PERIOD"])
    if macroP1 > 0 and len(periods) >= 1:
        periods[0] = macroP1
    if macroGradPeriod > 0 and len(periods) >= 2:
        periods[1] = macroGradPeriod
    if macroP3 > 0:
        if len(periods) >= 3:
            periods[2] = macroP3
        else:
            periods.append(macroP3)
    if not periods:
        return [1]
    return periods


def buildMacroView(
    ticker: str,
    days: int,
    holdoutDays: int,
    basePeriods: list[int],
    overrides: dict,
    targetTs: Iterable[Any],
    anchorMs: int | None = None,
) -> MacroView | None:
    intervalMacro = str(overrides["MACRO_INTERVAL"]).strip()
    if not intervalMacro or int(days) <= 0:
        return None

    periods = _macroPeriods(basePeriods, overrides)
    minCandles = max(int(max(periods) * 2 + 1), 1)
    klines = loadWindowedKlines(
        ticker,
        intervalMacro,
        int(days),
        minCandles,
        holdoutDays=holdoutDays,
        anchorMs=anchorMs,
    )
    ctx = buildContext(klines, periods)
    ctx["intervalStr"] = intervalMacro

    # Macro state is derived from closed candles, so expose it at close time.
    tsMacro = np.asarray([row[6] for row in klines], dtype=float)
    tsTarget = _targetMs(targetTs)

    dyn = macroDynFromContext(
        ctx,
        float(overrides["MACRO_NRG_WIN_DAYS"]),
        float(overrides["MACRO_NRG_Z_MIN"]),
        float(overrides["MACRO_NRG_Z_MAX"]),
        float(overrides["MACRO_DYN_PCT_MAX"]),
        float(overrides["MACRO_DYN_PCT_MIN"]),
        gradWinDays=float(overrides["MACRO_GRAD_WIN_DAYS"]),
        gradZMin=float(overrides["MACRO_GRAD_Z_MIN"]),
        gradZMax=float(overrides["MACRO_GRAD_Z_MAX"]),
        gradMultMin=float(overrides["MACRO_MULT_GRAD_MIN"]),
        gradMultMax=float(overrides["MACRO_MULT_GRAD_MAX"]),
    )
    mas = [np.asarray(ma, dtype=float) for ma in ctx["mas"][:3]]
    m1 = mas[0]
    m2 = mas[1]
    m3 = mas[2]
    trendRaw = trendCodes(m1, m2, m3)
    dirRaw = np.zeros_like(m1, dtype=int)
    dirRaw[m1 > m3] = 1
    dirRaw[m1 < m3] = -1
    momRaw = np.zeros_like(m1, dtype=int)
    momRaw[m1 > m2] = 1
    momRaw[m1 < m2] = -1

    return MacroView(
        dyn=alignMacroDyn(tsMacro, dyn, tsTarget),
        trend=alignMacroDyn(
            tsMacro,
            trendRaw.astype(float),
            tsTarget,
        ).astype(int),
        dir=alignMacroDyn(tsMacro, dirRaw.astype(float), tsTarget).astype(int),
        mom=alignMacroDyn(tsMacro, momRaw.astype(float), tsTarget).astype(int),
        close=alignMacroDyn(
            tsMacro,
            np.asarray(ctx["closes"], dtype=float),
            tsTarget,
        ),
        mas=[
            alignMacroDyn(tsMacro, series, tsTarget)
            for series in mas
        ],
        periods=periods,
    )


__all__ = [
    "MacroView",
    "buildMacroView",
]
