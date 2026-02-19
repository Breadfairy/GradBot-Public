#!/usr/bin/env python3
# tune_select.py – selection helpers for tuner outputs.

from __future__ import annotations

import math

import numpy as np


def _nanMin(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmin(arr))


def riskScoreFromRow(row: dict) -> float:
    """Risk-adjusted score used for selecting stats config.

    Worst-window weighted: prefers configs whose relative Sharpe/Sortino stay
    strong across both 4w and 13w horizons, with MAR (CAGR/MDD) as a smaller
    term.
    """
    cagrVal = float(row.get("cagr", float("nan")))
    mddVal = float(row.get("mdd", float("nan")))
    if (
        not math.isfinite(cagrVal)
        or not math.isfinite(mddVal)
        or mddVal <= 1e-12
    ):
        return float("-inf")
    mar = cagrVal / mddVal
    sharpeWorst = _nanMin([
        float(row.get("sharpe4w", float("nan"))),
        float(row.get("sharpe13w", float("nan"))),
    ])
    sortinoWorst = _nanMin([
        float(row.get("sortino4w", float("nan"))),
        float(row.get("sortino13w", float("nan"))),
    ])
    score = (0.10 * mar) + (0.45 * sharpeWorst) + (0.45 * sortinoWorst)
    return score if math.isfinite(score) else float("-inf")


__all__ = [
    "riskScoreFromRow",
]
