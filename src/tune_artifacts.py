#!/usr/bin/env python3
# tune_artifacts.py – artifact writers for tuned configs.

from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Any, Callable, Dict, Tuple


CONFIG_KEY_ORDER = [
    "ticker",
    "tickers",
    "primer_days",
    "tuner_days",
    "holdout_days",
    "intervals",
    "p1",
    "p2",
    "p3",
    "GRAD1_BUY_Z_MIN",
    "GRAD1_SELL_Z_MIN",
    "GRAD1_BUY_WIN_DAYS",
    "GRAD1_SELL_WIN_DAYS",
    "SPACING_Z_MIN_12",
    "SPACING_Z_MIN_23",
    "SPACING_WIN_DAYS_12",
    "SPACING_WIN_DAYS_23",
    "MICRO_NRG_MODEL",
    "MICRO_NRG_WIN_DAYS",
    "MICRO_NRG_MIN_12",
    "MICRO_NRG_MIN_23",
    "PHASE_BUY_PORTIONS",
    "MACRO_INTERVAL",
    "MACRO_P1",
    "MACRO_P2",
    "MACRO_P3",
    "MACRO_NRG_WIN_DAYS",
    "MACRO_NRG_Z_MIN",
    "MACRO_NRG_Z_MAX",
    "MACRO_DYN_PCT_MIN",
    "MACRO_DYN_PCT_MAX",
    "MACRO_GRAD_WIN_DAYS",
    "MACRO_GRAD_Z_MIN",
    "MACRO_GRAD_Z_MAX",
    "MACRO_MULT_GRAD_MIN",
    "MACRO_MULT_GRAD_MAX",
    "MACRO_BUY_MULT_BULL",
    "MACRO_BUY_MULT_BEAR",
    "MACRO_BUY_MULT_REV",
    "MACRO_BUY_MULT_ROLL",
    "MACRO_SELL_MULT_BULL",
    "MACRO_SELL_MULT_BEAR",
    "MACRO_SELL_MULT_REV",
    "MACRO_SELL_MULT_ROLL",
    "PHASE_SELL_PORTIONS",
    "FINAL_PORTION_PCT",
    "COOLDOWN",
    "SUMMARY_LABEL",
    "WALLET_SEED_QUOTE",
    "WALLET_FEE_RATE",
    "QUOTE_TO_AUD_RATE",
    "TAX_MODE",
    "ANNUAL_INCOME_BASE",
    "PROFIT_SWEEP_INTERVAL",
    "PROFIT_SWEEP_SHARE",
    "out",
    "CHART_CHUNK_SIZE",
]


def orderedConfig(
    data: Dict[str, Any],
    shape: Dict[str, Any] | None = None,
) -> "OrderedDict[str, Any]":
    ordered: "OrderedDict[str, Any]" = OrderedDict()
    if shape is not None:
        for key in shape.keys():
            if key in data:
                ordered[key] = data[key]
    for key in CONFIG_KEY_ORDER:
        if key in data and key not in ordered:
            ordered[key] = data[key]
    for key, value in data.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def roundForJson(value: Any, places: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, places)
    if isinstance(value, list):
        return [roundForJson(v, places) for v in value]
    if isinstance(value, tuple):
        return tuple(roundForJson(v, places) for v in value)
    if isinstance(value, dict):
        return {k: roundForJson(v, places) for k, v in value.items()}
    return value


def bestConfigFromRow(
    rowData: dict,
    config: dict,
) -> "OrderedDict[str, Any]":
    # Keep parity with the input profile shape/keys (incl. "out" and
    # metadata fields), but override tuned axes with the chosen row.
    bestConfigLocal = {
        key: value
        for key, value in config.items()
        if key != "ticker"
    }
    if isinstance(bestConfigLocal.get("intervals"), str):
        bestConfigLocal["intervals"] = str(rowData["interval"])
    else:
        bestConfigLocal["intervals"] = [str(rowData["interval"])]
    for key, value in rowData.items():
        if key in bestConfigLocal:
            bestConfigLocal[key] = value
    return orderedConfig(bestConfigLocal, config)


def writeBestArtifacts(
    rowData: dict,
    outDir: str,
    config: dict,
    ticker: str,
    days: int,
    klinesByInterval: dict,
    ensureKlinesFn: Callable[[str, str, int, int], list],
    suffix: str | None = None,
    containerDir: str | None = None,
) -> Tuple["OrderedDict[str, Any]", str, None, str]:
    baseRoot = containerDir if containerDir else outDir
    targetDir = os.path.join(baseRoot, "best-configs")
    os.makedirs(targetDir, exist_ok=True)

    bestPeriods = [rowData["p1"], rowData["p2"], rowData["p3"]]
    minCandlesLocal = (max(bestPeriods) * 2) + 1
    intervalValue = rowData["interval"]
    intervalKlines = klinesByInterval.get(intervalValue)
    if intervalKlines is None or len(intervalKlines) < minCandlesLocal:
        intervalKlines = ensureKlinesFn(ticker, intervalValue, days, minCandlesLocal)
        klinesByInterval[intervalValue] = intervalKlines

    bestConfigOrdered = bestConfigFromRow(rowData, config)
    if suffix is None:
        fileName = "best-config.json"
    elif suffix == "stats":
        fileName = "beststats-config.json"
    else:
        fileName = f"best{suffix}-config.json"
    bestConfigPathLocal = os.path.join(targetDir, fileName)
    tmpPath = f"{bestConfigPathLocal}.tmp"
    with open(tmpPath, "w") as bestFile:
        json.dump(
            roundForJson(bestConfigOrdered, places=6),
            bestFile,
            indent=2,
        )
    os.replace(tmpPath, bestConfigPathLocal)
    return bestConfigOrdered, bestConfigPathLocal, None, targetDir


__all__ = [
    "CONFIG_KEY_ORDER",
    "bestConfigFromRow",
    "orderedConfig",
    "roundForJson",
    "writeBestArtifacts",
]
