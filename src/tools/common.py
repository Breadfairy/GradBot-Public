#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
os.makedirs(ROOT_DIR / ".mplconfig", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".mplconfig"))

from config import profile
from data.klines_io import loadWindowedKlines
from analysis.reporting import resultMetrics
from tune.trace import Trace


BASE_FIELDS = {
    "ticker",
    "tickers",
    "intervals",
    "p1",
    "p2",
    "p3",
    "primer_days",
    "training_days",
    "tuner_days",
    "holdout_days",
}


def loadConfig(path: str | Path) -> dict[str, Any]:
    cfg = profile.loadJson(str(path))
    profile.ensureFinalPortionPct(cfg)
    return cfg


def configParts(
    cfg: dict[str, Any],
) -> tuple[
    str,
    str,
    list[int],
    dict[str, Any],
    int,
    int,
    int,
    int,
    int,
]:
    ticker = profile._requireTickers(cfg)[0]
    interval = profile.intervalsFromConfig(cfg)[0]
    periods = [int(cfg["p1"]), int(cfg["p2"]), int(cfg["p3"])]
    primerDays, trainingDays, tunerDays, holdoutDays, totalDays = (
        profile.windowParts(cfg)
    )
    overrides = {k: v for k, v in cfg.items() if k not in BASE_FIELDS}
    overridesNorm = profile.overrides(overrides)
    profile.validate(overridesNorm, kind="backtest")
    return (
        ticker,
        interval,
        periods,
        overridesNorm,
        primerDays,
        trainingDays,
        tunerDays,
        holdoutDays,
        totalDays,
    )


def activePrimerDays(
    primerDays: int,
    trainingDays: int,
    tunerDays: int,
    window: str,
    startOffsetDays: int = 0,
) -> int:
    if str(window).strip().lower() == "tune":
        return primerDays + trainingDays + startOffsetDays
    return primerDays + trainingDays + tunerDays + startOffsetDays


def loadKlinesForWindow(
    cfg: dict[str, Any],
    window: str,
    anchorMs: int | None = None,
) -> tuple[list[Any], tuple[Any, ...]]:
    parts = configParts(cfg)
    ticker, interval, periods = parts[0], parts[1], parts[2]
    holdoutDays = int(parts[7])
    totalDays = int(parts[8])
    holdoutTrim = holdoutDays if str(window).lower() == "tune" else 0
    klines = loadWindowedKlines(
        ticker,
        interval,
        totalDays,
        (max(periods) * 2) + 1,
        holdoutDays=holdoutTrim,
        anchorMs=anchorMs,
    )
    return klines, parts


def runTraceForWindow(
    cfg: dict[str, Any],
    klines: list[Any],
    parts: tuple[Any, ...],
    window: str,
    anchorMs: int | None = None,
    startOffsetDays: int = 0,
    showCharts: bool = False,
    chartsDir: str | Path | None = None,
) -> tuple[Any, Trace]:
    ticker = str(parts[0])
    interval = str(parts[1])
    periods = list(parts[2])
    overrides = dict(parts[3])
    primerDays = int(parts[4])
    trainingDays = int(parts[5])
    tunerDays = int(parts[6])
    holdoutDays = int(parts[7])
    totalDays = int(parts[8])
    primerActive = activePrimerDays(
        primerDays,
        trainingDays,
        tunerDays,
        window,
        startOffsetDays=startOffsetDays,
    )
    if chartsDir is not None:
        os.makedirs(chartsDir, exist_ok=True)
        os.environ["CHARTS_OUT_DIR"] = str(chartsDir)
    bt = Trace(
        ticker,
        klines,
        interval,
        periods,
        days=totalDays,
        showCharts=showCharts,
        showPrints=False,
        showSummary=False,
        overrides=overrides,
        computeRisk=True,
        primerDays=primerActive,
        holdoutDays=0 if str(window).lower() == "holdout" else holdoutDays,
        anchorMs=anchorMs,
    )
    return bt.run(), bt

def metricRow(label: str, ticker: str, result: Any) -> dict[str, Any]:
    row = resultMetrics(label, ticker, result)
    row["grossVsHodl"] = float(row["pct"])
    row["edgePct"] = float(row["edge"])
    row["hodlPct"] = float(row["hodl"])
    row["mddPct"] = float(row["mdd"]) * 100.0
    row["cagrPct"] = float(row["cagr"]) * 100.0
    return row
