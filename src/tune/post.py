#!/usr/bin/env python3
# tune_post.py - cache-free post-run helpers for the host tune pipeline.

from __future__ import annotations

import contextlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT_DIR_LOCAL = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR_LOCAL / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT_DIR_LOCAL / ".cache"))

from analysis.charting import generateScatter, plotTimVal
from data.klines_io import loadWindowedKlines
from config import profile
from analysis.reporting import printMetricBlock, resultMetrics
from tune.trace import Trace
from tune.artifacts import (
    bestConfigFromRow,
    configsEqual,
    writeBestArtifacts,
)
from tune.axes import axesFromConfig
from tune.fingerprint import buildFingerprintAt
from tune.robust import robustRowsFromResults


BAR = "=" * 55


def buildIntervalKlines(
    loadKlinesFn: Callable[[str, str, int, int], list],
    ticker: str,
    intervals: List[str],
    days: int,
    minCandles: int,
) -> Dict[str, list]:
    return {
        intervalValue: loadKlinesFn(
            ticker,
            intervalValue,
            days,
            minCandles,
        )
        for intervalValue in intervals
    }


def _writeFingerprint(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


def _render_summary_to_path(
    rowObj: dict,
    cfgObj: dict,
    destPath: str,
    ticker: str,
    days: int,
    klinesByInterval: dict,
    holdoutDays: int = 0,
    anchorMs: int | None = None,
) -> None:
    primerDays, trainingDays, _tunerDays, holdoutLocal, totalDays = (
        profile.profileWindows(cfgObj)
    )
    activePrimerDays = primerDays + trainingDays
    intervalValue = rowObj["interval"]
    periodsLocal = [
        int(rowObj["p1"]),
        int(rowObj["p2"]),
        int(rowObj["p3"]),
    ]
    klines = klinesByInterval.get(intervalValue)
    minCandlesLocal = (max(periodsLocal) * 2) + 1
    if klines is None or len(klines) < minCandlesLocal:
        klines = loadWindowedKlines(
            ticker,
            intervalValue,
            totalDays,
            minCandlesLocal,
            holdoutDays=holdoutLocal,
            anchorMs=anchorMs,
        )
        klinesByInterval[intervalValue] = klines
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        Trace(
            ticker,
            klines,
            intervalValue,
            periodsLocal,
            days=days,
            showCharts=False,
            showPrints=False,
            showSummary=True,
            overrides=cfgObj,
            ctx=None,
            signals=None,
            primerDays=activePrimerDays,
            holdoutDays=holdoutDays,
            anchorMs=anchorMs,
        ).run()
    tmp = f"{destPath}.tmp"
    with open(tmp, "w") as outf:
        outf.write(buf.getvalue())
    os.replace(tmp, destPath)


def _chartEnabled(cfgObj: dict, key: str) -> bool:
    return bool(cfgObj.get(key, True))


def traceTuneConfig(
    cfgObj: dict,
    outDir: str | Path,
    label: str,
    anchorMs: int | None = None,
):
    tickers = cfgObj["tickers"]
    ticker = str(tickers[0])
    primerDays, trainingDays, _tunerDays, holdoutDays, totalDays = (
        profile.profileWindows(cfgObj)
    )
    activePrimerDays = primerDays + trainingDays
    intervals = profile.intervalsFromConfig(cfgObj)
    intervalValue = str(intervals[0])
    periodsLocal = [
        int(cfgObj["p1"]),
        int(cfgObj["p2"]),
        int(cfgObj["p3"]),
    ]
    minCandlesLocal = (max(periodsLocal) * 2) + 1
    chartsRoot = Path(outDir) / "charts" / "tune"
    traceChartsDir = chartsRoot / str(label)
    chartTrades = _chartEnabled(cfgObj, "CHARTS_TRADES")
    chartTimeVal = _chartEnabled(cfgObj, "CHARTS_TIMEVAL")
    prevChartsDir = os.environ.get("CHARTS_OUT_DIR")
    klines = loadWindowedKlines(
        ticker,
        intervalValue,
        totalDays,
        minCandlesLocal,
        holdoutDays=holdoutDays,
        anchorMs=anchorMs,
    )

    chartsRoot.mkdir(parents=True, exist_ok=True)
    if chartTrades:
        traceChartsDir.mkdir(parents=True, exist_ok=True)
        os.environ["CHARTS_OUT_DIR"] = str(traceChartsDir)
    else:
        os.environ.pop("CHARTS_OUT_DIR", None)

    result = Trace(
        ticker,
        klines,
        intervalValue,
        periodsLocal,
        days=totalDays,
        showCharts=bool(chartTrades),
        showPrints=False,
        showSummary=False,
        overrides=cfgObj,
        ctx=None,
        signals=None,
        computeRisk=True,
        primerDays=activePrimerDays,
        holdoutDays=holdoutDays,
        anchorMs=anchorMs,
    ).run()

    if (
        chartTimeVal
        and result.curveTs is not None
        and result.curveSim is not None
        and result.curveBench is not None
        and result.curveAssetFrac is not None
        and result.curveQuoteFrac is not None
    ):
        timValPath = chartsRoot / f"{label}-timVal.png"
        title = f"{ticker} {intervalValue} - tune {label} timVal"
        plotTimVal(
            result.curveTs,
            result.curveSim,
            result.curveBench,
            result.curveAssetFrac,
            result.curveQuoteFrac,
            title,
            str(timValPath),
            cfgObj,
        )

    if prevChartsDir is None:
        os.environ.pop("CHARTS_OUT_DIR", None)
    else:
        os.environ["CHARTS_OUT_DIR"] = prevChartsDir
    return result


def renderSelectedTuneCharts(
    runDir: str | Path,
    anchorMs: int | None = None,
) -> None:
    cfgPath = Path(runDir) / "best-configs" / "best-config.json"
    with open(cfgPath) as fh:
        cfgObj = json.load(fh)
    traceTuneConfig(cfgObj, runDir, "best", anchorMs=anchorMs)


def finalizeTunerRun(
    config: dict,
    out_dir: str,
    chosenRow: dict,
    statsRow: dict | None = None,
    startTime: float | None = None,
    anchorMs: int | None = None,
    anchorKind: str = "runtime",
    anchorDate: str | None = None,
    charts: bool = True,
    peakLockBaseRows: int = 6,
) -> str:
    tickers = config["tickers"]
    ticker = tickers[0]
    _primerDays, _trainingDays, _tunerDays, holdoutDays, totalDays = (
        profile.profileWindows(config)
    )
    intervals = profile.intervalsFromConfig(config)
    outDir = os.path.abspath(out_dir)
    resultsCsvPath = os.path.join(outDir, "results.csv")
    chartsRoot = os.path.join(outDir, "charts")
    tuneChartsDir = os.path.join(chartsRoot, "tune")
    if charts:
        os.makedirs(tuneChartsDir, exist_ok=True)

    axes = axesFromConfig(config)
    maxPeriod = max(
        max(axes["p1Values"]),
        max(axes["p2Values"]),
        max(axes["p3Values"]),
    )
    globalMinCandles = (maxPeriod * 2) + 1
    klinesByInterval = buildIntervalKlines(
        lambda tkr, iv, d, mc: loadWindowedKlines(
            tkr,
            iv,
            totalDays,
            mc,
            holdoutDays=holdoutDays,
            anchorMs=anchorMs,
        ),
        ticker,
        intervals,
        totalDays,
        globalMinCandles,
    )

    if statsRow is None:
        statsRow = chosenRow
    statsConfigPreview = bestConfigFromRow(statsRow, config)

    ensureFn = lambda tkr, iv, d, mc: loadWindowedKlines(
        tkr,
        iv,
        totalDays,
        mc,
        holdoutDays=holdoutDays,
        anchorMs=anchorMs,
    )
    bestConfig, bestConfigPath, _, _bestDirPath = writeBestArtifacts(
        chosenRow,
        outDir,
        config,
        ticker,
        totalDays,
        klinesByInterval,
        ensureFn,
    )

    statsSameAsBest = configsEqual(bestConfig, statsConfigPreview)
    if statsSameAsBest:
        statsConfig = bestConfig
        statsConfigPath = bestConfigPath
    else:
        statsConfig, statsConfigPath, _, _ = writeBestArtifacts(
            statsRow,
            outDir,
            config,
            ticker,
            totalDays,
            klinesByInterval,
            ensureFn,
            "stats",
        )

    robustRows = robustRowsFromResults(resultsCsvPath, outDir, maxRows=5)
    for i, robustRow in enumerate(robustRows, start=1):
        robustConfigPreview = bestConfigFromRow(robustRow, config)
        if configsEqual(robustConfigPreview, bestConfig):
            continue
        if (
            not statsSameAsBest
            and configsEqual(robustConfigPreview, statsConfig)
        ):
            continue
        suffix = f"robust{i:02d}"
        writeBestArtifacts(
            robustRow,
            outDir,
            config,
            ticker,
            totalDays,
            klinesByInterval,
            ensureFn,
            suffix,
        )

    if not charts:
        fingerprint = buildFingerprintAt(
            config,
            anchorMs=anchorMs,
            anchorKind=anchorKind,
            anchorDate=anchorDate,
        )
        fpPath = os.path.join(outDir, "fingerprint.json")
        _writeFingerprint(fpPath, fingerprint)
        if startTime is not None:
            duration = time.time() - startTime
            print(f"[tune] duration: {duration/60.0:.1f} minutes")
        return resultsCsvPath

    resTuneBest = traceTuneConfig(
        bestConfig,
        outDir,
        "best",
        anchorMs=anchorMs,
    )
    resTuneStats = resTuneBest
    if not statsSameAsBest:
        resTuneStats = traceTuneConfig(
            statsConfig,
            outDir,
            "stats",
            anchorMs=anchorMs,
        )

    bestMetrics = resultMetrics("best", ticker, resTuneBest)
    statsMetrics = resultMetrics("stats", ticker, resTuneStats)
    printMetricBlock("tune", bestMetrics, BAR)
    printMetricBlock("tune", statsMetrics, BAR)

    summaryPath = os.path.join(outDir, "run.log")
    _render_summary_to_path(
        chosenRow,
        bestConfig,
        summaryPath,
        ticker,
        totalDays,
        klinesByInterval,
        holdoutDays=holdoutDays,
        anchorMs=anchorMs,
    )

    scatterPath = os.path.join(tuneChartsDir, "scatter.png")
    if os.path.exists(resultsCsvPath):
        generateScatter(resultsCsvPath, scatterPath, tickers=tickers)

    fingerprint = buildFingerprintAt(
        config,
        anchorMs=anchorMs,
        anchorKind=anchorKind,
        anchorDate=anchorDate,
    )
    fpPath = os.path.join(outDir, "fingerprint.json")
    _writeFingerprint(fpPath, fingerprint)

    if startTime is not None:
        duration = time.time() - startTime
        print(f"[tune] duration: {duration/60.0:.1f} minutes")
    return resultsCsvPath
