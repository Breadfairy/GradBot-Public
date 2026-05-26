#!/usr/bin/env python3
"""Explore DSP/PID feature value for peak-lock timing."""

from __future__ import annotations

########################################################################
# Imports
########################################################################

import argparse
import csv
import json
import math
import os
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import profile
from data.klines_io import loadWindowedKlines
from engine.shared import buildContext, buildSignals, bars_per_day, emaLpf
from repo_paths import rootPath


########################################################################
# Constants
########################################################################

HORIZON_DAYS = [7, 14, 30, 60]
REPORT_HORIZON = 30
EVENT_DRAW_PCT = 10.0
DAILY_STRONG_CLUSTER = 2
DAILY_DOWN_MASK = 9
LEAK_DECAY = 0.985


########################################################################
# Scalar Helpers
########################################################################

def loadProfile(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def scalarFloat(config: dict[str, Any], key: str, default: float) -> float:
    return float(profile.scalarValue(config.get(key), default))


def scalarInt(config: dict[str, Any], key: str, default: int) -> int:
    return int(profile.scalarValue(config.get(key), default))


def scalarText(config: dict[str, Any], key: str, default: str) -> str:
    return str(profile.scalarValue(config.get(key), default))


def firstTicker(config: dict[str, Any]) -> str:
    raw = config["tickers"]
    return str(raw[0])


def intervalText(config: dict[str, Any]) -> str:
    raw = profile.scalarValue(config["intervals"], "1h")
    return str(raw)


########################################################################
# Numeric Helpers
########################################################################

def safeCorr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)[mask]
    y = np.asarray(b, dtype=float)[mask]
    xStd = float(np.nanstd(x))
    yStd = float(np.nanstd(y))
    corr = 0.0
    if x.size >= 30 and xStd > 1e-12 and yStd > 1e-12:
        corr = float(np.corrcoef(x, y)[0, 1])
    return corr


def finiteMask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(int(arrays[0].size), dtype=bool)
    for i in arrays:
        mask &= np.isfinite(np.asarray(i, dtype=float))
    return mask


def leakySum(values: np.ndarray, decay: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.zeros(arr.size, dtype=float)
    running = 0.0
    for i in range(arr.size):
        running = (float(decay) * running) + float(arr[i])
        out[i] = running
    return out


def leakyMean(values: np.ndarray, decay: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.zeros(arr.size, dtype=float)
    running = 0.0
    keep = float(decay)
    add = 1.0 - keep
    for i in range(arr.size):
        running = (keep * running) + (add * float(arr[i]))
        out[i] = running
    return out


def futureExtrema(
    values: np.ndarray,
    window: int,
    wantMax: bool,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.size, np.nan, dtype=float)
    dq: deque[int] = deque()
    for i in range(arr.size - 1, -1, -1):
        while dq and dq[0] > i + window:
            dq.popleft()
        if i + window < arr.size and dq:
            out[i] = arr[dq[0]]
        if wantMax:
            while dq and arr[i] >= arr[dq[-1]]:
                dq.pop()
        else:
            while dq and arr[i] <= arr[dq[-1]]:
                dq.pop()
        dq.append(i)
    return out


def pctChange(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    den = np.where(np.abs(bottom) > 1e-12, bottom, np.nan)
    return ((top - bottom) / den) * 100.0


def diffSeries(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.zeros(arr.size, dtype=float)
    if arr.size > 1:
        out[1:] = arr[1:] - arr[:-1]
    return out


########################################################################
# PID Helpers
########################################################################

def pidParts(
    close: np.ndarray,
    barsDay: float,
    config: dict[str, Any],
) -> dict[str, np.ndarray | float | int]:
    maDays = scalarFloat(config, "PEAK_LOCK_MA_DAYS", 30.0)
    kp = scalarFloat(config, "PEAK_LOCK_KP", 6.0)
    ki = scalarFloat(config, "PEAK_LOCK_KI", 0.0)
    kd = scalarFloat(config, "PEAK_LOCK_KD", 0.0)
    decay = scalarFloat(config, "PEAK_LOCK_INTEGRAL_DECAY", 0.985)
    entry = scalarFloat(config, "PEAK_LOCK_ENTRY_THRESHOLD", 0.25)
    exitVal = scalarFloat(config, "PEAK_LOCK_EXIT_THRESHOLD", 0.05)
    confirm = scalarInt(config, "PEAK_LOCK_CONFIRM_BARS", 6)
    maBars = max(2, int(round(maDays * barsDay)))
    ma = emaLpf(close, maBars)
    err = np.divide(
        close - ma,
        np.where(np.abs(ma) > 1e-12, ma, np.nan),
    )
    deriv = diffSeries(err)
    integ = leakySum(err, decay)
    rawP = kp * err
    rawI = ki * integ
    rawD = kd * deriv
    rawFull = rawP + rawI + rawD
    return {
        "maDays": maDays,
        "maBars": maBars,
        "kp": kp,
        "ki": ki,
        "kd": kd,
        "decay": decay,
        "entry": entry,
        "exit": exitVal,
        "confirmBars": confirm,
        "ma": ma,
        "err": err,
        "deriv": deriv,
        "integral": integ,
        "rawP": rawP,
        "rawI": rawI,
        "rawD": rawD,
        "rawFull": rawFull,
    }


def pidLongState(
    raw: np.ndarray,
    entry: float,
    exitVal: float,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(raw, dtype=float)
    long = np.zeros(arr.size, dtype=bool)
    bear = np.zeros(arr.size, dtype=int)
    state = False
    count = 0
    for i in range(arr.size):
        if arr[i] > entry:
            state = True
        elif arr[i] < exitVal:
            state = False
        if state:
            count = 0
        else:
            count += 1
        long[i] = state
        bear[i] = count
    return long, bear


def decisionSummary(
    parts: dict[str, np.ndarray | float | int],
    segmentMask: np.ndarray,
    futureDraw: dict[int, np.ndarray],
) -> dict[str, Any]:
    rawP = np.asarray(parts["rawP"], dtype=float)
    rawFull = np.asarray(parts["rawFull"], dtype=float)
    rawI = np.asarray(parts["rawI"], dtype=float)
    rawD = np.asarray(parts["rawD"], dtype=float)
    entry = float(parts["entry"])
    exitVal = float(parts["exit"])
    confirm = int(parts["confirmBars"])
    longP, bearP = pidLongState(rawP, entry, exitVal)
    longFull, bearFull = pidLongState(rawFull, entry, exitVal)
    delta = longP != longFull
    early = (~longFull) & longP
    late = longFull & (~longP)
    riskP = bearP >= confirm
    riskFull = bearFull >= confirm
    mask = segmentMask & finiteMask(rawP, rawFull)
    draw30 = np.asarray(futureDraw[REPORT_HORIZON], dtype=float)
    earlyMask = mask & early & np.isfinite(draw30)
    lateMask = mask & late & np.isfinite(draw30)
    pRiskMask = mask & riskP & np.isfinite(draw30)
    fullRiskMask = mask & riskFull & np.isfinite(draw30)
    return {
        "bars": int(mask.sum()),
        "fullVsPDeltaPct": pctMean(delta, mask),
        "fullEarlyExitPct": pctMean(early, mask),
        "fullLateExitPct": pctMean(late, mask),
        "pRiskReadyPct": pctMean(riskP, mask),
        "fullRiskReadyPct": pctMean(riskFull, mask),
        "p95AbsI": pctileAbs(rawI, mask, 95.0),
        "p95AbsD": pctileAbs(rawD, mask, 95.0),
        "p95AbsRawFull": pctileAbs(rawFull, mask, 95.0),
        "entryThreshold": entry,
        "earlyExitDraw30Mean": meanOrNan(draw30, earlyMask),
        "lateExitDraw30Mean": meanOrNan(draw30, lateMask),
        "pRiskDraw30Mean": meanOrNan(draw30, pRiskMask),
        "fullRiskDraw30Mean": meanOrNan(draw30, fullRiskMask),
    }


def decisionRows(
    parts: dict[str, np.ndarray | float | int],
    segments: dict[str, np.ndarray],
    regimes: dict[str, np.ndarray],
    futureDraw: dict[int, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rawP = np.asarray(parts["rawP"], dtype=float)
    rawFull = np.asarray(parts["rawFull"], dtype=float)
    entry = float(parts["entry"])
    exitVal = float(parts["exit"])
    confirm = int(parts["confirmBars"])
    longP, bearP = pidLongState(rawP, entry, exitVal)
    longFull, bearFull = pidLongState(rawFull, entry, exitVal)
    draw30 = np.asarray(futureDraw[REPORT_HORIZON], dtype=float)
    delta = longP != longFull
    early = (~longFull) & longP
    late = longFull & (~longP)
    riskP = bearP >= confirm
    riskFull = bearFull >= confirm
    valid = finiteMask(rawP, rawFull, draw30)
    for segmentName, segmentMask in segments.items():
        for regimeName, regimeMask in regimes.items():
            mask = segmentMask & regimeMask & valid
            earlyMask = mask & early
            lateMask = mask & late
            pRiskMask = mask & riskP
            fullRiskMask = mask & riskFull
            if int(mask.sum()) < 60:
                continue
            rows.append({
                "segment": segmentName,
                "regime": regimeName,
                "count": int(mask.sum()),
                "fullVsPDeltaPct": pctMean(delta, mask),
                "fullEarlyExitPct": pctMean(early, mask),
                "fullLateExitPct": pctMean(late, mask),
                "pRiskReadyPct": pctMean(riskP, mask),
                "fullRiskReadyPct": pctMean(riskFull, mask),
                "earlyExitDraw30Mean": meanOrNan(draw30, earlyMask),
                "lateExitDraw30Mean": meanOrNan(draw30, lateMask),
                "pRiskDraw30Mean": meanOrNan(draw30, pRiskMask),
                "fullRiskDraw30Mean": meanOrNan(draw30, fullRiskMask),
            })
    return rows


def pctMean(flag: np.ndarray, mask: np.ndarray) -> float:
    out = 0.0
    if int(mask.sum()) > 0:
        out = float(np.mean(np.asarray(flag, dtype=float)[mask]) * 100.0)
    return out


def pctileAbs(values: np.ndarray, mask: np.ndarray, pct: float) -> float:
    out = 0.0
    arr = np.abs(np.asarray(values, dtype=float)[mask])
    if arr.size > 0:
        out = float(np.nanpercentile(arr, pct))
    return out


def meanOrNan(values: np.ndarray, mask: np.ndarray) -> float:
    out = math.nan
    arr = np.asarray(values, dtype=float)[mask]
    if arr.size > 0:
        out = float(np.nanmean(arr))
    return out


########################################################################
# Cluster Helpers
########################################################################

def clusterPath(config: dict[str, Any]) -> str:
    raw = config.get("DAILY_CLUSTER_PATH", "")
    return str(profile.scalarValue(raw, ""))


def clusterIsDown(cluster: np.ndarray) -> np.ndarray:
    arr = np.asarray(cluster, dtype=int)
    out = np.zeros(arr.size, dtype=bool)
    valid = (arr >= 0) & (arr < 30)
    bits = np.left_shift(np.ones(arr.size, dtype=np.int64), arr.clip(0, 29))
    out[valid] = (bits[valid] & DAILY_DOWN_MASK) != 0
    return out


def loadClusterSeries(
    config: dict[str, Any],
    openMs: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    pathText = clusterPath(config)
    labels = np.full(openMs.size, -99, dtype=int)
    meta: dict[str, Any] = {
        "path": pathText,
        "mode": "none",
        "labelCounts": {},
    }
    if pathText == "":
        return labels, meta

    dataPath = rootPath(pathText)
    frame = pd.read_csv(dataPath)
    timeKey = "openMs" if "openMs" in frame.columns else "closeMs"
    timeVals = frame[timeKey].to_numpy(dtype=np.int64)
    clusterVals = frame["cluster"].to_numpy(dtype=int)
    idx = np.searchsorted(timeVals, openMs, side="right") - 1
    valid = idx >= 0
    labels[valid] = clusterVals[idx[valid]]
    unique, counts = np.unique(labels, return_counts=True)
    meta["mode"] = "static_csv"
    meta["rows"] = int(frame.shape[0])
    meta["timeKey"] = timeKey
    meta["labelCounts"] = {
        str(int(k)): int(v) for k, v in zip(unique, counts)
    }
    return labels, meta


def regimeMasks(cluster: np.ndarray) -> dict[str, np.ndarray]:
    valid = np.asarray(cluster, dtype=int) >= 0
    strong = np.asarray(cluster, dtype=int) == DAILY_STRONG_CLUSTER
    down = clusterIsDown(cluster)
    crab = valid & (~strong) & (~down)
    return {
        "all": np.ones(cluster.size, dtype=bool),
        "ultra": strong,
        "down": down,
        "crab": crab,
    }


########################################################################
# Feature Assembly
########################################################################

def featureMap(
    close: np.ndarray,
    signals: dict[str, object],
    parts: dict[str, np.ndarray | float | int],
) -> dict[str, np.ndarray]:
    g1P1 = np.asarray(signals["g1P1"], dtype=float)
    g1P3 = np.asarray(signals["g1P3"], dtype=float)
    s12 = np.asarray(signals["s12"], dtype=float) * 100.0
    s23 = np.asarray(signals["s23"], dtype=float) * 100.0
    trendCode = np.asarray(signals["trendCode"], dtype=float)
    spread = s12 + s23
    g1Accel = diffSeries(g1P1)
    spreadDelta = diffSeries(spread)
    bearFlag = (trendCode < 0.0).astype(float)
    bullFlag = (trendCode > 0.0).astype(float)
    rawP = np.asarray(parts["rawP"], dtype=float)
    rawI = np.asarray(parts["rawI"], dtype=float)
    rawD = np.asarray(parts["rawD"], dtype=float)
    rawFull = np.asarray(parts["rawFull"], dtype=float)
    err = np.asarray(parts["err"], dtype=float) * 100.0
    deriv = np.asarray(parts["deriv"], dtype=float) * 100.0
    integral = np.asarray(parts["integral"], dtype=float)
    closeRet = pctChange(close, np.roll(close, 1))
    closeRet[0] = 0.0
    return {
        "pidErrPct": err,
        "pidDerivPct": deriv,
        "pidIntegral": integral,
        "pidRawP": rawP,
        "pidRawI": rawI,
        "pidRawD": rawD,
        "pidRawFull": rawFull,
        "pidRawIdDelta": rawFull - rawP,
        "g1P1": g1P1,
        "g1P3": g1P3,
        "g1Accel": g1Accel,
        "g1Curv": diffSeries(g1Accel),
        "s12Pct": s12,
        "s23Pct": s23,
        "spreadPct": spread,
        "spreadDelta": spreadDelta,
        "spreadPersist": leakySum(spreadDelta, LEAK_DECAY),
        "bearPersist": leakyMean(bearFlag, LEAK_DECAY),
        "bullPersist": leakyMean(bullFlag, LEAK_DECAY),
        "trendCode": trendCode,
        "closeRetPct": closeRet,
    }


def futureLabels(close: np.ndarray, barsDay: float) -> dict[int, np.ndarray]:
    labels: dict[int, np.ndarray] = {}
    for i in HORIZON_DAYS:
        bars = max(1, int(round(float(i) * barsDay)))
        fmin = futureExtrema(close, bars, wantMax=False)
        den = np.where(np.abs(close) > 1e-12, close, np.nan)
        draw = np.maximum(0.0, ((close - fmin) / den) * 100.0)
        labels[int(i)] = draw
    return labels


def futureUpside(close: np.ndarray, barsDay: float) -> dict[int, np.ndarray]:
    labels: dict[int, np.ndarray] = {}
    for i in HORIZON_DAYS:
        bars = max(1, int(round(float(i) * barsDay)))
        fmax = futureExtrema(close, bars, wantMax=True)
        labels[int(i)] = np.maximum(0.0, pctChange(fmax, close))
    return labels


def segmentMasks(
    nBars: int,
    barsDay: float,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    primer, training, tuner, holdout, _total = profile.windowParts(config)
    primerBars = int(round(float(primer) * barsDay))
    trainBars = int(round(float(training) * barsDay))
    tunerBars = int(round(float(tuner) * barsDay))
    holdoutBars = int(round(float(holdout) * barsDay))
    tuneStart = min(nBars, primerBars + trainBars)
    holdoutStart = max(0, nBars - holdoutBars)
    tuneEnd = min(nBars, tuneStart + tunerBars)
    idx = np.arange(nBars)
    visible = idx >= primerBars
    tune = (idx >= tuneStart) & (idx < min(tuneEnd, holdoutStart))
    hold = idx >= holdoutStart
    return {
        "visible": visible,
        "tune": tune,
        "holdout": hold,
    }


########################################################################
# Scoring
########################################################################

def decileStats(
    featureVals: np.ndarray,
    drawVals: np.ndarray,
    eventVals: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    vals = np.asarray(featureVals, dtype=float)
    draw = np.asarray(drawVals, dtype=float)
    event = np.asarray(eventVals, dtype=bool)
    arr = vals[mask]
    lowVal = float(np.nanpercentile(arr, 10.0))
    highVal = float(np.nanpercentile(arr, 90.0))
    lowMask = mask & (vals <= lowVal)
    highMask = mask & (vals >= highVal)
    lowDraw = meanOrNan(draw, lowMask)
    highDraw = meanOrNan(draw, highMask)
    lowEvent = pctMean(event, lowMask)
    highEvent = pctMean(event, highMask)
    baseEvent = pctMean(event, mask)
    highLift = highDraw - lowDraw
    lowLift = lowDraw - highDraw
    direction = "high"
    lift = highLift
    eventLift = highEvent - baseEvent
    riskDraw = highDraw
    riskEvent = highEvent
    if lowLift > highLift:
        direction = "low"
        lift = lowLift
        eventLift = lowEvent - baseEvent
        riskDraw = lowDraw
        riskEvent = lowEvent
    return {
        "riskDirection": direction,
        "riskDrawMean": riskDraw,
        "riskEventPct": riskEvent,
        "baseEventPct": baseEvent,
        "riskDrawLiftPct": lift,
        "riskEventLiftPct": eventLift,
        "lowDrawMean": lowDraw,
        "highDrawMean": highDraw,
    }


def scoreFeatures(
    features: dict[str, np.ndarray],
    futureDraw: dict[int, np.ndarray],
    futureUp: dict[int, np.ndarray],
    segments: dict[str, np.ndarray],
    regimes: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segmentName, segmentMask in segments.items():
        for regimeName, regimeMask in regimes.items():
            for days, drawVals in futureDraw.items():
                upVals = futureUp[days]
                eventVals = np.asarray(drawVals, dtype=float) >= EVENT_DRAW_PCT
                for featureName, featureVals in features.items():
                    mask = (
                        segmentMask
                        & regimeMask
                        & finiteMask(featureVals, drawVals, upVals)
                    )
                    if int(mask.sum()) < 60:
                        continue
                    stats = decileStats(
                        featureVals,
                        drawVals,
                        eventVals,
                        mask,
                    )
                    rows.append({
                        "segment": segmentName,
                        "regime": regimeName,
                        "horizonDays": int(days),
                        "feature": featureName,
                        "count": int(mask.sum()),
                        "corrDrawdown": safeCorr(featureVals, drawVals, mask),
                        "corrUpside": safeCorr(featureVals, upVals, mask),
                        **stats,
                    })
    return rows


########################################################################
# IO
########################################################################

def atomicText(path: Path, text: str) -> None:
    tmpPath = path.with_suffix(path.suffix + ".tmp")
    with open(tmpPath, "w") as fh:
        fh.write(text)
    os.replace(tmpPath, path)


def atomicJson(path: Path, data: dict[str, Any]) -> None:
    tmpPath = path.with_suffix(path.suffix + ".tmp")
    with open(tmpPath, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    os.replace(tmpPath, path)


def writeCsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldNames = list(rows[0].keys())
    tmpPath = path.with_suffix(path.suffix + ".tmp")
    with open(tmpPath, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldNames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmpPath, path)


########################################################################
# Report
########################################################################

def topScoreRows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filt = [
        i for i in rows
        if i["segment"] == "holdout"
        and i["regime"] == "all"
        and int(i["horizonDays"]) == REPORT_HORIZON
    ]
    if not filt:
        filt = [
            i for i in rows
            if i["segment"] == "visible"
            and i["regime"] == "all"
            and int(i["horizonDays"]) == REPORT_HORIZON
        ]
    return sorted(
        filt,
        key=lambda i: float(i["riskDrawLiftPct"]),
        reverse=True,
    )[:12]


def formatTop(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for i in rows:
        lines.append(
            "- "
            f"`{i['feature']}` {i['riskDirection']} side: "
            f"draw lift `{float(i['riskDrawLiftPct']):.3f}%`, "
            f"event lift `{float(i['riskEventLiftPct']):.3f}pp`, "
            f"corr draw `{float(i['corrDrawdown']):.3f}`, "
            f"corr upside `{float(i['corrUpside']):.3f}`"
        )
    return lines


def reportText(
    profilePath: Path,
    config: dict[str, Any],
    meta: dict[str, Any],
    clusterMeta: dict[str, Any],
    decision: dict[str, Any],
    decisionTable: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> str:
    topRows = topScoreRows(rows)
    topLines = formatTop(topRows)
    regimeLines = formatDecisionRows(decisionTable)
    lines = [
        "# PID Feature Explore",
        "",
        f"- profile: `{profilePath}`",
        f"- ticker: `{meta['ticker']}`",
        f"- interval: `{meta['interval']}`",
        f"- bars: `{meta['bars']}`",
        f"- date range ms: `{meta['startMs']}` to `{meta['endMs']}`",
        f"- cluster mode: `{clusterMeta['mode']}`",
        f"- cluster path: `{clusterMeta['path']}`",
        "",
        "## Runtime Contract",
        "",
        "The C sweep decodes PID and posture keys per row, then resets "
        "daily posture and peak-lock state for that row before the wallet "
        "pass. Static `DAILY_CLUSTER_PATH` lanes provide fixed causal "
        "labels; `DAILY_CLUSTER_MODEL_PATH` is the native causal inference "
        "path when model files are used.",
        "",
        "## PID Candidate",
        "",
        f"- MA days/bars: `{decision['maDays']:.6g}` / "
        f"`{decision['maBars']}`",
        f"- Kp/Ki/Kd: `{decision['kp']:.6g}`, "
        f"`{decision['ki']:.6g}`, `{decision['kd']:.6g}`",
        f"- entry/exit: `{decision['entryThreshold']:.6g}`, "
        f"`{decision['exitThreshold']:.6g}`",
        f"- full-vs-P state delta: "
        f"`{decision['fullVsPDeltaPct']:.3f}%`",
        f"- full exits while P remains long: "
        f"`{decision['fullEarlyExitPct']:.3f}%`",
        f"- full stays long while P exits: "
        f"`{decision['fullLateExitPct']:.3f}%`",
        f"- p95 abs I/D contribution: `{decision['p95AbsI']:.6g}` / "
        f"`{decision['p95AbsD']:.6g}`",
        "",
        "## Decision Delta By Regime",
        "",
        *regimeLines,
        "",
        "## Best Holdout 30d Drawdown Features",
        "",
        *topLines,
        "",
        "## Interpretation",
        "",
        "Features with positive draw/event lift separate future drawdown "
        "risk. If `pidRawI`, `pidRawD`, or `pidRawIdDelta` do not rank "
        "above DSP structure features, then KI/KD are not carrying robust "
        "timing information for this profile.",
        "",
    ]
    return "\n".join(lines)


def formatDecisionRows(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    wanted = [
        i for i in rows
        if i["segment"] == "holdout"
        and i["regime"] in {"all", "ultra", "down", "crab"}
    ]
    for i in wanted:
        lines.append(
            "- "
            f"`{i['regime']}`: delta "
            f"`{float(i['fullVsPDeltaPct']):.3f}%`, "
            f"early `{float(i['fullEarlyExitPct']):.3f}%`, "
            f"late `{float(i['fullLateExitPct']):.3f}%`, "
            f"P risk draw `{float(i['pRiskDraw30Mean']):.3f}%`, "
            f"full risk draw `{float(i['fullRiskDraw30Mean']):.3f}%`"
        )
    return lines


########################################################################
# Runtime
########################################################################

def buildRun(profilePath: Path, anchorMs: int | None) -> dict[str, Any]:
    config = loadProfile(profilePath)
    ticker = firstTicker(config)
    interval = intervalText(config)
    periods = [
        scalarInt(config, "p1", 10),
        scalarInt(config, "p2", 20),
        scalarInt(config, "p3", 55),
    ]
    _primer, _training, _tuner, _holdout, totalDays = profile.windowParts(
        config
    )
    rows = loadWindowedKlines(
        ticker,
        interval,
        totalDays,
        None,
        holdoutDays=0,
        anchorMs=anchorMs,
    )
    ctx = buildContext(rows, periods)
    ctx["intervalStr"] = interval
    signals = buildSignals(ctx, [])
    close = np.asarray(ctx["closes"], dtype=float)
    openMs = np.asarray([int(i[0]) for i in rows], dtype=np.int64)
    barsDay = bars_per_day(ctx)
    return {
        "config": config,
        "ticker": ticker,
        "interval": interval,
        "rows": rows,
        "ctx": ctx,
        "signals": signals,
        "close": close,
        "openMs": openMs,
        "barsDay": barsDay,
    }


def explore(
    profilePath: Path,
    outputDir: Path,
    anchorMs: int | None,
) -> None:
    run = buildRun(profilePath, anchorMs)
    config = run["config"]
    close = np.asarray(run["close"], dtype=float)
    openMs = np.asarray(run["openMs"], dtype=np.int64)
    barsDay = float(run["barsDay"])
    parts = pidParts(close, barsDay, config)
    features = featureMap(close, run["signals"], parts)
    draw = futureLabels(close, barsDay)
    up = futureUpside(close, barsDay)
    cluster, clusterMeta = loadClusterSeries(config, openMs)
    segments = segmentMasks(close.size, barsDay, config)
    regimes = regimeMasks(cluster)
    scores = scoreFeatures(features, draw, up, segments, regimes)
    visibleMask = segments["visible"]
    decision = decisionSummary(parts, visibleMask, draw)
    decisionTable = decisionRows(parts, segments, regimes, draw)
    decision.update({
        "maDays": float(parts["maDays"]),
        "maBars": int(parts["maBars"]),
        "kp": float(parts["kp"]),
        "ki": float(parts["ki"]),
        "kd": float(parts["kd"]),
        "decay": float(parts["decay"]),
        "exitThreshold": float(parts["exit"]),
    })
    meta = {
        "profile": str(profilePath),
        "ticker": run["ticker"],
        "interval": run["interval"],
        "bars": int(close.size),
        "barsDay": barsDay,
        "startMs": int(openMs[0]),
        "endMs": int(openMs[-1]),
        "anchorMs": anchorMs,
    }
    outputDir.mkdir(parents=True, exist_ok=True)
    writeCsv(outputDir / "feature_scores.csv", scores)
    writeCsv(outputDir / "decision_scores.csv", decisionTable)
    atomicJson(outputDir / "pid_feature_summary.json", {
        "meta": meta,
        "cluster": clusterMeta,
        "decision": decision,
        "decisionByRegime": decisionTable,
        "topHoldout30d": topScoreRows(scores),
    })
    atomicText(
        outputDir / "pid_feature_report.md",
        reportText(
            profilePath,
            config,
            meta,
            clusterMeta,
            decision,
            decisionTable,
            scores,
        ),
    )
    print(f"[pid-feature] wrote {outputDir}")


########################################################################
# CLI
########################################################################

def parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tune.pid_feature_explore",
        description="Score DSP/PID features against future drawdown labels.",
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--anchor-ms", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parseArgs(argv)
    profilePath = rootPath(args.profile)
    outputDir = rootPath(args.out)
    explore(profilePath, outputDir, args.anchor_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
