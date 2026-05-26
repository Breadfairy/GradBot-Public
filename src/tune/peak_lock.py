#!/usr/bin/env python3
# peak_lock_tune.py - legacy/manual post-tune peak-lock selector.

from __future__ import annotations

########################################################################
# Imports
########################################################################

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT_DIR / ".cache"))

from analysis.charting import plotTimVal
from data.klines_io import loadWindowedKlines
from analysis.metrics import timeRegionStats
from config import profile
from data.time_bounds import resolveAnchorMs
from tune.artifacts import (
    bestConfigFromRow,
    configsEqual,
    orderedConfig,
    roundForJson,
)
from tune.schema import rowIntFields, rowStrFields
from tune.trace import Trace


########################################################################
# Constants
########################################################################

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
    *profile.HOLDOUT_START_KEYS,
    "CHARTS_TIMEVAL",
    "CHARTS_TRADES",
}
PEAK_KEYS = {
    "PEAK_LOCK_CAP_PCT",
    "PEAK_LOCK_UNLOCK_GAIN_PCT",
    "PEAK_LOCK_REENTRY_STEP_PCT",
    "PEAK_LOCK_ARM_GAIN_PCT",
    "PEAK_LOCK_GIVEBACK_PCT",
    "PEAK_LOCK_MAX_DAYS",
    "PEAK_LOCK_EDGE_DRAW_PCT",
    "PEAK_LOCK_EDGE_SLOPE_DAYS",
    "PEAK_LOCK_REQUIRE_EDGE_RISK",
    "PEAK_LOCK_MA_DAYS",
    "PEAK_LOCK_KP",
    "PEAK_LOCK_KI",
    "PEAK_LOCK_KD",
    "PEAK_LOCK_INTEGRAL_DECAY",
    "PEAK_LOCK_ENTRY_THRESHOLD",
    "PEAK_LOCK_EXIT_THRESHOLD",
    "PEAK_LOCK_CONFIRM_BARS",
    "PEAK_LOCK_RELEASE_TARGET_PCT",
    "PEAK_LOCK_ULTRA_GRACE_DAYS",
}
BASE_ROW_COUNT = 6


########################################################################
# File helpers
########################################################################

def _writeFrame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _writeJson(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def _anchorFromRun(runDir: Path) -> int | None:
    fp = runDir / "fingerprint.json"
    if not fp.is_file():
        return None
    data = json.loads(fp.read_text())
    value = data.get("anchorMs")
    return None if value is None else int(value)


def _rowDict(row: pd.Series) -> dict[str, Any]:
    strFields = rowStrFields()
    intFields = rowIntFields()
    out: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            out[key] = ""
        elif key in strFields:
            out[key] = str(value)
        elif key in intFields:
            out[key] = int(float(value))
        elif isinstance(value, float):
            out[key] = float(value)
        else:
            out[key] = value
    return out


########################################################################
# Config helpers
########################################################################

def _label(spec: dict[str, Any]) -> str:
    ma = int(float(spec["PEAK_LOCK_MA_DAYS"]))
    kp = int(float(spec["PEAK_LOCK_KP"]))
    cap = int(round(float(spec["PEAK_LOCK_CAP_PCT"]) * 100.0))
    unlock = int(round(float(spec["PEAK_LOCK_UNLOCK_GAIN_PCT"])))
    giveback = int(round(float(spec["PEAK_LOCK_GIVEBACK_PCT"])))
    confirm = int(spec["PEAK_LOCK_CONFIRM_BARS"])
    release = int(round(float(
        spec.get("PEAK_LOCK_RELEASE_TARGET_PCT", 0.0)
    ) * 100.0))
    grace = int(round(float(
        spec.get("PEAK_LOCK_ULTRA_GRACE_DAYS", 0.0)
    )))
    suffix = ""
    if release > 0:
        suffix += f"-rel{release}"
    if grace > 0:
        suffix += f"-gr{grace}"
    return (
        f"peaklock-ma{ma}-kp{kp}-cap{cap}-ug{unlock}"
        f"-gb{giveback}-c{confirm}{suffix}"
    )


def _isSweepSpec(value: Any) -> bool:
    if isinstance(value, list):
        return True
    if isinstance(value, dict) and "range" in value:
        return True
    if isinstance(value, dict) and all(
        k in value for k in ("start", "stop", "step")
    ):
        return True
    return False


def _hasProfilePeakGrid(cfg: dict[str, Any]) -> bool:
    return any(_isSweepSpec(cfg.get(k)) for k in PEAK_KEYS)


def _expandRange(
    start: float,
    stop: float,
    step: float,
    caster: Callable[[float], Any],
) -> list[Any]:
    values: list[Any] = []
    cur = float(start)
    end = float(stop)
    inc = float(step)
    if inc == 0:
        raise ValueError("peak-lock range step cannot be 0")
    if inc > 0:
        while cur <= end + 1e-12:
            values.append(caster(cur))
            cur += inc
    else:
        while cur >= end - 1e-12:
            values.append(caster(cur))
            cur += inc
    return values


def _expandSpec(
    spec: Any,
    caster: Callable[[float], Any],
) -> list[Any]:
    if isinstance(spec, list):
        return [caster(i) for i in spec]
    if isinstance(spec, dict) and "range" in spec:
        arr = spec["range"]
        return _expandRange(arr[0], arr[1], arr[2], caster)
    if isinstance(spec, dict):
        return _expandRange(
            spec["start"],
            spec["stop"],
            spec["step"],
            caster,
        )
    return [caster(spec)]


def _profileValues(
    cfg: dict[str, Any],
    key: str,
    default: Any,
    caster: Callable[[float], Any],
) -> list[Any]:
    return _expandSpec(cfg.get(key, default), caster)


def _defaultPidSpecs() -> list[dict[str, float]]:
    return [
        {
            "PEAK_LOCK_MA_DAYS": 14.0,
            "PEAK_LOCK_KP": 12.0,
            "PEAK_LOCK_KI": 0.0,
            "PEAK_LOCK_KD": 0.0,
            "PEAK_LOCK_ENTRY_THRESHOLD": 0.25,
            "PEAK_LOCK_EXIT_THRESHOLD": 0.05,
        },
        {
            "PEAK_LOCK_MA_DAYS": 30.0,
            "PEAK_LOCK_KP": 6.0,
            "PEAK_LOCK_KI": 0.0,
            "PEAK_LOCK_KD": 0.0,
            "PEAK_LOCK_ENTRY_THRESHOLD": 0.25,
            "PEAK_LOCK_EXIT_THRESHOLD": 0.05,
        },
    ]


def _profilePidSpecs(cfg: dict[str, Any]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    maValues = _profileValues(cfg, "PEAK_LOCK_MA_DAYS", 30.0, float)
    kpValues = _profileValues(cfg, "PEAK_LOCK_KP", 6.0, float)
    kiValues = _profileValues(cfg, "PEAK_LOCK_KI", 0.0, float)
    kdValues = _profileValues(cfg, "PEAK_LOCK_KD", 0.0, float)
    entryValues = _profileValues(
        cfg,
        "PEAK_LOCK_ENTRY_THRESHOLD",
        0.25,
        float,
    )
    exitValues = _profileValues(
        cfg,
        "PEAK_LOCK_EXIT_THRESHOLD",
        0.05,
        float,
    )
    for ma, kp, ki, kd, entry, exitValue in itertools.product(
        maValues,
        kpValues,
        kiValues,
        kdValues,
        entryValues,
        exitValues,
    ):
        out.append({
            "PEAK_LOCK_MA_DAYS": float(ma),
            "PEAK_LOCK_KP": float(kp),
            "PEAK_LOCK_KI": float(ki),
            "PEAK_LOCK_KD": float(kd),
            "PEAK_LOCK_ENTRY_THRESHOLD": float(entry),
            "PEAK_LOCK_EXIT_THRESHOLD": float(exitValue),
        })
    return out


def _variantAxes(
    cfg: dict[str, Any] | None,
) -> tuple[dict[str, list[Any]], list[dict[str, float]], str]:
    if cfg is None or not _hasProfilePeakGrid(cfg):
        return (
            {
                "caps": [0.35, 0.50, 0.65],
                "unlockGains": [10.0, 15.0, 25.0],
                "reentrySteps": [0.15],
                "armGains": [15.0],
                "givebacks": [4.0],
                "maxDays": [120.0],
                "edgeDraws": [5.0],
                "edgeSlopes": [7.0],
                "edgeRequires": [1],
                "integralDecays": [0.985],
                "confirms": [6, 12],
                "releaseTargets": [0.0, 0.5, 0.65, 0.8],
                "graceDays": [0.0],
            },
            _defaultPidSpecs(),
            "default",
        )
    return (
        {
            "caps": _profileValues(
                cfg,
                "PEAK_LOCK_CAP_PCT",
                0.35,
                float,
            ),
            "unlockGains": _profileValues(
                cfg,
                "PEAK_LOCK_UNLOCK_GAIN_PCT",
                15.0,
                float,
            ),
            "reentrySteps": _profileValues(
                cfg,
                "PEAK_LOCK_REENTRY_STEP_PCT",
                0.15,
                float,
            ),
            "armGains": _profileValues(
                cfg,
                "PEAK_LOCK_ARM_GAIN_PCT",
                15.0,
                float,
            ),
            "givebacks": _profileValues(
                cfg,
                "PEAK_LOCK_GIVEBACK_PCT",
                4.0,
                float,
            ),
            "maxDays": _profileValues(
                cfg,
                "PEAK_LOCK_MAX_DAYS",
                120.0,
                float,
            ),
            "edgeDraws": _profileValues(
                cfg,
                "PEAK_LOCK_EDGE_DRAW_PCT",
                5.0,
                float,
            ),
            "edgeSlopes": _profileValues(
                cfg,
                "PEAK_LOCK_EDGE_SLOPE_DAYS",
                7.0,
                float,
            ),
            "edgeRequires": _profileValues(
                cfg,
                "PEAK_LOCK_REQUIRE_EDGE_RISK",
                1,
                int,
            ),
            "integralDecays": _profileValues(
                cfg,
                "PEAK_LOCK_INTEGRAL_DECAY",
                0.985,
                float,
            ),
            "confirms": _profileValues(
                cfg,
                "PEAK_LOCK_CONFIRM_BARS",
                12,
                int,
            ),
            "releaseTargets": _profileValues(
                cfg,
                "PEAK_LOCK_RELEASE_TARGET_PCT",
                0.8,
                float,
            ),
            "graceDays": _profileValues(
                cfg,
                "PEAK_LOCK_ULTRA_GRACE_DAYS",
                0.0,
                float,
            ),
        },
        _profilePidSpecs(cfg),
        "profile",
    )


def _variants(cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    axes, pidSpecs, _source = _variantAxes(cfg)
    for pid in pidSpecs:
        for (
            cap,
            unlockGain,
            reentryStep,
            armGain,
            giveback,
            maxDays,
            edgeDraw,
            edgeSlope,
            edgeRequire,
            integralDecay,
            confirm,
            releaseTarget,
            graceDay,
        ) in itertools.product(
            axes["caps"],
            axes["unlockGains"],
            axes["reentrySteps"],
            axes["armGains"],
            axes["givebacks"],
            axes["maxDays"],
            axes["edgeDraws"],
            axes["edgeSlopes"],
            axes["edgeRequires"],
            axes["integralDecays"],
            axes["confirms"],
            axes["releaseTargets"],
            axes["graceDays"],
        ):
            spec = dict(pid)
            spec.update({
                "PEAK_LOCK_CAP_PCT": float(cap),
                "PEAK_LOCK_UNLOCK_GAIN_PCT": float(unlockGain),
                "PEAK_LOCK_REENTRY_STEP_PCT": float(reentryStep),
                "PEAK_LOCK_ARM_GAIN_PCT": float(armGain),
                "PEAK_LOCK_GIVEBACK_PCT": float(giveback),
                "PEAK_LOCK_MAX_DAYS": float(maxDays),
                "PEAK_LOCK_EDGE_DRAW_PCT": float(edgeDraw),
                "PEAK_LOCK_EDGE_SLOPE_DAYS": float(edgeSlope),
                "PEAK_LOCK_REQUIRE_EDGE_RISK": int(edgeRequire),
                "PEAK_LOCK_INTEGRAL_DECAY": float(integralDecay),
                "PEAK_LOCK_CONFIRM_BARS": int(confirm),
                "PEAK_LOCK_RELEASE_TARGET_PCT": float(releaseTarget),
                "PEAK_LOCK_ULTRA_GRACE_DAYS": float(graceDay),
            })
            spec["label"] = _label(spec)
            out.append(spec)
    return out


def _configWithSpec(
    cfg: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    out = dict(cfg)
    for key in PEAK_KEYS:
        out.pop(key, None)
    for key, value in spec.items():
        if key != "label":
            out[key] = value
    return dict(orderedConfig(out, cfg))


def _stripPeakConfig(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    for key in PEAK_KEYS:
        out.pop(key, None)
    return dict(orderedConfig(out, cfg))


def _candidateConfigs(
    runDir: Path,
    cfg: dict[str, Any],
    baseRows: int,
) -> list[dict[str, Any]]:
    resultsPath = runDir / "results.csv"
    out = [{
        "baseLabel": "base00",
        "baseRank": 0,
        "baseSource": "best-config",
        "config": _stripPeakConfig(cfg),
    }]
    if not resultsPath.is_file() or int(baseRows) <= 1:
        return out

    frame = pd.read_csv(resultsPath)
    scoreCol = (
        "scoreMetric" if "scoreMetric" in frame.columns else "edgeVsBench"
    )
    frame["_rankScore"] = pd.to_numeric(frame[scoreCol], errors="coerce")
    frame = frame.dropna(subset=["_rankScore"])
    frame = frame.sort_values("_rankScore", ascending=False)
    for i, (_idx, row) in enumerate(frame.head(int(baseRows)).iterrows()):
        rowData = _rowDict(row)
        candidate = _stripPeakConfig(dict(bestConfigFromRow(rowData, cfg)))
        duplicate = False
        for item in out:
            if configsEqual(candidate, item["config"]):
                duplicate = True
                break
        if duplicate:
            continue
        label = f"base{len(out):02d}"
        out.append({
            "baseLabel": label,
            "baseRank": int(i) + 1,
            "baseSource": "results.csv",
            "config": candidate,
        })
    return out


########################################################################
# Trace helpers
########################################################################

def _configParts(
    cfg: dict[str, Any],
) -> tuple[str, str, list[int], dict[str, Any], int, int, int, int, int]:
    ticker = cfg["tickers"][0]
    interval = profile.intervalsFromConfig(cfg)[0]
    periods = [int(cfg["p1"]), int(cfg["p2"]), int(cfg["p3"])]
    primerDays, trainingDays, tunerDays, holdoutDays, totalDays = (
        profile.windowParts(cfg)
    )
    rawOverrides = {k: v for k, v in cfg.items() if k not in BASE_FIELDS}
    overrides = profile.overrides(rawOverrides)
    profile.validate(overrides, kind="backtest")
    return (
        ticker,
        interval,
        periods,
        overrides,
        primerDays,
        trainingDays,
        tunerDays,
        holdoutDays,
        totalDays,
    )


def _runWindow(
    cfg: dict[str, Any],
    label: str,
    window: str,
    anchorMs: int | None,
    chartsDir: Path | None = None,
) -> dict[str, Any]:
    (
        ticker,
        interval,
        periods,
        overrides,
        primerDays,
        trainingDays,
        tunerDays,
        holdoutDays,
        totalDays,
    ) = _configParts(cfg)
    key = str(window).strip().lower()
    trimHoldout = holdoutDays if key == "tune" else 0
    activePrimer = primerDays + trainingDays
    if key == "holdout":
        activePrimer += tunerDays
    rows = loadWindowedKlines(
        ticker,
        interval,
        totalDays,
        (max(periods) * 2) + 1,
        holdoutDays=trimHoldout,
        anchorMs=anchorMs,
    )
    result = Trace(
        ticker,
        rows,
        interval,
        periods,
        days=totalDays,
        showCharts=False,
        showPrints=False,
        showSummary=False,
        overrides=overrides,
        ctx=None,
        signals=None,
        computeRisk=True,
        primerDays=activePrimer,
        holdoutDays=trimHoldout,
        anchorMs=anchorMs,
    ).run()
    simVal = float(result.sim.get("portfolio_value", 0.0))
    benchVal = float(result.bench.get("portfolio_value", 0.0))
    curveSim = result.curveSim if result.curveSim is not None else []
    curveBench = result.curveBench if result.curveBench is not None else []
    timeStats = timeRegionStats(curveSim, curveBench)
    posture = result.postureStats or {}
    out = {
        "label": str(label),
        "window": key,
        "finalValue": simVal,
        "benchValue": benchVal,
        "grossVsHodl": ((simVal / max(benchVal, 1e-12)) - 1.0) * 100.0,
        "mdd": float(result.mdd),
        "mddPct": float(result.mdd) * 100.0,
        "trades": int(result.sim.get("trades", 0)),
        "buyTrades": int(result.buyTrades),
        "sellTrades": int(result.sellTrades),
        "feePaid": float(result.sim.get("fees_paid_quote", 0.0)),
        "peakLocks": int(posture.get("peakLocks", 0)),
        "cappedBuys": int(posture.get("peakCappedBuys", 0)),
        "unlockSteps": int(posture.get("peakUnlockSteps", 0)),
        "lockHours": int(posture.get("peakLockHours", 0)),
        "lockGainMax": float(posture.get("peakLockGainMax", 0.0)),
        "strongReleases": int(posture.get("peakStrongReleases", 0)),
    }
    for key in PEAK_KEYS:
        out[key] = cfg.get(key, "")
    out.update(timeStats)
    if (
        chartsDir is not None
        and result.curveTs is not None
        and result.curveSim is not None
        and result.curveBench is not None
        and result.curveAssetFrac is not None
        and result.curveQuoteFrac is not None
    ):
        chartsDir.mkdir(parents=True, exist_ok=True)
        outPath = chartsDir / f"{window}-{label}-timVal.png"
        title = f"{ticker} {interval} - {window} {label} peak-lock"
        plotTimVal(
            result.curveTs,
            result.curveSim,
            result.curveBench,
            result.curveAssetFrac,
            result.curveQuoteFrac,
            title,
            str(outPath),
            cfg,
        )
    return out


########################################################################
# Selection
########################################################################

def _paired(tune: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    frame = holdout.merge(tune, on="label", suffixes=("_holdout", "_tune"))
    frame["derived_minTimeRegionScore"] = frame[
        ["timeRegionScore_holdout", "timeRegionScore_tune"]
    ].min(axis=1)
    frame["derived_timeGap"] = (
        frame["timeRegionScore_tune"] - frame["timeRegionScore_holdout"]
    )
    return frame


def _tuneRowForLabel(frame: pd.DataFrame, label: str) -> pd.Series:
    return frame[frame["label"] == label].iloc[0]


def _selectionRow(
    selection: str,
    tuneRow: pd.Series,
    holdRow: pd.Series,
) -> dict[str, Any]:
    return {
        "selection": selection,
        "label": tuneRow["label"],
        "baseLabel": tuneRow["baseLabel"],
        "peakLabel": tuneRow["peakLabel"],
        "releaseTargetPct": tuneRow.get(
            "PEAK_LOCK_RELEASE_TARGET_PCT",
            "",
        ),
        "ultraGraceDays": tuneRow.get(
            "PEAK_LOCK_ULTRA_GRACE_DAYS",
            "",
        ),
        "tune_timeRegionScore": tuneRow["timeRegionScore"],
        "holdout_timeRegionScore": holdRow["timeRegionScore"],
        "holdout_grossVsHodl": holdRow["grossVsHodl"],
        "holdout_mddPct": holdRow["mddPct"],
        "holdout_trades": holdRow["trades"],
        "holdout_peakLocks": holdRow["peakLocks"],
        "holdout_cappedBuys": holdRow["cappedBuys"],
        "holdout_strongReleases": holdRow["strongReleases"],
    }


def runPeakLockTune(
    runDir: Path,
    anchorMs: int | None = None,
    charts: int = 4,
    baseRows: int = BASE_ROW_COUNT,
) -> Path | None:
    bestCfgPath = runDir / "best-configs" / "best-config.json"
    outDir = runDir / "peak-lock"
    cfg = profile.loadJson(str(bestCfgPath))
    profile.ensureFinalPortionPct(cfg)
    resolvedAnchorMs = (
        anchorMs if anchorMs is not None else _anchorFromRun(runDir)
    )
    _primerDays, _trainingDays, _tunerDays, holdoutDays, _totalDays = (
        profile.windowParts(cfg)
    )
    if holdoutDays <= 0:
        return None
    if not str(profile.scalarValue(cfg.get("DAILY_CLUSTER_PATH", ""), "")):
        return None

    variantSource = _variantAxes(cfg)[2]
    variants = _variants(cfg)
    baseCandidates = _candidateConfigs(runDir, cfg, int(baseRows))
    configByLabel: dict[str, dict[str, Any]] = {}
    tuneRows: list[dict[str, Any]] = []
    for item in baseCandidates:
        baseLabel = str(item["baseLabel"])
        baseCfg = dict(item["config"])
        currentLabel = f"{baseLabel}-current"
        configByLabel[currentLabel] = baseCfg
        row = _runWindow(baseCfg, currentLabel, "tune", resolvedAnchorMs)
        row.update({
            "baseLabel": baseLabel,
            "baseRank": int(item["baseRank"]),
            "baseSource": str(item["baseSource"]),
            "peakLabel": "current",
        })
        tuneRows.append(row)
        for spec in variants:
            peakLabel = str(spec["label"])
            label = f"{baseLabel}-{peakLabel}"
            candidate = _configWithSpec(baseCfg, spec)
            configByLabel[label] = candidate
            row = _runWindow(candidate, label, "tune", resolvedAnchorMs)
            row.update({
                "baseLabel": baseLabel,
                "baseRank": int(item["baseRank"]),
                "baseSource": str(item["baseSource"]),
                "peakLabel": peakLabel,
            })
            tuneRows.append(row)
    tuneFrame = pd.DataFrame(tuneRows)
    tuneFrame = tuneFrame.sort_values("timeRegionScore", ascending=False)
    _writeFrame(outDir / "summary-tune.csv", tuneFrame)

    baseFrame = pd.DataFrame([
        {
            "baseLabel": str(i["baseLabel"]),
            "baseRank": int(i["baseRank"]),
            "baseSource": str(i["baseSource"]),
            "DAILY_CLUSTER_PATH": i["config"].get("DAILY_CLUSTER_PATH", ""),
        }
        for i in baseCandidates
    ])
    _writeFrame(outDir / "base-candidates.csv", baseFrame)

    tuneVariants = tuneFrame[tuneFrame["peakLabel"] != "current"].copy()
    bestTune = tuneVariants.iloc[0]
    currentTune = tuneFrame[tuneFrame["peakLabel"] == "current"].iloc[0]
    bestPeakPath = runDir / "best-configs" / "bestpeaklock-config.json"
    if (
        float(bestTune["timeRegionScore"])
        <= float(currentTune["timeRegionScore"]) + 1e-9
    ):
        if bestPeakPath.is_file():
            bestPeakPath.unlink()
        _writeJson(outDir / "metadata.json", {
            "schema": "gradbot-peak-lock-tune-v1",
            "baseConfig": str(bestCfgPath),
            "selectedConfig": None,
            "anchorMs": resolvedAnchorMs,
            "selectionMetric": "timeRegionScore",
            "variantCount": len(variants),
            "variantSource": variantSource,
            "baseCandidateCount": len(baseCandidates),
            "notes": "No peak-lock variant beat current_model on tune.",
        })
        print("[peak-lock] no tune improvement; skipping selected config")
        return None
    bestPeakCfg = configByLabel[str(bestTune["label"])]
    _writeJson(bestPeakPath, roundForJson(bestPeakCfg, places=6))

    holdRows: list[dict[str, Any]] = []
    for item in baseCandidates:
        baseLabel = str(item["baseLabel"])
        baseCfg = dict(item["config"])
        currentLabel = f"{baseLabel}-current"
        row = _runWindow(baseCfg, currentLabel, "holdout", resolvedAnchorMs)
        row.update({
            "baseLabel": baseLabel,
            "baseRank": int(item["baseRank"]),
            "baseSource": str(item["baseSource"]),
            "peakLabel": "current",
        })
        holdRows.append(row)
        for spec in variants:
            peakLabel = str(spec["label"])
            label = f"{baseLabel}-{peakLabel}"
            row = _runWindow(
                configByLabel[label],
                label,
                "holdout",
                resolvedAnchorMs,
            )
            row.update({
                "baseLabel": baseLabel,
                "baseRank": int(item["baseRank"]),
                "baseSource": str(item["baseSource"]),
                "peakLabel": peakLabel,
            })
            holdRows.append(row)
    holdFrame = pd.DataFrame(holdRows)
    holdFrame = holdFrame.sort_values("timeRegionScore", ascending=False)
    _writeFrame(outDir / "summary-holdout.csv", holdFrame)
    paired = _paired(tuneFrame, holdFrame)
    _writeFrame(outDir / "paired-summary.csv", paired)

    bestHold = holdFrame[holdFrame["peakLabel"] != "current"].iloc[0]
    selectedHold = holdFrame[holdFrame["label"] == bestTune["label"]].iloc[0]
    bestHoldTune = _tuneRowForLabel(tuneFrame, str(bestHold["label"]))
    selection = pd.DataFrame([
        _selectionRow("tune_selected", bestTune, selectedHold),
        _selectionRow("holdout_best_diagnostic", bestHoldTune, bestHold),
    ])
    _writeFrame(outDir / "selection.csv", selection)

    chartCount = int(charts)
    chartLabels = list(tuneVariants.head(chartCount)["label"])
    for label in chartLabels:
        _runWindow(
            configByLabel[str(label)],
            str(label),
            "tune",
            resolvedAnchorMs,
            chartsDir=outDir / "charts",
        )
    if chartCount > 0:
        _runWindow(
            bestPeakCfg,
            str(bestTune["label"]),
            "holdout",
            resolvedAnchorMs,
            chartsDir=outDir / "charts",
        )
    _writeJson(outDir / "metadata.json", {
        "schema": "gradbot-peak-lock-tune-v1",
        "baseConfig": str(bestCfgPath),
        "selectedConfig": str(bestPeakPath),
        "anchorMs": resolvedAnchorMs,
        "selectionMetric": "timeRegionScore",
        "variantCount": len(variants),
        "variantSource": variantSource,
        "baseCandidateCount": len(baseCandidates),
        "baseRows": int(baseRows),
        "notes": (
            "Tune selects across bounded DSP base rows and peak-lock/PID "
            "supervisor params. Holdout rows are reported after selection "
            "and should not drive parameter choice."
        ),
    })
    print(
        "[peak-lock] selected "
        f"{bestTune['label']} tune={bestTune['timeRegionScore']:.2f} "
        f"holdout={selectedHold['timeRegionScore']:.2f}"
    )
    return bestPeakPath


########################################################################
# CLI
########################################################################

def parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="peak_lock_tune",
        description="Select peak-lock supervisor params from tune window.",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--anchor-ms", type=int, default=None)
    parser.add_argument("--anchor-date", default=None)
    parser.add_argument("--charts", type=int, default=4)
    parser.add_argument("--base-rows", type=int, default=BASE_ROW_COUNT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parseArgs(argv)
    runPeakLockTune(
        Path(args.run_dir),
        anchorMs=resolveAnchorMs(args.anchor_ms, args.anchor_date),
        charts=int(args.charts),
        baseRows=int(args.base_rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
