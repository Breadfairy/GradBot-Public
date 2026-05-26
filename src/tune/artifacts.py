#!/usr/bin/env python3
"""Artifact writes and copies for tune orchestration."""

from __future__ import annotations

from collections import OrderedDict
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable
import json
import os
import shutil

import pandas as pd

from tune.schema import configKeyOrder


CONFIG_KEY_ORDER = configKeyOrder()


###############################################################################
# Config Formatting
###############################################################################

def normalizeForCompare(value: Any, places: int = 6) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        rounded = round(float(value), int(places))
        if rounded.is_integer():
            return int(rounded)
        return rounded
    if isinstance(value, list):
        return [normalizeForCompare(i, places) for i in value]
    if isinstance(value, tuple):
        return tuple(normalizeForCompare(i, places) for i in value)
    if isinstance(value, dict):
        return {k: normalizeForCompare(v, places) for k, v in value.items()}
    return value


def configsEqual(a: dict, b: dict, places: int = 6) -> bool:
    return normalizeForCompare(a, places) == normalizeForCompare(b, places)


def orderedConfig(
    data: dict[str, Any],
    shape: dict[str, Any] | None = None,
) -> "OrderedDict[str, Any]":
    ordered: "OrderedDict[str, Any]" = OrderedDict()
    if shape is not None:
        for k in shape.keys():
            if k in data:
                ordered[k] = data[k]
    for k in CONFIG_KEY_ORDER:
        if k in data and k not in ordered:
            ordered[k] = data[k]
    for k, v in data.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def roundForJson(value: Any, places: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, places)
    if isinstance(value, list):
        return [roundForJson(i, places) for i in value]
    if isinstance(value, tuple):
        return tuple(roundForJson(i, places) for i in value)
    if isinstance(value, dict):
        return {k: roundForJson(v, places) for k, v in value.items()}
    return value


def bestConfigFromRow(
    rowData: dict,
    config: dict,
) -> "OrderedDict[str, Any]":
    # Keep only active profile keys so deleted params do not leak forward.
    bestConfigLocal = {
        k: v
        for k, v in config.items()
        if k != "ticker" and k in CONFIG_KEY_ORDER
    }
    if isinstance(bestConfigLocal.get("intervals"), str):
        bestConfigLocal["intervals"] = str(rowData["interval"])
    else:
        bestConfigLocal["intervals"] = [str(rowData["interval"])]
    for k, v in rowData.items():
        if k in bestConfigLocal:
            bestConfigLocal[k] = v
    return orderedConfig(bestConfigLocal, config)


###############################################################################
# Atomic Writes
###############################################################################

def copyConfigAtomic(src: Path, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)


def writeJsonAtomic(dest: Path, data: dict) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, dest)


def writeFrameAtomic(dest: Path, frame: pd.DataFrame) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tmp, index=False)
    os.replace(tmp, dest)


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
) -> tuple["OrderedDict[str, Any]", str, None, str]:
    baseRoot = containerDir if containerDir else outDir
    targetDir = os.path.join(baseRoot, "best-configs")
    os.makedirs(targetDir, exist_ok=True)

    bestPeriods = [rowData["p1"], rowData["p2"], rowData["p3"]]
    minCandlesLocal = (max(bestPeriods) * 2) + 1
    intervalValue = rowData["interval"]
    intervalKlines = klinesByInterval.get(intervalValue)
    if intervalKlines is None or len(intervalKlines) < minCandlesLocal:
        intervalKlines = ensureKlinesFn(
            ticker,
            intervalValue,
            days,
            minCandlesLocal,
        )
        klinesByInterval[intervalValue] = intervalKlines

    bestConfigOrdered = bestConfigFromRow(rowData, config)
    if suffix is None:
        fileName = "best-config.json"
    elif suffix == "stats":
        fileName = "beststats-config.json"
    else:
        fileName = f"best{suffix}-config.json"
    bestConfigPath = os.path.join(targetDir, fileName)
    writeJsonAtomic(
        Path(bestConfigPath),
        roundForJson(bestConfigOrdered, places=6),
    )
    return bestConfigOrdered, bestConfigPath, None, targetDir


###############################################################################
# Profile Sync
###############################################################################

def _resultsDirForProfile(
    profilesDir: Path,
    profilePath: Path | None,
) -> Path:
    root = Path(profilesDir).resolve()
    source = Path(profilePath).resolve() if profilePath is not None else None
    for name in ("user", "codex"):
        bucket = root / name
        if source is None:
            continue
        try:
            source.relative_to(bucket.resolve())
            return bucket / "results"
        except ValueError:
            continue
    return root / "user" / "results"


def syncProfiles(
    runDir: Path,
    profilesDir: Path,
    profilePath: Path | None = None,
) -> None:
    bestDir = Path(runDir) / "best-configs"
    bestCfg = bestDir / "best-config.json"
    statsCfg = bestDir / "beststats-config.json"
    resultsDir = _resultsDirForProfile(profilesDir, profilePath)
    if bestCfg.is_file():
        copyConfigAtomic(bestCfg, resultsDir / "best-config.json")
    if statsCfg.is_file():
        copyConfigAtomic(statsCfg, resultsDir / "stats-config.json")
    elif bestCfg.is_file():
        copyConfigAtomic(bestCfg, resultsDir / "stats-config.json")
    for path in sorted(bestDir.glob("bestrobust*.json")):
        copyConfigAtomic(path, resultsDir / path.name)


def copyLaneArtifacts(
    laneDir: Path,
    runDir: Path,
    selectedConfig: Path,
) -> None:
    bestDir = Path(runDir) / "best-configs"
    bestDir.mkdir(parents=True, exist_ok=True)
    copyConfigAtomic(selectedConfig, bestDir / "best-config.json")
    copyConfigAtomic(selectedConfig, bestDir / "beststats-config.json")
    for name in (
        "results.csv",
        "best-row.csv",
        "stats-row.csv",
        "robust-row.csv",
        "robust-candidates.csv",
    ):
        src = Path(laneDir) / name
        if src.is_file():
            copyConfigAtomic(src, Path(runDir) / name)
