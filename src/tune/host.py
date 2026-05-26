#!/usr/bin/env python3
# host_tune.py – pack and post helpers for the C host tuner.

from __future__ import annotations

import csv
import itertools
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

from data.klines_io import KLINES_ROOT
from config import profile
from repo_paths import LIVE_INPUT_DIR, NATIVE_HOST_DIR, NATIVE_TUNE_BIN
from repo_paths import ROOT_DIR
from tune.axes import axesFromConfig, buildIntervalGroups
from tune.schema import hostAxisNameMap, rowIntFields, rowStrFields


HOST_SPEC_DIR = "host-spec"
HOST_META_PATH = "meta.txt"
HOST_INTERVAL_PATH = "interval_groups.csv"
HOST_MACRO_PATH = "macro_groups.csv"
HOST_AXES_PATH = "axes.txt"
HOST_CLUSTER_MODEL_PATH = "daily_cluster_model.txt"
BEST_ROW_PATH = "best-row.csv"
STATS_ROW_PATH = "stats-row.csv"
ROW_STR_FIELDS = rowStrFields()
ROW_INT_FIELDS = rowIntFields()
AXIS_NAME_MAP = hostAxisNameMap()


def _writeText(path: Path, text: str) -> None:
    tmpPath = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmpPath, "w") as fh:
        fh.write(text)
    os.replace(tmpPath, path)


def _hostDir() -> Path:
    return NATIVE_HOST_DIR


def _hostBin() -> Path:
    return NATIVE_TUNE_BIN


def _rowPath(outDir: Path, name: str) -> Path:
    return outDir / name


def _dailyClusterPath(config: dict) -> str:
    raw = str(profile.scalarValue(config.get("DAILY_CLUSTER_PATH", ""), ""))
    if not raw:
        return ""
    path = Path(raw)
    return str(path if path.is_absolute() else ROOT_DIR / path)


def _clusterModelSourcePath(config: dict) -> Path | None:
    raw = str(
        profile.scalarValue(config.get("DAILY_CLUSTER_MODEL_PATH", ""), "")
    )
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    rootPath = ROOT_DIR / path
    if rootPath.is_file():
        return rootPath
    return LIVE_INPUT_DIR / path


def _flatCsv(values: list[Any]) -> str:
    return ",".join(str(i) for i in values)


def _modelList(model: dict, key: str) -> list[Any]:
    value = model.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"cluster model field must be a list: {key}")
    return value


def _writeClusterModelSpec(
    config: dict,
    specDir: Path,
    tuneStartMs: int,
) -> str:
    source = _clusterModelSourcePath(config)
    if source is None:
        return ""
    with source.open() as fh:
        model = json.load(fh)
    periods = model.get("periods", {})
    featureIds = _modelList(model, "featureIds")
    center = _modelList(model, "center")
    scale = _modelList(model, "scale")
    pcaMean = _modelList(model, "pcaMean")
    pcaComponents = _modelList(model, "pcaComponents")
    centroids = _modelList(model, "centroids")
    remap = model.get("clusterRemap", [])
    if remap is None:
        remap = []
    if not isinstance(remap, list):
        raise ValueError("cluster model field must be a list: clusterRemap")
    fitEndMs = int(model.get("fitEndMs", 0) or 0)
    if fitEndMs and fitEndMs > int(tuneStartMs):
        raise ValueError(
            "cluster model fitEndMs is after tune window start: "
            f"fitEndMs={fitEndMs} tuneStartMs={int(tuneStartMs)}"
        )
    lines = [
        "schema=gradbot-cluster-model-v1",
        f"sourcePath={source}",
        f"view={model['view']}",
        f"featureFamily={model['featureFamily']}",
        f"clusterMethod={model.get('clusterMethod', 'kmeans')}",
        f"ticker={model['ticker']}",
        f"interval={model['interval']}",
        f"windowBars={int(model['windowBars'])}",
        f"periodFast={int(periods['fast'])}",
        f"periodMid={int(periods['mid'])}",
        f"periodSlow={int(periods['slow'])}",
        f"clusterCount={int(model['clusterCount'])}",
        f"featureCount={int(model['featureCount'])}",
        f"pcaCount={int(model['pcaCount'])}",
        f"featureIds={_flatCsv(featureIds)}",
        f"center={_flatCsv(center)}",
        f"scale={_flatCsv(scale)}",
        f"pcaMean={_flatCsv(pcaMean)}",
        f"pcaComponents={_flatCsv(pcaComponents)}",
        f"centroids={_flatCsv(centroids)}",
        f"clusterRemap={_flatCsv(remap)}",
        f"fitEndMs={fitEndMs}",
    ]
    dest = specDir / HOST_CLUSTER_MODEL_PATH
    _writeText(dest, "\n".join(lines) + "\n")
    return str(dest)


def buildHostSpec(
    config: dict,
    out_dir: str,
    anchorMs: int | None = None,
) -> Path:
    outDir = Path(out_dir).resolve()
    specDir = outDir / HOST_SPEC_DIR
    axes = axesFromConfig(config)
    intervals = profile.intervalsFromConfig(config)
    primerDays, trainingDays, tunerDays, holdoutDays, totalDays = (
        profile.profileWindows(config)
    )
    runAnchorMs = (
        int(anchorMs)
        if anchorMs is not None
        else int(time.time() * 1000)
    )
    intervalGroups = buildIntervalGroups(
        intervals,
        axes["p1Values"],
        axes["p2Values"],
        axes["p3Values"],
    )
    tuneStartMs = int(runAnchorMs) - (
        (int(holdoutDays) + int(tunerDays)) * 24 * 60 * 60 * 1000
    )
    clusterModelPath = _writeClusterModelSpec(config, specDir, tuneStartMs)

    metaLines = [
        f"ticker={config['tickers'][0]}",
        f"cacheRoot={KLINES_ROOT}",
        f"outDir={outDir}",
        f"resultsCsvPath={outDir / 'results.csv'}",
        f"bestRowCsvPath={_rowPath(outDir, BEST_ROW_PATH)}",
        f"statsRowCsvPath={_rowPath(outDir, STATS_ROW_PATH)}",
        f"anchorMs={int(runAnchorMs)}",
        f"totalDays={int(totalDays)}",
        f"primerDays={int(primerDays)}",
        f"trainingDays={int(trainingDays)}",
        f"tunerDays={int(tunerDays)}",
        f"holdoutDays={int(holdoutDays)}",
        f"feeRate={float(config['WALLET_FEE_RATE'])}",
        f"seedQuote={float(config['WALLET_SEED_QUOTE'])}",
        f"dailyClusterPath={_dailyClusterPath(config)}",
        f"dailyClusterModelPath={clusterModelPath}",
    ]
    _writeText(specDir / HOST_META_PATH, "\n".join(metaLines) + "\n")

    intervalLines = [
        f"{iv},{int(p1)},{int(p2)},{int(p3)}"
        for iv, p1, p2, p3 in intervalGroups
    ]
    _writeText(
        specDir / HOST_INTERVAL_PATH,
        "\n".join(intervalLines) + "\n",
    )

    macroLines = [
        (
            f"{vals[0]},{int(vals[1])},{float(vals[2])},{float(vals[3])},"
            f"{float(vals[4])},{float(vals[5])},{int(vals[6])},"
            f"{int(vals[7])},{int(vals[8])},{int(vals[9])},"
            f"{float(vals[10])},{float(vals[11])},{float(vals[12])},"
            f"{float(vals[13])}"
        )
        for vals in itertools.product(
            axes["macroIntervalValues"],
            axes["macroDynWinValues"],
            axes["macroDynZMinValues"],
            axes["macroDynZMaxValues"],
            axes["macroDynPctMinValues"],
            axes["macroDynPctMaxValues"],
            axes["macroP1Values"],
            axes["macroP3Values"],
            axes["macroGradPeriodValues"],
            axes["macroGradWinValues"],
            axes["macroGradZMinValues"],
            axes["macroGradZMaxValues"],
            axes["macroGradMultMinValues"],
            axes["macroGradMultMaxValues"],
        )
    ]
    _writeText(
        specDir / HOST_MACRO_PATH,
        "\n".join(macroLines) + "\n",
    )

    axisLines = []
    for hostName, axisName in AXIS_NAME_MAP.items():
        values = axes[axisName]
        axisLines.append(
            f"{hostName}=" + ",".join(str(value) for value in values)
        )
    _writeText(specDir / HOST_AXES_PATH, "\n".join(axisLines) + "\n")
    return specDir


def _readWinnerRow(path: Path) -> dict:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        row = next(reader)
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key in ROW_STR_FIELDS:
            out[key] = value
        elif key in ROW_INT_FIELDS:
            out[key] = int(float(value))
        else:
            out[key] = float(value)
    return out


def runHostTuner(spec_dir: Path) -> None:
    subprocess.run(
        ["make", "gradbot_tune"],
        cwd=_hostDir(),
        stdout=subprocess.DEVNULL,
        check=True,
    )
    subprocess.run(
        [str(_hostBin()), str(spec_dir)],
        check=True,
    )


def postHostRun(
    config: dict,
    out_dir: str,
    startTime: float | None,
    anchorMs: int | None = None,
    anchorKind: str = "runtime",
    anchorDate: str | None = None,
    charts: bool = True,
    peakLockBaseRows: int = 6,
) -> str:
    from tune.post import finalizeTunerRun

    outDir = Path(out_dir).resolve()
    bestRow = _readWinnerRow(outDir / BEST_ROW_PATH)
    statsRow = _readWinnerRow(outDir / STATS_ROW_PATH)
    return finalizeTunerRun(
        config,
        str(outDir),
        bestRow,
        statsRow,
        startTime=startTime,
        anchorMs=anchorMs,
        anchorKind=anchorKind,
        anchorDate=anchorDate,
        charts=bool(charts),
        peakLockBaseRows=int(peakLockBaseRows),
    )
