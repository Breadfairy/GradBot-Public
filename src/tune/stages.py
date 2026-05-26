#!/usr/bin/env python3
"""Stage functions for the tuning-first offline workflow."""

from __future__ import annotations

from pathlib import Path
import time

import pandas as pd

from data.causality_audit import writeCausalityAudit
from tune.host import buildHostSpec, postHostRun, runHostTuner
from data.prepare_klines import ensureKlinesForProfile
from tune.artifacts import (
    copyLaneArtifacts,
    syncProfiles,
    writeFrameAtomic,
    writeJsonAtomic,
)
from tune.context import TuneContext
from tune.paths import safeLabel
from tune.selection import laneSelection, posturePaths
from tune.post import renderSelectedTuneCharts
from tune.trace import traceHoldoutRun
from config import profile


###############################################################################
# Host Stages
###############################################################################

def printAnchor(ctx: TuneContext) -> None:
    if ctx.anchorKind == "date":
        print(f"[tune] anchor: {ctx.anchorDate} UTC")
    elif ctx.anchorKind == "ms":
        print(f"[tune] anchor-ms: {ctx.anchorMs}")
    else:
        print("[tune] anchor: runtime-now")


def prepareKlines(ctx: TuneContext) -> None:
    ensureKlinesForProfile(ctx.cfg, anchorMs=ctx.anchorMs)


def auditCausality(ctx: TuneContext) -> None:
    auditPath = ctx.runDir / "causality-audit.json"
    audit = writeCausalityAudit(ctx.cfg, auditPath, ctx.anchorMs)
    print(f"[tune] causality audit: {audit['status']}")
    if str(audit["status"]) == "fail":
        raise SystemExit(f"causality audit failed: {auditPath}")


def runHostPipeline(
    cfg: dict,
    runDir: Path,
    startTime: float,
    ctx: TuneContext,
    charts: bool,
) -> None:
    specDir = buildHostSpec(cfg, str(runDir), anchorMs=ctx.anchorMs)
    runHostTuner(specDir)
    postHostRun(
        cfg,
        str(runDir),
        startTime,
        anchorMs=ctx.anchorMs,
        anchorKind=ctx.anchorKind,
        anchorDate=ctx.anchorDate,
        charts=bool(charts),
        peakLockBaseRows=int(ctx.peakLockBaseRows),
    )


###############################################################################
# Posture Sweep
###############################################################################

def runPostureSweep(ctx: TuneContext, startTime: float) -> None:
    paths = posturePaths(ctx.cfg)
    rows = []
    lanesDir = ctx.runDir / "posture-sweep"
    print(f"[tune] posture outer sweep paths={len(paths)}")
    for i, posturePath in enumerate(paths):
        laneLabel = safeLabel(posturePath, i)
        laneDir = lanesDir / laneLabel
        laneCfg = dict(ctx.cfg)
        laneCfg["DAILY_CLUSTER_PATH"] = posturePath
        profile.validate(laneCfg, kind="tuner")
        laneStart = time.time()
        print(f"[tune] posture lane {i + 1}/{len(paths)}: {laneLabel}")
        runHostPipeline(
            laneCfg,
            laneDir,
            laneStart,
            ctx,
            charts=False,
        )
        rows.append(laneSelection(laneDir, posturePath))

    frame = pd.DataFrame(rows)
    frame = frame.sort_values("tune_timeRegionScore", ascending=False)
    writeFrameAtomic(ctx.runDir / "posture-sweep-summary.csv", frame)
    winner = frame.iloc[0]
    selectedConfig = Path(str(winner["selectedConfig"]))
    selectedLane = Path(str(winner["laneDir"]))
    copyLaneArtifacts(selectedLane, ctx.runDir, selectedConfig)
    if ctx.charts:
        renderSelectedTuneCharts(ctx.runDir, anchorMs=ctx.anchorMs)
    writeJsonAtomic(ctx.runDir / "posture-sweep-metadata.json", {
        "schema": "gradbot-posture-outer-sweep-v1",
        "selectionMetric": "tune_timeRegionScore",
        "posturePaths": paths,
        "selectedLane": str(selectedLane),
        "selectedConfig": str(selectedConfig),
        "charts": bool(ctx.charts),
        "peakLockBaseRows": int(ctx.peakLockBaseRows),
        "notes": (
            "Posture selection uses tune metrics only. Holdout columns in "
            "the summary are diagnostics from the already selected lane."
        ),
    })
    writeJsonAtomic(ctx.runDir / "fingerprint.json", {
        "anchorMs": int(ctx.anchorMs),
        "anchorKind": ctx.anchorKind,
        "anchorDate": ctx.anchorDate,
        "postureSweep": True,
    })
    duration = time.time() - startTime
    print(
        "[tune] posture winner "
        f"{winner['label']} tune={winner['tune_timeRegionScore']:.2f}"
    )
    print(f"[tune] duration: {duration/60.0:.1f} minutes")


###############################################################################
# Orchestration Stages
###############################################################################

def runSweepStage(ctx: TuneContext, startTime: float) -> None:
    if len(posturePaths(ctx.cfg)) > 1:
        runPostureSweep(ctx, startTime)
    else:
        runHostPipeline(
            ctx.cfg,
            ctx.runDir,
            startTime,
            ctx,
            charts=ctx.charts,
        )


def traceHoldoutStage(ctx: TuneContext) -> None:
    traceHoldoutRun(ctx.runDir)


def syncProfilesStage(ctx: TuneContext) -> None:
    syncProfiles(ctx.runDir, ctx.profilesDir, ctx.profilePath)
