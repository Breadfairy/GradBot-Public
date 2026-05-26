#!/usr/bin/env python3
"""Run context for the tuning-first offline workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import time

from config import profile
from data.time_bounds import resolveAnchorMs


###############################################################################
# Data
###############################################################################

@dataclass
class TuneContext:
    rootDir: Path
    profilesDir: Path
    outputsDir: Path
    profilePath: Path
    runDir: Path
    label: str
    cfg: dict
    anchorMs: int
    anchorKind: str
    anchorDate: str | None
    flash: bool
    charts: bool
    peakLockBaseRows: int


###############################################################################
# Helpers
###############################################################################

def truthyEnv(name: str) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def anchorKind(
    anchorMs: int | None,
    anchorDate: str | None,
) -> str:
    if anchorDate is not None:
        return "date"
    if anchorMs is not None:
        return "ms"
    return "runtime"


def resolveRunAnchorMs(
    anchorMs: int | None,
    anchorDate: str | None,
) -> int:
    resolved = resolveAnchorMs(
        anchorMs=anchorMs,
        anchorDate=anchorDate,
    )
    if resolved is not None:
        return int(resolved)
    return int(time.time() * 1000)


def runDirHasFiles(path: Path) -> bool:
    runDir = Path(path)
    if not runDir.exists():
        return False
    return any(runDir.iterdir())


def buildTuneContext(
    profileInput: str,
    label: str,
    outDir: Path | None,
    anchorMs: int | None = None,
    anchorDate: str | None = None,
    flash: bool | None = None,
) -> TuneContext:
    rootDir = Path(__file__).resolve().parents[2]
    profilesDir = rootDir / "inputs" / "profiles"
    outputsDir = rootDir / "outputs" / "tuning"
    mplDir = rootDir / ".mplconfig"
    mplDir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplDir))
    profilePath = profile.resolveProfilePath(profileInput, profilesDir)
    runDir = outDir if outDir is not None else outputsDir / str(label)
    if runDirHasFiles(runDir):
        raise SystemExit(
            f"output directory exists and is not empty: {runDir}"
        )
    runDir.mkdir(parents=True, exist_ok=True)

    cfg = profile.loadJson(profilePath)
    profile.ensureFinalPortionPct(cfg)
    profile.validate(cfg, kind="tuner")

    flashFlag = truthyEnv("TUNE_FLASH") if flash is None else bool(flash)
    charts = not flashFlag
    peakLockBaseRows = 2 if flashFlag else 6
    if flashFlag:
        cfg["CHARTS_TIMEVAL"] = False
        cfg["CHARTS_TRADES"] = False

    return TuneContext(
        rootDir=rootDir,
        profilesDir=profilesDir,
        outputsDir=outputsDir,
        profilePath=profilePath,
        runDir=runDir,
        label=str(label),
        cfg=cfg,
        anchorMs=resolveRunAnchorMs(anchorMs, anchorDate),
        anchorKind=anchorKind(anchorMs, anchorDate),
        anchorDate=anchorDate,
        flash=flashFlag,
        charts=charts,
        peakLockBaseRows=peakLockBaseRows,
    )
