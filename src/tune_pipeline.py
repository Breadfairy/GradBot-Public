#!/usr/bin/env python3
# tune_pipeline.py – Shell runner for tuning + holdout summaries.

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from prepare_klines import ensureKlinesForProfile
import profile
from tune import runTuner


def copyConfigAtomic(src: Path, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)


def syncProfiles(runDir: Path, profilesDir: Path) -> None:
    bestDir = runDir / "best-configs"
    bestCfg = bestDir / "best-config.json"
    statsCfg = bestDir / "beststats-config.json"
    resultsDir = profilesDir / "results"
    if bestCfg.is_file():
        copyConfigAtomic(bestCfg, resultsDir / "best-config.json")
    if statsCfg.is_file():
        copyConfigAtomic(statsCfg, resultsDir / "stats-config.json")
    elif bestCfg.is_file():
        copyConfigAtomic(bestCfg, resultsDir / "stats-config.json")
    for path in sorted(bestDir.glob("best[0-9]*-config.json")):
        copyConfigAtomic(path, profilesDir / "brackets" / path.name)
        copyConfigAtomic(path, resultsDir / "brackets" / path.name)


def runPipeline(profileInput: str, runLabel: str, outDir: Path | None) -> None:
    rootDir = Path(__file__).resolve().parent.parent
    profilesDir = rootDir / "inputs" / "profiles"
    outputsDir = rootDir / "outputs" / "tuning"

    profilePath = profile.resolveProfilePath(profileInput, profilesDir)
    runDir = outDir if outDir is not None else outputsDir / runLabel
    runDir.mkdir(parents=True, exist_ok=True)

    cfg = profile.loadJson(profilePath)
    profile.ensureFinalPortionPct(cfg)
    profile.validate(cfg, kind="tuner")
    ensureKlinesForProfile(cfg)

    runTuner(cfg, str(runDir))
    syncProfiles(runDir, profilesDir)
    print("[tune] finished.")


def parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tune_pipeline",
        description="Run tuner + holdout printing from a single entrypoint.",
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="Profile path or short name under inputs/profiles/",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Run label used under outputs/tuning/<label>/",
    )
    parser.add_argument(
        "--out",
        help="Optional explicit output directory for the tuner run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parseArgs(argv)
    outDir = Path(args.out) if args.out else None
    runPipeline(args.profile, args.label, outDir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
