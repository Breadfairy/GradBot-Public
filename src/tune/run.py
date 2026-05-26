#!/usr/bin/env python3
"""Direct tuning-first offline entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from tune.context import buildTuneContext
from tune.stages import (
    auditCausality,
    prepareKlines,
    printAnchor,
    runSweepStage,
    syncProfilesStage,
    traceHoldoutStage,
)


###############################################################################
# Tuning
###############################################################################

def runTune(
    profileInput: str,
    runLabel: str,
    outDir: Path | None,
    anchorMs: int | None = None,
    anchorDate: str | None = None,
    flash: bool | None = None,
) -> None:
    ctx = buildTuneContext(
        profileInput,
        runLabel,
        outDir,
        anchorMs=anchorMs,
        anchorDate=anchorDate,
        flash=flash,
    )
    printAnchor(ctx)
    prepareKlines(ctx)
    auditCausality(ctx)
    startTime = time.time()
    runSweepStage(ctx, startTime)
    traceHoldoutStage(ctx)
    syncProfilesStage(ctx)
    print("[tune] finished.")


###############################################################################
# CLI
###############################################################################

def parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tune.run",
        description="Run tuning plus selected-config traces.",
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
    parser.add_argument(
        "--anchor-ms",
        type=int,
        default=None,
        help="Optional UTC millisecond anchor for historical runs",
    )
    parser.add_argument(
        "--anchor-date",
        default=None,
        help="Optional UTC anchor date for historical runs (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--flash",
        action="store_true",
        help="Disable charts and use flash-sized post processing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parseArgs(argv)
    outDir = Path(args.out) if args.out else None
    flash = True if bool(args.flash) else None
    runTune(
        args.profile,
        args.label,
        outDir,
        anchorMs=args.anchor_ms,
        anchorDate=args.anchor_date,
        flash=flash,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
