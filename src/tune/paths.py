#!/usr/bin/env python3
"""Path and label helpers for tune orchestration."""

from __future__ import annotations

from pathlib import Path


###############################################################################
# Labels
###############################################################################

def safeLabel(text: str, index: int) -> str:
    stem = Path(str(text)).stem
    safe = "".join(
        i.lower() if i.isalnum() else "-"
        for i in stem
    ).strip("-")
    if not safe:
        safe = "posture"
    return f"p{int(index):02d}-{safe[:48]}"


###############################################################################
# Run Paths
###############################################################################

def chartsHoldoutDir(runDir: Path) -> Path:
    return Path(runDir) / "charts" / "holdout"


def holdoutLogPath(runDir: Path) -> Path:
    return Path(runDir) / "holdout.log"


def fingerprintPath(runDir: Path) -> Path:
    return Path(runDir) / "fingerprint.json"
