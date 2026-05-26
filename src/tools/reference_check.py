#!/usr/bin/env python3

########################################################################
# Imports
########################################################################

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from tune.host import buildHostSpec, runHostTuner
from data.prepare_klines import ensureKlinesForProfile
from config import profile


########################################################################
# Constants
########################################################################

FLOAT_FIELDS = {
    "GRAD1_SELL_Z_MIN",
    "MACRO_DYN_PCT_MAX",
    "MACRO_SELL_RELAX_PCT",
    "netPctVsHodl",
    "simValue",
    "benchValue",
    "lifecycleEdgeScore",
    "scoreMetric",
    "mdd",
    "cagr",
    "sharpe4w",
    "sortino4w",
    "sharpe13w",
    "sortino13w",
}

INT_FIELDS = {
    "trades",
    "GRAD1_SELL_WIN_DAYS",
    "MACRO_NRG_WIN_DAYS",
    "MACRO_GRAD_WIN_DAYS",
}

ROW_FIELDS = [
    "GRAD1_SELL_Z_MIN",
    "GRAD1_SELL_WIN_DAYS",
    "MACRO_NRG_WIN_DAYS",
    "MACRO_DYN_PCT_MAX",
    "MACRO_GRAD_WIN_DAYS",
    "MACRO_SELL_RELAX_PCT",
    "trades",
    "netPctVsHodl",
    "simValue",
    "benchValue",
    "lifecycleEdgeScore",
    "scoreMetric",
    "mdd",
    "cagr",
    "sharpe4w",
    "sortino4w",
    "sharpe13w",
    "sortino13w",
]


########################################################################
# Helpers
########################################################################

def rowValue(row: dict[str, str], key: str) -> Any:
    value = row[key]
    if key in INT_FIELDS:
        return int(float(value))
    if key in FLOAT_FIELDS:
        return round(float(value), 6)
    return value


def readRow(path: Path) -> dict[str, Any]:
    row = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        source = next(reader)
    for key in ROW_FIELDS:
        row[key] = rowValue(source, key)
    return row


def resultRows(outDir: Path) -> int:
    count = 0
    with open(outDir / "results.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        for _row in reader:
            count += 1
    return count


def loadConfig(path: Path) -> dict[str, Any]:
    cfg = profile.loadJson(str(path))
    profile.ensureFinalPortionPct(cfg)
    profile.validate(cfg, kind="tuner")
    return cfg


def buildActual(outDir: Path, anchorMs: int) -> dict[str, Any]:
    actual = {
        "anchorMs": anchorMs,
        "comboCount": resultRows(outDir),
        "best": readRow(outDir / "best-row.csv"),
        "stats": readRow(outDir / "stats-row.csv"),
    }
    return actual


def assertSame(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    actualText = json.dumps(actual, sort_keys=True, indent=2)
    expectedText = json.dumps(expected, sort_keys=True, indent=2)
    if actualText != expectedText:
        print("[reference] mismatch", file=sys.stderr)
        print("[reference] actual:", file=sys.stderr)
        print(actualText, file=sys.stderr)
        print("[reference] expected:", file=sys.stderr)
        print(expectedText, file=sys.stderr)
        raise SystemExit(1)


########################################################################
# Main
########################################################################

def main(argv: list[str]) -> int:
    if len(argv) != 5:
        raise SystemExit(
            "usage: reference_check.py PROFILE EXPECTED OUT_DIR ANCHOR_MS"
        )

    profilePath = Path(argv[1])
    expectedPath = Path(argv[2])
    outDir = Path(argv[3])
    anchorMs = int(argv[4])
    cfg = loadConfig(profilePath)
    start = time.time()

    os.makedirs(outDir, exist_ok=True)
    ensureKlinesForProfile(cfg, anchorMs=anchorMs)
    specDir = buildHostSpec(cfg, str(outDir), anchorMs=anchorMs)
    runHostTuner(specDir)

    actual = buildActual(outDir, anchorMs)
    with open(expectedPath) as fh:
        expected = json.load(fh)
    assertSame(actual, expected)

    print(
        "[reference] phase9 fast check passed: "
        f"{actual['comboCount']} combos in {time.time() - start:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
