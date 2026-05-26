#!/usr/bin/env python3

from __future__ import annotations

########################################################################
# Imports
########################################################################

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from clustering.render_regime_bands import _finalStates


########################################################################
# Constants
########################################################################

STRONG_CLUSTER = 2
NEUTRAL_CLUSTER = 1
DOWN_CLUSTER = 0
POST_ULTRA_CRAB_CLUSTER = 3


########################################################################
# Helpers
########################################################################

def _stepMs(rows: pd.DataFrame) -> int:
    vals = rows["openMs"].astype(np.int64).to_numpy()
    diffs = np.diff(vals)
    diffs = diffs[diffs > 0]
    return int(np.median(diffs))


def _writeFrame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


########################################################################
# Public API
########################################################################

def exportPosture(
    timeValsPath: Path,
    outPath: Path,
    model: str,
    confirmBars: int,
) -> Path:
    rows = pd.read_csv(timeValsPath)
    rows = rows[rows["model"].eq(model)].copy()
    rows = rows.drop_duplicates("openMs", keep="last")
    rows = rows.sort_values("openMs").reset_index(drop=True)
    states = _finalStates(rows, True, 1, int(confirmBars))
    stepMs = _stepMs(rows)
    stateVals = states.astype(str)
    cluster = np.select(
        [
            stateVals.isin(["ultraBull", "bullChop"]),
            stateVals.isin(["bear", "flush"]),
            stateVals.isin(["crab_post", "crab_c2", "lock"]),
        ],
        [
            STRONG_CLUSTER,
            DOWN_CLUSTER,
            POST_ULTRA_CRAB_CLUSTER,
        ],
        default=NEUTRAL_CLUSTER,
    )

    out = pd.DataFrame({
        "openMs": rows["openMs"].astype(np.int64),
        "closeMs": rows["openMs"].astype(np.int64) + int(stepMs),
        "close": rows["close"].astype(float),
        "cluster": cluster.astype(int),
        "confirmedRegime": stateVals,
    })
    _writeFrame(outPath, out)
    return outPath


########################################################################
# CLI
########################################################################

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="export_regime_posture",
        description="Export confirmed regime states as posture clusters.",
    )
    parser.add_argument("--timevals", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--confirm-bars", type=int, default=12)
    args = parser.parse_args()
    path = exportPosture(
        Path(args.timevals),
        Path(args.out),
        str(args.model),
        int(args.confirm_bars),
    )
    print(path)


if __name__ == "__main__":
    main()
