#!/usr/bin/env python3

from __future__ import annotations

########################################################################
# Imports
########################################################################

import argparse
import os
from pathlib import Path

import pandas as pd

from clustering.cluster_state_machine import runStateMachine


########################################################################
# Helpers
########################################################################

def _writeFrame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _featurePaths(root: Path, family: str) -> list[Path]:
    return sorted(root.glob(f"*/**/{family}/k*/clustered_features.csv"))


def _parts(path: Path, family: str) -> dict[str, str]:
    parts = list(path.parts)
    idx = parts.index(family)
    if parts[idx - 1] in {"kmeans", "gmm"}:
        periods = parts[idx - 2]
        method = parts[idx - 1]
    else:
        periods = parts[idx - 1]
        method = "kmeans"
    return {
        "periods": periods,
        "method": method,
        "family": family,
        "clusters": parts[idx + 1],
    }


def _pairName(regimePath: Path, eventPath: Path) -> str:
    reg = _parts(regimePath, "raw_market_ema_expanded")
    evt = _parts(eventPath, "capitulation_state")
    return (
        f"{reg['periods']}_{reg['method']}_"
        f"raw{reg['clusters']}_cap{evt['clusters']}"
    )


def _readScores(outDir: Path) -> pd.DataFrame:
    path = outDir / "state_machine_scores.csv"
    data = pd.read_csv(path)
    data["runDir"] = str(outDir)
    return data


########################################################################
# Run
########################################################################

def runSweep(
    root: Path,
    outDir: Path,
    seedPct: float,
    fee: float,
    rebalanceMin: float,
) -> dict[str, object]:
    regimePaths = _featurePaths(root, "raw_market_ema_expanded")
    eventPaths = _featurePaths(root, "capitulation_state")
    scoreFrames: list[pd.DataFrame] = []
    pairs = 0
    for regimePath in regimePaths:
        reg = _parts(regimePath, "raw_market_ema_expanded")
        for eventPath in eventPaths:
            evt = _parts(eventPath, "capitulation_state")
            if reg["periods"] != evt["periods"]:
                continue
            if reg["method"] != evt["method"]:
                continue
            name = _pairName(regimePath, eventPath)
            runDir = outDir / name
            result = runStateMachine(
                regimePath,
                eventPath,
                runDir,
                seedPct,
                fee,
                rebalanceMin,
                False,
            )
            scores = _readScores(runDir)
            scores["pair"] = name
            scores["regimeClusters"] = reg["clusters"]
            scores["eventClusters"] = evt["clusters"]
            scores["periods"] = reg["periods"]
            scores["method"] = reg["method"]
            scoreFrames.append(scores)
            pairs += 1
            print(
                f"[state-sweep] {name}: rows={result['rows']} "
                f"scores={result['scores']}"
            )
    allScores = pd.concat(scoreFrames, ignore_index=True)
    _writeFrame(outDir / "all_state_machine_scores.csv", allScores)
    fit = allScores[allScores["partition"] == "fit"].copy()
    holdout = allScores[allScores["partition"] == "holdout"].copy()
    selected = fit.sort_values("score", ascending=False).head(100)
    chosen = set(zip(selected["pair"], selected["model"]))
    rows = [
        row for _i, row in holdout.iterrows()
        if (row["pair"], row["model"]) in chosen
    ]
    _writeFrame(outDir / "selected_holdout_scores.csv", pd.DataFrame(rows))
    bestHoldout = holdout.sort_values("score", ascending=False).head(100)
    _writeFrame(outDir / "best_holdout_scores.csv", bestHoldout)
    return {"pairs": pairs, "outDir": str(outDir)}


########################################################################
# CLI
########################################################################

def parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cluster_state_machine_sweep",
        description="Run state-machine tests across clustering outputs.",
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed-pct", type=float, default=55.0)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--rebalance-min", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parseArgs(argv)
    result = runSweep(
        Path(args.root),
        Path(args.out),
        float(args.seed_pct),
        float(args.fee),
        float(args.rebalance_min),
    )
    print(f"[state-sweep] output: {result['outDir']}")
    print(f"[state-sweep] pairs: {result['pairs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
