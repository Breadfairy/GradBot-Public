#!/usr/bin/env python3
"""Selected config discovery for tune trace stages."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


###############################################################################
# Posture Selection
###############################################################################

def posturePaths(cfg: dict) -> list[str]:
    raw = cfg.get("DAILY_CLUSTER_PATH", "")
    if isinstance(raw, list):
        return [str(i).strip() for i in raw if str(i).strip()]
    return [str(raw).strip()] if str(raw).strip() else []


def _num(row, key: str, default: float = float("nan")) -> float:
    value = row.get(key, default)
    out = float(value)
    return out if math.isfinite(out) else default


def _minFinite(a: float, b: float) -> float:
    vals = [i for i in (a, b) if math.isfinite(i)]
    return min(vals) if vals else float("nan")


def _tradePenalty(row) -> float:
    return 0.03 * max(_num(row, "trades", 0.0) - 500.0, 0.0)


def _selectionScore(row) -> float:
    score = _num(row, "scoreMetric", _num(row, "lifecycleEdgeScore"))
    mdd = _num(row, "mdd", 1.0)
    if not math.isfinite(score) or not math.isfinite(mdd):
        return float("-inf")
    drawPenalty = (0.35 * mdd * 100.0)
    drawPenalty += 1.25 * max(mdd - 0.55, 0.0) * 100.0
    return score - drawPenalty - _tradePenalty(row)


def _statsScore(row) -> float:
    life = _num(row, "lifecycleEdgeScore")
    cagr = _num(row, "cagr")
    mdd = _num(row, "mdd")
    sharpe = _minFinite(_num(row, "sharpe4w"), _num(row, "sharpe13w"))
    sortino = _minFinite(_num(row, "sortino4w"), _num(row, "sortino13w"))
    if not all(
        math.isfinite(i)
        for i in (life, cagr, mdd, sharpe, sortino)
    ):
        return float("-inf")
    if mdd <= 1e-12:
        return float("-inf")
    drawPenalty = (0.35 * mdd) + (1.25 * max(mdd - 0.55, 0.0))
    return (
        life
        + (2.0 * (cagr / mdd))
        + (4.0 * sharpe)
        + (4.0 * sortino)
        - (100.0 * drawPenalty)
        - _tradePenalty(row)
    )


def _candidate(
    laneDir: Path,
    rowPath: Path,
    cfgPath: Path,
    kind: str,
) -> dict | None:
    if not rowPath.is_file() or not cfgPath.is_file():
        return None
    frame = pd.read_csv(rowPath)
    row = frame.iloc[0]
    score = _selectionScore(row)
    if not math.isfinite(score):
        score = _statsScore(row)
    if not math.isfinite(score):
        return None
    return {
        "label": Path(laneDir).name,
        "selectedConfig": str(cfgPath),
        "selectedKind": kind,
        "tune_timeRegionScore": float(score),
    }


def laneSelection(laneDir: Path, posturePath: str) -> dict:
    bestDir = Path(laneDir) / "best-configs"
    bestCfg = bestDir / "best-config.json"
    statsCfg = bestDir / "beststats-config.json"
    robustCfg = bestDir / "bestrobust01-config.json"
    bestRowPath = Path(laneDir) / "best-row.csv"
    statsRowPath = Path(laneDir) / "stats-row.csv"
    robustRowPath = Path(laneDir) / "robust-row.csv"
    out = {
        "posturePath": posturePath,
        "laneDir": str(laneDir),
        "selectedConfig": str(bestCfg),
        "selectedKind": "best",
        "tune_timeRegionScore": float("-inf"),
        "holdout_timeRegionScore": float("nan"),
        "holdout_grossVsHodl": float("nan"),
        "holdout_mddPct": float("nan"),
        "releaseTargetPct": float("nan"),
        "ultraGraceDays": float("nan"),
        "holdout_strongReleases": float("nan"),
        "label": "",
    }
    candidates = [
        _candidate(laneDir, bestRowPath, bestCfg, "best"),
        _candidate(
            laneDir,
            statsRowPath,
            statsCfg if statsCfg.is_file() else bestCfg,
            "stats",
        ),
        _candidate(laneDir, robustRowPath, robustCfg, "robust01"),
    ]
    candidates = [i for i in candidates if i is not None]
    if candidates:
        best = max(candidates, key=lambda i: i["tune_timeRegionScore"])
        out.update(best)
    return out


###############################################################################
# Holdout Selection
###############################################################################

def selectedConfigPaths(runDir: Path) -> tuple[Path, Path | None, list]:
    bestDir = Path(runDir) / "best-configs"
    bestCfg = bestDir / "best-config.json"
    statsCfg = bestDir / "beststats-config.json"
    statsPath = statsCfg if statsCfg.is_file() else None
    extraCfgs = []
    for path in sorted(bestDir.glob("bestrobust*-config.json")):
        label = path.name
        label = label.removeprefix("best")
        label = label.removesuffix("-config.json")
        extraCfgs.append((label, path))
    return bestCfg, statsPath, extraCfgs
