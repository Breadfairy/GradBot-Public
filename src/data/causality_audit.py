#!/usr/bin/env python3
# causality_audit.py - split and alignment checks for tune pipeline runs.

from __future__ import annotations

import argparse
import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.klines_io import (
    DAY_MS,
    INTERVAL_MS,
    loadWindowedKlines,
    normalizeInterval,
)
from config import profile
from data.time_bounds import resolveAnchorMs
from tune.axes import axesFromConfig


########################################################################
# Paths + Serialization
########################################################################

ROOT_DIR = Path(__file__).resolve().parents[2]


def _utc(ms: int | None) -> str | None:
    if ms is None:
        return None
    dt = datetime.fromtimestamp(int(ms) / 1000.0, tz=timezone.utc)
    return dt.isoformat()


def _writeJson(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def _resolvePath(rawPath: str) -> Path:
    path = Path(str(rawPath))
    return path if path.is_absolute() else ROOT_DIR / path


########################################################################
# Config Helpers
########################################################################

def _posturePaths(cfg: dict[str, Any]) -> list[str]:
    raw = cfg.get("DAILY_CLUSTER_PATH", "")
    if isinstance(raw, list):
        return [str(i).strip() for i in raw if str(i).strip()]
    path = str(raw).strip()
    return [path] if path else []


def _periodMaxMin(cfg: dict[str, Any]) -> int:
    axes = axesFromConfig(cfg)
    vals = (
        max(int(a), int(b), int(c))
        for a, b, c in itertools.product(
            axes["p1Values"],
            axes["p2Values"],
            axes["p3Values"],
        )
    )
    return int(min(vals))


def _barsPerDay(interval: str) -> float:
    key = normalizeInterval(interval)
    return float(DAY_MS) / float(INTERVAL_MS[key])


def _startIdx(
    klines: list,
    interval: str,
    periodMax: int,
    warmupDays: int,
) -> int:
    bpd = _barsPerDay(interval)
    start = (int(periodMax) * 2) + int(round(int(warmupDays) * bpd))
    return min(max(start, 0), max(len(klines) - 1, 0))


def _opens(klines: list) -> np.ndarray:
    return np.asarray([int(float(i[0])) for i in klines], dtype=np.int64)


def _closes(klines: list) -> np.ndarray:
    return np.asarray([int(float(i[6])) for i in klines], dtype=np.int64)


def _status(rows: list[dict[str, Any]]) -> str:
    statuses = {str(i.get("status", "pass")) for i in rows}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


########################################################################
# Window Separation
########################################################################

def _windowRows(
    cfg: dict[str, Any],
    anchorMs: int,
    periodMax: int,
) -> list[dict[str, Any]]:
    ticker = str(cfg["tickers"][0]).upper()
    intervals = profile.intervalsFromConfig(cfg)
    primerDays, trainingDays, tunerDays, holdoutDays, totalDays = (
        profile.profileWindows(cfg)
    )
    tuneBoundary = int(anchorMs) - (int(holdoutDays) * DAY_MS)
    rows: list[dict[str, Any]] = []
    for i in intervals:
        tuneKlines = loadWindowedKlines(
            ticker,
            i,
            totalDays,
            1,
            holdoutDays=holdoutDays,
            anchorMs=anchorMs,
        )
        fullKlines = loadWindowedKlines(
            ticker,
            i,
            totalDays,
            1,
            holdoutDays=0,
            anchorMs=anchorMs,
        )
        tuneOpen = _opens(tuneKlines)
        fullOpen = _opens(fullKlines)
        tuneStart = _startIdx(
            tuneKlines,
            i,
            periodMax,
            primerDays + trainingDays,
        )
        holdStart = _startIdx(
            fullKlines,
            i,
            periodMax,
            primerDays + trainingDays + tunerDays,
        )
        tuneActive = tuneOpen[tuneStart:]
        holdActive = fullOpen[holdStart:]
        sharedActive = np.intersect1d(
            tuneActive,
            holdActive,
            assume_unique=False,
        )
        tuneLast = int(tuneActive[-1]) if tuneActive.size else None
        holdFirst = int(holdActive[0]) if holdActive.size else None
        ordered = (
            tuneLast is not None
            and holdFirst is not None
            and tuneLast < holdFirst
        )
        rows.append({
            "stage": "window",
            "interval": normalizeInterval(i),
            "status": "pass" if ordered and sharedActive.size == 0 else "fail",
            "periodMaxMin": int(periodMax),
            "tuneRows": int(tuneOpen.size),
            "holdoutRows": int(fullOpen.size),
            "tuneStartIdx": int(tuneStart),
            "holdoutStartIdx": int(holdStart),
            "tuneBoundaryMs": int(tuneBoundary),
            "tuneBoundaryUtc": _utc(tuneBoundary),
            "tuneActiveFirstMs": (
                int(tuneActive[0]) if tuneActive.size else None
            ),
            "tuneActiveFirstUtc": _utc(
                int(tuneActive[0]) if tuneActive.size else None
            ),
            "tuneActiveLastMs": tuneLast,
            "tuneActiveLastUtc": _utc(tuneLast),
            "holdoutActiveFirstMs": holdFirst,
            "holdoutActiveFirstUtc": _utc(holdFirst),
            "holdoutActiveLastMs": (
                int(holdActive[-1]) if holdActive.size else None
            ),
            "holdoutActiveLastUtc": _utc(
                int(holdActive[-1]) if holdActive.size else None
            ),
            "sharedActiveRows": int(sharedActive.size),
            "holdoutWarmupRowsBeforeBoundary": int(
                np.count_nonzero(fullOpen < tuneBoundary)
            ),
            "note": (
                "Holdout loads earlier candles as warmup, but active "
                "holdout scoring starts after the tune boundary."
            ),
        })
    return rows


########################################################################
# Alignment Checks
########################################################################

def _alignCounts(
    sourceTimes: np.ndarray,
    targetTimes: np.ndarray,
    activeStart: int,
) -> dict[str, int]:
    pos = np.searchsorted(sourceTimes, targetTimes, side="right") - 1
    valid = pos >= 0
    used = np.zeros(targetTimes.size, dtype=np.int64)
    used[valid] = sourceTimes[pos[valid]]
    future = valid & (used > targetTimes)
    active = np.zeros(targetTimes.size, dtype=bool)
    active[int(activeStart):] = True
    return {
        "futureRows": int(np.count_nonzero(future)),
        "activeFutureRows": int(np.count_nonzero(future & active)),
        "missingRows": int(np.count_nonzero(~valid)),
        "activeMissingRows": int(np.count_nonzero((~valid) & active)),
    }


def _postureRows(
    cfg: dict[str, Any],
    anchorMs: int,
    periodMax: int,
) -> list[dict[str, Any]]:
    ticker = str(cfg["tickers"][0]).upper()
    intervals = profile.intervalsFromConfig(cfg)
    primerDays, trainingDays, tunerDays, _holdoutDays, totalDays = (
        profile.profileWindows(cfg)
    )
    rows: list[dict[str, Any]] = []
    for i in _posturePaths(cfg):
        path = _resolvePath(i)
        frame = pd.read_csv(path, usecols=["closeMs", "close", "cluster"])
        frame = frame[frame["cluster"] >= 0].copy()
        source = frame["closeMs"].to_numpy(dtype=np.int64)
        sortedOk = bool(np.all(source[:-1] <= source[1:]))
        for j in intervals:
            fullKlines = loadWindowedKlines(
                ticker,
                j,
                totalDays,
                1,
                holdoutDays=0,
                anchorMs=anchorMs,
            )
            target = _opens(fullKlines)
            activeStart = _startIdx(
                fullKlines,
                j,
                periodMax,
                primerDays + trainingDays + tunerDays,
            )
            counts = _alignCounts(source, target, activeStart)
            failed = (
                source.size <= 0
                or not sortedOk
                or counts["futureRows"] > 0
                or counts["activeFutureRows"] > 0
            )
            status = "pass"
            if counts["activeMissingRows"] > 0:
                status = "warn"
            if failed:
                status = "fail"
            rows.append({
                "stage": "dailyPosture",
                "interval": normalizeInterval(j),
                "path": str(path),
                "status": status,
                "sourceRows": int(source.size),
                "targetRows": int(target.size),
                "activeStartIdx": int(activeStart),
                "sourceSorted": sortedOk,
                **counts,
                "note": (
                    "Daily posture aligns by last closed daily closeMs at "
                    "or before the micro candle open."
                ),
            })
    return rows


def _macroRows(
    cfg: dict[str, Any],
    anchorMs: int,
    periodMax: int,
) -> list[dict[str, Any]]:
    ticker = str(cfg["tickers"][0]).upper()
    axes = axesFromConfig(cfg)
    intervals = profile.intervalsFromConfig(cfg)
    macroIntervals = [
        str(i).strip()
        for i in dict.fromkeys(axes["macroIntervalValues"])
        if str(i).strip()
    ]
    primerDays, trainingDays, tunerDays, holdoutDays, totalDays = (
        profile.profileWindows(cfg)
    )
    rows: list[dict[str, Any]] = []
    for i in intervals:
        tuneMicro = loadWindowedKlines(
            ticker,
            i,
            totalDays,
            1,
            holdoutDays=holdoutDays,
            anchorMs=anchorMs,
        )
        fullMicro = loadWindowedKlines(
            ticker,
            i,
            totalDays,
            1,
            holdoutDays=0,
            anchorMs=anchorMs,
        )
        tuneOpen = _opens(tuneMicro)
        fullOpen = _opens(fullMicro)
        tuneStart = _startIdx(
            tuneMicro,
            i,
            periodMax,
            primerDays + trainingDays,
        )
        holdStart = _startIdx(
            fullMicro,
            i,
            periodMax,
            primerDays + trainingDays + tunerDays,
        )
        for j in macroIntervals:
            tuneMacro = loadWindowedKlines(
                ticker,
                j,
                totalDays,
                1,
                holdoutDays=holdoutDays,
                anchorMs=anchorMs,
            )
            fullMacro = loadWindowedKlines(
                ticker,
                j,
                totalDays,
                1,
                holdoutDays=0,
                anchorMs=anchorMs,
            )
            tuneCounts = _alignCounts(_closes(tuneMacro), tuneOpen, tuneStart)
            holdCounts = _alignCounts(_closes(fullMacro), fullOpen, holdStart)
            failed = (
                tuneCounts["activeFutureRows"] > 0
                or tuneCounts["activeMissingRows"] > 0
                or holdCounts["activeFutureRows"] > 0
                or holdCounts["activeMissingRows"] > 0
            )
            rows.append({
                "stage": "macro",
                "interval": normalizeInterval(i),
                "macroInterval": normalizeInterval(str(j)),
                "status": "fail" if failed else "pass",
                "tuneActiveStartIdx": int(tuneStart),
                "holdoutActiveStartIdx": int(holdStart),
                "tuneFutureRows": tuneCounts["futureRows"],
                "tuneActiveFutureRows": tuneCounts["activeFutureRows"],
                "tuneMissingRows": tuneCounts["missingRows"],
                "tuneActiveMissingRows": tuneCounts["activeMissingRows"],
                "holdoutFutureRows": holdCounts["futureRows"],
                "holdoutActiveFutureRows": holdCounts["activeFutureRows"],
                "holdoutMissingRows": holdCounts["missingRows"],
                "holdoutActiveMissingRows": holdCounts["activeMissingRows"],
                "note": (
                    "Macro state aligns from closed macro candles to micro "
                    "opens by last-known sample."
                ),
            })
    return rows


########################################################################
# Public API
########################################################################

def auditConfig(
    cfg: dict[str, Any],
    anchorMs: int,
) -> dict[str, Any]:
    primerDays, trainingDays, tunerDays, holdoutDays, totalDays = (
        profile.profileWindows(cfg)
    )
    periodMax = _periodMaxMin(cfg)
    rows: list[dict[str, Any]] = []
    rows.extend(_windowRows(cfg, anchorMs, periodMax))
    rows.extend(_postureRows(cfg, anchorMs, periodMax))
    rows.extend(_macroRows(cfg, anchorMs, periodMax))
    status = _status(rows)
    return {
        "schema": "gradbot-causality-audit-v1",
        "status": status,
        "anchorMs": int(anchorMs),
        "anchorUtc": _utc(anchorMs),
        "primerDays": int(primerDays),
        "trainingDays": int(trainingDays),
        "tunerDays": int(tunerDays),
        "holdoutDays": int(holdoutDays),
        "totalDays": int(totalDays),
        "periodMaxMin": int(periodMax),
        "checks": rows,
        "notes": [
            "Selection metrics use tune rows only; holdout columns are "
            "diagnostic after a tune-selected config exists.",
            "Holdout traces still load pre-holdout candles as warmup; "
            "the audit checks that active tune and active holdout scoring "
            "candles do not overlap.",
        ],
    }


def writeCausalityAudit(
    cfg: dict[str, Any],
    outPath: Path,
    anchorMs: int,
) -> dict[str, Any]:
    payload = auditConfig(cfg, anchorMs)
    _writeJson(outPath, payload)
    return payload


########################################################################
# CLI
########################################################################

def parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="causality_audit",
        description="Audit tune/holdout split and causal alignments.",
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--anchor-ms", type=int, default=None)
    parser.add_argument("--anchor-date", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parseArgs(argv)
    cfg = profile.loadJson(args.profile)
    profile.ensureFinalPortionPct(cfg)
    profile.validate(cfg, kind="tuner")
    anchorMs = resolveAnchorMs(
        anchorMs=args.anchor_ms,
        anchorDate=args.anchor_date,
    )
    if anchorMs is None:
        raise SystemExit(
            "causality audit requires --anchor-ms or --anchor-date"
        )
    payload = writeCausalityAudit(cfg, Path(args.out), int(anchorMs))
    print(f"[audit] causality status={payload['status']}")
    return 0 if payload["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
