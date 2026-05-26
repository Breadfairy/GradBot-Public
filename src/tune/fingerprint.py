#!/usr/bin/env python3
# tune_fingerprint.py - shared tuner fingerprint helpers.

from __future__ import annotations

import hashlib
import os
from typing import Dict

from data.klines_io import REPO_ROOT, klinesMeta, loadWindowedKlines
from config import profile
from tune.axes import axesFromConfig


def _engineFiles() -> tuple[str, ...]:
    return (
        "src/native/engine/Makefile",
        "src/native/engine/engine.h",
        "src/native/engine/engine.c",
        "src/native/engine/batch.c",
        "src/engine/core.py",
        "src/engine/shared.py",
        "src/runtime/diag.py",
        "src/runtime/gates.py",
        "src/engine/dynamics.py",
        "src/portfolio/wallet.py",
        "src/tune/trace.py",
    )


def engineFingerprint() -> str:
    digest = hashlib.blake2b(digest_size=16)
    for relPath in _engineFiles():
        absPath = os.path.join(REPO_ROOT, relPath)
        with open(absPath, "rb") as fh:
            digest.update(fh.read())
    return digest.hexdigest()


def _fingerprintPayload(
    config: dict,
    intervals: list[str],
    axes: dict,
    klineMetas: Dict[str, dict],
    tuneKlineMetas: Dict[str, dict],
    primerDays: int,
    trainingDays: int,
    tunerDays: int,
    holdoutDays: int,
    totalDays: int,
    anchorMs: int | None,
    anchorKind: str,
    anchorDate: str | None,
) -> dict:
    axesCopy = {key: list(values) for key, values in axes.items()}
    return {
        "version": 1,
        "engineHash": engineFingerprint(),
        "tickers": list(config["tickers"]),
        "baseTicker": config["tickers"][0],
        "intervals": list(intervals),
        "primerDays": int(primerDays),
        "trainingDays": int(trainingDays),
        "tunerDays": int(tunerDays),
        "holdoutDays": int(holdoutDays),
        "totalDays": int(totalDays),
        "anchorMs": int(anchorMs) if anchorMs is not None else None,
        "anchorKind": str(anchorKind),
        "anchorDate": str(anchorDate) if anchorDate is not None else None,
        "axes": axesCopy,
        "klines": klineMetas,
        "klinesTune": tuneKlineMetas,
    }


def buildFingerprint(config: dict) -> dict:
    return buildFingerprintAt(config)


def buildFingerprintAt(
    config: dict,
    anchorMs: int | None = None,
    anchorKind: str = "runtime",
    anchorDate: str | None = None,
) -> dict:
    cfg = dict(config)
    profile.ensureFinalPortionPct(cfg)
    profile.validate(cfg, kind="tuner")
    intervals = profile.intervalsFromConfig(cfg)
    primerDays, trainingDays, tunerDays, holdoutDays, totalDays = (
        profile.profileWindows(cfg)
    )
    axes = axesFromConfig(cfg)
    maxPeriod = max(
        max(axes["p1Values"]),
        max(axes["p2Values"]),
        max(axes["p3Values"]),
    )
    minCandles = (maxPeriod * 2) + 1
    klineMetas: Dict[str, dict] = {}
    tuneKlineMetas: Dict[str, dict] = {}
    ticker = cfg["tickers"][0]
    for intervalValue in intervals:
        fullKlines = loadWindowedKlines(
            ticker,
            intervalValue,
            totalDays,
            minCandles,
            holdoutDays=0,
            anchorMs=anchorMs,
        )
        tuneKlines = loadWindowedKlines(
            ticker,
            intervalValue,
            totalDays,
            minCandles,
            holdoutDays=holdoutDays,
            anchorMs=anchorMs,
        )
        klineMetas[intervalValue] = klinesMeta(fullKlines)
        tuneKlineMetas[intervalValue] = klinesMeta(tuneKlines)
    return _fingerprintPayload(
        cfg,
        intervals,
        axes,
        klineMetas,
        tuneKlineMetas,
        primerDays,
        trainingDays,
        tunerDays,
        holdoutDays,
        totalDays,
        anchorMs,
        anchorKind,
        anchorDate,
    )


__all__ = [
    "buildFingerprintAt",
    "buildFingerprint",
    "engineFingerprint",
]
