#!/usr/bin/env python3
# control_error_audit.py - feedback-style diagnostics for existing flags.

from __future__ import annotations

########################################################################
# Imports
########################################################################

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import profile
from portfolio.accounting import FIXED_INCOME_BASE, marginalIncomeTaxRate
from engine.shared import bars_per_day, buildContext, buildSignals
from data.klines_io import loadWindowedKlines
from engine.macro_view import buildMacroView
from analysis.metrics import equityCurveFromTrades, maxDrawdown
from config.params import overridesFromDict
from runtime.diag import flagDiagnostics
from runtime.gates import paramsFromSettings
from data.time_bounds import resolveAnchorMs
from portfolio.wallet import Wallet, simulateFromFlags


########################################################################
# Constants
########################################################################

BASE_FIELDS = {
    "ticker",
    "tickers",
    "intervals",
    "p1",
    "p2",
    "p3",
    "primer_days",
    "training_days",
    "tuner_days",
    "holdout_days",
    *profile.HOLDOUT_START_KEYS,
    "CHARTS_TIMEVAL",
    "CHARTS_TRADES",
}

DAY_MS = 86_400_000


########################################################################
# Helpers
########################################################################

def _writeFrame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _configParts(
    cfg: dict,
    window: str,
) -> tuple[str, str, list[int], dict, int, int, int]:
    ticker = cfg["tickers"][0]
    interval = profile.intervalsFromConfig(cfg)[0]
    periods = [int(cfg["p1"]), int(cfg["p2"]), int(cfg["p3"])]
    primerDays, trainingDays, tunerDays, holdoutDays, totalDays = (
        profile.windowParts(cfg)
    )
    overrides = {k: v for k, v in cfg.items() if k not in BASE_FIELDS}
    overridesNorm = profile.overrides(overrides)
    profile.validate(overridesNorm, kind="backtest")
    win = str(window).strip().lower()
    if win == "tune":
        warmupDays = primerDays + trainingDays
        dataDays = primerDays + trainingDays + tunerDays
    elif win == "holdout":
        warmupDays = primerDays + trainingDays + tunerDays
        dataDays = totalDays
    else:
        warmupDays = primerDays + trainingDays
        dataDays = totalDays
    return (
        ticker,
        interval,
        periods,
        overridesNorm,
        warmupDays,
        dataDays,
        holdoutDays,
    )


def _startIdx(ctx: dict, periods: list[int], warmupDays: int) -> int:
    startIdx = max(periods) * 2
    if int(warmupDays) > 0:
        startIdx += int(round(int(warmupDays) * bars_per_day(ctx)))
    return startIdx


def _timestamps(klines: list) -> list[Any]:
    ts = pd.to_datetime([i[0] for i in klines], unit="ms", utc=True)
    return ts.tz_convert(None).to_pydatetime().tolist()


def _macroArrays(
    ticker: str,
    days: int,
    periods: list[int],
    overrides: dict,
    ts: list[Any],
    anchorMs: int | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    macro = buildMacroView(
        ticker,
        int(days),
        0,
        periods,
        overrides,
        ts,
        anchorMs=anchorMs,
    )
    if macro is None:
        return None, None, None
    return macro.dyn, macro.dir, macro.mom


def _candidateRows(
    diag: dict,
    ctx: dict,
    ts: list[Any],
    errMin: float,
    gradMin: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    closes = np.asarray(ctx["closes"], dtype=float)
    sideSpecs = [
        (
            "BUY",
            np.asarray(diag["buyIdxF"], dtype=int),
            np.asarray(diag["buyIdxSp"], dtype=int),
            np.asarray(diag["buyDeltaPct"], dtype=float),
            np.asarray(diag["buyReqPct"], dtype=float),
        ),
        (
            "SELL",
            np.asarray(diag["sellIdxF"], dtype=int),
            np.asarray(diag["sellIdxSp"], dtype=int),
            np.asarray(diag["sellDeltaPct"], dtype=float),
            np.asarray(diag["sellReqPct"], dtype=float),
        ),
    ]
    for side, idxs, accepted, deltaArr, reqArr in sideSpecs:
        acceptedSet = set(int(i) for i in accepted.tolist())
        prevErr = np.nan
        for idx in idxs.tolist():
            movePct = float(deltaArr[idx])
            reqPct = float(reqArr[idx])
            err = movePct - reqPct
            grad = err - prevErr if np.isfinite(prevErr) else 0.0
            passed = err >= float(errMin) and grad >= float(gradMin)
            rows.append(
                {
                    "index": int(idx),
                    "openMs": int(ctx["klines"][idx][0]),
                    "ts": ts[idx],
                    "side": side,
                    "close": float(closes[idx]),
                    "movePct": movePct,
                    "requiredPct": reqPct,
                    "stepErrorPct": err,
                    "stepErrorGradPct": grad,
                    "currentAccepted": int(idx) in acceptedSet,
                    "controlAccepted": bool(passed),
                }
            )
            prevErr = err
    out = pd.DataFrame(rows).sort_values(["index", "side"])
    return out.reset_index(drop=True)


def _flagsFromRows(rows: pd.DataFrame, col: str) -> list[tuple[int, str]]:
    use = rows[rows[col].astype(bool)].copy()
    use["sideRank"] = np.where(use["side"].astype(str) == "SELL", 0, 1)
    use = use.sort_values(["index", "sideRank"])
    return [
        (int(i["index"]), str(i["side"]))
        for _, i in use.iterrows()
    ]


def _walletSummary(
    name: str,
    ctx: dict,
    flags: list[tuple[int, str]],
    startIdx: int,
    overrides: dict,
    trendCode: np.ndarray,
) -> dict[str, object]:
    closes = np.asarray(ctx["closes"], dtype=float)
    ts = _timestamps(ctx["klines"])
    seed = float(overrides["WALLET_SEED_QUOTE"])
    feeRate = float(overrides["WALLET_FEE_RATE"])
    taxMode = str(overrides["TAX_MODE"]).lower()
    taxRate = marginalIncomeTaxRate(FIXED_INCOME_BASE)
    seedAssetPct = float(overrides.get("WALLET_SEED_ASSET_PCT", 1.0))
    finalPortionPct = float(overrides["FINAL_PORTION_PCT"])
    wallet = simulateFromFlags(
        ctx,
        flags,
        baseSymbol=str(ctx["ticker"]),
        startingCash=0.0,
        feeRate=feeRate,
        taxRate=taxRate,
        discountDays=365,
        discountRate=0.50,
        seedInvestQuote=seed,
        seedAssetPct=seedAssetPct,
        seedIndex=startIdx,
        doPrints=False,
        phaseBuyPortions=int(overrides["PHASE_BUY_PORTIONS"]),
        phaseSellPortions=int(overrides["PHASE_SELL_PORTIONS"]),
        taxMode=taxMode,
        annualIncomeBase=FIXED_INCOME_BASE,
        finalPortionPct=finalPortionPct,
        trendCodes=trendCode,
        overrides=overrides,
    )
    bench = Wallet(
        baseSymbol=str(ctx["ticker"]),
        startingCash=seed,
        feeRate=feeRate,
        taxRate=taxRate,
        taxMode=taxMode,
        annualIncomeBase=FIXED_INCOME_BASE,
    )
    bench.buyAll(startIdx, ts[startIdx], float(closes[startIdx]))
    lastPrice = float(closes[-1])
    sim = wallet.summary(currentPrice=lastPrice, currentTs=ts[-1])
    hodl = bench.summary(currentPrice=lastPrice, currentTs=ts[-1])
    curve = equityCurveFromTrades(
        closes=closes,
        trades=wallet.trades,
        startIndex=startIdx,
        seedQuote=seed,
    )
    return {
        "mode": name,
        "flags": int(len(flags)),
        "trades": int(len(wallet.trades)),
        "simValue": float(sim["portfolio_value"]),
        "benchValue": float(hodl["portfolio_value"]),
        "grossVsHodlPct": (
            ((float(sim["portfolio_value"]) / float(hodl["portfolio_value"]))
             - 1.0)
            * 100.0
        ),
        "mdd": float(maxDrawdown(curve)),
        "fees": float(sim["fees_paid_quote"]),
    }


########################################################################
# Main Audit
########################################################################

def runAudit(
    profilePath: Path,
    outDir: Path,
    stepErrorMin: float,
    stepGradMin: float,
    window: str,
    anchorMs: int | None,
) -> dict[str, Path]:
    cfg = profile.loadJson(str(profilePath))
    profile.ensureFinalPortionPct(cfg)
    ticker, interval, periods, overrides, warmupDays, totalDays, holdoutDays = (
        _configParts(cfg, window)
    )
    effectiveAnchorMs = anchorMs
    if str(window).strip().lower() == "tune" and anchorMs is not None:
        effectiveAnchorMs = int(anchorMs) - (int(holdoutDays) * DAY_MS)
    minCandles = (max(periods) * 2) + 1
    klines = loadWindowedKlines(
        ticker,
        interval,
        totalDays,
        minCandles,
        holdoutDays=0,
        anchorMs=effectiveAnchorMs,
    )
    ctx = buildContext(klines, periods)
    ctx["ticker"] = ticker
    ctx["days"] = int(totalDays)
    ctx["intervalStr"] = interval
    ctx["_cache"] = {
        "ticker": ticker,
        "interval": interval,
        "days": int(totalDays),
        "anchorMs": effectiveAnchorMs,
    }
    ts = _timestamps(klines)
    startIdx = _startIdx(ctx, periods, warmupDays)
    params = paramsFromSettings(overridesFromDict(overrides))
    signals = buildSignals(ctx, [])
    macroDyn, macroDir, macroMom = _macroArrays(
        ticker,
        totalDays,
        periods,
        overrides,
        ts,
        effectiveAnchorMs,
    )
    diag = flagDiagnostics(
        ctx,
        signals,
        params,
        startIdx,
        overrides,
        macroDyn=macroDyn,
        macroDir=macroDir,
        macroMom=macroMom,
    )
    candidates = _candidateRows(
        diag,
        ctx,
        ts,
        stepErrorMin,
        stepGradMin,
    )
    currentFlags = _flagsFromRows(candidates, "currentAccepted")
    controlFlags = _flagsFromRows(candidates, "controlAccepted")
    trendCode = np.asarray(signals["trendCode"], dtype=int)
    summary = pd.DataFrame(
        [
            _walletSummary(
                "current",
                ctx,
                currentFlags,
                startIdx,
                overrides,
                trendCode,
            ),
            _walletSummary(
                "step_error_control",
                ctx,
                controlFlags,
                startIdx,
                overrides,
                trendCode,
            ),
        ]
    )
    candidatesPath = outDir / "control_error_candidates.csv"
    summaryPath = outDir / "control_error_summary.csv"
    _writeFrame(candidatesPath, candidates)
    _writeFrame(summaryPath, summary)
    return {
        "candidates": candidatesPath,
        "summary": summaryPath,
    }


def parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="control_error_audit",
        description="Audit macro-move step error and positive error gradient.",
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--step-error-min", type=float, default=0.0)
    parser.add_argument("--step-grad-min", type=float, default=0.0)
    parser.add_argument(
        "--window",
        choices=["all", "tune", "holdout"],
        default="all",
    )
    parser.add_argument("--anchor-ms", type=int, default=None)
    parser.add_argument("--anchor-date", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parseArgs(argv)
    paths = runAudit(
        Path(args.profile),
        Path(args.out),
        float(args.step_error_min),
        float(args.step_grad_min),
        str(args.window),
        resolveAnchorMs(
            anchorMs=args.anchor_ms,
            anchorDate=args.anchor_date,
        ),
    )
    print(f"[control-error] candidates: {paths['candidates']}")
    print(f"[control-error] summary: {paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
