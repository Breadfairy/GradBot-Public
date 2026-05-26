#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from common import (
    activePrimerDays,
    loadConfig,
    loadKlinesForWindow,
    metricRow,
)
from runtime.posture_feed import DAILY_STRONG_CLUSTER, dailyPostureArrays
from engine.shared import bars_per_day
from data.time_bounds import resolveAnchorMs


########################################################################
# Trace Helpers
########################################################################

def _startIndex(parts: tuple, ctx: dict, window: str, offsetDays: int) -> int:
    periods = list(parts[2])
    primerDays = int(parts[4])
    trainingDays = int(parts[5])
    tunerDays = int(parts[6])
    primerActive = activePrimerDays(
        primerDays,
        trainingDays,
        tunerDays,
        window,
        startOffsetDays=offsetDays,
    )
    return max(periods) * 2 + int(round(primerActive * bars_per_day(ctx)))


def _runWallet(
    cfgPath: str,
    anchorMs: int | None,
    window: str,
    offsetDays: int,
) -> tuple[Any, dict, list, list, np.ndarray, dict]:
    from tune.trace import Trace

    cfg = loadConfig(cfgPath)
    klines, parts = loadKlinesForWindow(cfg, window, anchorMs=anchorMs)
    ticker = str(parts[0])
    interval = str(parts[1])
    periods = list(parts[2])
    overrides = dict(parts[3])
    holdoutDays = int(parts[7])
    macroHoldoutDays = (
        0 if str(window).strip().lower() == "holdout" else holdoutDays
    )
    primerActive = activePrimerDays(
        int(parts[4]),
        int(parts[5]),
        int(parts[6]),
        window,
        startOffsetDays=offsetDays,
    )
    bt = Trace(
        ticker,
        klines,
        interval,
        periods,
        days=int(parts[8]),
        showCharts=False,
        showSummary=False,
        overrides=overrides,
        computeRisk=True,
        primerDays=primerActive,
        holdoutDays=macroHoldoutDays,
        anchorMs=anchorMs,
    )
    ctx = bt._ensureContext()
    ts = bt._timestamps()
    startIdx = _startIndex(parts, ctx, window, offsetDays)
    params, signals = bt._flagParamsAndSignals(ctx)
    flagsIdx, _flagsTs = bt._generateFlags(
        ctx,
        ts,
        startIdx,
        params,
        signals,
    )
    trendArr = np.asarray(signals["trendCode"], dtype=int)
    seed, seedAssetPct, taxMode, incomeBase, _audRate, finalPortionPct = (
        bt._walletOverrides()
    )
    wallet, _bench = bt._simulateWallets(
        ctx,
        flagsIdx,
        startIdx,
        taxMode,
        incomeBase,
        seed,
        seedAssetPct,
        finalPortionPct=finalPortionPct,
        trendCodeArr=trendArr,
    )
    result = bt.run()
    return result, ctx, ts, flagsIdx, trendArr, {
        "wallet": wallet,
        "overrides": overrides,
        "ticker": ticker,
        "startIdx": startIdx,
    }


########################################################################
# Sell Outcome Metrics
########################################################################

def _futureRet(closes: np.ndarray, index: int, bars: int) -> float:
    j = min(int(index) + int(bars), closes.size - 1)
    price = float(closes[int(index)])
    future = float(closes[j])
    return ((future / price) - 1.0) * 100.0 if price > 0.0 else 0.0


def _futureDrop(closes: np.ndarray, index: int, bars: int) -> float:
    j = min(int(index) + int(bars), closes.size - 1)
    price = float(closes[int(index)])
    futureMin = float(np.min(closes[int(index):j + 1]))
    return ((price / futureMin) - 1.0) * 100.0 if futureMin > 0.0 else 0.0


def _mean(vals: list[float]) -> float:
    finite = [float(i) for i in vals if np.isfinite(float(i))]
    return float(mean(finite)) if finite else 0.0


def _sellRows(
    ctx: dict,
    flagsIdx: list,
    wallet: Any,
    overrides: dict,
    horizonHours: int,
) -> list[dict[str, Any]]:
    closes = np.asarray(ctx["closes"], dtype=float)
    daily = dailyPostureArrays(ctx, overrides)
    clusters = (
        daily["cluster"] if daily is not None
        else np.full(closes.size, -1, dtype=int)
    )
    bpd = max(bars_per_day(ctx), 1.0)
    bars = max(int(round((float(horizonHours) / 24.0) * bpd)), 1)
    tradesByIndex = {
        int(i.index): i for i in wallet.trades
        if str(i.side) == "SELL" and not str(i.note or "")
    }
    rows = []
    for i, label in flagsIdx:
        idx = int(i)
        strong = int(clusters[idx]) == DAILY_STRONG_CLUSTER
        trade = tradesByIndex.get(idx)
        if str(label) != "SELL":
            continue
        fwdRet = _futureRet(closes, idx, bars)
        edge = -fwdRet
        drop = _futureDrop(closes, idx, bars)
        executed = trade is not None
        rows.append({
            "index": idx,
            "strong": strong,
            "executed": executed,
            "edge": edge,
            "drawdownAvoided": drop,
            "cashDelta": float(trade.cashDelta) if executed else 0.0,
            "realizedGain": float(trade.realizedGain) if executed else 0.0,
        })
    return rows


def auditConfig(
    label: str,
    cfgPath: str,
    anchorMs: int | None,
    window: str,
    offsetDays: int,
    horizonHours: int,
) -> dict[str, Any]:
    result = None
    ctx = {}
    flagsIdx = []
    extra = {}
    sellRows = []
    executedRows = []
    blockedStrongRows = []
    usefulBlockedRows = []
    earlySuppressedRows = []
    row = {}
    result, ctx, _ts, flagsIdx, _trendArr, extra = _runWallet(
        cfgPath,
        anchorMs,
        window,
        offsetDays,
    )
    wallet = extra["wallet"]
    overrides = extra["overrides"]
    sellRows = _sellRows(ctx, flagsIdx, wallet, overrides, horizonHours)
    executedRows = [i for i in sellRows if bool(i["executed"])]
    blockedStrongRows = [
        i for i in sellRows
        if bool(i["strong"]) and not bool(i["executed"])
    ]
    usefulBlockedRows = [
        i for i in blockedStrongRows if float(i["edge"]) > 0.0
    ]
    earlySuppressedRows = [
        i for i in blockedStrongRows if float(i["edge"]) <= 0.0
    ]
    metric = metricRow(label, str(extra["ticker"]), result)
    sellFlags = len(sellRows)
    sellExecPct = (
        (len(executedRows) / sellFlags) * 100.0 if sellFlags > 0 else 0.0
    )
    realizedQuote = sum(float(i["cashDelta"]) for i in executedRows)
    realizedGain = sum(float(i["realizedGain"]) for i in executedRows)
    sellEdge = _mean([float(i["edge"]) for i in executedRows])
    blockedEdge = _mean([float(i["edge"]) for i in blockedStrongRows])
    realizedScore = (
        float(metric["grossVsHodl"])
        + (0.25 * max(sellEdge, 0.0))
        + (0.35 * len(earlySuppressedRows))
        - (1.25 * len(usefulBlockedRows))
    )
    row = {
        "label": label,
        "window": window,
        "offsetDays": offsetDays,
        "grossVsHodl": float(metric["grossVsHodl"]),
        "edgePct": float(metric["edgePct"]),
        "trades": int(metric["trades"]),
        "buys": int(metric["buys"]),
        "sells": int(metric["sells"]),
        "mddPct": float(metric["mddPct"]),
        "sellFlags": sellFlags,
        "sellExecuted": len(executedRows),
        "sellExecPct": sellExecPct,
        "strongSellBlocked": len(blockedStrongRows),
        "usefulBlockedSells": len(usefulBlockedRows),
        "earlySuppressedSells": len(earlySuppressedRows),
        "sellEdgeMean": sellEdge,
        "blockedSellEdgeMean": blockedEdge,
        "drawdownAvoidedMean": _mean([
            float(i["drawdownAvoided"]) for i in executedRows
        ]),
        "realizedQuote": realizedQuote,
        "realizedGain": realizedGain,
        "realizationScore": realizedScore,
        "ultraSellMult": float(overrides.get("ULTRA_SELL_MULT", 0.0)),
        "dailyBuyMult": float(overrides.get("DAILY_DOWN_BUY_MULT", 0.0)),
        "ultraGainMinPct": float(overrides.get("ULTRA_GAIN_MIN_PCT", 0.0)),
        "ultraGainMaxPct": float(overrides.get("ULTRA_GAIN_MAX_PCT", 0.0)),
        "ultraExitHoldDays": int(overrides.get("ULTRA_EXIT_HOLD_DAYS", 0)),
        "profile": cfgPath,
    }
    return row


########################################################################
# CLI
########################################################################

def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit sell realization and strong-posture suppression.",
    )
    parser.add_argument("--profile", action="append", required=True)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--anchor-ms", type=int, default=None)
    parser.add_argument("--anchor-date", default=None)
    parser.add_argument(
        "--window",
        choices=("holdout", "tune"),
        default="holdout",
    )
    parser.add_argument("--start-offset-days", type=int, default=0)
    parser.add_argument("--horizon-hours", type=int, default=168)
    return parser.parse_args()


def main() -> int:
    args = parseArgs()
    anchorMs = resolveAnchorMs(args.anchor_ms, args.anchor_date)
    labels = list(args.label)
    profiles = list(args.profile)
    rows = []
    fields = []
    for i, path in enumerate(profiles):
        label = labels[i] if i < len(labels) else Path(path).stem
        rows.append(auditConfig(
            label,
            path,
            anchorMs,
            str(args.window),
            int(args.start_offset_days),
            int(args.horizon_hours),
        ))
    rows.sort(key=lambda i: float(i["realizationScore"]), reverse=True)
    fields = list(rows[0].keys())
    outPath = Path(args.out)
    outPath.parent.mkdir(parents=True, exist_ok=True)
    tmpPath = Path(f"{outPath}.tmp")
    with tmpPath.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmpPath.replace(outPath)
    print(
        "label,grossVsHodl,realizationScore,sellExecPct,"
        "usefulBlockedSells,earlySuppressedSells,sellEdgeMean"
    )
    for i in rows:
        print(
            f"{i['label']},{i['grossVsHodl']:.2f},"
            f"{i['realizationScore']:.2f},{i['sellExecPct']:.1f},"
            f"{i['usefulBlockedSells']},"
            f"{i['earlySuppressedSells']},{i['sellEdgeMean']:.2f}"
        )
    print(f"wrote {outPath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
