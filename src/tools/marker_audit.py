#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from common import (
    activePrimerDays,
    loadConfig,
    loadKlinesForWindow,
)
from data.time_bounds import resolveAnchorMs


def _startIndex(parts: tuple, ctx: dict, window: str, offsetDays: int) -> int:
    from engine.shared import bars_per_day

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


def audit(
    cfgPath: str,
    anchorMs: int | None,
    window: str,
    startOffsetDays: int,
) -> dict:
    from tune.trace import Trace

    cfg = loadConfig(cfgPath)
    klines, parts = loadKlinesForWindow(cfg, window, anchorMs=anchorMs)
    ticker = str(parts[0])
    interval = str(parts[1])
    periods = list(parts[2])
    overrides = dict(parts[3])
    bt = Trace(
        ticker,
        klines,
        interval,
        periods,
        days=int(parts[8]),
        showCharts=False,
        showSummary=False,
        overrides=overrides,
        computeRisk=False,
        primerDays=0,
        anchorMs=anchorMs,
    )
    ctx = bt._ensureContext()
    ts = bt._timestamps()
    startIdx = _startIndex(parts, ctx, window, startOffsetDays)
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
    flagsByIndex = {
        (int(i), str(label)): True for i, label in flagsIdx
    }
    missing = []
    noteCounts = Counter()
    sideCounts = Counter()
    matched = 0
    for tr in wallet.trades:
        note = str(tr.note or "")
        side = str(tr.side)
        noteCounts[note or "signal_trade"] += 1
        sideCounts[side] += 1
        synthetic = note in {"seed_buy", "daily_posture_lock"}
        if synthetic:
            continue
        if flagsByIndex.get((int(tr.index), side)):
            matched += 1
        else:
            missing.append({
                "index": int(tr.index),
                "ts": tr.ts.isoformat(),
                "side": side,
                "note": note,
            })
    return {
        "startIdx": startIdx,
        "activeStart": ts[startIdx].isoformat(),
        "activeEnd": ts[-1].isoformat(),
        "flags": len(flagsIdx),
        "walletTrades": len(wallet.trades),
        "matchedSignalTrades": matched,
        "missingSignalTrades": len(missing),
        "noteCounts": dict(noteCounts),
        "sideCounts": dict(sideCounts),
        "firstMissing": missing[:20],
    }


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit chart wallet markers against generated flags.",
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--anchor-ms", type=int, default=None)
    parser.add_argument("--anchor-date", default=None)
    parser.add_argument(
        "--window",
        choices=("holdout", "tune"),
        default="holdout",
    )
    parser.add_argument("--start-offset-days", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parseArgs()
    anchorMs = resolveAnchorMs(args.anchor_ms, args.anchor_date)
    out = audit(
        args.profile,
        anchorMs,
        args.window,
        int(args.start_offset_days),
    )
    print(f"active: {out['activeStart']} -> {out['activeEnd']}")
    print(f"startIdx: {out['startIdx']}")
    print(f"flags: {out['flags']}")
    print(f"wallet trades: {out['walletTrades']}")
    print(f"matched signal trades: {out['matchedSignalTrades']}")
    print(f"missing signal trades: {out['missingSignalTrades']}")
    print(f"notes: {out['noteCounts']}")
    print(f"sides: {out['sideCounts']}")
    for i in out["firstMissing"]:
        print(
            "missing "
            f"{i['side']} idx={i['index']} ts={i['ts']} note={i['note']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
