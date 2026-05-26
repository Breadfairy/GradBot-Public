#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from common import (
    ROOT_DIR,
    loadConfig,
    loadKlinesForWindow,
    metricRow,
    runTraceForWindow,
)
from analysis.charting import plotTimVal
from data.time_bounds import resolveAnchorMs


def _lockBase(depth: float, gain: float) -> dict:
    return {
        "ULTRA_EXIT_DEPTH": depth,
        "ULTRA_GAIN_MIN_PCT": gain,
    }


def _variants() -> list[tuple[str, dict]]:
    out = [
        ("base", {}),
    ]
    for depth in [0.45, 0.75, 1.0]:
        for gain in [5.0, 10.0, 15.0]:
            for gainMax in [25.0, 35.0, 45.0]:
                changes = _lockBase(depth, gain)
                changes.update({
                    "ULTRA_GAIN_MAX_PCT": gainMax,
                    "ULTRA_EXIT_HOLD_DAYS": 60,
                })
                label = (
                    f"exit-d{int(depth * 100)}-g{int(gain)}-"
                    f"m{int(gainMax)}-h60"
                )
                out.append((label, changes))
    return out


def runSweep(
    profilePath: str,
    outDir: Path,
    anchorMs: int | None,
    window: str,
    charts: int,
) -> list[dict]:
    base = loadConfig(profilePath)
    outDir.mkdir(parents=True, exist_ok=True)
    rows = []
    saved = {}
    for label, changes in _variants():
        cfg = dict(base)
        cfg.update(changes)
        klines, parts = loadKlinesForWindow(cfg, window, anchorMs=anchorMs)
        result, _trace = runTraceForWindow(
            cfg,
            klines,
            parts,
            window,
            anchorMs=anchorMs,
        )
        row = metricRow(label, str(parts[0]), result)
        row["config"] = ""
        row["chart"] = ""
        rows.append(row)
        saved[label] = (cfg, result)
    rows.sort(key=lambda r: float(r["grossVsHodl"]), reverse=True)
    for row in rows[:charts]:
        label = str(row["label"])
        cfg, result = saved[label]
        cfgPath = outDir / f"{label}.json"
        chartPath = outDir / f"{label}-timVal.png"
        tmpPath = Path(f"{cfgPath}.tmp")
        tmpPath.write_text(json.dumps(cfg, indent=2))
        os.replace(tmpPath, cfgPath)
        plotTimVal(
            result.curveTs,
            result.curveSim,
            result.curveBench,
            result.curveAssetFrac,
            result.curveQuoteFrac,
            label,
            str(chartPath),
        )
        row["config"] = str(cfgPath.relative_to(ROOT_DIR))
        row["chart"] = str(chartPath.relative_to(ROOT_DIR))
    return rows


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compact daily profit-lock sweep for one profile.",
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--anchor-ms", type=int, default=None)
    parser.add_argument("--anchor-date", default=None)
    parser.add_argument(
        "--window",
        choices=("holdout", "tune"),
        default="holdout",
    )
    parser.add_argument("--charts", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parseArgs()
    anchorMs = resolveAnchorMs(args.anchor_ms, args.anchor_date)
    rows = runSweep(
        args.profile,
        Path(args.out),
        anchorMs,
        args.window,
        int(args.charts),
    )
    fields = [
        "label",
        "grossVsHodl",
        "edgePct",
        "hodlPct",
        "trades",
        "buys",
        "sells",
        "mddPct",
        "cagrPct",
        "config",
        "chart",
    ]
    outPath = Path(args.out) / "summary.csv"
    with outPath.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r[k] for k in fields} for r in rows])
    print("label,grossVsHodl,edgePct,trades,mddPct,cagrPct")
    for r in rows[:20]:
        print(
            f"{r['label']},{r['grossVsHodl']:.2f},"
            f"{r['edgePct']:.2f},{r['trades']},"
            f"{r['mddPct']:.2f},{r['cagrPct']:.2f}"
        )
    print(f"wrote {outPath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
