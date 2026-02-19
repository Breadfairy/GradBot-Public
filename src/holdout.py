#!/usr/bin/env python3
# holdout.py – Backtest holdout summaries for tuned configs.

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

from backtest import Backtest
from charting import plotTimVal
from cache import getKlinesCached
from metrics import grossPctVsBench
import profile
from config_compare import configsEqual


BAR = "=" * 54


def _config_parts(
    cfg: dict,
) -> Tuple[str, str, list[int], dict, int, int, int, int]:
    ticker = cfg["tickers"][0]
    interval = profile.intervalsFromConfig(cfg)[0]
    periods = [int(cfg["p1"]), int(cfg["p2"]), int(cfg["p3"])]
    primerDays, tunerDays, holdoutDays, totalDays = (
        profile.windowParts(cfg)
    )
    baseFields = {
        "ticker",
        "tickers",
        "intervals",
        "p1",
        "p2",
        "p3",
        "primer_days",
        "tuner_days",
        "holdout_days",
    }
    overrides = {k: v for k, v in cfg.items() if k not in baseFields}
    overridesNorm = profile.overrides(overrides)
    profile.validate(overridesNorm, kind="backtest")
    return (
        ticker,
        interval,
        periods,
        overridesNorm,
        primerDays,
        tunerDays,
        holdoutDays,
        totalDays,
    )


def _run_one(
    label: str,
    cfgPath: Path,
    holdoutOverride: int,
    chartsRoot: Path | None = None,
) -> dict[str, float | int | str]:
    cfg = profile.loadJson(cfgPath)
    profile.ensureFinalPortionPct(cfg)
    (
        ticker,
        interval,
        periods,
        overrides,
        primerDays,
        tunerDays,
        holdoutDays,
        totalDays,
    ) = _config_parts(cfg)
    dataDays = totalDays
    warmupDays = primerDays + tunerDays
    if holdoutOverride > 0:
        holdoutDays = holdoutOverride
        dataDays = warmupDays + holdoutDays
    if dataDays <= 0 or holdoutDays <= 0:
        raise SystemExit("holdout requires holdout_days in config or --days")
    minCandles = (max(periods) * 2) + 1
    klines = getKlinesCached(
        ticker,
        interval,
        dataDays,
        minCandles,
        holdoutDays=0,
    )
    chartsDir = None
    if chartsRoot is not None:
        chartsDir = Path(chartsRoot) / label
        os.makedirs(chartsDir, exist_ok=True)
    if chartsDir is not None:
        os.environ["CHARTS_OUT_DIR"] = str(chartsDir)

    bt = Backtest(
        ticker,
        klines,
        interval,
        periods,
        days=dataDays,
        doOracles=False,
        showCharts=True,
        showPrints=False,
        showSummary=False,
        overrides=overrides,
        ctx=None,
        signals=None,
        computeRisk=True,
        primerDays=warmupDays,
        holdoutDays=0,
    )
    res = bt.run()
    if (
        chartsRoot is not None
        and res.curveTs is not None
        and res.curveSim is not None
        and res.curveBench is not None
    ):
        outPath = Path(chartsRoot) / f"{label}-timVal.png"
        title = f"{ticker} {interval} – {label} timVal"
        plotTimVal(
            res.curveTs,
            res.curveSim,
            res.curveBench,
            title,
            str(outPath),
        )
    simVal = float(res.sim.get("portfolio_value", 0.0))
    benchVal = float(res.bench.get("portfolio_value", 0.0))
    seed = float(res.seedQuote or 0.0)
    pct = grossPctVsBench(simVal, benchVal)
    if seed > 0.0:
        edgePct = ((simVal / seed) - 1.0) * 100.0
        hodlPct = ((benchVal / seed) - 1.0) * 100.0
    else:
        edgePct = float("nan")
        hodlPct = float("nan")
    return {
        "pct": pct,
        "edge": edgePct,
        "hodl": hodlPct,
        "trades": res.sim.get("trades", 0),
        "buys": res.buyTrades,
        "sells": res.sellTrades,
        "sh1": res.sharpe1wAbs,
        "sh1_edge": res.sharpe1w,
        "sh4": res.sharpe4wAbs,
        "sh13": res.sharpe13wAbs,
        "sor1": res.sortino1wAbs,
        "sor1_edge": res.sortino1w,
        "sor4": res.sortino4wAbs,
        "sor13": res.sortino13wAbs,
        "cagr": res.cagr,
        "sh4_edge": res.sharpe4w,
        "sh13_edge": res.sharpe13w,
        "sor4_edge": res.sortino4w,
        "sor13_edge": res.sortino13w,
        "mdd": res.mdd,
        "mar": (res.cagr / res.mdd) if res.mdd > 1e-12 else float("nan"),
        "ticker": ticker,
        "label": label,
    }


def _print_block(data: dict[str, float | int | str]) -> None:
    prefix = "holdout"
    label = str(data.get("label", ""))
    print(BAR)
    print(f"[{prefix}] GrossVhodl: {label}")
    print(
        f"[{prefix}] - {data['ticker']}: "
        f"{data['pct']:+.2f}%  TRADES: {int(data['trades'])}"
    )
    print(
        f"[{prefix}] - EDGE%: {data['edge']:+.2f}%, "
        f"HODL%: {data['hodl']:+.2f}%"
    )
    print(
        f"[{prefix}] - BUYS: {int(data['buys'])}, "
        f"SELLS: {int(data['sells'])}"
    )
    print(f"[{prefix}] RiskStats")
    print(
        f"[{prefix}] - SHARPE 1w  (abs/rel): "
        f"{data['sh1']:.2f} / {data['sh1_edge']:.2f}"
    )
    print(
        f"[{prefix}] - SHARPE 4w  (abs/rel): "
        f"{data['sh4']:.2f} / {data['sh4_edge']:.2f}"
    )
    print(
        f"[{prefix}] - SHARPE 13w (abs/rel): "
        f"{data['sh13']:.2f} / {data['sh13_edge']:.2f}"
    )
    print(
        f"[{prefix}] - SORTINO 1w  (abs/rel): "
        f"{data['sor1']:.2f} / {data['sor1_edge']:.2f}"
    )
    print(
        f"[{prefix}] - SORTINO 4w  (abs/rel): "
        f"{data['sor4']:.2f} / {data['sor4_edge']:.2f}"
    )
    print(
        f"[{prefix}] - SORTINO 13w (abs/rel): "
        f"{data['sor13']:.2f} / {data['sor13_edge']:.2f}"
    )
    print(f"[{prefix}] - MAR (CAGR/MDD): {data['mar']:.2f}")
    print(f"[{prefix}] - MDD: {data['mdd']*100.0:.2f}%")
    print(f"[{prefix}] - CAGR: {data['cagr']*100.0:.2f}%")
    print(BAR)


def runHoldout(
    bestCfg: Path,
    statsCfg: Path | None,
    holdoutDays: int,
    chartsRoot: Path | None,
) -> None:
    statsSameAsBest = False
    if statsCfg is not None and statsCfg.is_file():
        bestCfgObj = profile.loadJson(str(bestCfg))
        statsCfgObj = profile.loadJson(str(statsCfg))
        profile.ensureFinalPortionPct(bestCfgObj)
        profile.ensureFinalPortionPct(statsCfgObj)
        statsSameAsBest = configsEqual(bestCfgObj, statsCfgObj)

    if chartsRoot is not None:
        print("[holdout] preparing charts...")
    if statsSameAsBest:
        print("[holdout] stats == best; skipping duplicate run...")
    best = _run_one("best", bestCfg, holdoutDays, chartsRoot)
    _print_block(best)
    if (
        statsCfg is not None
        and statsCfg.is_file()
        and not statsSameAsBest
    ):
        stats = _run_one("stats", statsCfg, holdoutDays, chartsRoot)
        _print_block(stats)


def parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="holdout",
        description="Run holdout backtests for best and stats configs.",
    )
    parser.add_argument(
        "--best",
        required=True,
        help="Path to best-config.json",
    )
    parser.add_argument(
        "--stats",
        help="Path to beststats-config.json",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=0,
        help="Override holdout window in days (optional)",
    )
    parser.add_argument(
        "--charts-root",
        default=None,
        help="Directory for holdout charts (subdirs per label)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parseArgs(argv)
    bestPath = Path(args.best)
    statsPath = Path(args.stats) if args.stats else None
    chartsRoot = Path(args.charts_root) if args.charts_root else None
    runHoldout(bestPath, statsPath, args.days, chartsRoot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
