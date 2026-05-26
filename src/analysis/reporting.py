#!/usr/bin/env python3
# reporting.py - shared post-run metric summaries.

from __future__ import annotations

import io
import math
from typing import Any

from analysis.metrics import grossPctVsBench


def _num(value: Any) -> float:
    out = float(value)
    return out if math.isfinite(out) else float("nan")


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else float("nan")


def _minFinite(a: float, b: float) -> float:
    values = [i for i in (a, b) if math.isfinite(i)]
    return min(values) if values else float("nan")


def _fmtScore(value: float) -> str:
    return f"{value:.2f}" if math.isfinite(value) else "nan"


def _fmtPct(value: float) -> str:
    return f"{value * 100.0:.2f}%" if math.isfinite(value) else "nan"


def _fmtSignedPct(value: float) -> str:
    return f"{value:+.2f}%" if math.isfinite(value) else "nan"


def _fmtTs(value: Any) -> str:
    if value is None:
        return ""
    return value.strftime("%Y-%m-%d %H:%M UTC")


def _daySpan(start: Any, end: Any) -> int:
    if start is None or end is None:
        return 0
    seconds = (end - start).total_seconds()
    return max(0, int(round(seconds / 86400.0)))


def riskScore(data: dict[str, float | int | str]) -> float:
    mar = _num(data["mar"])
    sharpe = _minFinite(_num(data["sh4_edge"]), _num(data["sh13_edge"]))
    sortino = _minFinite(_num(data["sor4_edge"]), _num(data["sor13_edge"]))
    if not all(math.isfinite(i) for i in (mar, sharpe, sortino)):
        return float("nan")
    return (0.10 * mar) + (0.45 * sharpe) + (0.45 * sortino)


def resultMetrics(
    label: str,
    ticker: str,
    result,
) -> dict[str, float | int | str]:
    simVal = float(result.sim.get("portfolio_value", 0.0))
    benchVal = float(result.bench.get("portfolio_value", 0.0))
    seed = float(result.seedQuote or 0.0)
    health = getattr(result, "executionHealth", {}) or {}
    if seed > 0.0:
        edgePct = ((simVal / seed) - 1.0) * 100.0
        hodlPct = ((benchVal / seed) - 1.0) * 100.0
    else:
        edgePct = float("nan")
        hodlPct = float("nan")
    return {
        "label": label,
        "ticker": ticker,
        "pct": grossPctVsBench(simVal, benchVal),
        "edge": edgePct,
        "hodl": hodlPct,
        "trades": int(result.sim.get("trades", 0)),
        "buys": int(result.buyTrades),
        "sells": int(result.sellTrades),
        "same_bar_flips": int(health.get("same_bar_opposite_flips", 0)),
        "day_flips": int(health.get("day_opposite_flips", 0)),
        "neutral_low_pct": float(
            health.get("neutral_low_exposure_pct", float("nan"))
        ),
        "neutral_half_pct": float(
            health.get("neutral_half_exposure_pct", float("nan"))
        ),
        "sh1": float(result.sharpe1wAbs),
        "sh1_edge": float(result.sharpe1w),
        "sh4": float(result.sharpe4wAbs),
        "sh4_edge": float(result.sharpe4w),
        "sh13": float(result.sharpe13wAbs),
        "sh13_edge": float(result.sharpe13w),
        "sor1": float(result.sortino1wAbs),
        "sor1_edge": float(result.sortino1w),
        "sor4": float(result.sortino4wAbs),
        "sor4_edge": float(result.sortino4w),
        "sor13": float(result.sortino13wAbs),
        "sor13_edge": float(result.sortino13w),
        "mdd": float(result.mdd),
        "life_score": float(result.lifecycleEdgeScore),
        "life_p25": float(result.lifecycleEdgeP25),
        "life_min": float(result.lifecycleEdgeMin),
        "life_under": float(result.lifecycleUnderwaterPct),
        "life_track": float(result.lifecycleTrackingPct),
        "life_edge_mdd": float(result.lifecycleEdgeMdd),
        "cagr": float(result.cagr),
        "mar": (
            float(result.cagr) / float(result.mdd)
            if float(result.mdd) > 1e-12
            else float("nan")
        ),
        "days": _daySpan(
            getattr(result, "visibleStartTs", None),
            getattr(result, "visibleEndTs", None),
        ),
        "raw_start": _fmtTs(getattr(result, "rawStartTs", None)),
        "raw_end": _fmtTs(getattr(result, "rawEndTs", None)),
        "active_start": _fmtTs(getattr(result, "visibleStartTs", None)),
        "active_end": _fmtTs(getattr(result, "visibleEndTs", None)),
    }


def metricBlockText(
    prefix: str,
    data: dict[str, float | int | str],
    bar: str,
) -> str:
    buf = io.StringIO()
    label = str(data.get("label", ""))
    print(bar, file=buf)
    print(f"[{prefix}] GrossVhodl: {label}", file=buf)
    print(
        f"[{prefix}] - {data['ticker']}: "
        f"{data['pct']:+.2f}% - edge: {data['edge']:+.2f}%, "
        f"hodl: {data['hodl']:+.2f}%"
        ,
        file=buf,
    )
    print(
        f"[{prefix}] - TRADES: {int(data['trades'])} - "
        f"buys: {int(data['buys'])}, sells: {int(data['sells'])}"
        ,
        file=buf,
    )
    print(
        f"[{prefix}] - EXEC: same-bar flips "
        f"{int(data['same_bar_flips'])}, 24h flips "
        f"{int(data['day_flips'])}, neutral low/half "
        f"{data['neutral_low_pct']:.2f}% / "
        f"{data['neutral_half_pct']:.2f}%",
        file=buf,
    )
    print(f"[{prefix}] - DAYS: {int(data['days'])}", file=buf)
    print(f"[{prefix}] RiskStats", file=buf)
    print(f"[{prefix}] - SCORE: {_fmtScore(riskScore(data))}", file=buf)
    print(
        f"[{prefix}] - SHARPE 1w  (abs/rel): "
        f"{data['sh1']:.2f} / {data['sh1_edge']:.2f}"
        ,
        file=buf,
    )
    print(
        f"[{prefix}] - SHARPE 4w  (abs/rel): "
        f"{data['sh4']:.2f} / {data['sh4_edge']:.2f}"
        ,
        file=buf,
    )
    print(
        f"[{prefix}] - SHARPE 13w (abs/rel): "
        f"{data['sh13']:.2f} / {data['sh13_edge']:.2f}"
        ,
        file=buf,
    )
    print(
        f"[{prefix}] - SORTINO 1w  (abs/rel): "
        f"{data['sor1']:.2f} / {data['sor1_edge']:.2f}"
        ,
        file=buf,
    )
    print(
        f"[{prefix}] - SORTINO 4w  (abs/rel): "
        f"{data['sor4']:.2f} / {data['sor4_edge']:.2f}"
        ,
        file=buf,
    )
    print(
        f"[{prefix}] - SORTINO 13w (abs/rel): "
        f"{data['sor13']:.2f} / {data['sor13_edge']:.2f}"
        ,
        file=buf,
    )
    print(
        f"[{prefix}] - MAR (CAGR/MDD): {data['mar']:.2f}",
        file=buf,
    )
    print(f"[{prefix}] - MDD: {data['mdd']*100.0:.2f}%", file=buf)
    print(
        f"[{prefix}] - LIFE: {data['life_score']:.2f} "
        f"(p25 {data['life_p25']:+.2f}%, "
        f"min {data['life_min']:+.2f}%)",
        file=buf,
    )
    print(f"[{prefix}] - CAGR: {data['cagr']*100.0:.2f}%", file=buf)
    print(bar, file=buf)
    return buf.getvalue()


def _startTag(baseLabel: str, label: str) -> str:
    suffix = label.replace(baseLabel, "", 1).lstrip("-")
    return suffix if suffix else "s00"


def holdoutTableText(
    prefix: str,
    series: list[tuple[str, list[dict[str, float | int | str]]]],
    bar: str,
) -> str:
    buf = io.StringIO()
    print(bar, file=buf)
    print(f"[{prefix}] profile comparison across start offsets", file=buf)
    print(
        f"[{prefix}] profile   starts  gross best      mean       min  "
        "life mean/min  mdd max  trades avg  flip avg  neut low  worst",
        file=buf,
    )
    for baseLabel, rows in series:
        pctVals = [_num(i["pct"]) for i in rows]
        scoreVals = [_num(i["life_score"]) for i in rows]
        mddVals = [_num(i["mdd"]) for i in rows]
        tradeVals = [_num(i["trades"]) for i in rows]
        flipVals = [_num(i["day_flips"]) for i in rows]
        neutralLowVals = [_num(i["neutral_low_pct"]) for i in rows]
        scoreFinite = [i for i in scoreVals if math.isfinite(i)]
        neutralFinite = [i for i in neutralLowVals if math.isfinite(i)]
        worst = min(rows, key=lambda i: _num(i["pct"]))
        scoreMean = _mean(scoreFinite)
        scoreMin = min(scoreFinite) if scoreFinite else float("nan")
        tradesAvg = int(round(_mean(tradeVals)))
        flipsAvg = int(round(_mean(flipVals)))
        neutralMax = max(neutralFinite) if neutralFinite else float("nan")
        print(
            f"[{prefix}] {baseLabel:<9} {len(rows):>6}  "
            f"{_fmtSignedPct(max(pctVals)):>10} "
            f"{_fmtSignedPct(_mean(pctVals)):>9} "
            f"{_fmtSignedPct(min(pctVals)):>9}  "
            f"{_fmtScore(scoreMean):>5} / {_fmtScore(scoreMin):<5}  "
            f"{_fmtPct(max(mddVals)):>7}  "
            f"{tradesAvg:>10}  "
            f"{flipsAvg:>8}  "
            f"{neutralMax:>8.2f}  "
            f"{_startTag(baseLabel, str(worst['label']))}",
            file=buf,
        )
    print(bar, file=buf)
    return buf.getvalue()


def printMetricBlock(
    prefix: str,
    data: dict[str, float | int | str],
    bar: str,
) -> None:
    print(metricBlockText(prefix, data, bar), end="")


__all__ = [
    "holdoutTableText",
    "metricBlockText",
    "printMetricBlock",
    "resultMetrics",
    "riskScore",
]
