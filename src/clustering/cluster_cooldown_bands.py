#!/usr/bin/env python3

from __future__ import annotations

########################################################################
# Imports
########################################################################

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gradbot-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gradbot-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


########################################################################
# Constants
########################################################################

CLUSTER_COLORS = [
    "#2f6f4e",
    "#c94f2d",
    "#2d6fc9",
    "#c9a22d",
    "#7c4fc9",
    "#4aa8a8",
    "#8c4a2f",
    "#5b7f2a",
    "#b04f83",
    "#4d5db8",
]

DAY_MS = 86_400_000
YEAR_DAYS = 365.25


########################################################################
# IO Helpers
########################################################################

def _writeFrame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


########################################################################
# Band Helpers
########################################################################

def _clusterColors(clusterCount: int) -> list[str]:
    colors: list[str] = []
    while len(colors) < int(clusterCount):
        colors.extend(CLUSTER_COLORS)
    return colors[:int(clusterCount)]


def _rawBands(rows: pd.DataFrame) -> pd.Series:
    clusters = rows["cluster"].astype(int)
    labels = np.full(int(rows.shape[0]), "none", dtype=object)
    valid = clusters.to_numpy() >= 0
    labels[valid] = "c" + clusters[valid].astype(str)
    return pd.Series(labels, index=rows.index)


def _cooldownBands(
    rawBands: pd.Series,
    confirmBars: int,
    cooldownBars: int,
) -> pd.Series:
    rawVals = rawBands.astype(str).tolist()
    active = "none"
    pending = ""
    pendingBars = 0
    coolLeft = 0
    out: list[str] = []
    confirm = max(int(confirmBars), 1)
    cooldown = max(int(cooldownBars), 0)

    for val in rawVals:
        if val == "none":
            out.append(active)
            continue
        if active == "none":
            active = val
            out.append(active)
            continue
        if coolLeft > 0:
            coolLeft -= 1
            out.append(active)
            continue
        if val == active:
            pending = ""
            pendingBars = 0
        elif val == pending:
            pendingBars += 1
        else:
            pending = val
            pendingBars = 1
        if pending and pendingBars >= confirm:
            active = pending
            pending = ""
            pendingBars = 0
            coolLeft = cooldown
        out.append(active)
    return pd.Series(out, index=rawBands.index)


def _runIds(values: pd.Series) -> pd.Series:
    return (values != values.shift(1)).cumsum()


def _runFrame(rows: pd.DataFrame, bandCol: str) -> pd.DataFrame:
    use = rows[rows[bandCol].astype(str) != "none"].copy()
    runIds = _runIds(use[bandCol].astype(str))
    out = use.assign(runId=runIds).groupby("runId", sort=False).agg(
        band=(bandCol, "first"),
        startMs=("openMs", "first"),
        endMs=("openMs", "last"),
        bars=(bandCol, "size"),
        startClose=("close", "first"),
        endClose=("close", "last"),
    ).reset_index(drop=True)
    out["retPct"] = ((out["endClose"] / out["startClose"]) - 1.0) * 100.0
    return out


def _fwdColumns(rows: pd.DataFrame) -> list[str]:
    return [
        str(i) for i in rows.columns
        if str(i).startswith("fwdRet") and str(i).endswith("h")
    ]


def _modeMetrics(
    rows: pd.DataFrame,
    bandCol: str,
    mode: str,
    partition: str,
) -> dict[str, object]:
    use = rows.copy()
    if partition != "all" and "partition" in use.columns:
        use = use[use["partition"].astype(str) == partition].copy()
    runs = _runFrame(use, bandCol)
    years = max(
        (float(use["openMs"].max() - use["openMs"].min()) / DAY_MS)
        / YEAR_DAYS,
        1e-9,
    )
    switches = max(int(runs.shape[0]) - 1, 0)
    row: dict[str, object] = {
        "mode": mode,
        "partition": partition,
        "rows": int(use.shape[0]),
        "runs": int(runs.shape[0]),
        "switches": int(switches),
        "switchesPerYear": float(switches / years),
        "medianRunBars": float(runs["bars"].median()),
        "meanRunBars": float(runs["bars"].mean()),
        "minRunBars": int(runs["bars"].min()),
        "maxRunBars": int(runs["bars"].max()),
        "medianRunRetPct": float(runs["retPct"].median()),
    }
    for col in _fwdColumns(use):
        grouped = use.groupby(bandCol)[col].median()
        vals = grouped.dropna().astype(float)
        row[f"{col}BandSpread"] = (
            float(vals.max() - vals.min()) if int(vals.shape[0]) else 0.0
        )
    return row


def _comparisonMetrics(
    rows: pd.DataFrame,
    partitions: list[str],
) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    for part in partitions:
        raw = _modeMetrics(rows, "rawBand", "raw", part)
        cool = _modeMetrics(rows, "cooldownBand", "cooldown", part)
        out.append(raw)
        out.append(cool)
        out.append(
            {
                "mode": "comparison",
                "partition": part,
                "rows": int(raw["rows"]),
                "runs": "",
                "switches": "",
                "switchesPerYear": "",
                "medianRunBars": "",
                "meanRunBars": "",
                "minRunBars": "",
                "maxRunBars": "",
                "medianRunRetPct": "",
                "switchReductionPct": (
                    (
                        1.0
                        - (float(cool["switches"])
                           / max(float(raw["switches"]), 1.0))
                    )
                    * 100.0
                ),
                "barsChangedPct": float(
                    (
                        rows["rawBand"].astype(str)
                        != rows["cooldownBand"].astype(str)
                    ).mean()
                    * 100.0
                ),
            }
        )
    return pd.DataFrame(out)


########################################################################
# Charting
########################################################################

def _bandCodes(rows: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    states = sorted(
        {
            str(i) for i in pd.concat(
                [rows["rawBand"], rows["cooldownBand"]],
                ignore_index=True,
            ).tolist()
            if str(i) != "none"
        },
        key=lambda i: int(i[1:]) if i.startswith("c") else 999,
    )
    colors = _clusterColors(max(len(states), 1))
    byState = {state: i for i, state in enumerate(states)}
    raw = rows["rawBand"].astype(str).map(byState).fillna(-1).astype(int)
    cool = (
        rows["cooldownBand"].astype(str).map(byState).fillna(-1).astype(int)
    )
    return np.vstack([raw.to_numpy(), cool.to_numpy()]), states, colors


def _plotBands(rows: pd.DataFrame, outPath: Path, title: str) -> None:
    if rows.empty:
        return
    xTime = pd.to_datetime(rows["openMs"], unit="ms", utc=True)
    x = mdates.date2num(xTime.dt.tz_convert(None).to_numpy())
    close = rows["close"].astype(float).to_numpy()
    codes, states, colors = _bandCodes(rows)
    cmap = ListedColormap(colors)
    cmap.set_bad(color="#111111")
    bounds = np.arange(-0.5, float(len(states)) + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    masked = np.ma.masked_less(codes.astype(float), 0.0)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [8, 1]},
    )
    ax = axes[0]
    strip = axes[1]
    ax.plot(x, close, color="#202020", linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel("close")
    ax.grid(True, alpha=0.2)
    strip.imshow(
        masked,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        origin="lower",
        extent=[x[0], x[-1], 0, 2],
    )
    strip.set_yticks([0.5, 1.5])
    strip.set_yticklabels(["raw", "cooldown"])
    strip.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    handles = [
        Patch(facecolor=color, edgecolor="none", label=state)
        for state, color in zip(states, colors)
    ]
    ax.legend(handles=handles, loc="upper left", ncols=5, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    outPath.parent.mkdir(parents=True, exist_ok=True)
    tmp = outPath.with_name(outPath.stem + ".tmp" + outPath.suffix)
    fig.savefig(tmp, dpi=140)
    plt.close(fig)
    os.replace(tmp, outPath)


########################################################################
# Main
########################################################################

def _partitions(rows: pd.DataFrame) -> list[str]:
    out = ["all"]
    if "partition" in rows.columns:
        vals = rows["partition"].dropna().astype(str).unique().tolist()
        out += [i for i in ["fit", "holdout", "policy"] if i in vals]
    return out


def run(
    featuresPath: Path,
    outDir: Path,
    confirmBars: int,
    cooldownBars: int,
    chartBars: int,
) -> dict[str, Path]:
    rows = pd.read_csv(featuresPath).sort_values("openMs").reset_index(
        drop=True,
    )
    rows["rawBand"] = _rawBands(rows)
    rows["cooldownBand"] = _cooldownBands(
        rows["rawBand"],
        confirmBars,
        cooldownBars,
    )
    keep = [
        i for i in [
            "ticker",
            "openMs",
            "closeMs",
            "close",
            "partition",
            "cluster",
            "clusterConfidence",
            "clusterDistance",
            "rawBand",
            "cooldownBand",
        ]
        if i in rows.columns
    ]
    bandRows = rows[keep].copy()
    metrics = _comparisonMetrics(rows, _partitions(rows))
    chartRows = rows.tail(int(chartBars)).copy() if chartBars > 0 else rows
    bandPath = outDir / "cooldown_band_rows.csv"
    metricPath = outDir / "cooldown_band_metrics.csv"
    chartPath = outDir / "cooldown_band_chart.png"
    _writeFrame(bandPath, bandRows)
    _writeFrame(metricPath, metrics)
    _plotBands(
        chartRows,
        chartPath,
        (
            f"raw vs cooldown bands "
            f"(confirm={int(confirmBars)}, cooldown={int(cooldownBars)})"
        ),
    )
    return {
        "bands": bandPath,
        "metrics": metricPath,
        "chart": chartPath,
    }


def parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cluster_cooldown_bands",
        description="Compare raw cluster bands with causal cooldown bands.",
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--confirm-bars", type=int, default=3)
    parser.add_argument("--cooldown-bars", type=int, default=6)
    parser.add_argument("--chart-bars", type=int, default=2160)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parseArgs(argv)
    paths = run(
        Path(args.features),
        Path(args.out),
        int(args.confirm_bars),
        int(args.cooldown_bars),
        int(args.chart_bars),
    )
    print(f"[cluster-cooldown] bands: {paths['bands']}")
    print(f"[cluster-cooldown] metrics: {paths['metrics']}")
    print(f"[cluster-cooldown] chart: {paths['chart']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
