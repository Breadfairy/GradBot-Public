#!/usr/bin/env python3

from __future__ import annotations

########################################################################
# Imports
########################################################################

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gradbot-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gradbot-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"], "fontconfig").mkdir(
    parents=True,
    exist_ok=True,
)

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib.colors import BoundaryNorm, ListedColormap

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
]


########################################################################
# Helpers
########################################################################

def _clusterColors(clusterCount: int) -> list[str]:
    colors: list[str] = []
    while len(colors) < int(clusterCount):
        colors.extend(CLUSTER_COLORS)
    return colors[:int(clusterCount)]


def _periodsFromPath(path: Path) -> list[int]:
    pattern = re.compile(r"ema(\d{3})_(\d{3})_(\d{3})")
    for part in path.parts:
        match = pattern.fullmatch(part)
        if match:
            return [int(match.group(i)) for i in range(1, 4)]
    return []


def _filterDates(
    frame: pd.DataFrame,
    startDate: str,
    endDate: str,
) -> pd.DataFrame:
    out = frame.copy()
    ts = pd.to_datetime(out["openMs"], unit="ms", utc=True)
    if startDate:
        start = pd.Timestamp(startDate, tz="UTC")
        out = out[ts >= start]
        ts = pd.to_datetime(out["openMs"], unit="ms", utc=True)
    if endDate:
        end = pd.Timestamp(endDate, tz="UTC")
        out = out[ts <= end]
    return out


def _plotFlags(ax, chart: pd.DataFrame, x: np.ndarray) -> None:
    buyMask = chart["acceptedBuy"].astype(float).to_numpy() > 0.0
    sellMask = chart["acceptedSell"].astype(float).to_numpy() > 0.0
    low = chart["low"].astype(float).to_numpy()
    high = chart["high"].astype(float).to_numpy()
    ax.scatter(
        x[buyMask],
        low[buyMask],
        marker="^",
        s=42,
        color="#15803d",
        edgecolor="white",
        linewidth=0.4,
        label="DSP buy",
        zorder=5,
    )
    ax.scatter(
        x[sellMask],
        high[sellMask],
        marker="v",
        s=42,
        color="#b91c1c",
        edgecolor="white",
        linewidth=0.4,
        label="DSP sell",
        zorder=5,
    )


def _plot(
    frame: pd.DataFrame,
    path: Path,
    title: str,
    tailBars: int,
    periods: list[int],
) -> bool:
    chart = frame.tail(int(tailBars)).copy() if int(tailBars) else frame.copy()
    chart = chart[chart["cluster"] >= 0].copy()
    if chart.empty:
        return False
    ts = pd.to_datetime(chart["openMs"], unit="ms", utc=True)
    x = mdates.date2num(ts.dt.tz_convert(None).to_numpy())
    close = chart["close"].astype(float).to_numpy()
    clusters = chart["cluster"].astype(int).to_numpy()
    clusterCount = int(np.max(clusters)) + 1
    colors = _clusterColors(clusterCount)
    bounds = np.arange(-0.5, float(clusterCount) + 0.5, 1.0)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, (ax, strip) = plt.subplots(
        2,
        1,
        figsize=(16, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [8, 1]},
    )
    ax.plot(x, close, "-", color="#202020", linewidth=1.0, label="close")
    for period in periods:
        ema = chart["close"].ewm(span=int(period), adjust=False).mean()
        ax.plot(x, ema, "-", linewidth=0.8, label=f"EMA {int(period)}")
    if "acceptedBuy" in chart.columns and "acceptedSell" in chart.columns:
        _plotFlags(ax, chart, x)

    start = 0
    while start < clusters.shape[0]:
        end = start + 1
        while end < clusters.shape[0] and clusters[end] == clusters[start]:
            end += 1
        right = x[end - 1] if end < clusters.shape[0] else x[-1]
        ax.axvspan(
            x[start],
            right,
            color=colors[int(clusters[start])],
            alpha=0.08,
            linewidth=0,
        )
        start = end

    strip.imshow(
        clusters.reshape(1, -1),
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=[x[0], x[-1], 0, 1],
    )
    strip.set_yticks([])
    strip.set_ylabel("cluster")
    ax.set_title(title)
    ax.set_ylabel("price")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", ncols=4, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.png")
    fig.tight_layout()
    fig.savefig(tmp, dpi=140)
    plt.close(fig)
    os.replace(tmp, path)
    return True


def renderCharts(
    featuresPath: Path,
    outDir: Path,
    title: str,
    tailBars: int,
    yearly: bool,
    startDate: str,
    endDate: str,
    name: str,
) -> list[Path]:
    frame = _filterDates(pd.read_csv(featuresPath), startDate, endDate)
    paths: list[Path] = []
    useTitle = title if title else str(featuresPath.parent.name)
    stem = name if name else "full"
    periods = _periodsFromPath(featuresPath)
    fullPath = outDir / f"cluster_chart_{stem}.png"
    chartTitle = f"{useTitle} {stem}"
    if _plot(frame, fullPath, chartTitle, int(tailBars), periods):
        paths.append(fullPath)
    if yearly:
        ts = pd.to_datetime(frame["openMs"], unit="ms", utc=True)
        frame["chartYear"] = ts.dt.year
        for year in sorted(frame["chartYear"].dropna().unique().astype(int)):
            part = frame[frame["chartYear"] == int(year)].drop(
                columns=["chartYear"],
            )
            path = outDir / f"cluster_chart_{int(year)}.png"
            if _plot(part, path, f"{useTitle} {int(year)}", 0, periods):
                paths.append(path)
    return paths


########################################################################
# CLI
########################################################################

def parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="render_cluster_charts",
        description="Render cluster charts from clustered_features.csv.",
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--tail-bars", type=int, default=0)
    parser.add_argument("--yearly", action="store_true")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--name", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parseArgs(argv)
    paths = renderCharts(
        Path(args.features),
        Path(args.out),
        str(args.title),
        int(args.tail_bars),
        bool(args.yearly),
        str(args.start_date),
        str(args.end_date),
        str(args.name),
    )
    for path in paths:
        print(f"[cluster-chart] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
