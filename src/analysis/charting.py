#!/usr/bin/env python3
# charting.py – plotting helpers (closes line only).

import os
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
from matplotlib.patches import Patch

matplotlib.use("Agg")  # headless-safe for batch chart saving
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.colors import BoundaryNorm, ListedColormap
from engine.shared import (
    buildSignals,
    macroDynCarry,
    bars_per_day,
)
from engine.macro_view import buildMacroView


# RGB-style colors tuned for closes mode (0–1 floats).
BG_COLOR = [0.08, 0.03, 0.03]
TEXT_COLOR = [0.98, 0.95, 0.82]
CLOSE_COLOR = [0.98, 0.98, 0.98]
GRID_COLOR = [0.45, 0.40, 0.35]

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
    "#8f6d1f",
    "#3b7f7a",
    "#9a3f3f",
    "#5571a8",
    "#6f8f2a",
    "#8a5aa8",
]

REGIME_COLORS = {
    "bear": "#dc2626",
    "flush": "#9333ea",
    "crab": "#64748b",
    "crab_c1": "#94a3b8",
    "crab_c2": "#475569",
    "crab_post": "#2563eb",
    "ultraBull": "#16a34a",
    "bullChop": "#84cc16",
    "lock": "#ca8a04",
}

REGIME_ORDER = [
    "bear",
    "flush",
    "crab",
    "crab_c1",
    "crab_c2",
    "crab_post",
    "ultraBull",
    "bullChop",
    "lock",
]


def seriesLike(arr, index):
    """Return Series aligned to index; pad/trim so size always matches."""
    a = np.asarray(arr, dtype=float)
    n = len(index)
    if a.size < n:
        a = np.pad(a, (0, n - a.size), constant_values=np.nan)
    elif a.size > n:
        a = a[:n]
    return pd.Series(a, index=index)


def _clusterColors(clusterCount: int) -> list[str]:
    colors: list[str] = []
    clusterCount = int(clusterCount)
    while len(colors) < clusterCount:
        colors.extend(CLUSTER_COLORS)
    return colors[:clusterCount]


def _posturePath(overrides: dict | None) -> Path | None:
    ov = overrides if isinstance(overrides, dict) else {}
    pathRaw = str(ov.get("DAILY_CLUSTER_PATH", "")).strip()
    if not pathRaw:
        return None
    path = Path(pathRaw)
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).resolve().parents[2] / path
    if not path.exists():
        return None
    return path


def _postureFrame(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    cols = ["openMs", "cluster"]
    if "closeMs" in header.columns:
        cols.append("closeMs")
    if "confirmedRegime" in header.columns:
        cols.append("confirmedRegime")
    frame = pd.read_csv(path, usecols=cols).copy()
    if "closeMs" not in frame.columns:
        frame["closeMs"] = frame["openMs"]
    if "confirmedRegime" not in frame.columns:
        frame["confirmedRegime"] = (
            "cluster_" + frame["cluster"].astype(int).astype(str)
        )
    return frame.sort_values("closeMs").reset_index(drop=True)


def _regimeNames(values: np.ndarray) -> list[str]:
    present = {str(i) for i in values.tolist() if str(i) != "none"}
    names = [i for i in REGIME_ORDER if i in present]
    names += sorted(present.difference(names))
    return names


def _colorsForNames(names: list[str]) -> list[str]:
    fallback = _clusterColors(max(len(names), 1))
    out = []
    for i, name in enumerate(names):
        out.append(REGIME_COLORS.get(str(name), fallback[i]))
    return out


def _alignPostureToMs(
    frame: pd.DataFrame,
    targetMs: np.ndarray,
) -> tuple[np.ndarray, list[str], list[str]]:
    closeMs = frame["closeMs"].to_numpy(dtype=np.int64)
    regimes = frame["confirmedRegime"].astype(str).to_numpy()
    pos = np.searchsorted(closeMs, targetMs, side="right") - 1
    valid = (pos >= 0) & (pos < regimes.size)
    labelsRaw = np.full(targetMs.shape[0], "none", dtype=object)
    labelsRaw[valid] = regimes[pos[valid]]
    names = _regimeNames(labelsRaw)
    byName = {name: i for i, name in enumerate(names)}
    labels = np.full(targetMs.shape[0], -1, dtype=int)
    for name, idx in byName.items():
        labels[labelsRaw == name] = int(idx)
    return labels, names, _colorsForNames(names)


def _shadeRegimes(
    ax,
    xVals,
    labels: np.ndarray | None,
    colors: list[str],
    alpha: float = 0.08,
) -> None:
    if labels is None or labels.size == 0 or not colors:
        return
    vals = np.asarray(labels, dtype=int)
    start = 0
    while start < vals.size:
        end = start + 1
        while end < vals.size and vals[end] == vals[start]:
            end += 1
        if vals[start] >= 0:
            right = xVals[end - 1] if end < vals.size else xVals[-1]
            ax.axvspan(
                xVals[start],
                right,
                color=colors[int(vals[start])],
                alpha=alpha,
                linewidth=0,
                zorder=0.1,
            )
        start = end


def _regimeLegend(ax, names: list[str], colors: list[str]) -> None:
    if not names or not colors:
        return
    handles = [
        Patch(facecolor=color, edgecolor="none", label=name, alpha=0.28)
        for name, color in zip(names, colors)
    ]
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        fontsize=7,
        ncols=min(len(handles), 4),
    )
    if leg:
        for txt in leg.get_texts():
            txt.set_color(TEXT_COLOR)
        frame = leg.get_frame()
        if frame:
            frame.set_facecolor(BG_COLOR)
            frame.set_edgecolor(GRID_COLOR)
    ax.add_artist(leg)


def _dailyPostureLabelsForPlot(
    overrides: dict,
    ctx,
) -> tuple[np.ndarray, list[str], list[str]] | tuple[None, list[str], list[str]]:
    path = _posturePath(overrides)
    if path is None:
        return None, [], []
    frame = _postureFrame(path)
    kOpen = np.asarray([int(k[0]) for k in ctx["klines"]], dtype=np.int64)
    return _alignPostureToMs(frame, kOpen)


def _timValPostureLabelsForPlot(
    overrides: dict | None,
    ts: List[Any],
) -> tuple[np.ndarray, list[str], list[str]] | tuple[None, list[str], list[str]]:
    path = _posturePath(overrides)
    if path is None:
        return None, [], []
    frame = _postureFrame(path)
    xTime = pd.to_datetime(pd.Series(list(ts)), utc=True)
    xMs = (xTime.astype("int64") // 1_000_000).to_numpy(dtype=np.int64)
    return _alignPostureToMs(frame, xMs)


def _dailyClusterLabelsForPlot(
    overrides: dict,
    ctx,
) -> tuple[np.ndarray, int] | tuple[None, int]:
    pathRaw = str(overrides.get("DAILY_CLUSTER_PATH", "")).strip()
    if not pathRaw:
        return None, 0
    path = Path(pathRaw)
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).resolve().parents[2] / path
    if not path.exists():
        return None, 0

    frame = pd.read_csv(path, usecols=["openMs", "cluster"])
    dayOpen = frame["openMs"].to_numpy(dtype=np.int64)
    dayCluster = frame["cluster"].to_numpy(dtype=int)
    kOpen = np.asarray([int(k[0]) for k in ctx["klines"]], dtype=np.int64)
    pos = np.searchsorted(dayOpen, kOpen, side="right") - 1
    labels = np.full(kOpen.shape[0], -1, dtype=int)
    valid = (pos >= 0) & (pos < dayCluster.size)
    labels[valid] = dayCluster[pos[valid]]
    nonWarm = labels[labels >= 0]
    count = int(nonWarm.max()) + 1 if nonWarm.size else 0
    return labels, count


def _timValClusterLabelsForPlot(
    overrides: dict | None,
    ts: List[Any],
) -> tuple[np.ndarray, int] | tuple[None, int]:
    ov = overrides if isinstance(overrides, dict) else {}
    pathRaw = str(ov.get("DAILY_CLUSTER_PATH", "")).strip()
    if not pathRaw:
        return None, 0
    path = Path(pathRaw)
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).resolve().parents[2] / path
    if not path.exists():
        return None, 0

    frame = pd.read_csv(path, usecols=["openMs", "cluster"])
    dayOpen = frame["openMs"].to_numpy(dtype=np.int64)
    dayCluster = frame["cluster"].to_numpy(dtype=int)
    xTime = pd.to_datetime(pd.Series(list(ts)), utc=True)
    xMs = (xTime.astype("int64") // 1_000_000).to_numpy(dtype=np.int64)
    pos = np.searchsorted(dayOpen, xMs, side="right") - 1
    labels = np.full(xMs.shape[0], -1, dtype=int)
    valid = (pos >= 0) & (pos < dayCluster.size)
    labels[valid] = dayCluster[pos[valid]]
    nonWarm = labels[labels >= 0]
    count = int(nonWarm.max()) + 1 if nonWarm.size else 0
    return labels, count


def plotTimVal(
    ts: List[Any],
    edgeVals: np.ndarray,
    hodlVals: np.ndarray,
    assetFrac: np.ndarray,
    quoteFrac: np.ndarray,
    title: str,
    savePath: str,
    overrides: dict | None = None,
) -> None:
    x = list(ts)
    edge = np.asarray(edgeVals, dtype=float)
    hodl = np.asarray(hodlVals, dtype=float)
    asset = np.asarray(assetFrac, dtype=float)
    quote = np.asarray(quoteFrac, dtype=float)
    if edge.size == 0 or hodl.size == 0 or len(x) == 0:
        return

    n = min(
        len(x),
        int(edge.size),
        int(hodl.size),
        int(asset.size),
        int(quote.size),
    )
    x = x[:n]
    edge = edge[:n]
    hodl = hodl[:n]
    asset = asset[:n] * 100.0
    quote = quote[:n] * 100.0
    postureLabels, postureNames, postureColors = (
        _timValPostureLabelsForPlot(overrides, x)
    )
    showCluster = postureLabels is not None and len(postureNames) > 0

    if showCluster:
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(12, 6.9),
            sharex=True,
            gridspec_kw={"height_ratios": [3.0, 1.2, 0.35]},
        )
        topAx = axes[0]
        botAx = axes[1]
        clusterAx = axes[2]
    else:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12, 6.4),
            sharex=True,
            gridspec_kw={"height_ratios": [3.0, 1.2]},
        )
        topAx = axes[0]
        botAx = axes[1]
        clusterAx = None
    fig.patch.set_facecolor(BG_COLOR)
    for ax in axes:
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)

    topAx.set_title(title, color=TEXT_COLOR, pad=10)
    topAx.set_ylabel("Gross Value", color=TEXT_COLOR)

    _shadeRegimes(topAx, x, postureLabels, postureColors)
    topAx.plot(x, edge, color="orange", linewidth=1.2, label="EDGE")
    topAx.plot(
        x,
        hodl,
        color="deepskyblue",
        linewidth=1.2,
        label="HODL",
    )
    topAx.grid(
        color=GRID_COLOR,
        linestyle=":",
        linewidth=0.6,
        alpha=0.7,
    )

    _regimeLegend(topAx, postureNames, postureColors)
    legTop = topAx.legend(loc="best", frameon=True, fontsize=8)
    if legTop:
        for txt in legTop.get_texts():
            txt.set_color(TEXT_COLOR)
        frame = legTop.get_frame()
        if frame:
            frame.set_facecolor(BG_COLOR)
            frame.set_edgecolor(GRID_COLOR)

    botAx.plot(
        x,
        asset,
        color="mediumseagreen",
        linewidth=1.1,
        label="EDGE Asset %",
    )
    botAx.plot(
        x,
        quote,
        color="gold",
        linewidth=1.1,
        label="EDGE USDT %",
    )
    botAx.set_ylabel("Allocation %", color=TEXT_COLOR)
    botAx.set_ylim(-2.0, 102.0)
    botAx.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    botAx.grid(
        color=GRID_COLOR,
        linestyle=":",
        linewidth=0.6,
        alpha=0.7,
    )

    if clusterAx is not None:
        xVals = mdates.date2num(pd.to_datetime(x).to_pydatetime())
        stripVals = np.ma.masked_less(
            np.asarray(postureLabels[:n], dtype=float).reshape(1, -1),
            0.0,
        )
        cmap = ListedColormap(postureColors)
        cmap.set_bad(color=BG_COLOR)
        norm = BoundaryNorm(
            np.arange(-0.5, float(len(postureNames)) + 0.5, 1.0),
            cmap.N,
        )
        clusterAx.imshow(
            stripVals,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            origin="lower",
            extent=[xVals[0], xVals[-1], 0, 1],
        )
        clusterAx.set_yticks([0.5])
        clusterAx.set_yticklabels(["6h"])
        clusterAx.set_ylabel("regime", color=TEXT_COLOR)
        clusterAx.grid(False)

    legBot = botAx.legend(loc="best", frameon=True, fontsize=8)
    if legBot:
        for txt in legBot.get_texts():
            txt.set_color(TEXT_COLOR)
        frame = legBot.get_frame()
        if frame:
            frame.set_facecolor(BG_COLOR)
            frame.set_edgecolor(GRID_COLOR)

    fig.tight_layout(pad=0.8)
    if savePath:
        fig.savefig(savePath, facecolor=BG_COLOR)
    plt.close(fig)


def _filteredResultsDf(
    path: str,
    tickers: list[str] | None,
    minTrades: int | None,
    maxTrades: int | None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if tickers:
        allowed = {str(item).upper() for item in tickers}
        df = df[df["ticker"].str.upper().isin(allowed)]
    if minTrades is not None:
        df = df[df["trades"] >= minTrades]
    if maxTrades is not None:
        df = df[df["trades"] <= maxTrades]
    return df


def _resultPctVsBenchmark(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"simValue", "benchValue"}.issubset(out.columns):
        bench = out["benchValue"].astype(float)
        sim = out["simValue"].astype(float)
        out["pctVsBench"] = ((sim / bench) - 1.0) * 100.0
        return out
    if {"simPostTax", "benchPostTax"}.issubset(out.columns):
        bench = out["benchPostTax"].astype(float)
        sim = out["simPostTax"].astype(float)
        out["pctVsBench"] = ((sim / bench) - 1.0) * 100.0
        return out
    out["pctVsBench"] = 0.0
    return out


def generateScatter(
    resultsCsv: str,
    outputPng: str,
    tickers: list[str] | None = None,
    minTrades: int | None = None,
    maxTrades: int | None = None,
    titleSuffix: str | None = None,
) -> None:
    df = _filteredResultsDf(resultsCsv, tickers, minTrades, maxTrades)
    if df.empty:
        return
    df = _resultPctVsBenchmark(df)
    df = df[pd.notna(df["pctVsBench"])]
    if df.empty:
        return

    title = "Trades vs % vs Benchmark"
    if not titleSuffix:
        parts = []
        if minTrades is not None:
            parts.append(f">= {minTrades} trades")
        if maxTrades is not None:
            parts.append(f"<= {maxTrades} trades")
        if parts:
            titleSuffix = ", ".join(parts)
    if titleSuffix:
        title = f"{title} ({titleSuffix})"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)

    if "interval" in df.columns:
        for intervalValue, grp in df.groupby("interval"):
            ax.scatter(
                grp["trades"],
                grp["pctVsBench"],
                label=str(intervalValue),
                s=14,
                alpha=0.75,
                color=CLOSE_COLOR,
                edgecolors="none",
            )
        leg = ax.legend(
            title="interval",
            loc="best",
            frameon=True,
            fontsize=8,
        )
        if leg:
            leg.get_title().set_color(TEXT_COLOR)
            for txt in leg.get_texts():
                txt.set_color(TEXT_COLOR)
            frame = leg.get_frame()
            if frame:
                frame.set_facecolor(BG_COLOR)
                frame.set_edgecolor(GRID_COLOR)
    else:
        ax.scatter(
            df["trades"],
            df["pctVsBench"],
            s=14,
            alpha=0.75,
            color=CLOSE_COLOR,
            edgecolors="none",
        )

    tradeMedian = float(df["trades"].median())
    metricMedian = float(df["pctVsBench"].median())
    ax.axvline(
        tradeMedian,
        color=GRID_COLOR,
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
    )
    ax.axhline(
        metricMedian,
        color=GRID_COLOR,
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
    )
    ax.set_xlabel("Number of Trades", color=TEXT_COLOR)
    ax.set_ylabel("Increase vs Benchmark (%)", color=TEXT_COLOR)
    ax.set_title(title, color=TEXT_COLOR, pad=8)
    ax.grid(
        True,
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
        color=GRID_COLOR,
    )

    os.makedirs(os.path.dirname(outputPng) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(outputPng)
    plt.close(fig)


class Chart:
    """Plot closes, EMAs, markers, and gradient panels."""

    def __init__(
        self,
        klines,
        ticker,
        markers,
        mas,
        grads,
        switchInfo=None,
        secondaryMarkers=None,
        gradSlowHeat=None,
        macroClose=None,
        macroMas=None,
        macroPeriods=None,
        macroInterval=None,
        dailyClusterIds=None,
        dailyClusterCount=0,
        dailyClusterNames=None,
        dailyClusterColors=None,
    ):
        self.klines = klines
        self.ticker = ticker
        self.markers = markers
        self.mas = mas
        self.grads = grads
        self.switchInfo = switchInfo or {}
        self.secondaryMarkers = secondaryMarkers
        self.gradSlowHeat = gradSlowHeat or []
        self.macroClose = macroClose
        self.macroMas = macroMas
        self.macroPeriods = macroPeriods
        self.macroInterval = macroInterval
        self.dailyClusterIds = dailyClusterIds
        self.dailyClusterCount = int(dailyClusterCount or 0)
        self.dailyClusterNames = list(dailyClusterNames or [])
        self.dailyClusterColors = list(dailyClusterColors or [])
        self.hideGrads = False
        if isinstance(self.switchInfo, dict):
            self.hideGrads = bool(
                self.switchInfo.get("HIDE_GRADS")
                or self.switchInfo.get("hide_grads")
            )

    def plot(self, title=None, savePath=None):
        df = pd.DataFrame(
            self.klines,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trade_count",
                "taker_base_volume",
                "taker_quote_volume",
                "ignore",
            ],
        )

        ts = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["open_time"] = ts.dt.tz_convert(None)

        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        df.set_index("open_time", inplace=True)

        self._plot_closes(df, title, savePath)

    def _marker_series(self, df, markers=None):
        idx = df.index
        sell = pd.Series(np.nan, index=idx)
        buy = pd.Series(np.nan, index=idx)
        wSell = pd.Series(np.nan, index=idx)
        wLockSell = pd.Series(np.nan, index=idx)
        wCrabCap = pd.Series(np.nan, index=idx)
        wBuy = pd.Series(np.nan, index=idx)
        wSeedBuy = pd.Series(np.nan, index=idx)
        srcMarkers = self.markers if markers is None else markers
        if srcMarkers:
            times = [ts_ for ts_, _fl in srcMarkers]
            positions = idx.get_indexer(times, method="nearest")
            for (ts_, fl), pos in zip(srcMarkers, positions):
                if pos < 0:
                    continue
                high = df["high"].iloc[pos]
                low = df["low"].iloc[pos]
                if fl == "SELL":
                    sell.iloc[pos] = high * 1.02
                elif fl == "BUY":
                    buy.iloc[pos] = low * 0.98
                elif fl == "W_SELL":
                    wSell.iloc[pos] = high * 1.03
                elif fl == "W_LOCK_SELL":
                    wLockSell.iloc[pos] = high * 1.045
                elif fl == "W_CRAB_CAP":
                    wCrabCap.iloc[pos] = high * 1.04
                elif fl == "W_BUY":
                    wBuy.iloc[pos] = low * 0.97
                elif fl == "W_SEED_BUY":
                    wSeedBuy.iloc[pos] = low * 0.955
        return {
            "sell": sell,
            "buy": buy,
            "wSell": wSell,
            "wLockSell": wLockSell,
            "wCrabCap": wCrabCap,
            "wBuy": wBuy,
            "wSeedBuy": wSeedBuy,
        }

    def _plot_closes(self, df, title, savePath):
        idx = df.index
        showSecondary = (
            self.hideGrads
            and self.secondaryMarkers is not None
        )
        showMacro = (
            self.macroClose is not None
            and self.macroMas is not None
        )
        showDailyCluster = (
            self.dailyClusterIds is not None
            and self.dailyClusterCount > 0
        )
        showClusterStrip = showDailyCluster
        clusterRows = int(showDailyCluster)
        clusterRatio = 0.35 * max(clusterRows, 1)
        clusterAx = None
        if self.hideGrads:
            if showSecondary:
                nRows = 3 if showMacro else 2
                ratios = [3.0, 1.7, 1.7] if showMacro else [3.0, 1.7]
                figHeight = 8.0 if showMacro else 6.5
                if showClusterStrip:
                    nRows += 1
                    ratios.append(clusterRatio)
                    figHeight += 0.45 * max(clusterRows, 1)
                fig, axes = plt.subplots(
                    nRows,
                    1,
                    sharex=True,
                    figsize=(12, figHeight),
                    gridspec_kw={"height_ratios": ratios},
                )
                mainAx = axes[0]
                macroAx = axes[1] if showMacro else None
                filtAx = axes[-2] if showClusterStrip else axes[-1]
                clusterAx = axes[-1] if showClusterStrip else None
            else:
                if showMacro:
                    nRows = 3 if showClusterStrip else 2
                    ratios = (
                        [3.0, 1.7, clusterRatio]
                        if showClusterStrip else [3.0, 1.7]
                    )
                    figHeight = (
                        6.0 + (0.45 * max(clusterRows, 1))
                        if showClusterStrip else 6.0
                    )
                    fig, axes = plt.subplots(
                        nRows,
                        1,
                        sharex=True,
                        figsize=(12, figHeight),
                        gridspec_kw={
                            "height_ratios": ratios,
                        },
                    )
                    mainAx = axes[0]
                    macroAx = axes[1]
                    clusterAx = axes[-1] if showClusterStrip else None
                else:
                    if showClusterStrip:
                        fig, axes = plt.subplots(
                            2,
                            1,
                            sharex=True,
                            figsize=(12, 4.5 + (0.45 * max(clusterRows, 1))),
                            gridspec_kw={
                                "height_ratios": [3.0, clusterRatio],
                            },
                        )
                        mainAx = axes[0]
                        clusterAx = axes[1]
                    else:
                        fig, mainAx = plt.subplots(
                            1,
                            1,
                            sharex=True,
                            figsize=(12, 4.5),
                        )
                        axes = (mainAx,)
                    macroAx = None
        else:
            if showMacro:
                nRows = 4 if showClusterStrip else 3
                ratios = (
                    [3.0, 1.7, 1.1, clusterRatio]
                    if showClusterStrip else [3.0, 1.7, 1.1]
                )
                figHeight = (
                    7.2 + (0.45 * max(clusterRows, 1))
                    if showClusterStrip else 7.2
                )
                fig, axes = plt.subplots(
                    nRows,
                    1,
                    sharex=True,
                    figsize=(12, figHeight),
                    gridspec_kw={
                        "height_ratios": ratios,
                    },
                )
                mainAx = axes[0]
                macroAx = axes[1]
                gradSlowAx = axes[2]
                clusterAx = axes[-1] if showClusterStrip else None
            else:
                nRows = 3 if showClusterStrip else 2
                ratios = (
                    [3.0, 1.1, clusterRatio]
                    if showClusterStrip else [3.0, 1.1]
                )
                figHeight = (
                    5.6 + (0.45 * max(clusterRows, 1))
                    if showClusterStrip else 5.6
                )
                fig, axes = plt.subplots(
                    nRows,
                    1,
                    sharex=True,
                    figsize=(12, figHeight),
                    gridspec_kw={"height_ratios": ratios},
                )
                mainAx = axes[0]
                gradSlowAx = axes[1]
                clusterAx = axes[-1] if showClusterStrip else None
                macroAx = None
        fig.patch.set_facecolor(BG_COLOR)
        for ax in axes:
            ax.set_facecolor(BG_COLOR)
            ax.tick_params(colors=TEXT_COLOR, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(GRID_COLOR)
        mainAx.set_title(title or self.ticker, color=TEXT_COLOR, pad=10)
        mainAx.set_ylabel("Price", color=TEXT_COLOR)
        if macroAx is not None:
            macroLabel = str(self.macroInterval or "").strip()
            macroAx.set_ylabel(
                f"Macro {macroLabel}" if macroLabel else "Macro",
                color=TEXT_COLOR,
            )
        if showSecondary:
            filtAx.set_ylabel("Filtered", color=TEXT_COLOR)

        _shadeRegimes(
            mainAx,
            idx.to_pydatetime(),
            (
                seriesLike(self.dailyClusterIds, idx).to_numpy(dtype=int)
                if showDailyCluster else None
            ),
            self.dailyClusterColors,
        )
        mainAx.plot(
            idx,
            df["close"],
            color=CLOSE_COLOR,
            linewidth=1.2,
            label="close",
        )
        for ma, color, label in zip(
            self.mas,
            ("cyan", "magenta", "yellow"),
            ("ema1", "ema2", "ema3"),
        ):
            mainAx.plot(
                idx,
                seriesLike(ma, idx),
                color=color,
                linewidth=1.0,
                label=label,
            )

        if macroAx is not None:
            macroLabel = str(self.macroInterval or "").strip()
            closeLabel = (
                f"close({macroLabel})" if macroLabel else "close"
            )
            macroAx.plot(
                idx,
                seriesLike(self.macroClose, idx),
                color=CLOSE_COLOR,
                linewidth=1.1,
                alpha=0.9,
                label=closeLabel,
            )
            periods = (
                list(self.macroPeriods)
                if isinstance(self.macroPeriods, (list, tuple))
                else []
            )
            labels = ["ema1", "ema2", "ema3"]
            if len(periods) >= 3:
                labels = [
                    f"ema1(p{periods[0]})",
                    f"ema2(p{periods[1]})",
                    f"ema3(p{periods[2]})",
                ]
            for ma, color, label in zip(
                self.macroMas,
                ("cyan", "magenta", "yellow"),
                labels,
            ):
                macroAx.plot(
                    idx,
                    seriesLike(ma, idx),
                    color=color,
                    linewidth=0.95,
                    alpha=0.9,
                    label=label,
                )

        markers = self._marker_series(df, self.markers)

        def addMark(series, marker, color, size=50, ax=None):
            valid = series.dropna()
            if not valid.empty:
                targetAx = mainAx if ax is None else ax
                targetAx.scatter(
                    valid.index,
                    valid.values,
                    color=color,
                    marker=marker,
                    s=size,
                    edgecolors="none",
                )

        addMark(markers["sell"], "^", "orange")
        addMark(markers["buy"], "v", "yellow")
        addMark(markers["wSell"], "o", "orange")
        addMark(markers["wBuy"], "o", "yellow")
        addMark(markers["wLockSell"], "P", "magenta", size=65)
        addMark(markers["wCrabCap"], "X", "cyan", size=60)
        addMark(markers["wSeedBuy"], "*", "white", size=90)

        if showSecondary:
            filtAx.plot(
                idx,
                df["close"],
                color=CLOSE_COLOR,
                linewidth=1.0,
                alpha=0.85,
            )
            filtMarks = self._marker_series(df, self.secondaryMarkers)
            addMark(filtMarks["buy"], "^", "red", size=55, ax=filtAx)
            addMark(
                filtMarks["sell"],
                "v",
                "green",
                size=55,
                ax=filtAx,
            )

        if not self.hideGrads:
            for t0, t1, color in self.gradSlowHeat or []:
                gradSlowAx.axvspan(
                    t0,
                    t1,
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                    zorder=0.5,
                )
        if not self.hideGrads:
            periods = list(self.grads.keys())
            slowLabel = None
            if isinstance(self.switchInfo, dict):
                slowLabel = self.switchInfo.get("GRAD_SLOW_LABEL")
            if periods:
                p3 = periods[-1]
                labelSlow = slowLabel if slowLabel else f"g1(p{p3})"
                slowGrad = self.grads.get(p3, {})
                seriesSlow = seriesLike(slowGrad.get("grad1", []), idx)
                carryVals = slowGrad.get("carry")
            else:
                labelSlow = slowLabel if slowLabel else "g1"
                seriesSlow = pd.Series(np.nan, index=idx)
                carryVals = None
            labelCarry = None
            seriesCarry = None
            if carryVals is not None:
                labelCarry = self.switchInfo.get(
                    "GRAD_CARRY_LABEL",
                    "macro dyn carry%",
                )
                seriesCarry = seriesLike(carryVals, idx)
            gradSlowAx.plot(
                idx,
                seriesSlow,
                color="steelblue",
                linewidth=0.9,
                label=labelSlow,
            )
            if seriesCarry is not None:
                gradSlowAx.plot(
                    idx,
                    seriesCarry,
                    color="darkorange",
                    linewidth=0.9,
                    alpha=0.9,
                    label=labelCarry,
                )
            gradSlowAx.set_ylabel(labelSlow, color=TEXT_COLOR)
            if seriesCarry is not None:
                legSlow = gradSlowAx.legend(
                    loc="best",
                    frameon=True,
                    fontsize=8,
                )
                if legSlow:
                    for txt in legSlow.get_texts():
                        txt.set_color(TEXT_COLOR)
                    frame = legSlow.get_frame()
                    if frame:
                        frame.set_facecolor(BG_COLOR)
                        frame.set_edgecolor(GRID_COLOR)

        if clusterAx is not None:
            rows = []
            labels = []
            if showDailyCluster:
                rows.append(seriesLike(self.dailyClusterIds, idx).to_numpy())
                labels.append("6h")
            clusterVals = np.vstack(rows).astype(float)
            xVals = mdates.date2num(idx.to_pydatetime())
            count = self.dailyClusterCount
            colors = (
                self.dailyClusterColors
                if self.dailyClusterColors else _clusterColors(count)
            )
            cmap = ListedColormap(colors)
            cmap.set_bad(color=BG_COLOR)
            norm = BoundaryNorm(
                np.arange(-0.5, float(count) + 0.5, 1.0),
                cmap.N,
            )
            stripVals = np.ma.masked_less(clusterVals, 0.0)
            clusterAx.imshow(
                stripVals,
                aspect="auto",
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                origin="lower",
                extent=[xVals[0], xVals[-1], 0, len(rows)],
            )
            clusterAx.set_yticks(np.arange(len(labels)) + 0.5)
            clusterAx.set_yticklabels(labels)
            clusterAx.set_ylabel("regime", color=TEXT_COLOR)

        bottomAx = axes[-1]
        bottomAx.xaxis.set_major_formatter(
            mdates.DateFormatter("%d-%b %H:%M")
        )
        bottomAx.tick_params(axis="x", colors=TEXT_COLOR, labelsize=8)
        for ax in axes:
            ax.grid(
                color=GRID_COLOR,
                linestyle=":",
                linewidth=0.6,
                alpha=0.7,
            )
        if clusterAx is not None:
            clusterAx.grid(False)
        def styleLegend(leg):
            if leg:
                for txt in leg.get_texts():
                    txt.set_color(TEXT_COLOR)
                frame = leg.get_frame()
                if frame:
                    frame.set_facecolor(BG_COLOR)
                    frame.set_edgecolor(GRID_COLOR)

        _regimeLegend(
            mainAx,
            self.dailyClusterNames,
            self.dailyClusterColors,
        )
        styleLegend(
            mainAx.legend(
                loc="best",
                frameon=True,
                fontsize=8,
            )
        )
        if macroAx is not None:
            styleLegend(
                macroAx.legend(
                    loc="best",
                    frameon=True,
                    fontsize=8,
                )
            )

        plt.tight_layout(pad=0.8)
        if savePath:
            fig.savefig(savePath, facecolor=BG_COLOR)
        plt.close(fig)


# ======================================================================
# Trace chart orchestration
# ======================================================================


def plotTraceCharts(
    showCharts: bool,
    ctx,
    ts: List[Any],
    startIdx: int,
    flagsTs: List[Tuple[Any, str]],
    walletMarkers: List[Tuple[Any, str]],
    signals: dict | None,
    overrides: dict,
    klines: list,
    ticker: str,
    intervalStr: str,
) -> None:
    """Render trace charts if enabled."""
    if not showCharts:
        return
    cgr = ctx.get("_cgr") or {}
    ov = overrides
    chartsOutDir = os.environ.get("CHARTS_OUT_DIR")
    if chartsOutDir:
        os.makedirs(chartsOutDir, exist_ok=True)
    seq = 1
    daysChunk = float(ov['CHART_CHUNK_SIZE'])
    barsPerDayVal = max(bars_per_day(ctx), 1.0)
    chunk = int(round(daysChunk * barsPerDayVal))
    if chunk <= 0:
        raise ValueError("CHART_CHUNK_SIZE must be > 0")

    sigLoc = signals if signals is not None else buildSignals(ctx, [])
    trendCode = np.asarray(sigLoc["trendCode"], dtype=int)

    macroInterval = str(ov['MACRO_INTERVAL']).strip()
    macroDynFull: np.ndarray | None = None
    macroCarryFull: np.ndarray | None = None
    macroCloseFull: np.ndarray | None = None
    macroMasFull: list[np.ndarray] | None = None
    macroPeriodsUsed: list[int] | None = None
    tsMicro = ts
    if macroInterval:
        meta = ctx.get("_cache") if isinstance(ctx, dict) else None
        baseDays = meta.get("days") if isinstance(meta, dict) else 0
        anchorMs = meta.get("anchorMs") if isinstance(meta, dict) else None
        if not baseDays and isinstance(ctx, dict):
            baseDays = int(ctx.get("days", 0) or 0)
        baseTicker = meta.get("ticker") if isinstance(meta, dict) else ticker
        periodsBase = list(ctx.get("periods", []))
        macro = buildMacroView(
            str(baseTicker),
            int(baseDays),
            0,
            periodsBase,
            ov,
            tsMicro,
            anchorMs=anchorMs,
        )
        if macro is not None:
            macroDynFull = macro.dyn
            macroCarryFull = macroDynCarry(macroDynFull, trendCode)
            macroCloseFull = macro.close
            macroMasFull = macro.mas
            macroPeriodsUsed = macro.periods

    dailyClusterFull = None
    dailyClusterCount = 0
    dailyClusterNames: list[str] = []
    dailyClusterColors: list[str] = []
    if str(ov.get("DAILY_CLUSTER_PATH", "")).strip():
        (
            dailyClusterFull,
            dailyClusterNames,
            dailyClusterColors,
        ) = _dailyPostureLabelsForPlot(ov, ctx)
        dailyClusterCount = len(dailyClusterNames)

    periodsCtx = ctx.get("periods", [])

    for start in range(startIdx, len(ts), chunk):
        end = min(start + chunk, len(ts))
        segment = slice(start, end)
        title = (
            f"{ticker} – {ts[start].date()} → "
            f"{ts[end - 1].date()} (UTC)"
        )
        markerPool = list(flagsTs) + list(walletMarkers)
        markers = [
            m for m in markerPool if ts[start] <= m[0] <= ts[end - 1]
        ]

        masSeg = [m[segment] for m in ctx["mas"]]

        gradsSeg = {}
        if periodsCtx and cgr:
            p1 = periodsCtx[0]
            p3 = periodsCtx[-1]
            gFast = cgr.get(p1, {})
            gFastArr = gFast.get("grad1")
            if isinstance(gFastArr, np.ndarray):
                gradsSeg[p1] = {"grad1": gFastArr[segment]}
            if macroDynFull is not None:
                gradsSeg[p3] = {"grad1": macroDynFull[segment]}
                if macroCarryFull is not None:
                    gradsSeg[p3]["carry"] = macroCarryFull[segment]
                ov['GRAD_FAST_LABEL'] = f"g1(p{p1}) micro"
                ov['GRAD_SLOW_LABEL'] = "macro dyn%"
                ov['GRAD_CARRY_LABEL'] = "macro dyn carry%"
            else:
                gSlow = cgr.get(p3, {})
                gSlowArr = gSlow.get("grad1")
                if isinstance(gSlowArr, np.ndarray):
                    gradsSeg[p3] = {"grad1": gSlowArr[segment]}

        gradSlowHeat: list[tuple[Any, Any, tuple[float, float, float]]] = []
        if macroDynFull is not None:
            pctMax = float(ov['MACRO_DYN_PCT_MAX'])
            if pctMax > 0.0:
                dynSeg = macroDynFull[segment]
                mag = np.abs(dynSeg) / pctMax
                mag = np.clip(mag, 0.0, 1.0)
                for iLocal, mval in enumerate(mag):
                    iAbs = start + iLocal
                    t0 = ts[iAbs]
                    t1 = ts[iAbs + 1] if (iAbs + 1) < len(ts) else ts[iAbs]
                    if mval <= 0.5:
                        t = mval / 0.5 if 0.5 > 0 else 0.0
                        r = 0.0 + t * (1.0 - 0.0)
                        g = 1.0 + t * (0.65 - 1.0)
                        b = 0.0
                    else:
                        t = (mval - 0.5) / 0.5 if 0.5 > 0 else 0.0
                        r = 1.0
                        g = 0.65 + t * (0.0 - 0.65)
                        b = 0.0
                    gradSlowHeat.append((t0, t1, (r, g, b)))

        savePath = None
        if chartsOutDir:
            savePath = os.path.join(chartsOutDir, f"chart-{seq:04d}.png")
            seq += 1
        macroCloseSeg = None
        macroMasSeg = None
        if macroCloseFull is not None and macroMasFull is not None:
            macroCloseSeg = macroCloseFull[segment]
            macroMasSeg = [m[segment] for m in macroMasFull]
        Chart(
            klines=klines[segment],
            ticker=ticker,
            markers=markers,
            mas=masSeg,
            grads=gradsSeg,
            switchInfo=ov,
            gradSlowHeat=gradSlowHeat,
            macroClose=macroCloseSeg,
            macroMas=macroMasSeg,
            macroPeriods=macroPeriodsUsed,
            macroInterval=macroInterval,
            dailyClusterIds=(
                dailyClusterFull[segment]
                if dailyClusterFull is not None else None
            ),
            dailyClusterCount=dailyClusterCount,
            dailyClusterNames=dailyClusterNames,
            dailyClusterColors=dailyClusterColors,
        ).plot(title=title, savePath=savePath)
