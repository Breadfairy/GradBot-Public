#!/usr/bin/env python3
# charting.py – plotting helpers (closes line only).

import os
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless-safe for batch chart saving
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from engine_shared import (
    buildSignals,
    buildContext,
    spacingState,
    zscoreSeries,
    bars_per_day,
)
from dynamics import macroDynFromContext, alignMacroDyn
from cache import getKlinesCached


# RGB-style colors tuned for closes mode (0–1 floats).
BG_COLOR = [0.08, 0.03, 0.03]
TEXT_COLOR = [0.98, 0.95, 0.82]
CLOSE_COLOR = [0.98, 0.98, 0.98]
GRID_COLOR = [0.45, 0.40, 0.35]
ENERGY_FILL_COLOR = [0.5, 0.05, 0.05]
CHART_CHUNK_SIZE = 30


def seriesLike(arr, index):
    """Return Series aligned to index; pad/trim so size always matches."""
    a = np.asarray(arr, dtype=float)
    n = len(index)
    if a.size < n:
        a = np.pad(a, (0, n - a.size), constant_values=np.nan)
    elif a.size > n:
        a = a[:n]
    return pd.Series(a, index=index)


def plotBellCurve(values, title, savePath):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 2:
        return
    mu = float(arr.mean())
    sigma = float(arr.std(ddof=0))
    if sigma <= 0.0:
        return

    fig, ax = plt.subplots(figsize=(8, 4.0))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)

    titleStr = str(title)
    titleKey = titleStr.upper()
    span = 4.0 * sigma
    xMin = mu - span
    xMax = mu + span
    isZscore = (
        "ZSCORE" in titleKey
        or "MACRO_DYN_Z" in titleKey
    )
    isPct = (
        "PCT" in titleKey
        or "PERCENT" in titleKey
    )
    if isZscore:
        xMin = -1.0
        xMax = 10.0
    elif isPct:
        xMin = -1.0
        xMax = 30.0

    binWidth = 0.25
    binEdges = np.arange(xMin, xMax + binWidth, binWidth)
    if binEdges.size < 2:
        binEdges = 40

    counts, binEdges, _ = ax.hist(
        arr,
        bins=binEdges,
        density=False,
        color=CLOSE_COLOR,
        alpha=0.55,
        edgecolor=GRID_COLOR,
        linewidth=0.6,
    )

    x = np.linspace(xMin, xMax, 400)
    norm = (1.0 / (sigma * np.sqrt(2.0 * np.pi)))
    yPdf = norm * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    binWidth = float(binEdges[1] - binEdges[0]) if binEdges.size > 1 else 1.0
    y = yPdf * float(n) * binWidth
    ax.plot(x, y, color="deepskyblue", linewidth=1.2)
    ax.set_xlim(xMin, xMax)
    if isZscore:
        ticks = np.arange(
            np.floor(xMin),
            np.ceil(xMax) + 1.0,
            1.0,
        )
        ax.set_xticks(ticks)

    ax.set_title(title, color=TEXT_COLOR, pad=8)
    ax.set_xlabel("Value", color=TEXT_COLOR)
    ax.set_ylabel("Count", color=TEXT_COLOR)
    ax.grid(
        color=GRID_COLOR,
        linestyle=":",
        linewidth=0.6,
        alpha=0.7,
    )
    fig.tight_layout(pad=0.6)
    if savePath:
        fig.savefig(savePath, facecolor=BG_COLOR)
    plt.close(fig)


def plotTimVal(
    ts: List[Any],
    edgeVals: np.ndarray,
    hodlVals: np.ndarray,
    title: str,
    savePath: str,
) -> None:
    x = list(ts)
    edge = np.asarray(edgeVals, dtype=float)
    hodl = np.asarray(hodlVals, dtype=float)
    if edge.size == 0 or hodl.size == 0 or len(x) == 0:
        return

    n = min(len(x), int(edge.size), int(hodl.size))
    x = x[:n]
    edge = edge[:n]
    hodl = hodl[:n]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)

    ax.set_title(title, color=TEXT_COLOR, pad=10)
    ax.set_ylabel("Gross Value", color=TEXT_COLOR)

    ax.plot(x, edge, color="orange", linewidth=1.2, label="EDGE")
    ax.plot(x, hodl, color="deepskyblue", linewidth=1.2, label="HODL")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    ax.grid(
        color=GRID_COLOR,
        linestyle=":",
        linewidth=0.6,
        alpha=0.7,
    )

    leg = ax.legend(loc="best", frameon=True, fontsize=8)
    if leg:
        for txt in leg.get_texts():
            txt.set_color(TEXT_COLOR)
        frame = leg.get_frame()
        if frame:
            frame.set_facecolor(BG_COLOR)
            frame.set_edgecolor(GRID_COLOR)

    fig.tight_layout(pad=0.8)
    if savePath:
        fig.savefig(savePath, facecolor=BG_COLOR)
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
        energySpans=None,
        gradFastSpans=None,
        gradSlowHeat=None,
        macroClose=None,
        macroMas=None,
        macroPeriods=None,
        macroInterval=None,
    ):
        self.klines = klines
        self.ticker = ticker
        self.markers = markers
        self.mas = mas
        self.grads = grads
        self.switchInfo = switchInfo or {}
        self.secondaryMarkers = secondaryMarkers
        self.energySpans = energySpans or []
        self.gradFastSpans = gradFastSpans or []
        self.gradSlowHeat = gradSlowHeat or []
        self.macroClose = macroClose
        self.macroMas = macroMas
        self.macroPeriods = macroPeriods
        self.macroInterval = macroInterval
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
        mSell = pd.Series(np.nan, index=idx)
        oSell = pd.Series(np.nan, index=idx)
        buy = pd.Series(np.nan, index=idx)
        mBuy = pd.Series(np.nan, index=idx)
        oBuy = pd.Series(np.nan, index=idx)
        wSell = pd.Series(np.nan, index=idx)
        wBuy = pd.Series(np.nan, index=idx)
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
                elif fl == "M_SELL":
                    mSell.iloc[pos] = high * 1.02
                elif fl == "O_SELL":
                    oSell.iloc[pos] = high * 1.04
                elif fl == "BUY":
                    buy.iloc[pos] = low * 0.98
                elif fl == "M_BUY":
                    mBuy.iloc[pos] = low * 0.98
                elif fl == "O_BUY":
                    oBuy.iloc[pos] = low * 0.96
                elif fl == "W_SELL":
                    wSell.iloc[pos] = high * 1.03
                elif fl == "W_BUY":
                    wBuy.iloc[pos] = low * 0.97
        return {
            "sell": sell,
            "mSell": mSell,
            "oSell": oSell,
            "buy": buy,
            "mBuy": mBuy,
            "oBuy": oBuy,
            "wSell": wSell,
            "wBuy": wBuy,
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
        if self.hideGrads:
            if showSecondary:
                nRows = 3 if showMacro else 2
                ratios = [3.0, 1.7, 1.7] if showMacro else [3.0, 1.7]
                figHeight = 8.0 if showMacro else 6.5
                fig, axes = plt.subplots(
                    nRows,
                    1,
                    sharex=True,
                    figsize=(12, figHeight),
                    gridspec_kw={"height_ratios": ratios},
                )
                mainAx = axes[0]
                macroAx = axes[1] if showMacro else None
                filtAx = axes[-1]
            else:
                if showMacro:
                    fig, axes = plt.subplots(
                        2,
                        1,
                        sharex=True,
                        figsize=(12, 6.0),
                        gridspec_kw={
                            "height_ratios": [3.0, 1.7],
                        },
                    )
                    mainAx, macroAx = axes
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
                fig, axes = plt.subplots(
                    3,
                    1,
                    sharex=True,
                    figsize=(12, 7.2),
                    gridspec_kw={
                        "height_ratios": [3.0, 1.7, 1.1],
                    },
                )
                mainAx, macroAx, gradSlowAx = axes
            else:
                fig, axes = plt.subplots(
                    2,
                    1,
                    sharex=True,
                    figsize=(12, 5.6),
                    gridspec_kw={"height_ratios": [3.0, 1.1]},
                )
                mainAx, gradSlowAx = axes
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

        addMark(markers["oBuy"], "^", "red")
        addMark(markers["oSell"], "v", "green")
        addMark(markers["sell"], "^", "orange")
        addMark(markers["buy"], "v", "yellow")
        addMark(markers["mSell"], "^", "deepskyblue")
        addMark(markers["mBuy"], "v", "deepskyblue")
        addMark(markers["wSell"], "o", "orange")
        addMark(markers["wBuy"], "o", "yellow")

        if showSecondary:
            filtAx.plot(
                idx,
                df["close"],
                color=CLOSE_COLOR,
                linewidth=1.0,
                alpha=0.85,
            )
            filtMarks = self._marker_series(df, self.secondaryMarkers)
            addMark(filtMarks["oBuy"], "^", "red", size=55, ax=filtAx)
            addMark(
                filtMarks["oSell"],
                "v",
                "green",
                size=55,
                ax=filtAx,
            )

        if not self.hideGrads:
            periods = list(self.grads.keys())
            slowLabel = None
            if isinstance(self.switchInfo, dict):
                slowLabel = self.switchInfo.get("GRAD_SLOW_LABEL")
            if periods:
                p3 = periods[-1]
                labelSlow = slowLabel if slowLabel else f"g1(p{p3})"
                seriesSlow = seriesLike(
                    self.grads.get(p3, {}).get("grad1", []), idx
                )
            else:
                labelSlow = slowLabel if slowLabel else "g1"
                seriesSlow = pd.Series(np.nan, index=idx)
            gradSlowAx.plot(idx, seriesSlow, color="steelblue", linewidth=0.9)
            gradSlowAx.set_ylabel(labelSlow, color=TEXT_COLOR)

        for (t0, t1) in self.energySpans or []:
            mainAx.axvspan(
                t0,
                t1,
                color=ENERGY_FILL_COLOR,
                alpha=0.4,
                linewidth=0,
                zorder=0.5,
            )

        if not self.hideGrads:
            for (t0, t1, color) in self.gradSlowHeat or []:
                gradSlowAx.axvspan(
                    t0,
                    t1,
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                    zorder=0.5,
                )

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
        def styleLegend(leg):
            if leg:
                for txt in leg.get_texts():
                    txt.set_color(TEXT_COLOR)
                frame = leg.get_frame()
                if frame:
                    frame.set_facecolor(BG_COLOR)
                    frame.set_edgecolor(GRID_COLOR)

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
# Backtest chart orchestration
# ======================================================================


def plotBacktestCharts(
    showCharts: bool,
    ctx,
    ts: List[Any],
    startIdx: int,
    flagsTs: List[Tuple[Any, str]],
    walletMarkers: List[Tuple[Any, str]],
    oracles: List[Tuple[Any, str]],
    signals: dict | None,
    overrides: dict,
    klines: list,
    ticker: str,
    intervalStr: str,
) -> None:
    """Render backtest charts if enabled."""
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
    trend = np.asarray(sigLoc["trendCode"], dtype=int)
    allowBuy = (trend == -1)
    allowSell = (trend == 1)
    s12 = np.asarray(sigLoc["s12"], dtype=float)
    s23 = np.asarray(sigLoc["s23"], dtype=float)
    buyEsp = spacingState(ctx, trend, allowBuy, s12, s23, ov)
    sellEsp = spacingState(ctx, trend, allowSell, s12, s23, ov)
    energyActive = (allowBuy & ~buyEsp.energyMask) | (
        allowSell & ~sellEsp.energyMask
    )

    macroInterval = str(ov['MACRO_INTERVAL']).strip()
    macroDynFull: np.ndarray | None = None
    macroCloseFull: np.ndarray | None = None
    macroMasFull: list[np.ndarray] | None = None
    macroPeriodsUsed: list[int] | None = None
    tsMicro = np.array(
        [t.timestamp() * 1000.0 for t in ts],
        dtype=float,
    )
    if macroInterval:
        meta = ctx.get("_cache") if isinstance(ctx, dict) else None
        baseDays = meta.get("days") if isinstance(meta, dict) else 0
        baseTicker = meta.get("ticker") if isinstance(meta, dict) else ticker
        periodsBase = ctx.get("periods", [])
        macroP1 = ov['MACRO_P1']
        macroP2 = ov['MACRO_P2']
        macroP3 = ov['MACRO_P3']
        periodsMacro = list(periodsBase) if periodsBase else []
        if macroP1 is not None and int(macroP1) > 0 and len(periodsMacro) >= 1:
            periodsMacro[0] = int(macroP1)
        if macroP2 is not None and int(macroP2) > 0 and len(periodsMacro) >= 2:
            periodsMacro[1] = int(macroP2)
        if macroP3 is not None and int(macroP3) > 0:
            p3Macro = int(macroP3)
            if len(periodsMacro) >= 3:
                periodsMacro[2] = p3Macro
            else:
                periodsMacro.append(p3Macro)
        if not periodsMacro:
            periodsMacro = [1]
        macroPeriodsUsed = list(periodsMacro)
        minCandles = max(int(max(periodsMacro) * 2 + 1), 1)
        klMacro = getKlinesCached(
            str(baseTicker),
            str(macroInterval),
            int(baseDays),
            minCandles,
            holdoutDays=0,
        )
        winDays = float(ov['MACRO_NRG_WIN_DAYS'])
        zmin = float(ov['MACRO_NRG_Z_MIN'])
        zmax = float(ov['MACRO_NRG_Z_MAX'])
        pctMax = float(ov['MACRO_DYN_PCT_MAX'])
        ctxMacroFull = buildContext(klMacro, periodsMacro)
        ctxMacroFull["intervalStr"] = str(macroInterval)
        tsMacro = np.array([k[0] for k in klMacro], dtype=float)
        macroCloseFull = alignMacroDyn(
            tsMacro, np.asarray(ctxMacroFull["closes"], dtype=float), tsMicro
        )
        macroMasFull = [
            alignMacroDyn(tsMacro, np.asarray(ma, dtype=float), tsMicro)
            for ma in ctxMacroFull["mas"][:3]
        ]

        gradWinDays = float(ov['MACRO_GRAD_WIN_DAYS'])
        gradZMin = float(ov['MACRO_GRAD_Z_MIN'])
        gradZMax = float(ov['MACRO_GRAD_Z_MAX'])
        gradMultMin = float(ov['MACRO_MULT_GRAD_MIN'])
        gradMultMax = float(ov['MACRO_MULT_GRAD_MAX'])
        pctMin = float(ov['MACRO_DYN_PCT_MIN'])
        dynMacro = macroDynFromContext(
            ctxMacroFull,
            winDays,
            zmin,
            zmax,
            pctMax,
            pctMin,
            gradWinDays=gradWinDays,
            gradZMin=gradZMin,
            gradZMax=gradZMax,
            gradMultMin=gradMultMin,
            gradMultMax=gradMultMax,
        )
        macroDynFull = alignMacroDyn(tsMacro, dynMacro, tsMicro)

    periodsCtx = ctx.get("periods", [])
    g1All: np.ndarray | None = None
    if "g1P1" in sigLoc:
        g1All = np.asarray(sigLoc["g1P1"], dtype=float)
    nBars = len(sigLoc["trendCode"])
    blockedGradMask = np.zeros(nBars, dtype=bool)
    if periodsCtx and g1All is not None:
        barsPerDayVal = max(bars_per_day(ctx), 1.0)
        winBuyDays = float(ov['GRAD1_BUY_WIN_DAYS'])
        threshBuy = float(ov['GRAD1_BUY_Z_MIN'])
        winBuyBars = max(int(round(winBuyDays * barsPerDayVal)), 1)
        zBuy, validBuy = zscoreSeries(
            ctx,
            g1All,
            winBuyBars,
            "g1p1",
        )
        signedBuy = -zBuy
        readyBuy = allowBuy & validBuy
        gateBuyOpen = readyBuy & (signedBuy >= threshBuy)
        gateBuyBlocked = readyBuy & ~gateBuyOpen

        winSellDays = float(ov['GRAD1_SELL_WIN_DAYS'])
        threshSell = float(ov['GRAD1_SELL_Z_MIN'])
        winSellBars = max(int(round(winSellDays * barsPerDayVal)), 1)
        zSell, validSell = zscoreSeries(
            ctx,
            g1All,
            winSellBars,
            "g1p1",
        )
        signedSell = zSell
        readySell = allowSell & validSell
        gateSellOpen = readySell & (signedSell >= threshSell)
        gateSellBlocked = readySell & ~gateSellOpen

        blockedGradMask = gateBuyBlocked | gateSellBlocked

    for start in range(startIdx, len(ts), chunk):
        end = min(start + chunk, len(ts))
        segment = slice(start, end)
        title = (
            f"{ticker} – {ts[start].date()} → "
            f"{ts[end - 1].date()} (UTC)"
        )
        markerPool = list(flagsTs) + list(walletMarkers) + list(oracles)
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
                ov['GRAD_FAST_LABEL'] = f"g1(p{p1}) micro"
                ov['GRAD_SLOW_LABEL'] = "macro dyn%"
            else:
                gSlow = cgr.get(p3, {})
                gSlowArr = gSlow.get("grad1")
                if isinstance(gSlowArr, np.ndarray):
                    gradsSeg[p3] = {"grad1": gSlowArr[segment]}

        spans = []
        blk = energyActive[segment].astype(int)
        if blk.size > 0:
            pad = np.pad(blk, (1, 1))
            diff = np.diff(pad)
            starts = np.flatnonzero(diff == 1)
            ends = np.flatnonzero(diff == -1)
            for i0, i1 in zip(starts, ends):
                a = start + max(0, int(i0))
                b = start + max(0, int(i1) - 1)
                if a < 0 or b < 0 or a >= len(ts) or b >= len(ts):
                    continue
                if a > b:
                    continue
                idxEnd = b + 1 if (b + 1) < len(ts) else b
                spans.append((ts[a], ts[idxEnd]))

        gradFastSpans: list[tuple[Any, Any]] = []
        blkSegment = blockedGradMask[segment].astype(int)
        if blkSegment.size > 0:
            padM = np.pad(blkSegment, (1, 1))
            diffM = np.diff(padM)
            startsM = np.flatnonzero(diffM == 1)
            endsM = np.flatnonzero(diffM == -1)
            for i0, i1 in zip(startsM, endsM):
                a = start + max(0, int(i0))
                b = start + max(0, int(i1) - 1)
                if a < 0 or b < 0 or a >= len(ts) or b >= len(ts):
                    continue
                if a > b:
                    continue
                idxEndM = b + 1 if (b + 1) < len(ts) else b
                gradFastSpans.append((ts[a], ts[idxEndM]))

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
            energySpans=spans,
            gradFastSpans=gradFastSpans,
            gradSlowHeat=gradSlowHeat,
            macroClose=macroCloseSeg,
            macroMas=macroMasSeg,
            macroPeriods=macroPeriodsUsed,
            macroInterval=macroInterval,
        ).plot(title=title, savePath=savePath)
