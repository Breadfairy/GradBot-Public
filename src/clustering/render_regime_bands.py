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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


########################################################################
# Constants
########################################################################

DEFAULT_PARTITIONS = [
    "all",
    "holdout",
    "tail365d",
    "tail730d",
    "tail1095d",
    "tail1460d",
]

STATE_COLORS = {
    "ultraBull": "#16a34a",
    "bullChop": "#84cc16",
    "crab": "#64748b",
    "crab_c1": "#94a3b8",
    "crab_c2": "#475569",
    "crab_post": "#2563eb",
    "bear": "#dc2626",
    "flush": "#9333ea",
    "lock": "#ca8a04",
}

STATE_ORDER = {
    "bear": 0,
    "flush": 1,
    "crab": 2,
    "crab_c1": 3,
    "crab_c2": 4,
    "crab_post": 5,
    "lock": 6,
    "bullChop": 7,
    "ultraBull": 8,
}

DAY_MS = 86_400_000
YEAR_DAYS = 365.25
PUMP_RET_PCT = 12.0
PUMP_BARS = 4
BEAR_DROP_PCT = -12.0
BEAR_BARS = 8
DEFAULT_MIN_RUN_BARS = 6


########################################################################
# Helpers
########################################################################

def _parseList(value: str, default: list[str]) -> list[str]:
    if value:
        return [i.strip() for i in value.split(",") if i.strip()]
    return default


def _dateNums(rows: pd.DataFrame) -> np.ndarray:
    ts = pd.to_datetime(rows["openMs"], unit="ms", utc=True)
    return mdates.date2num(ts.dt.tz_convert(None).to_numpy())


def _bandState(rows: pd.DataFrame, splitCrab: bool) -> pd.Series:
    phase = rows["phase"].astype(str)
    eff = rows["effectiveRegime"].astype(str)
    out = np.full(int(rows.shape[0]), "crab", dtype=object)
    out[phase.eq("ultraRide").to_numpy()] = "ultraBull"
    out[phase.eq("bullChop").to_numpy()] = "bullChop"
    out[phase.eq("profitLock").to_numpy()] = "lock"
    out[phase.eq("postUltraCrab").to_numpy()] = "crab"
    out[phase.eq("bearRisk").to_numpy()] = "bear"
    out[phase.eq("flush").to_numpy()] = "flush"
    out[eff.eq("bearRisk").to_numpy()] = "bear"
    if splitCrab:
        crabMask = out == "crab"
        cluster = rows["regimeCluster"].astype(int).to_numpy()
        out[crabMask & (cluster == 1)] = "crab_c1"
        out[crabMask & (cluster != 1)] = "crab_c2"
        out[phase.eq("postUltraCrab").to_numpy()] = "crab_post"
    return pd.Series(out, index=rows.index)


def _runIds(values: pd.Series) -> pd.Series:
    return (values != values.shift(1)).cumsum()


def _futureRet(close: pd.Series, bars: int) -> pd.Series:
    future = close.shift(-int(bars))
    return ((future / close) - 1.0) * 100.0


def _stateRuns(rows: pd.DataFrame) -> pd.DataFrame:
    states = rows["bandState"].astype(str)
    runIds = _runIds(states)
    grouped = rows.assign(runId=runIds).groupby("runId", sort=False)
    runs = grouped.agg(
        state=("bandState", "first"),
        startMs=("openMs", "first"),
        endMs=("openMs", "last"),
        bars=("bandState", "size"),
        startClose=("close", "first"),
        endClose=("close", "last"),
    ).reset_index(drop=True)
    runs["durationDays"] = (
        ((runs["endMs"] - runs["startMs"]) / DAY_MS)
        + (6.0 / 24.0)
    )
    runs["retPct"] = ((runs["endClose"] / runs["startClose"]) - 1.0) * 100.0
    return runs


def _smoothStates(states: pd.Series, minRunBars: int) -> pd.Series:
    values = states.astype(str).tolist()
    if int(minRunBars) <= 1 or not values:
        return pd.Series(values, index=states.index)
    changed = True
    while changed:
        changed = False
        out = list(values)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[end] == values[start]:
                end += 1
            length = end - start
            left = values[start - 1] if start > 0 else ""
            right = values[end] if end < len(values) else ""
            if length < int(minRunBars) and left:
                fill = left
                if right and right == left:
                    fill = right
                for i in range(start, end):
                    out[i] = fill
                changed = True
            start = end
        values = out
    return pd.Series(values, index=states.index)


def _causalConfirmStates(states: pd.Series, confirmBars: int) -> pd.Series:
    values = states.astype(str).tolist()
    if int(confirmBars) <= 1 or not values:
        return pd.Series(values, index=states.index)
    active = values[0]
    pending = ""
    pendingBars = 0
    out: list[str] = []
    for raw in values:
        if raw == active:
            pending = ""
            pendingBars = 0
        elif raw == pending:
            pendingBars += 1
        else:
            pending = raw
            pendingBars = 1
        if pending and pendingBars >= int(confirmBars):
            active = pending
            pending = ""
            pendingBars = 0
        out.append(active)
    return pd.Series(out, index=states.index)


def _finalStates(
    rows: pd.DataFrame,
    splitCrab: bool,
    minRunBars: int,
    causalConfirmBars: int,
) -> pd.Series:
    rawState = _bandState(rows, splitCrab)
    if int(causalConfirmBars) > 1:
        return _causalConfirmStates(rawState, causalConfirmBars)
    return _smoothStates(rawState, minRunBars)


def _postUltraGiveback(rows: pd.DataFrame) -> float:
    runs = _stateRuns(rows)
    vals: list[float] = []
    for i in range(int(runs.shape[0]) - 1):
        state = str(runs.loc[i, "state"])
        nextState = str(runs.loc[i + 1, "state"])
        endMs = int(runs.loc[i, "endMs"])
        if state not in {"ultraBull", "bullChop"}:
            continue
        if nextState in {"ultraBull", "bullChop"}:
            continue
        use = rows[
            (rows["openMs"] > endMs)
            & (rows["openMs"] <= endMs + (30 * DAY_MS))
        ].copy()
        if use.empty:
            continue
        entry = float(runs.loc[i, "endClose"])
        low = float(use["close"].min())
        vals.append(((entry / max(low, 1e-12)) - 1.0) * 100.0)
    if not vals:
        return 0.0
    return float(np.median(vals))


def _metricRows(
    rows: pd.DataFrame,
    model: str,
    partition: str,
    minRunBars: int,
    splitCrab: bool,
    causalConfirmBars: int,
) -> dict[str, object]:
    rows = rows.sort_values("openMs").copy()
    rows["bandState"] = _finalStates(
        rows,
        splitCrab,
        minRunBars,
        causalConfirmBars,
    )
    runs = _stateRuns(rows)
    close = rows["close"].astype(float)
    fwdPump = _futureRet(close, PUMP_BARS)
    fwdBear = _futureRet(close, BEAR_BARS)
    pumpMask = fwdPump >= PUMP_RET_PCT
    bearMask = fwdBear <= BEAR_DROP_PCT
    ultraMask = rows["bandState"].isin(["ultraBull", "bullChop"])
    bearState = rows["bandState"].isin(["bear", "flush"])
    years = max(
        (float(rows["openMs"].max() - rows["openMs"].min()) / DAY_MS)
        / YEAR_DAYS,
        1e-9,
    )
    switches = max(int(runs.shape[0]) - 1, 0)
    falseStarts = runs[
        runs["state"].eq("ultraBull")
        & (runs["retPct"] < 5.0)
        & (runs["bars"] < 8)
    ]
    above = rows["value"].astype(float) > rows["hodlValue"].astype(float)
    exposure = rows["exposurePct"].astype(float) / 100.0
    return {
        "model": model,
        "partition": partition,
        "minRunBars": int(minRunBars),
        "splitCrabClusters": bool(splitCrab),
        "causalConfirmBars": int(causalConfirmBars),
        "rows": int(rows.shape[0]),
        "years": float(years),
        "stateSwitchesPerYear": float(switches / years),
        "medianStateDurationDays": float(runs["durationDays"].median()),
        "ultraBullCapturePct": float(
            ultraMask[pumpMask].mean() * 100.0 if pumpMask.any() else 0.0
        ),
        "bearDrawdownAvoidancePct": float(
            bearState[bearMask].mean() * 100.0 if bearMask.any() else 0.0
        ),
        "falseUltraStarts": int(falseStarts.shape[0]),
        "falseUltraStartsPerYear": float(falseStarts.shape[0] / years),
        "missedPumpBars": int((pumpMask & ~ultraMask).sum()),
        "pumpBars": int(pumpMask.sum()),
        "postUltraGivebackPct": _postUltraGiveback(rows),
        "timeAboveHodlPct": float(above.mean() * 100.0),
        "timeAboveHodlExposureAdjustedPct": float(
            (above.astype(float) * exposure).sum()
            / max(exposure.sum(), 1e-12)
            * 100.0
        ),
    }


def _shadeBands(ax, x: np.ndarray, states: pd.Series) -> None:
    values = states.astype(str).to_numpy()
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and values[end] == values[start]:
            end += 1
        state = str(values[start])
        color = STATE_COLORS.get(state, "#64748b")
        right = x[end - 1] if end < values.shape[0] else x[-1]
        ax.axvspan(x[start], right, color=color, alpha=0.18, linewidth=0)
        start = end


def _plotPartition(
    rows: pd.DataFrame,
    model: str,
    partition: str,
    outDir: Path,
    title: str,
    minRunBars: int,
    splitCrab: bool,
    holdoutOpenMs: int,
    causalConfirmBars: int,
) -> Path | None:
    if partition == "all":
        part = rows[rows["model"].eq(model)].copy()
        part = part.drop_duplicates("openMs", keep="last")
    else:
        part = rows[
            rows["model"].eq(model)
            & rows["partition"].eq(partition)
        ].copy()
    if part.empty:
        return None
    part = part.sort_values("openMs").reset_index(drop=True)
    part["bandState"] = _finalStates(
        part,
        splitCrab,
        minRunBars,
        causalConfirmBars,
    )
    x = _dateNums(part)
    stateCodes = part["bandState"].map(STATE_ORDER).astype(int).to_numpy()

    fig, (priceAx, stateAx) = plt.subplots(
        2,
        1,
        figsize=(18, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [8, 1]},
    )
    _shadeBands(priceAx, x, part["bandState"])
    priceAx.plot(
        x,
        part["close"].astype(float),
        color="#111827",
        linewidth=1.1,
        label="close",
    )
    change = part["bandState"] != part["bandState"].shift(1)
    change.iloc[0] = False
    for _, row in part[change].iterrows():
        state = str(row["bandState"])
        priceAx.axvline(
            mdates.date2num(
                pd.to_datetime(row["openMs"], unit="ms", utc=True)
                .tz_convert(None)
                .to_pydatetime()
            ),
            color=STATE_COLORS.get(state, "#64748b"),
            linewidth=0.7,
            alpha=0.85,
        )
    if holdoutOpenMs > 0:
        holdoutX = mdates.date2num(
            pd.to_datetime(holdoutOpenMs, unit="ms", utc=True)
            .tz_convert(None)
            .to_pydatetime()
        )
        priceAx.axvline(
            holdoutX,
            color="#111827",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
        )
        priceAx.text(
            holdoutX,
            float(part["close"].max()),
            " holdout",
            fontsize=8,
            va="top",
            ha="left",
            color="#111827",
        )

    stateAx.imshow(
        stateCodes.reshape(1, -1),
        aspect="auto",
        interpolation="nearest",
        extent=[x[0], x[-1], 0, 1],
        cmap=ListedColormap(
            [STATE_COLORS[i] for i in sorted(STATE_ORDER, key=STATE_ORDER.get)]
        ),
        vmin=0,
        vmax=max(STATE_ORDER.values()),
    )
    stateAx.set_yticks([])
    stateAx.set_ylabel("state")
    mode = f"confirm={int(causalConfirmBars)}"
    if int(causalConfirmBars) <= 1:
        mode = f"minRun={int(minRunBars)}"
    priceAx.set_title(f"{title} {partition} {mode}")
    priceAx.set_ylabel("close")
    priceAx.grid(True, alpha=0.18)
    visible = set(part["bandState"].astype(str).unique())
    handles = [
        Patch(facecolor=color, alpha=0.35, label=state)
        for state, color in STATE_COLORS.items()
        if state in visible
    ]
    priceAx.legend(handles=handles, loc="upper left", ncols=6, fontsize=8)
    priceAx.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()

    outDir.mkdir(parents=True, exist_ok=True)
    path = outDir / f"regime_bands_{partition}.png"
    tmp = path.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=140)
    plt.close(fig)
    os.replace(tmp, path)
    return path


########################################################################
# Public API
########################################################################

def renderBands(
    timeValsPath: Path,
    outDir: Path,
    model: str,
    partitions: list[str],
    title: str,
    minRunBars: int,
    splitCrab: bool,
    holdoutOpenMs: int,
    causalConfirmBars: int,
) -> list[Path]:
    rows = pd.read_csv(timeValsPath)
    paths: list[Path] = []
    metrics: list[dict[str, object]] = []
    for i in partitions:
        if i == "all":
            part = rows[rows["model"].eq(model)].drop_duplicates(
                "openMs",
                keep="last",
            ).copy()
        else:
            part = rows[
                rows["model"].eq(model) & rows["partition"].eq(i)
            ].copy()
        if part.empty:
            continue
        metrics.append(
            _metricRows(
                part,
                model,
                i,
                minRunBars,
                splitCrab,
                causalConfirmBars,
            )
        )
        path = _plotPartition(
            rows,
            model,
            i,
            outDir,
            title,
            minRunBars,
            splitCrab,
            holdoutOpenMs,
            causalConfirmBars,
        )
        if path is not None:
            paths.append(path)
    metricFrame = pd.DataFrame(metrics)
    tmp = outDir / "regime_band_metrics.csv.tmp"
    outDir.mkdir(parents=True, exist_ok=True)
    metricFrame.to_csv(tmp, index=False)
    os.replace(tmp, outDir / "regime_band_metrics.csv")
    return paths


########################################################################
# CLI
########################################################################

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="render_regime_bands",
        description="Render close-price regime bands from state timeVals.",
    )
    parser.add_argument("--timevals", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--partitions", default="")
    parser.add_argument("--min-run-bars", type=int, default=DEFAULT_MIN_RUN_BARS)
    parser.add_argument("--causal-confirm-bars", type=int, default=0)
    parser.add_argument("--split-crab-clusters", action="store_true")
    parser.add_argument("--holdout-open-ms", type=int, default=0)
    parser.add_argument("--title", default="regime bands")
    args = parser.parse_args()
    paths = renderBands(
        Path(args.timevals),
        Path(args.out),
        str(args.model),
        _parseList(str(args.partitions), DEFAULT_PARTITIONS),
        str(args.title),
        int(args.min_run_bars),
        bool(args.split_crab_clusters),
        int(args.holdout_open_ms),
        int(args.causal_confirm_bars),
    )
    for i in paths:
        print(i)


if __name__ == "__main__":
    main()
