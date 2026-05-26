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

matplotlib.use("Agg")
import matplotlib.pyplot as plt


########################################################################
# Constants
########################################################################

DEFAULT_MODELS = [
    "sm_b2_x4_u080_r090_c025_d015_p030",
    "pm_p1_l2_bc055_lk025_cb40_bb015_by055_sy015",
    "pm_p1_l4_bc055_lk035_cb40_bb015_by055_sy015",
    "dsp_step",
    "static_seed",
]

DEFAULT_PARTITIONS = [
    "holdout",
    "tail365d",
    "tail730d",
    "tail1095d",
    "tail1460d",
]

PHASE_COLORS = {
    "ultraRide": "#16a34a",
    "bullChop": "#65a30d",
    "profitLock": "#ca8a04",
    "postUltraCrab": "#2563eb",
    "bearRisk": "#dc2626",
    "flush": "#9333ea",
    "normal": "#64748b",
    "base": "#64748b",
}


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


def _shadePhases(ax, x: np.ndarray, phases: pd.Series) -> None:
    values = phases.astype(str).to_numpy()
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and values[end] == values[start]:
            end += 1
        phase = str(values[start])
        color = PHASE_COLORS.get(phase, "#64748b")
        right = x[end - 1] if end < values.shape[0] else x[-1]
        ax.axvspan(x[start], right, color=color, alpha=0.08, linewidth=0)
        start = end


def _plotPartition(
    rows: pd.DataFrame,
    partition: str,
    models: list[str],
    focusModel: str,
    outDir: Path,
    title: str,
) -> Path | None:
    part = rows[rows["partition"] == partition].copy()
    part = part[part["model"].isin(models)].copy()
    if part.empty:
        return None

    focus = part[part["model"] == focusModel].copy()
    if focus.empty:
        focus = part[part["model"] == str(part["model"].iloc[0])].copy()
        focusModel = str(focus["model"].iloc[0])
    focus = focus.sort_values("openMs")
    x = _dateNums(focus)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(18, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [7, 2, 2, 1]},
    )
    valueAx, exposureAx, signalAx, phaseAx = axes
    _shadePhases(valueAx, x, focus["phase"])

    valueAx.plot(
        x,
        focus["hodlValue"].astype(float),
        "--",
        color="#111827",
        linewidth=1.1,
        label="HODL",
    )
    for i in models:
        use = part[part["model"] == i].sort_values("openMs")
        if use.empty:
            continue
        valueAx.plot(
            _dateNums(use),
            use["value"].astype(float),
            linewidth=1.0,
            label=i,
        )
        if not i.startswith("static"):
            exposureAx.plot(
                _dateNums(use),
                use["exposurePct"].astype(float),
                linewidth=0.8,
                label=i,
            )

    signalAx.plot(
        x,
        focus["targetPct"].astype(float),
        color="#334155",
        linewidth=0.9,
        label=f"{focusModel} target",
    )
    if "parentPreviewProb" in focus.columns:
        signalAx.plot(
            x,
            focus["parentPreviewProb"].astype(float) * 100.0,
            color="#f97316",
            linewidth=0.9,
            label="parent preview prob",
        )

    buyMask = focus["acceptedBuy"].astype(float).to_numpy() > 0.0
    sellMask = focus["acceptedSell"].astype(float).to_numpy() > 0.0
    signalAx.scatter(
        x[buyMask],
        np.full(int(buyMask.sum()), 104.0),
        marker="^",
        s=24,
        color="#15803d",
        label="DSP buy",
    )
    signalAx.scatter(
        x[sellMask],
        np.full(int(sellMask.sum()), -4.0),
        marker="v",
        s=24,
        color="#b91c1c",
        label="DSP sell",
    )

    phaseCodes = pd.Categorical(focus["phase"].astype(str))
    phaseAx.imshow(
        phaseCodes.codes.reshape(1, -1),
        aspect="auto",
        interpolation="nearest",
        extent=[x[0], x[-1], 0, 1],
        cmap="tab10",
    )
    phaseAx.set_yticks([])
    phaseAx.set_ylabel("phase")

    valueAx.set_title(f"{title} {partition}")
    valueAx.set_ylabel("value")
    exposureAx.set_ylabel("exposure %")
    signalAx.set_ylabel("signal %")
    signalAx.set_ylim(-8, 108)
    for ax in axes[:3]:
        ax.grid(True, alpha=0.18)
        ax.legend(loc="upper left", ncols=3, fontsize=7)
    valueAx.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()

    outDir.mkdir(parents=True, exist_ok=True)
    path = outDir / f"state_machine_{partition}.png"
    tmp = path.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=140)
    plt.close(fig)
    os.replace(tmp, path)
    return path


########################################################################
# Public API
########################################################################

def renderCharts(
    timeValsPath: Path,
    outDir: Path,
    models: list[str],
    partitions: list[str],
    focusModel: str,
    title: str,
) -> list[Path]:
    rows = pd.read_csv(timeValsPath)
    paths: list[Path] = []
    for i in partitions:
        path = _plotPartition(rows, i, models, focusModel, outDir, title)
        if path is not None:
            paths.append(path)
    return paths


########################################################################
# CLI
########################################################################

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="render_state_machine_charts",
        description="Render state-machine timeVals PNGs.",
    )
    parser.add_argument("--timevals", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", default="")
    parser.add_argument("--partitions", default="")
    parser.add_argument("--focus-model", default=DEFAULT_MODELS[1])
    parser.add_argument("--title", default="state machine")
    args = parser.parse_args()

    paths = renderCharts(
        Path(args.timevals),
        Path(args.out),
        _parseList(args.models, DEFAULT_MODELS),
        _parseList(args.partitions, DEFAULT_PARTITIONS),
        str(args.focus_model),
        str(args.title),
    )
    for i in paths:
        print(i)


if __name__ == "__main__":
    main()
