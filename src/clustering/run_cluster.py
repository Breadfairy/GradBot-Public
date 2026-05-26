#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.colors import BoundaryNorm, ListedColormap

from clustering.cluster_config import PeriodConfig
from clustering.cluster_config import loadConfig
from clustering.feature_families import (
    featureIds,
    featureNames,
    familyNames,
    manifestRows,
)
from clustering.features import (
    addForwardReturns,
    engineFeatures,
    exploreFeatures,
    featureColumns,
)


from repo_paths import CLUSTERING_INPUT_DIR, CLUSTERING_OUTPUT_DIR


DEFAULT_CONFIG = str(CLUSTERING_INPUT_DIR / "linkusdt-1d-regime-60d.json")

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


def _clusterColors(clusterCount: int) -> list[str]:
    colors: list[str] = []
    clusterCount = int(clusterCount)
    while len(colors) < clusterCount:
        colors.extend(CLUSTER_COLORS)
    return colors[:clusterCount]


def _writeFrame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _writeRows(
    path: Path,
    header: list[str],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp, path)


def _writeText(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _readRows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _float(row: dict[str, str], key: str) -> float:
    raw = str(row.get(key, "")).strip()
    return float(raw) if raw else float("nan")


def _int(row: dict[str, str], key: str) -> int:
    raw = str(row.get(key, "")).strip()
    return int(raw) if raw else 0


def _buildView(cfg, view: str) -> pd.DataFrame:
    if view == "engine":
        return engineFeatures(cfg)
    if view == "explore":
        return exploreFeatures(cfg)
    raise ValueError(f"unknown clustering view: {view}")


def _clusterFeatures(
    view: str,
    frame: pd.DataFrame,
    featureFamily: str,
) -> list[str]:
    if view == "engine":
        features = featureNames(featureFamily)
        missing = [
            name for name in features
            if name not in frame.columns
        ]
        if missing:
            raise KeyError(
                "engine clustering missing runtime features: "
                + ", ".join(missing)
            )
        return features
    return featureColumns(frame)


def _fitPredict(
    cfg,
    frame: pd.DataFrame,
    features: list[str],
    clusterCount: int,
    clusterMethod: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import RobustScaler

    out = frame.copy()
    xRaw = out[features].replace([np.inf, -np.inf], np.nan)
    valid = xRaw.notna().all(axis=1).to_numpy()
    validIdx = np.flatnonzero(valid)
    fitCount = max(
        int(validIdx.shape[0] * float(cfg.fitFraction)),
        int(clusterCount),
    )
    fitIdx = validIdx[:fitCount]
    xValid = xRaw.iloc[validIdx].to_numpy(dtype=float)
    xFit = xRaw.iloc[fitIdx].to_numpy(dtype=float)

    scaler = RobustScaler()
    xFitScaled = scaler.fit_transform(xFit)
    xValidScaled = scaler.transform(xValid)
    pcaCount = min(12, int(xFitScaled.shape[1]), int(xFitScaled.shape[0] - 1))
    pca = PCA(n_components=pcaCount, random_state=cfg.randomState)
    xFitModel = pca.fit_transform(xFitScaled)
    xValidModel = pca.transform(xValidScaled)

    if clusterMethod == "kmeans":
        model = KMeans(
            n_clusters=int(clusterCount),
            random_state=cfg.randomState,
            n_init=20,
        )
        model.fit(xFitModel)
        labels = model.predict(xValidModel)
        centers = model.cluster_centers_.astype(float)
        dist = np.linalg.norm(
            xValidModel[:, None, :] - centers[None, :, :],
            axis=2,
        )
        ordered = np.sort(dist, axis=1)
        bestDist = ordered[:, 0]
        secondDist = ordered[:, 1] if int(clusterCount) > 1 else ordered[:, 0]
        confidence = 1.0 - (bestDist / np.maximum(secondDist, 1e-12))
        confidence = np.clip(confidence, 0.0, 1.0)
    elif clusterMethod == "gmm":
        model = GaussianMixture(
            n_components=int(clusterCount),
            covariance_type="full",
            reg_covar=1e-5,
            n_init=5,
            random_state=cfg.randomState,
        )
        model.fit(xFitModel)
        prob = model.predict_proba(xValidModel)
        labels = np.argmax(prob, axis=1)
        centers = model.means_.astype(float)
        ordered = np.sort(prob, axis=1)
        bestProb = ordered[:, -1]
        secondProb = ordered[:, -2] if int(clusterCount) > 1 else 0.0
        confidence = bestProb - secondProb
        bestDist = -np.log(np.maximum(bestProb, 1e-12))
    else:
        raise ValueError(f"unknown cluster method: {clusterMethod}")

    out["cluster"] = -1
    out.loc[out.index[validIdx], "cluster"] = labels.astype(int)
    out["clusterConfidence"] = np.nan
    out.loc[out.index[validIdx], "clusterConfidence"] = confidence
    out["clusterDistance"] = np.nan
    out.loc[out.index[validIdx], "clusterDistance"] = bestDist
    out["partition"] = "warmup"
    out.loc[out.index[fitIdx], "partition"] = "fit"
    holdoutIdx = validIdx[fitCount:]
    out.loc[out.index[holdoutIdx], "partition"] = "holdout"
    modelInfo = {
        "clusterMethod": clusterMethod,
        "features": list(features),
        "fitStartMs": int(out.iloc[fitIdx[0]]["openMs"]),
        "fitEndMs": int(out.iloc[fitIdx[-1]]["closeMs"]),
        "center": scaler.center_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
        "pcaCount": int(pcaCount),
        "pcaMean": pca.mean_.astype(float).tolist(),
        "pcaComponents": pca.components_.astype(float).tolist(),
        "centroids": centers.tolist(),
    }
    return out, modelInfo


def _summaryRows(
    frame: pd.DataFrame,
    forwardBars: list[int],
    partition: str,
    cfg=None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if partition == "policy":
        use = _policyFrame(cfg, frame)
    else:
        use = frame[(frame["cluster"] >= 0) & (frame["partition"] == partition)]
    for cluster in sorted(use["cluster"].dropna().unique().astype(int)):
        part = use[use["cluster"] == cluster]
        row: dict[str, object] = {
            "partition": partition,
            "cluster": int(cluster),
            "rows": int(part.shape[0]),
            "firstOpenMs": int(part["openMs"].iloc[0]),
            "lastOpenMs": int(part["openMs"].iloc[-1]),
        }
        for bars in forwardBars:
            col = f"fwdRet{int(bars)}h"
            vals = part[col].dropna()
            row[f"{col}Mean"] = float(vals.mean())
            row[f"{col}Median"] = float(vals.median())
            row[f"{col}WinRate"] = float((vals > 0.0).mean())
        rows.append(row)
    return rows


def _median(values: pd.Series) -> float:
    vals = values.dropna()
    return float(vals.median()) if int(vals.shape[0]) else float("nan")


def _winRate(values: pd.Series) -> float:
    vals = values.dropna()
    return float((vals > 0.0).mean()) if int(vals.shape[0]) else float("nan")


def _runLengthRows(
    frame: pd.DataFrame,
    targetBars: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    col = f"fwdRet{int(targetBars)}h"
    use = frame[frame["cluster"] >= 0].copy()
    use["prevCluster"] = use["cluster"].shift(1)
    use["nextCluster"] = use["cluster"].shift(-1)
    use["clusterEntry"] = use["cluster"] != use["prevCluster"]
    runId = (use["cluster"] != use["cluster"].shift(1)).cumsum()
    for partition in ["fit", "holdout"]:
        partAll = use[use["partition"] == partition]
        for cluster in sorted(partAll["cluster"].dropna().unique().astype(int)):
            part = partAll[partAll["cluster"] == cluster]
            entry = part[part["clusterEntry"]]
            cont = part[~part["clusterEntry"]]
            lengths = part.groupby(runId).size()
            nextSame = part["nextCluster"] == part["cluster"]
            rows.append(
                {
                    "partition": partition,
                    "cluster": int(cluster),
                    "rows": int(part.shape[0]),
                    "runs": int(lengths.shape[0]),
                    "medianRunBars": float(lengths.median()),
                    "meanRunBars": float(lengths.mean()),
                    "maxRunBars": int(lengths.max()),
                    "selfTransitionPct": float(nextSame.mean() * 100.0),
                    "entryRows": int(entry.shape[0]),
                    "entryMedian": _median(entry[col]),
                    "entryWinRate": _winRate(entry[col]),
                    "continueRows": int(cont.shape[0]),
                    "continueMedian": _median(cont[col]),
                    "continueWinRate": _winRate(cont[col]),
                    "medianConfidence": _median(part["clusterConfidence"]),
                    "medianDistance": _median(part["clusterDistance"]),
                }
            )
    return rows


def _policyMask(cfg, frame: pd.DataFrame) -> pd.Series:
    if cfg.policyStartMs is None or cfg.policyEndMs is None:
        return frame["partition"] == "fit"
    openMs = frame["openMs"].astype(float)
    return (
        (frame["cluster"] >= 0)
        & (openMs >= float(cfg.policyStartMs))
        & (openMs <= float(cfg.policyEndMs))
    )


def _policyFrame(cfg, frame: pd.DataFrame) -> pd.DataFrame:
    return frame[_policyMask(cfg, frame)].copy()


def _featureRows(
    frame: pd.DataFrame,
    features: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    use = frame[frame["cluster"] >= 0]
    for cluster in sorted(use["cluster"].dropna().unique().astype(int)):
        part = use[use["cluster"] == cluster]
        for feature in features:
            vals = part[feature].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "cluster": int(cluster),
                    "feature": feature,
                    "median": float(vals.median()),
                    "mean": float(vals.mean()),
                }
            )
    return rows


def _policyRows(
    cfg,
    frame: pd.DataFrame,
    forwardBars: list[int],
    clusterCount: int,
    flagRows: list[dict[str, object]],
    policyMode: str = "heuristic_forward",
    policyTarget: int = 24,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    col = f"fwdRet{int(policyTarget)}h"
    flagCol = f"{col}Median"
    winCol = f"{col}WinRate"
    use = _policyFrame(cfg, frame)
    for cluster in range(int(clusterCount)):
        part = use[use["cluster"] == cluster]
        vals = part[col].dropna()
        median = float(vals.median())
        mean = float(vals.mean())
        winRate = float((vals > 0.0).mean())
        buyMult = 1.0
        sellMult = 1.0
        label = "neutral"
        flagRowsLocal = []
        if policyMode == "neutral":
            label = "neutral"
        elif (
            policyMode == "heuristic_forward"
            and median >= 1.0
            and winRate >= 0.57
        ):
            buyMult = 0.80
            sellMult = 2.50
            label = "forward_up"
        elif (
            policyMode == "heuristic_forward"
            and median <= -0.35
            and winRate <= 0.46
        ):
            buyMult = 1.25
            sellMult = 0.70
            label = "forward_down"
        elif policyMode == "flag_outcome":
            flagRowsLocal = flagRows
        for flagRow in flagRowsLocal:
            if str(flagRow["partition"]) != "policy":
                continue
            if int(flagRow["cluster"]) != int(cluster):
                continue
            if int(flagRow["flags"]) < 20:
                continue
            flagMedian = float(flagRow[flagCol])
            flagWin = float(flagRow[winCol])
            if (
                str(flagRow["side"]) == "BUY"
                and flagMedian >= 1.0
                and flagWin >= 0.60
            ):
                buyMult = min(buyMult, 0.80)
                if label == "neutral":
                    label = "buy_flag_up"
            if (
                str(flagRow["side"]) == "SELL"
                and flagMedian <= -1.0
                and flagWin <= 0.40
            ):
                sellMult = min(sellMult, 0.70)
                if label == "neutral":
                    label = "sell_flag_down"
            if (
                str(flagRow["side"]) == "SELL"
                and flagMedian >= 1.0
                and flagWin >= 0.55
            ):
                sellMult = max(sellMult, 2.50)
                if label == "neutral":
                    label = "sell_flag_up"
        rows.append(
            {
                "policyMode": policyMode,
                "policyTarget": int(policyTarget),
                "policyStartMs": cfg.policyStartMs,
                "policyEndMs": cfg.policyEndMs,
                "cluster": int(cluster),
                "label": label,
                "rows": int(part.shape[0]),
                "mean": mean,
                "median": median,
                "winRate": winRate,
                "buyReqMult": buyMult,
                "sellReqMult": sellMult,
            }
        )
    return rows


def _flat(values: list[list[float]]) -> list[float]:
    return [float(item) for row in values for item in row]


def _csvFloats(values: list[float]) -> str:
    return ",".join(f"{float(value):.17g}" for value in values)


def _writeModelFiles(
    outDir: Path,
    cfg,
    view: str,
    featureFamily: str,
    policyMode: str,
    policyTarget: int,
    clusterCount: int,
    modelInfo: dict[str, object],
    policyRows: list[dict[str, object]],
) -> tuple[Path, Path]:
    modelDir = outDir / "model"
    jsonPath = modelDir / "cluster_model.json"
    textPath = modelDir / "cluster_model.txt"
    features = list(modelInfo["features"])
    ids = featureIds(features)
    buyMult = [float(row["buyReqMult"]) for row in policyRows]
    sellMult = [float(row["sellReqMult"]) for row in policyRows]
    pcaComponents = modelInfo["pcaComponents"]
    centroids = modelInfo["centroids"]
    pcaCount = int(modelInfo["pcaCount"])
    featureCount = len(features)
    dimCount = pcaCount if pcaCount > 0 else featureCount
    payload = {
        "schema": "gradbot-cluster-model-v1",
        "name": cfg.name,
        "view": view,
        "featureFamily": featureFamily,
        "clusterMethod": modelInfo["clusterMethod"],
        "policyMode": policyMode,
        "policyTarget": int(policyTarget),
        "policyStartMs": cfg.policyStartMs,
        "policyEndMs": cfg.policyEndMs,
        "fitStartMs": int(modelInfo.get("fitStartMs", 0)),
        "fitEndMs": int(modelInfo.get("fitEndMs", 0)),
        "ticker": cfg.ticker,
        "interval": cfg.interval,
        "windowBars": int(cfg.windowBars),
        "periods": {
            "fast": int(cfg.periods.fast),
            "mid": int(cfg.periods.mid),
            "slow": int(cfg.periods.slow),
        },
        "engine": dict(cfg.engine),
        "clusterCount": int(clusterCount),
        "featureCount": int(featureCount),
        "featureIds": ids,
        "features": features,
        "center": modelInfo["center"],
        "scale": modelInfo["scale"],
        "pcaCount": int(pcaCount),
        "pcaMean": modelInfo["pcaMean"],
        "pcaComponents": pcaComponents,
        "centroids": centroids,
        "policy": policyRows,
    }
    textLines = [
        "schema=gradbot-cluster-model-v1",
        f"view={view}",
        f"featureFamily={featureFamily}",
        f"clusterMethod={modelInfo['clusterMethod']}",
        f"policyMode={policyMode}",
        f"policyTarget={int(policyTarget)}",
        f"policyStartMs={cfg.policyStartMs}",
        f"policyEndMs={cfg.policyEndMs}",
        f"fitStartMs={int(modelInfo.get('fitStartMs', 0))}",
        f"fitEndMs={int(modelInfo.get('fitEndMs', 0))}",
        f"ticker={cfg.ticker}",
        f"interval={cfg.interval}",
        f"windowBars={int(cfg.windowBars)}",
        f"periodFast={int(cfg.periods.fast)}",
        f"periodMid={int(cfg.periods.mid)}",
        f"periodSlow={int(cfg.periods.slow)}",
        f"clusterCount={int(clusterCount)}",
        f"featureCount={int(featureCount)}",
        f"pcaCount={int(pcaCount)}",
        f"dimCount={int(dimCount)}",
        "featureIds=" + ",".join(str(value) for value in ids),
        "features=" + ",".join(features),
        "center=" + _csvFloats(list(modelInfo["center"])),
        "scale=" + _csvFloats(list(modelInfo["scale"])),
        "pcaMean=" + _csvFloats(list(modelInfo["pcaMean"])),
        "pcaComponents=" + _csvFloats(_flat(pcaComponents)),
        "centroids=" + _csvFloats(_flat(centroids)),
        "buyReqMult=" + _csvFloats(buyMult),
        "sellReqMult=" + _csvFloats(sellMult),
    ]
    _writeText(jsonPath, json.dumps(payload, indent=2) + "\n")
    _writeText(textPath, "\n".join(textLines) + "\n")
    return jsonPath, textPath


def _flagSummaryRows(
    cfg,
    frame: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sideSpecs = [
        ("BUY", "acceptedBuy"),
        ("SELL", "acceptedSell"),
    ]
    for side, col in sideSpecs:
        flagged = frame[frame[col].astype(float) > 0.0]
        for partition in ["fit", "holdout", "policy"]:
            if partition == "policy":
                use = flagged[_policyMask(cfg, flagged)]
            else:
                use = flagged[flagged["partition"] == partition]
            for cluster in sorted(use["cluster"].dropna().unique().astype(int)):
                part = use[use["cluster"] == cluster]
                row: dict[str, object] = {
                    "partition": partition,
                    "side": side,
                    "cluster": int(cluster),
                    "flags": int(part.shape[0]),
                    "firstOpenMs": int(part["openMs"].iloc[0]),
                    "lastOpenMs": int(part["openMs"].iloc[-1]),
                }
                for bars in cfg.forwardBars:
                    col = f"fwdRet{int(bars)}h"
                    vals = part[col].dropna()
                    row[f"{col}Mean"] = float(vals.mean())
                    row[f"{col}Median"] = float(vals.median())
                    row[f"{col}WinRate"] = float((vals > 0.0).mean())
                rows.append(row)
    return rows


def _chart(
    cfg,
    frame: pd.DataFrame,
    path: Path,
    view: str,
    clusterCount: int,
    titleSuffix: str,
    tailBars: int | None = None,
) -> None:
    chart = frame.tail(int(tailBars)).copy() if tailBars else frame.copy()
    chart = chart[chart["cluster"] >= 0]
    if chart.empty:
        return
    ts = pd.to_datetime(chart["openMs"], unit="ms", utc=True)
    x = mdates.date2num(ts.dt.tz_convert(None).to_numpy())
    close = chart["close"].astype(float).to_numpy()
    clusters = chart["cluster"].astype(int).to_numpy()
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
    for period, label in [
        (cfg.periods.fast, "fast EMA"),
        (cfg.periods.mid, "mid EMA"),
        (cfg.periods.slow, "slow EMA"),
    ]:
        ema = chart["close"].ewm(span=period, adjust=False).mean()
        ax.plot(x, ema, "-", linewidth=0.8, label=label)

    start = 0
    while start < clusters.shape[0]:
        end = start + 1
        while end < clusters.shape[0] and clusters[end] == clusters[start]:
            end += 1
        left = x[start]
        right = x[end - 1] if end < clusters.shape[0] else x[-1]
        ax.axvspan(
            left,
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
    ax.set_title(
        f"{cfg.ticker} {cfg.interval} {view} clusters "
        f"(k={clusterCount}, window={cfg.windowBars}) {titleSuffix}"
    )
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


def _yearCharts(
    cfg,
    clustered: pd.DataFrame,
    chartDir: Path,
    view: str,
    clusterCount: int,
) -> list[Path]:
    frame = clustered[clustered["cluster"] >= 0].copy()
    ts = pd.to_datetime(frame["openMs"], unit="ms", utc=True)
    frame["chartYear"] = ts.dt.year
    paths: list[Path] = []
    for year in sorted(frame["chartYear"].dropna().unique().astype(int)):
        yearFrame = frame[frame["chartYear"] == year].drop(
            columns=["chartYear"],
        )
        path = chartDir / f"cluster_chart_{year}.png"
        _chart(
            cfg,
            yearFrame,
            path,
            view,
            clusterCount,
            str(year),
        )
        paths.append(path)
    return paths


def _familyOutDir(
    cfg,
    view: str,
    clusterMethod: str,
    featureFamily: str,
    clusterCount: int,
    periodLabel: str = "",
) -> Path:
    base = CLUSTERING_OUTPUT_DIR / cfg.name / view
    if str(periodLabel).strip():
        base = base / str(periodLabel)
    if len(cfg.clusterMethods) > 1 or clusterMethod != "kmeans":
        base = base / str(clusterMethod)
    if view == "engine" and (
        featureFamily != "runtime_mixed"
        or len(cfg.featureFamilies) > 1
    ):
        base = base / featureFamily
    return base / f"k{int(clusterCount):02d}"


def _noClusterFrame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["cluster"] = -1
    out["partition"] = "none"
    return out


def _periodLabel(periods: PeriodConfig) -> str:
    return (
        f"ema{int(periods.fast):03d}_"
        f"{int(periods.mid):03d}_"
        f"{int(periods.slow):03d}"
    )


def _bestFlag(
    rows: list[dict[str, str]],
    side: str,
    targetBars: int,
    high: bool,
) -> dict[str, object]:
    col = f"fwdRet{int(targetBars)}hMedian"
    winCol = f"fwdRet{int(targetBars)}hWinRate"
    use = [
        i for i in rows
        if i["partition"] == "holdout"
        and i["side"] == side
        and _int(i, "flags") >= 10
    ]
    if not use:
        return {
            "cluster": -1,
            "flags": 0,
            "median": float("nan"),
            "winRate": float("nan"),
        }
    key = max if high else min
    row = key(use, key=lambda i: _float(i, col))
    return {
        "cluster": _int(row, "cluster"),
        "flags": _int(row, "flags"),
        "median": _float(row, col),
        "winRate": _float(row, winCol),
    }


def _transitionByCluster(
    rows: list[dict[str, str]],
    cluster: int,
    partition: str = "holdout",
) -> dict[str, str]:
    for row in rows:
        if (
            row["partition"] == partition
            and _int(row, "cluster") == int(cluster)
        ):
            return row
    return {}


def _sweepScoreRow(
    cfg,
    periodLabel: str,
    clusterMethod: str,
    view: str,
    featureFamily: str,
    clusterCount: int,
    paths: dict[str, Path],
) -> dict[str, object]:
    policyRows = _readRows(paths["policy"])
    flagRows = _readRows(paths["flagSummary"])
    transRows = _readRows(paths["transition"])
    medians = [_float(i, "median") for i in policyRows]
    wins = [_float(i, "winRate") for i in policyRows]
    buy = _bestFlag(flagRows, "BUY", cfg.policyTarget, True)
    sell = _bestFlag(flagRows, "SELL", cfg.policyTarget, False)
    buyTrans = _transitionByCluster(transRows, int(buy["cluster"]))
    sellTrans = _transitionByCluster(transRows, int(sell["cluster"]))
    policyCounts = [_int(i, "rows") for i in policyRows]
    buyFlags = int(buy["flags"])
    sellFlags = int(sell["flags"])
    buyMedian = float(buy["median"])
    sellMedian = float(sell["median"])
    buyWin = float(buy["winRate"])
    sellWin = float(sell["winRate"])
    buyScore = (
        max(buyMedian, 0.0)
        * buyWin
        * np.sqrt(min(buyFlags, 30) / 30.0)
    )
    sellDownScore = (
        abs(min(sellMedian, 0.0))
        * max(1.0 - sellWin, 0.0)
        * np.sqrt(min(sellFlags, 20) / 20.0)
    )
    return {
        "periods": periodLabel if periodLabel else "single",
        "clusterMethod": clusterMethod,
        "view": view,
        "featureFamily": featureFamily,
        "clusters": int(clusterCount),
        "policyTarget": int(cfg.policyTarget),
        "minPolicyRows": int(np.nanmin(policyCounts)),
        "policyMedianMax": float(np.nanmax(medians)),
        "policyMedianMin": float(np.nanmin(medians)),
        "policyMedianSpread": float(np.nanmax(medians) - np.nanmin(medians)),
        "policyWinMax": float(np.nanmax(wins)),
        "policyWinMin": float(np.nanmin(wins)),
        "buyScore": float(buyScore),
        "sellDownScore": float(sellDownScore),
        "buyHoldoutCluster": buy["cluster"],
        "buyHoldoutFlags": buy["flags"],
        "buyHoldoutMedian": buy["median"],
        "buyHoldoutWinRate": buy["winRate"],
        "buyClusterEntryMedian": _float(buyTrans, "entryMedian"),
        "buyClusterEntryWinRate": _float(buyTrans, "entryWinRate"),
        "buyClusterSelfTransitionPct": _float(
            buyTrans,
            "selfTransitionPct",
        ),
        "sellHoldoutCluster": sell["cluster"],
        "sellHoldoutFlags": sell["flags"],
        "sellHoldoutMedian": sell["median"],
        "sellHoldoutWinRate": sell["winRate"],
        "sellClusterEntryMedian": _float(sellTrans, "entryMedian"),
        "sellClusterEntryWinRate": _float(sellTrans, "entryWinRate"),
        "sellClusterSelfTransitionPct": _float(
            sellTrans,
            "selfTransitionPct",
        ),
        "featuresPath": str(paths["features"]),
        "policyPath": str(paths["policy"]),
        "flagSummaryPath": str(paths["flagSummary"]),
        "transitionPath": str(paths["transition"]),
    }


def runView(
    cfg,
    view: str,
    clusterCount: int,
    featureFamily: str,
    clusterMethod: str = "kmeans",
    periodLabel: str = "",
    outDirOverride: Path | None = None,
    policyMode: str = "heuristic_forward",
    policyTarget: int | None = None,
) -> dict[str, Path]:
    frame = _buildView(cfg, view)
    features = _clusterFeatures(view, frame, featureFamily)
    frame = addForwardReturns(frame, cfg.forwardBars)
    if features:
        clustered, modelInfo = _fitPredict(
            cfg,
            frame,
            features,
            int(clusterCount),
            str(clusterMethod),
        )
    else:
        clustered = _noClusterFrame(frame)
        modelInfo = {}
    outDir = (
        Path(outDirOverride)
        if outDirOverride is not None
        else _familyOutDir(
            cfg,
            view,
            str(clusterMethod),
            featureFamily,
            int(clusterCount),
            periodLabel,
        )
    )

    featurePath = outDir / "clustered_features.csv"
    summaryPath = outDir / "cluster_summary.csv"
    manifestPath = outDir / "cluster_feature_manifest.csv"
    featureSummaryPath = outDir / "cluster_feature_summary.csv"
    flagSummaryPath = outDir / "cluster_flag_summary.csv"
    transitionPath = outDir / "cluster_transition_summary.csv"
    policyPath = outDir / "cluster_policy.csv"
    chartDir = outDir / "charts"
    chartPath = chartDir / "cluster_chart_latest.png"
    targetBars = cfg.policyTarget if policyTarget is None else int(policyTarget)

    _writeFrame(featurePath, clustered)
    _writeRows(
        manifestPath,
        ["featureFamily", "position", "featureId", "feature"],
        manifestRows(featureFamily, features),
    )
    summaryRows = (
        _summaryRows(clustered, cfg.forwardBars, "fit")
        + _summaryRows(clustered, cfg.forwardBars, "holdout")
        + _summaryRows(clustered, cfg.forwardBars, "policy", cfg)
    )
    summaryHeader = [
        "partition",
        "cluster",
        "rows",
        "firstOpenMs",
        "lastOpenMs",
    ]
    for bars in cfg.forwardBars:
        col = f"fwdRet{int(bars)}h"
        summaryHeader += [
            f"{col}Mean",
            f"{col}Median",
            f"{col}WinRate",
        ]
    _writeRows(summaryPath, summaryHeader, summaryRows)
    _writeRows(
        featureSummaryPath,
        ["cluster", "feature", "median", "mean"],
        _featureRows(clustered, features),
    )
    _writeRows(
        transitionPath,
        [
            "partition",
            "cluster",
            "rows",
            "runs",
            "medianRunBars",
            "meanRunBars",
            "maxRunBars",
            "selfTransitionPct",
            "entryRows",
            "entryMedian",
            "entryWinRate",
            "continueRows",
            "continueMedian",
            "continueWinRate",
            "medianConfidence",
            "medianDistance",
        ],
        _runLengthRows(clustered, int(targetBars)),
    )
    if view == "engine":
        flagRows = _flagSummaryRows(cfg, clustered)
        if features:
            policyRows = _policyRows(
                cfg,
                clustered,
                cfg.forwardBars,
                int(clusterCount),
                flagRows,
                policyMode,
                int(targetBars),
            )
            modelJsonPath, modelTextPath = _writeModelFiles(
                outDir,
                cfg,
                view,
                featureFamily,
                policyMode,
                int(targetBars),
                int(clusterCount),
                modelInfo,
                policyRows,
            )
        else:
            policyRows = []
            modelJsonPath = outDir / "model" / "cluster_model.json"
            modelTextPath = outDir / "model" / "cluster_model.txt"
        flagHeader = [
            "partition",
            "side",
            "cluster",
            "flags",
            "firstOpenMs",
            "lastOpenMs",
        ]
        for bars in cfg.forwardBars:
            col = f"fwdRet{int(bars)}h"
            flagHeader += [
                f"{col}Mean",
                f"{col}Median",
                f"{col}WinRate",
            ]
        _writeRows(
            flagSummaryPath,
            flagHeader,
            flagRows,
        )
        _writeRows(
            policyPath,
            [
                "policyMode",
                "policyTarget",
                "policyStartMs",
                "policyEndMs",
                "cluster",
                "label",
                "rows",
                "mean",
                "median",
                "winRate",
                "buyReqMult",
                "sellReqMult",
            ],
            policyRows,
        )
    else:
        modelJsonPath = outDir / "model" / "cluster_model.json"
        modelTextPath = outDir / "model" / "cluster_model.txt"
    if features:
        _chart(
            cfg,
            clustered,
            chartPath,
            view,
            int(clusterCount),
            "latest",
            tailBars=cfg.chartBars,
        )
        if cfg.yearlyCharts:
            _yearCharts(cfg, clustered, chartDir, view, int(clusterCount))
    return {
        "features": featurePath,
        "summary": summaryPath,
        "manifest": manifestPath,
        "featureSummary": featureSummaryPath,
        "flagSummary": flagSummaryPath,
        "transition": transitionPath,
        "policy": policyPath,
        "modelJson": modelJsonPath,
        "modelText": modelTextPath,
        "chart": chartPath,
    }


def parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_cluster",
        description="Run causal unsupervised clustering research",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parseArgs(argv)
    cfg = loadConfig(args.config)
    availableFamilies = ", ".join(familyNames())
    periodSweep = len(cfg.periodCombos) > 1
    sweepRows: list[dict[str, object]] = []
    for periods in cfg.periodCombos:
        runCfg = replace(cfg, periods=periods)
        periodLabel = _periodLabel(periods) if periodSweep else ""
        if periodLabel:
            print(f"[cluster] periods={periodLabel}")
        for clusterMethod in runCfg.clusterMethods:
            print(f"[cluster] method={clusterMethod}")
            for view in runCfg.views:
                families = (
                    runCfg.featureFamilies
                    if view == "engine"
                    else ["explore_all"]
                )
                for featureFamily in families:
                    for clusterCount in runCfg.clusters:
                        print(
                            f"[cluster] view={view} family={featureFamily} "
                            f"k={clusterCount}"
                        )
                        paths = runView(
                            runCfg,
                            view,
                            int(clusterCount),
                            str(featureFamily),
                            str(clusterMethod),
                            periodLabel,
                        )
                        print(f"[cluster] features: {paths['features']}")
                        print(f"[cluster] manifest: {paths['manifest']}")
                        print(f"[cluster] summary: {paths['summary']}")
                        if view == "engine":
                            print(f"[cluster] flags: {paths['flagSummary']}")
                            print(
                                f"[cluster] transitions: "
                                f"{paths['transition']}"
                            )
                            print(f"[cluster] model: {paths['modelText']}")
                            sweepRows.append(
                                _sweepScoreRow(
                                    runCfg,
                                    periodLabel,
                                    str(clusterMethod),
                                    view,
                                    str(featureFamily),
                                    int(clusterCount),
                                    paths,
                                )
                            )
                        print(f"[cluster] chart: {paths['chart']}")
    if sweepRows:
        _writeRows(
            CLUSTERING_OUTPUT_DIR / cfg.name
            / "sweep_scores.csv",
            [
                "periods",
                "clusterMethod",
                "view",
                "featureFamily",
                "clusters",
                "policyTarget",
                "minPolicyRows",
                "policyMedianMax",
                "policyMedianMin",
                "policyMedianSpread",
                "policyWinMax",
                "policyWinMin",
                "buyScore",
                "sellDownScore",
                "buyHoldoutCluster",
                "buyHoldoutFlags",
                "buyHoldoutMedian",
                "buyHoldoutWinRate",
                "buyClusterEntryMedian",
                "buyClusterEntryWinRate",
                "buyClusterSelfTransitionPct",
                "sellHoldoutCluster",
                "sellHoldoutFlags",
                "sellHoldoutMedian",
                "sellHoldoutWinRate",
                "sellClusterEntryMedian",
                "sellClusterEntryWinRate",
                "sellClusterSelfTransitionPct",
                "featuresPath",
                "policyPath",
                "flagSummaryPath",
                "transitionPath",
            ],
            sweepRows,
        )
    print(f"[cluster] feature families: {availableFamilies}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
