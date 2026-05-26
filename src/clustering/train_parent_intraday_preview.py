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
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from clustering.cluster_state_machine import _readParentFeature
from clustering.cluster_state_machine import inferParentRoles


########################################################################
# Constants
########################################################################

KLINE_COLS = [
    "openMs",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "closeMs",
    "quoteVolume",
    "trades",
    "takerBuyBaseVolume",
    "takerBuyQuoteVolume",
    "ignore",
]

REGIME_FEATURES = [
    "emaGapFastPct",
    "emaGapMidPct",
    "emaGapSlowPct",
    "emaSpreadFastMidPct",
    "emaSpreadMidSlowPct",
    "emaSpreadFastSlowPct",
    "gradFastPct",
    "gradMidPct",
    "gradSlowPct",
    "trendBull",
    "trendBear",
    "trendHalfBull",
    "trendHalfBear",
    "distHigh24Pct",
    "distLow24Pct",
    "range24Pct",
    "distHigh48Pct",
    "distLow48Pct",
    "range48Pct",
    "rangePos24",
    "rangePos48",
    "realVol12",
    "realVol24",
    "realVol48",
    "ret1h",
    "ret2h",
    "ret3h",
    "ret4h",
    "ret6h",
    "ret8h",
    "ret12h",
    "ret24h",
    "ret48h",
    "trendEfficiency24",
    "bodyPct",
    "bodyAbsPct",
    "upperWickPct",
    "lowerWickPct",
    "bodyAbsMean12",
    "bodyAbsMean24",
    "bodyAbsMean48",
    "rangeMean12",
    "rangeMean24",
    "rangeMean48",
    "takerBaseRatio",
    "takerQuoteRatio",
    "takerImbalance",
    "logVolumeZ168",
    "logQuoteZ168",
    "logTradesZ168",
    "takerImbalanceZ168",
    "allowBuy",
    "allowSell",
    "buyDeltaPct",
    "sellDeltaPct",
    "buyReqPct",
    "sellReqPct",
    "macroDynSigned",
    "macroDynMag",
    "macroDir",
    "macroMom",
    "macroBull",
    "macroBear",
    "macroRev",
    "macroRoll",
    "acceptedBuy",
    "acceptedSell",
    "cluster",
    "clusterConfidence",
    "clusterDistance",
]


########################################################################
# IO Helpers
########################################################################

def _writeFrame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _readKlines(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, header=None, names=KLINE_COLS)
    for i in KLINE_COLS:
        data[i] = pd.to_numeric(data[i])
    return data.sort_values("openMs").reset_index(drop=True)


########################################################################
# Feature Helpers
########################################################################

def _safePct(num: pd.Series, den: pd.Series) -> pd.Series:
    denSafe = den.where(den.abs() > 1e-12, np.nan)
    return (num / denSafe) * 100.0


def _retPct(values: pd.Series, bars: int) -> pd.Series:
    prev = values.shift(int(bars))
    return ((values / prev) - 1.0) * 100.0


def _ema(values: pd.Series, span: int) -> pd.Series:
    return values.ewm(span=int(span), adjust=False).mean()


def _rollingZ(values: pd.Series, window: int) -> pd.Series:
    prior = values.shift(1)
    roll = prior.rolling(int(window), min_periods=int(window))
    mean = roll.mean()
    std = roll.std(ddof=0)
    return (values - mean) / std.where(std > 1e-12, np.nan)


def _rangePos(value: pd.Series, low: pd.Series, high: pd.Series) -> pd.Series:
    span = high - low
    return (value - low) / span.where(span > 1e-12, np.nan)


def _dailyBaselines(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily[["openMs", "quoteVolume", "trades", "volume"]].copy()
    out = out.rename(columns={"openMs": "dayOpenMs"})
    out["dayQuoteMean30"] = (
        out["quoteVolume"].shift(1).rolling(30, min_periods=30).mean()
    )
    out["dayQuoteStd30"] = (
        out["quoteVolume"].shift(1).rolling(30, min_periods=30).std(ddof=0)
    )
    out["dayTradesMean30"] = (
        out["trades"].shift(1).rolling(30, min_periods=30).mean()
    )
    out["dayVolumeMean30"] = (
        out["volume"].shift(1).rolling(30, min_periods=30).mean()
    )
    return out[
        [
            "dayOpenMs",
            "dayQuoteMean30",
            "dayQuoteStd30",
            "dayTradesMean30",
            "dayVolumeMean30",
        ]
    ]


def _intradayFeatures(hourRows: pd.DataFrame, dailyRows: pd.DataFrame) -> pd.DataFrame:
    out = hourRows.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    quote = out["quoteVolume"].astype(float)
    volume = out["volume"].astype(float)
    trades = out["trades"].astype(float)
    takerQuote = out["takerBuyQuoteVolume"].astype(float)

    out["dayOpenMs"] = (out["openMs"] // 86_400_000) * 86_400_000
    dayGroup = out.groupby("dayOpenMs", sort=False)
    dayOpen = dayGroup["open"].transform("first").astype(float)
    dayHigh = dayGroup["high"].cummax().astype(float)
    dayLow = dayGroup["low"].cummin().astype(float)
    dayQuote = dayGroup["quoteVolume"].cumsum().astype(float)
    dayVolume = dayGroup["volume"].cumsum().astype(float)
    dayTrades = dayGroup["trades"].cumsum().astype(float)
    dayTakerQuote = dayGroup["takerBuyQuoteVolume"].cumsum().astype(float)
    hourIndex = dayGroup.cumcount().astype(float) + 1.0

    out["h1DayProgress"] = hourIndex / 24.0
    out["h1DayRetSoFarPct"] = ((close / dayOpen) - 1.0) * 100.0
    out["h1DayHighRetPct"] = ((dayHigh / dayOpen) - 1.0) * 100.0
    out["h1DayLowRetPct"] = ((dayLow / dayOpen) - 1.0) * 100.0
    out["h1DayRangePct"] = _safePct(dayHigh - dayLow, dayOpen)
    out["h1DayRangePos"] = _rangePos(close, dayLow, dayHigh)
    out["h1DayQuoteCum"] = dayQuote
    out["h1DayVolumeCum"] = dayVolume
    out["h1DayTradesCum"] = dayTrades
    out["h1DayTakerImbalance"] = (2.0 * dayTakerQuote / dayQuote) - 1.0

    base = _dailyBaselines(dailyRows)
    out = out.merge(base, on="dayOpenMs", how="left")
    pace = 24.0 / hourIndex
    out["h1DayQuotePaceZ30"] = (
        (dayQuote * pace) - out["dayQuoteMean30"]
    ) / out["dayQuoteStd30"].where(out["dayQuoteStd30"] > 1e-12, np.nan)
    out["h1DayQuotePaceRatio30"] = (
        (dayQuote * pace) / out["dayQuoteMean30"]
    )
    out["h1DayTradesPaceRatio30"] = (
        (dayTrades * pace) / out["dayTradesMean30"]
    )
    out["h1DayVolumePaceRatio30"] = (
        (dayVolume * pace) / out["dayVolumeMean30"]
    )

    for i in [1, 2, 3, 4, 6, 8, 12, 24]:
        out[f"h1Ret{i}"] = _retPct(close, i)
    for i in [12, 24, 48, 72]:
        out[f"h1RealVol{i}"] = out["h1Ret1"].rolling(
            i,
            min_periods=i,
        ).std(ddof=0)
    for i in [24, 48, 72]:
        rollHigh = high.rolling(i, min_periods=i).max()
        rollLow = low.rolling(i, min_periods=i).min()
        out[f"h1DistHigh{i}Pct"] = _safePct(close - rollHigh, close)
        out[f"h1DistLow{i}Pct"] = _safePct(close - rollLow, close)
        out[f"h1RangePos{i}"] = _rangePos(close, rollLow, rollHigh)

    emaFast = _ema(close, 5)
    emaMid = _ema(close, 13)
    emaSlow = _ema(close, 34)
    emaLong = _ema(close, 144)
    out["h1EmaFastGapPct"] = _safePct(emaFast - emaMid, close)
    out["h1EmaMidGapPct"] = _safePct(emaMid - emaSlow, close)
    out["h1EmaSlowGapPct"] = _safePct(emaSlow - emaLong, close)
    out["h1EmaFastGradPct"] = _safePct(emaFast - emaFast.shift(1), emaFast)
    out["h1EmaMidGradPct"] = _safePct(emaMid - emaMid.shift(1), emaMid)
    out["h1EmaSlowGradPct"] = _safePct(emaSlow - emaSlow.shift(1), emaSlow)
    out["h1QuoteZ168"] = _rollingZ(np.log1p(quote), 168)
    out["h1VolumeZ168"] = _rollingZ(np.log1p(volume), 168)
    out["h1TradesZ168"] = _rollingZ(np.log1p(trades), 168)
    out["h1TakerImbalance"] = (2.0 * takerQuote / quote) - 1.0
    out["h1TakerImbalanceZ168"] = _rollingZ(out["h1TakerImbalance"], 168)
    return out


def _sixFeatures(sixRows: pd.DataFrame) -> pd.DataFrame:
    out = sixRows.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    quote = out["quoteVolume"].astype(float)
    takerQuote = out["takerBuyQuoteVolume"].astype(float)
    for i in [1, 2, 4, 8, 12, 20]:
        out[f"h6Ret{i}"] = _retPct(close, i)
    for i in [4, 8, 12, 20]:
        rollHigh = high.rolling(i, min_periods=i).max()
        rollLow = low.rolling(i, min_periods=i).min()
        out[f"h6RangePos{i}"] = _rangePos(close, rollLow, rollHigh)
        out[f"h6DistHigh{i}Pct"] = _safePct(close - rollHigh, close)
        out[f"h6DistLow{i}Pct"] = _safePct(close - rollLow, close)
    emaFast = _ema(close, 5)
    emaMid = _ema(close, 13)
    emaSlow = _ema(close, 34)
    emaLong = _ema(close, 89)
    out["h6EmaFastGapPct"] = _safePct(emaFast - emaMid, close)
    out["h6EmaMidGapPct"] = _safePct(emaMid - emaSlow, close)
    out["h6EmaSlowGapPct"] = _safePct(emaSlow - emaLong, close)
    out["h6EmaFastGradPct"] = _safePct(emaFast - emaFast.shift(1), emaFast)
    out["h6EmaMidGradPct"] = _safePct(emaMid - emaMid.shift(1), emaMid)
    out["h6EmaSlowGradPct"] = _safePct(emaSlow - emaSlow.shift(1), emaSlow)
    out["h6QuoteZ120"] = _rollingZ(np.log1p(quote), 120)
    out["h6TakerImbalance"] = (2.0 * takerQuote / quote) - 1.0
    out["h6TakerImbalanceZ120"] = _rollingZ(out["h6TakerImbalance"], 120)
    keep = [
        i for i in out.columns
        if i in {"openMs", "closeMs"} or i.startswith("h6")
    ]
    return out[keep]


def _featureRows(
    regimePath: Path,
    oneHourPath: Path,
    sixHourPath: Path,
    dailyPath: Path,
    parentPath: Path,
) -> tuple[pd.DataFrame, list[str], dict[int, str]]:
    regime = pd.read_csv(regimePath)
    hourRows = _readKlines(oneHourPath)
    sixRows = _readKlines(sixHourPath)
    dailyRows = _readKlines(dailyPath)
    parent = _readParentFeature(parentPath)
    parentRoles = inferParentRoles(parent)
    parent["parentRole"] = (
        parent["parentCluster"].astype(int).map(parentRoles)
    )

    baseCols = [
        "ticker",
        "openMs",
        "closeMs",
        "partition",
        "close",
        "cluster",
    ]
    baseCols += [
        i for i in REGIME_FEATURES
        if i in regime.columns and i not in set(baseCols)
    ]
    rows = regime[baseCols].copy()
    rows = rows[
        rows["partition"].isin(["fit", "holdout"])
        & rows["cluster"].astype(float).ge(0.0)
    ].copy()
    rows = rows.sort_values("openMs").reset_index(drop=True)

    intraday = _intradayFeatures(hourRows, dailyRows)
    intradayFeatures = [
        i for i in intraday.columns
        if i.startswith("h1") or i == "closeMs"
    ]
    rows = pd.merge_asof(
        rows.sort_values("closeMs"),
        intraday[intradayFeatures].sort_values("closeMs"),
        on="closeMs",
        direction="backward",
    ).sort_values("openMs")

    six = _sixFeatures(sixRows)
    sixFeatures = [i for i in six.columns if i.startswith("h6")]
    rows = rows.merge(
        six[["openMs"] + sixFeatures],
        on="openMs",
        how="left",
        suffixes=("", "Six"),
    )

    rows = pd.merge_asof(
        rows.sort_values("openMs"),
        parent.sort_values("parentOpenMs")[
            [
                "parentOpenMs",
                "parentCloseMs",
                "parentPartition",
                "parentCluster",
                "parentRole",
            ]
        ],
        left_on="openMs",
        right_on="parentOpenMs",
        direction="backward",
    ).sort_values("openMs")
    rows["parentBullLabel"] = (
        rows["parentRole"].fillna("none") == "parentBull"
    ).astype(int)
    skip = {
        "ticker",
        "openMs",
        "closeMs",
        "partition",
        "close",
        "parentOpenMs",
        "parentCloseMs",
        "parentPartition",
        "parentCluster",
        "parentRole",
        "parentBullLabel",
    }
    features = [
        i for i in rows.columns
        if i not in skip and pd.api.types.is_numeric_dtype(rows[i])
    ]
    return rows.reset_index(drop=True), features, parentRoles


########################################################################
# Model Helpers
########################################################################

def _models() -> dict[str, object]:
    return {
        "logreg": Pipeline(
            [
                ("scale", RobustScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "randomForest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=12,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=1,
        ),
        "histGradient": HistGradientBoostingClassifier(
            learning_rate=0.045,
            max_iter=240,
            l2_regularization=0.02,
            random_state=42,
        ),
    }


def _featureFrame(rows: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x = rows[features].replace([np.inf, -np.inf], np.nan).copy()
    return x.fillna(0.0)


def _bestThreshold(y: np.ndarray, prob: np.ndarray) -> float:
    bestThresh = 0.50
    bestScore = -1.0
    for i in np.linspace(0.35, 0.85, 51):
        score = f1_score(y, prob >= float(i), zero_division=0)
        if score > bestScore:
            bestScore = float(score)
            bestThresh = float(i)
    return bestThresh


def _scoreRows(
    rows: pd.DataFrame,
    modelName: str,
    partition: str,
    threshold: float,
) -> dict[str, object]:
    use = rows[rows["partition"] == partition].copy()
    probCol = f"{modelName}Prob"
    y = use["parentBullLabel"].astype(int).to_numpy()
    pred = use[probCol].astype(float).to_numpy() >= float(threshold)
    return {
        "model": modelName,
        "partition": partition,
        "rows": int(use.shape[0]),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "positives": int(y.sum()),
        "predictedPositives": int(pred.sum()),
    }


def _confusionRows(
    rows: pd.DataFrame,
    modelName: str,
    partition: str,
    threshold: float,
) -> list[dict[str, object]]:
    use = rows[rows["partition"] == partition].copy()
    probCol = f"{modelName}Prob"
    y = use["parentBullLabel"].astype(int).to_numpy()
    pred = (use[probCol].astype(float).to_numpy() >= threshold).astype(int)
    mat = confusion_matrix(y, pred, labels=[0, 1])
    out: list[dict[str, object]] = []
    labels = ["notBull", "parentBull"]
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            out.append(
                {
                    "model": modelName,
                    "partition": partition,
                    "actual": actual,
                    "predicted": predicted,
                    "count": int(mat[i, j]),
                }
            )
    return out


def _writePdf(
    path: Path,
    rows: pd.DataFrame,
    thresholds: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    modelNames = list(thresholds.keys())
    fig, axes = plt.subplots(
        len(modelNames),
        2,
        figsize=(10, 4 * len(modelNames)),
    )
    axesArr = np.atleast_2d(axes)
    for r, modelName in enumerate(modelNames):
        for c, partition in enumerate(["fit", "holdout"]):
            ax = axesArr[r, c]
            use = rows[rows["partition"] == partition].copy()
            probCol = f"{modelName}Prob"
            y = use["parentBullLabel"].astype(int).to_numpy()
            pred = (
                use[probCol].astype(float).to_numpy()
                >= thresholds[modelName]
            ).astype(int)
            mat = confusion_matrix(y, pred, labels=[0, 1])
            ax.imshow(mat, cmap="Blues")
            ax.set_title(f"{modelName} {partition}")
            ax.set_xticks([0, 1], ["notBull", "bull"])
            ax.set_yticks([0, 1], ["notBull", "bull"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(int(mat[i, j])), ha="center", va="center")
            ax.set_xlabel("predicted")
            ax.set_ylabel("actual")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


########################################################################
# Training
########################################################################

def trainPreview(
    regimePath: Path,
    oneHourPath: Path,
    sixHourPath: Path,
    dailyPath: Path,
    parentPath: Path,
    outDir: Path,
) -> dict[str, object]:
    rows, features, parentRoles = _featureRows(
        regimePath,
        oneHourPath,
        sixHourPath,
        dailyPath,
        parentPath,
    )
    x = _featureFrame(rows, features)
    trainMask = (
        rows["partition"].eq("fit")
        & rows["parentPartition"].eq("fit")
        & rows["parentCluster"].astype(float).ge(0.0)
    )
    y = rows.loc[trainMask, "parentBullLabel"].astype(int).to_numpy()
    thresholds: dict[str, float] = {}
    scoreRows: list[dict[str, object]] = []
    confusionRows: list[dict[str, object]] = []
    modelNames: list[str] = []

    for name, model in _models().items():
        model.fit(x.loc[trainMask], y)
        prob = model.predict_proba(x)[:, 1]
        threshold = _bestThreshold(y, prob[trainMask.to_numpy()])
        rows[f"{name}Prob"] = prob
        thresholds[name] = threshold
        modelNames.append(name)
        for partition in ["fit", "holdout"]:
            scoreRows.append(_scoreRows(rows, name, partition, threshold))
            confusionRows.extend(
                _confusionRows(rows, name, partition, threshold)
            )

    scores = pd.DataFrame(scoreRows)
    fitScores = scores[scores["partition"] == "fit"].copy()
    best = str(fitScores.sort_values("f1", ascending=False).iloc[0]["model"])
    rows["parentPreviewProb"] = rows[f"{best}Prob"].astype(float)
    rows["parentPreviewRole"] = np.where(
        rows["parentPreviewProb"] >= thresholds[best],
        "parentBull",
        "parentNeutral",
    )

    previewCols = [
        "ticker",
        "openMs",
        "closeMs",
        "partition",
        "close",
        "parentOpenMs",
        "parentCloseMs",
        "parentPartition",
        "parentCluster",
        "parentRole",
        "parentBullLabel",
        "parentPreviewProb",
        "parentPreviewRole",
    ]
    for name in modelNames:
        previewCols.append(f"{name}Prob")
    manifest = pd.DataFrame({"feature": features})
    _writeFrame(outDir / "parent_preview.csv", rows[previewCols])
    _writeFrame(outDir / "model_scores.csv", scores)
    _writeFrame(outDir / "confusion_matrices.csv", pd.DataFrame(confusionRows))
    _writeFrame(outDir / "feature_manifest.csv", manifest)
    _writeFrame(
        outDir / "selected_model.csv",
        pd.DataFrame([{"model": best, "threshold": thresholds[best]}]),
    )
    _writePdf(outDir / "confusion_matrices.pdf", rows, thresholds)
    for name in modelNames:
        single = previewCols[:11] + [f"{name}Prob"]
        out = rows[single].copy()
        out["parentPreviewProb"] = out[f"{name}Prob"].astype(float)
        out["parentPreviewRole"] = np.where(
            out["parentPreviewProb"] >= thresholds[name],
            "parentBull",
            "parentNeutral",
        )
        out = out.drop(columns=[f"{name}Prob"])
        _writeFrame(outDir / f"parent_preview_{name}.csv", out)
    return {
        "outDir": str(outDir),
        "rows": int(rows.shape[0]),
        "features": int(len(features)),
        "selectedModel": best,
        "threshold": float(thresholds[best]),
        "parentRoles": parentRoles,
    }


########################################################################
# CLI
########################################################################

def parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train_parent_intraday_preview",
        description="Train causal 1h+6h previews of daily parent bull labels.",
    )
    parser.add_argument("--regime-features", required=True)
    parser.add_argument("--one-hour-klines", required=True)
    parser.add_argument("--six-hour-klines", required=True)
    parser.add_argument("--daily-klines", required=True)
    parser.add_argument("--parent-features", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parseArgs(argv)
    result = trainPreview(
        Path(args.regime_features),
        Path(args.one_hour_klines),
        Path(args.six_hour_klines),
        Path(args.daily_klines),
        Path(args.parent_features),
        Path(args.out),
    )
    print(f"[intraday-preview] output: {result['outDir']}")
    print(f"[intraday-preview] rows: {result['rows']}")
    print(f"[intraday-preview] features: {result['features']}")
    print(f"[intraday-preview] selected model: {result['selectedModel']}")
    print(f"[intraday-preview] threshold: {result['threshold']:.4f}")
    print(f"[intraday-preview] parent roles: {result['parentRoles']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
