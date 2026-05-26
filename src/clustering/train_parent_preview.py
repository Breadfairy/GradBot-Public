#!/usr/bin/env python3

from __future__ import annotations

########################################################################
# Imports
########################################################################

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

FEATURES = [
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


########################################################################
# Data Helpers
########################################################################

def _loadRows(regimePath: Path, parentPath: Path) -> tuple[pd.DataFrame, dict]:
    rows = pd.read_csv(regimePath)
    parent = _readParentFeature(parentPath)
    parentRoles = inferParentRoles(parent)
    parent["parentRole"] = (
        parent["parentCluster"].astype(int).map(parentRoles)
    )
    parent = parent.sort_values("parentOpenMs")
    out = pd.merge_asof(
        rows.sort_values("openMs"),
        parent[
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
    )
    out["parentBullLabel"] = (
        out["parentRole"].fillna("none") == "parentBull"
    ).astype(int)
    return out.sort_values("openMs").reset_index(drop=True), parentRoles


def _featureFrame(rows: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    features = [i for i in FEATURES if i in rows.columns]
    x = rows[features].replace([np.inf, -np.inf], np.nan).copy()
    x = x.fillna(0.0)
    return x, features


def _bestThreshold(y: np.ndarray, prob: np.ndarray) -> float:
    bestThresh = 0.50
    bestScore = -1.0
    for i in np.linspace(0.35, 0.75, 41):
        pred = prob >= float(i)
        score = f1_score(y, pred, zero_division=0)
        if score > bestScore:
            bestScore = float(score)
            bestThresh = float(i)
    return bestThresh


########################################################################
# Training
########################################################################

def _scoreRows(
    rows: pd.DataFrame,
    partition: str,
    threshold: float,
) -> dict[str, object]:
    use = rows[rows["partition"] == partition].copy()
    y = use["parentBullLabel"].astype(int).to_numpy()
    pred = (use["parentPreviewProb"].astype(float) >= threshold).to_numpy()
    return {
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
    partition: str,
    threshold: float,
) -> list[dict[str, object]]:
    use = rows[rows["partition"] == partition].copy()
    y = use["parentBullLabel"].astype(int).to_numpy()
    pred = (use["parentPreviewProb"].astype(float) >= threshold).astype(int)
    mat = confusion_matrix(y, pred, labels=[0, 1])
    out: list[dict[str, object]] = []
    labels = ["notBull", "parentBull"]
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            out.append(
                {
                    "partition": partition,
                    "actual": actual,
                    "predicted": predicted,
                    "count": int(mat[i, j]),
                }
            )
    return out


def _writePdf(path: Path, rows: pd.DataFrame, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [i for i in ["fit", "holdout"] if i in set(rows["partition"])]
    fig, axes = plt.subplots(1, len(parts), figsize=(5 * len(parts), 4))
    axesList = np.atleast_1d(axes)
    for ax, part in zip(axesList, parts):
        use = rows[rows["partition"] == part].copy()
        y = use["parentBullLabel"].astype(int).to_numpy()
        pred = (
            use["parentPreviewProb"].astype(float) >= threshold
        ).astype(int)
        mat = confusion_matrix(y, pred, labels=[0, 1])
        ax.imshow(mat, cmap="Blues")
        ax.set_title(part)
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


def trainPreview(
    regimePath: Path,
    parentPath: Path,
    outDir: Path,
) -> dict[str, object]:
    rows, parentRoles = _loadRows(regimePath, parentPath)
    x, features = _featureFrame(rows)
    trainMask = (
        rows["partition"].eq("fit")
        & rows["parentPartition"].eq("fit")
        & rows["cluster"].astype(float).ge(0.0)
    )
    model = Pipeline(
        [
            ("scale", RobustScaler()),
            (
                "logreg",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    y = rows.loc[trainMask, "parentBullLabel"].astype(int).to_numpy()
    model.fit(x.loc[trainMask], y)
    prob = model.predict_proba(x)[:, 1]
    threshold = _bestThreshold(y, prob[trainMask.to_numpy()])
    rows["parentPreviewProb"] = prob
    rows["parentPreviewRole"] = np.where(
        rows["parentPreviewProb"] >= threshold,
        "parentBull",
        "parentNeutral",
    )
    score = pd.DataFrame(
        [_scoreRows(rows, i, threshold) for i in ["fit", "holdout"]]
    )
    confusion = pd.DataFrame(
        [
            row
            for i in ["fit", "holdout"]
            for row in _confusionRows(rows, i, threshold)
        ]
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
    manifest = pd.DataFrame({"feature": features})
    _writeFrame(outDir / "parent_preview.csv", rows[previewCols])
    _writeFrame(outDir / "model_scores.csv", score)
    _writeFrame(outDir / "confusion_matrices.csv", confusion)
    _writeFrame(outDir / "feature_manifest.csv", manifest)
    _writePdf(outDir / "confusion_matrices.pdf", rows, threshold)
    return {
        "outDir": str(outDir),
        "rows": int(rows.shape[0]),
        "features": int(len(features)),
        "threshold": float(threshold),
        "parentRoles": parentRoles,
    }


########################################################################
# CLI
########################################################################

def parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train_parent_preview",
        description="Train a causal 6h preview of the daily parent regime.",
    )
    parser.add_argument("--regime-features", required=True)
    parser.add_argument("--parent-features", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parseArgs(argv)
    result = trainPreview(
        Path(args.regime_features),
        Path(args.parent_features),
        Path(args.out),
    )
    print(f"[parent-preview] output: {result['outDir']}")
    print(f"[parent-preview] rows: {result['rows']}")
    print(f"[parent-preview] features: {result['features']}")
    print(f"[parent-preview] threshold: {result['threshold']:.4f}")
    print(f"[parent-preview] parent roles: {result['parentRoles']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
