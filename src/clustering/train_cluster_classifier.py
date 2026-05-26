#!/usr/bin/env python3

from __future__ import annotations

########################################################################
# Imports
########################################################################

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gradbot-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gradbot-cache")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"], "fontconfig").mkdir(
    parents=True,
    exist_ok=True,
)

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


########################################################################
# Constants
########################################################################

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROB_THRESHOLDS = [0.0, 0.50, 0.60, 0.70, 0.80, 0.90]
CONF_THRESHOLDS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]


########################################################################
# IO Helpers
########################################################################

def _writeFrame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _manifest(path: Path) -> list[str]:
    manifest = path.parent / "cluster_feature_manifest.csv"
    with open(manifest, newline="") as fh:
        return [row["feature"] for row in csv.DictReader(fh)]


########################################################################
# Models
########################################################################

def _model(name: str, randomState: int):
    if name == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                solver="saga",
                random_state=int(randomState),
                n_jobs=4,
            ),
        )
    if name == "randomForest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=260,
                max_depth=12,
                min_samples_leaf=20,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=int(randomState),
                n_jobs=4,
            ),
        )
    if name == "histGradient":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=220,
                learning_rate=0.06,
                max_leaf_nodes=31,
                l2_regularization=0.02,
                random_state=int(randomState),
            ),
        )
    raise ValueError(f"unknown cluster classifier model: {name}")


def _classes(model) -> np.ndarray:
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    return np.asarray(model[-1].classes_)


def _probabilities(
    model,
    x: np.ndarray,
    labels: list[int],
) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=float)
    classes = _classes(model).astype(int).tolist()
    out = np.zeros((x.shape[0], len(labels)), dtype=float)
    for i, label in enumerate(labels):
        out[:, i] = raw[:, classes.index(int(label))]
    return out


########################################################################
# Run
########################################################################

def _matrixRows(
    modelName: str,
    partition: str,
    labels: list[int],
    matrix: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            rows.append(
                {
                    "model": modelName,
                    "partition": partition,
                    "actual": int(actual),
                    "predicted": int(predicted),
                    "count": int(matrix[i, j]),
                }
            )
    return rows


def _score(
    modelName: str,
    partition: str,
    actual: np.ndarray,
    guess: np.ndarray,
    labels: list[int],
) -> dict[str, object]:
    return {
        "model": modelName,
        "partition": partition,
        "rows": int(actual.shape[0]),
        "accuracy": float(accuracy_score(actual, guess)),
        "balancedAccuracy": float(balanced_accuracy_score(actual, guess)),
        "macroF1": float(
            f1_score(
                actual,
                guess,
                labels=labels,
                average="macro",
                zero_division=0.0,
            )
        ),
    }


def _forwardColumns(data: pd.DataFrame) -> list[str]:
    return [
        col for col in data.columns
        if col.startswith("fwdRet") and col.endswith("h")
    ]


def _baseColumns(data: pd.DataFrame) -> list[str]:
    preferred = [
        "ticker",
        "openMs",
        "close",
        "partition",
        "cluster",
        "clusterConfidence",
        "clusterDistance",
        "acceptedBuy",
        "acceptedSell",
    ]
    return [col for col in preferred + _forwardColumns(data) if col in data]


def _flagProbRows(
    data: pd.DataFrame,
    modelName: str,
    pred: np.ndarray,
    prob: np.ndarray,
    labels: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    frame = data.copy()
    frame["predCluster"] = pred
    frame["predProb"] = prob
    sideCols = [("BUY", "acceptedBuy"), ("SELL", "acceptedSell")]
    forwardCols = _forwardColumns(frame)
    for side, sideCol in sideCols:
        flagged = frame[frame[sideCol].astype(float) > 0.0]
        for partition in ["fit", "holdout"]:
            partBase = flagged[flagged["partition"] == partition]
            for label in labels:
                labelBase = partBase[partBase["predCluster"] == int(label)]
                for minProb in PROB_THRESHOLDS:
                    part = labelBase[
                        labelBase["predProb"] >= float(minProb)
                    ]
                    if part.empty:
                        continue
                    row: dict[str, object] = {
                        "model": modelName,
                        "partition": partition,
                        "side": side,
                        "predictedCluster": int(label),
                        "minProb": float(minProb),
                        "flags": int(part.shape[0]),
                        "purity": float(
                            (part["cluster"] == part["predCluster"]).mean()
                        ),
                        "medianProb": float(part["predProb"].median()),
                        "meanProb": float(part["predProb"].mean()),
                        "firstOpenMs": int(part["openMs"].iloc[0]),
                        "lastOpenMs": int(part["openMs"].iloc[-1]),
                    }
                    for col in forwardCols:
                        vals = part[col].dropna()
                        row[f"{col}Mean"] = float(vals.mean())
                        row[f"{col}Median"] = float(vals.median())
                        row[f"{col}WinRate"] = float((vals > 0.0).mean())
                    rows.append(row)
    return rows


def _clusterConfidenceRows(
    data: pd.DataFrame,
    labels: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sideCols = [("BUY", "acceptedBuy"), ("SELL", "acceptedSell")]
    forwardCols = _forwardColumns(data)
    for side, sideCol in sideCols:
        flagged = data[data[sideCol].astype(float) > 0.0]
        for partition in ["fit", "holdout"]:
            partBase = flagged[flagged["partition"] == partition]
            for label in labels:
                labelBase = partBase[partBase["cluster"] == int(label)]
                for minConf in CONF_THRESHOLDS:
                    part = labelBase[
                        labelBase["clusterConfidence"] >= float(minConf)
                    ]
                    if part.empty:
                        continue
                    row: dict[str, object] = {
                        "partition": partition,
                        "side": side,
                        "cluster": int(label),
                        "minConfidence": float(minConf),
                        "flags": int(part.shape[0]),
                        "medianConfidence": float(
                            part["clusterConfidence"].median()
                        ),
                        "meanConfidence": float(
                            part["clusterConfidence"].mean()
                        ),
                        "firstOpenMs": int(part["openMs"].iloc[0]),
                        "lastOpenMs": int(part["openMs"].iloc[-1]),
                    }
                    for col in forwardCols:
                        vals = part[col].dropna()
                        row[f"{col}Mean"] = float(vals.mean())
                        row[f"{col}Median"] = float(vals.median())
                        row[f"{col}WinRate"] = float((vals > 0.0).mean())
                    rows.append(row)
    return rows


def _writeMatrixPdf(
    path: Path,
    matrixItems: list[dict[str, object]],
    labels: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with PdfPages(tmp) as pdf:
        for item in matrixItems:
            matrix = np.asarray(item["matrix"], dtype=float)
            title = f"{item['model']} {item['partition']}"
            fig, ax = plt.subplots(figsize=(7.0, 6.0))
            image = ax.imshow(matrix, cmap="Blues")
            ax.set_title(title)
            ax.set_xlabel("Predicted cluster")
            ax.set_ylabel("Actual cluster")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            vmax = float(np.nanmax(matrix)) if matrix.size else 0.0
            threshold = vmax * 0.55
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    color = "white" if matrix[i, j] >= threshold else "black"
                    ax.text(
                        j,
                        i,
                        f"{int(matrix[i, j])}",
                        ha="center",
                        va="center",
                        color=color,
                    )
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    os.replace(tmp, path)


def runClassifier(
    featuresPath: Path,
    outDir: Path,
    randomState: int,
) -> dict[str, object]:
    features = _manifest(featuresPath)
    data = pd.read_csv(featuresPath)
    data = data[
        (data["cluster"] >= 0)
        & data["partition"].isin(["fit", "holdout"])
    ].copy()
    labels = sorted(data["cluster"].dropna().astype(int).unique().tolist())
    trainMask = data["partition"].to_numpy() == "fit"
    x = data[features].to_numpy(dtype=float)
    y = data["cluster"].to_numpy(dtype=int)
    scoreRows: list[dict[str, object]] = []
    matrixRows: list[dict[str, object]] = []
    matrixItems: list[dict[str, object]] = []
    probRows: list[dict[str, object]] = []
    classified = data[_baseColumns(data)].copy()
    for name in ["logreg", "randomForest", "histGradient"]:
        model = _model(name, randomState)
        model.fit(x[trainMask], y[trainMask])
        pred = np.asarray(model.predict(x), dtype=int)
        proba = _probabilities(model, x, labels)
        maxProb = np.max(proba, axis=1)
        classified[f"{name}Cluster"] = pred
        classified[f"{name}Prob"] = maxProb
        for i, label in enumerate(labels):
            classified[f"{name}Prob{int(label)}"] = proba[:, i]
        probRows += _flagProbRows(
            data,
            name,
            pred,
            maxProb,
            labels,
        )
        for partition in ["fit", "holdout"]:
            mask = data["partition"].to_numpy() == partition
            actual = y[mask]
            guess = pred[mask]
            scoreRows.append(
                _score(name, partition, actual, guess, labels)
            )
            matrixRows += _matrixRows(
                name,
                partition,
                labels,
                confusion_matrix(actual, guess, labels=labels),
            )
            matrixItems.append(
                {
                    "model": name,
                    "partition": partition,
                    "matrix": confusion_matrix(actual, guess, labels=labels),
                }
            )
    _writeFrame(outDir / "model_scores.csv", pd.DataFrame(scoreRows))
    _writeFrame(
        outDir / "confusion_matrices.csv",
        pd.DataFrame(matrixRows),
    )
    _writeFrame(
        outDir / "probability_flag_summary.csv",
        pd.DataFrame(probRows),
    )
    _writeFrame(
        outDir / "cluster_confidence_flag_summary.csv",
        pd.DataFrame(_clusterConfidenceRows(data, labels)),
    )
    _writeFrame(outDir / "classified_features.csv", classified)
    _writeMatrixPdf(outDir / "confusion_matrices.pdf", matrixItems, labels)
    return {
        "outDir": str(outDir),
        "scores": scoreRows,
        "features": len(features),
        "classes": labels,
    }


########################################################################
# CLI
########################################################################

def parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train_cluster_classifier",
        description="Train supervised classifiers to imitate cluster labels.",
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parseArgs(argv)
    result = runClassifier(
        Path(args.features),
        Path(args.out),
        int(args.random_state),
    )
    print(f"[cluster-classifier] output: {result['outDir']}")
    print(f"[cluster-classifier] features: {result['features']}")
    print(f"[cluster-classifier] classes: {result['classes']}")
    for row in result["scores"]:
        print(
            f"[cluster-classifier] {row['model']} {row['partition']} "
            f"acc={row['accuracy']:.4f} "
            f"bal={row['balancedAccuracy']:.4f} "
            f"f1={row['macroF1']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
