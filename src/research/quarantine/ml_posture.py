#!/usr/bin/env python3

########################################################################
# Imports
########################################################################

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


########################################################################
# Paths
########################################################################

ROOT_DIR = Path(__file__).resolve().parents[3]
ROLE_ORDER = ["down", "neutral", "strong"]
ROLE_ID = {
    "down": 0,
    "neutral": 1,
    "strong": 2,
}
ID_ROLE = {
    0: "down",
    1: "neutral",
    2: "strong",
}
ROLE_CLUSTER = {
    "down": 0,
    "neutral": 1,
    "strong": 2,
}

SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


########################################################################
# Config
########################################################################

@dataclass(frozen=True)
class MlConfig:
    name: str
    clusterConfig: str
    labelPath: str
    view: str
    featureFamily: str
    targetShiftBars: int
    clusterRemap: list[int]
    roleClusters: dict[str, list[int]]
    trainFraction: float
    selectFraction: float
    randomState: int
    models: list[str]
    topK: int


def _rootPath(rawPath: str) -> Path:
    path = Path(rawPath)
    return path if path.is_absolute() else ROOT_DIR / path


def loadMlConfig(path: str | Path) -> MlConfig:
    with open(Path(path), "r") as fh:
        raw = json.load(fh)
    return MlConfig(
        name=str(raw["name"]),
        clusterConfig=str(raw["clusterConfig"]),
        labelPath=str(raw["labelPath"]),
        view=str(raw["view"]),
        featureFamily=str(raw["featureFamily"]),
        targetShiftBars=int(raw["targetShiftBars"]),
        clusterRemap=[int(i) for i in raw["clusterRemap"]],
        roleClusters={
            str(k): [int(i) for i in v]
            for k, v in dict(raw["roleClusters"]).items()
        },
        trainFraction=float(raw["trainFraction"]),
        selectFraction=float(raw["selectFraction"]),
        randomState=int(raw["randomState"]),
        models=[str(i) for i in raw["models"]],
        topK=int(raw["topK"]),
    )


########################################################################
# Write Helpers
########################################################################

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
        writer.writerows(rows)
    os.replace(tmp, path)


def _writeJson(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


########################################################################
# Dataset
########################################################################

def _featureFrame(cfg: MlConfig) -> pd.DataFrame:
    from clustering.cluster_config import loadConfig
    from clustering.features import engineFeatures
    from clustering.features import exploreFeatures

    clusterCfg = loadConfig(_rootPath(cfg.clusterConfig))
    if cfg.view == "engine":
        return engineFeatures(clusterCfg)
    if cfg.view == "explore":
        return exploreFeatures(clusterCfg)
    raise ValueError(f"unknown ML view: {cfg.view}")


def _features(cfg: MlConfig, frame: pd.DataFrame) -> list[str]:
    if cfg.view == "engine":
        from clustering.feature_families import featureNames

        return featureNames(cfg.featureFamily)
    from clustering.features import featureColumns

    return featureColumns(frame)


def _roleMap(cfg: MlConfig) -> dict[int, str]:
    out: dict[int, str] = {}
    for k, v in cfg.roleClusters.items():
        for i in v:
            out[int(i)] = str(k)
    return out


def _labelFrame(cfg: MlConfig) -> pd.DataFrame:
    labels = pd.read_csv(
        _rootPath(cfg.labelPath),
        usecols=["openMs", "cluster"],
    ).sort_values("openMs")
    remap = np.asarray(cfg.clusterRemap, dtype=int)
    raw = labels["cluster"].to_numpy(dtype=int)
    mapped = np.full(raw.shape[0], -1, dtype=int)
    good = raw >= 0
    mapped[good] = remap[raw[good]]
    # Train on the future role while keeping the feature row causal.
    target = pd.Series(mapped, index=labels.index)
    target = target.shift(-int(cfg.targetShiftBars)).fillna(-1)
    labels["targetCluster"] = target.astype(int)
    return labels[["openMs", "targetCluster"]]


def _partition(validIndex: np.ndarray, cfg: MlConfig) -> pd.Series:
    part = pd.Series("drop", index=validIndex)
    count = int(validIndex.shape[0])
    trainEnd = int(count * cfg.trainFraction)
    selectEnd = trainEnd + int(count * cfg.selectFraction)
    part.iloc[:trainEnd] = "train"
    part.iloc[trainEnd:selectEnd] = "select"
    part.iloc[selectEnd:] = "holdout"
    return part


def buildDataset(cfg: MlConfig) -> tuple[pd.DataFrame, list[str]]:
    frame = _featureFrame(cfg)
    labels = _labelFrame(cfg)
    features = _features(cfg, frame)
    roles = _roleMap(cfg)
    data = frame.merge(labels, on="openMs", how="left")
    data["targetRole"] = data["targetCluster"].map(roles)
    xRaw = data[features].replace([np.inf, -np.inf], np.nan)
    valid = (
        xRaw.notna().all(axis=1)
        & data["targetRole"].notna()
    ).to_numpy()
    validIndex = data.index[valid].to_numpy()
    data["mlPartition"] = "drop"
    data.loc[validIndex, "mlPartition"] = _partition(validIndex, cfg).to_numpy()
    data["targetRole"] = data["targetRole"].fillna("drop")
    return data, features


########################################################################
# Models
########################################################################

def buildModel(name: str, randomState: int):
    if name == "logreg":
        return make_pipeline(
            RobustScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=int(randomState),
            ),
        )
    if name == "randomForest":
        return RandomForestClassifier(
            n_estimators=450,
            max_depth=9,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=int(randomState),
            n_jobs=4,
        )
    if name == "histGradientBoost":
        return HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.02,
            random_state=int(randomState),
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=450,
            max_depth=4,
            learning_rate=0.035,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=int(randomState),
            eval_metric="mlogloss",
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=450,
            max_depth=5,
            learning_rate=0.035,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=int(randomState),
            objective="multiclass",
        )
    raise ValueError(f"unknown ML model: {name}")


def _targetIds(values: np.ndarray) -> np.ndarray:
    return np.asarray([ROLE_ID[str(i)] for i in values], dtype=int)


def _targetNames(values: np.ndarray) -> np.ndarray:
    return np.asarray([ID_ROLE[int(i)] for i in values], dtype=object)


def _alignedProb(classes: np.ndarray, prob: np.ndarray) -> np.ndarray:
    out = np.zeros((prob.shape[0], len(ROLE_ORDER)), dtype=float)
    classVals = [int(i) for i in classes]
    for i in range(len(ROLE_ORDER)):
        if i in classVals:
            out[:, i] = prob[:, classVals.index(i)]
    return out


########################################################################
# Metrics
########################################################################

def _matrixFrame(matrix: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(matrix, columns=ROLE_ORDER)
    out.insert(0, "actual", ROLE_ORDER)
    return out


def _matrixRows(
    modelName: str,
    partition: str,
    matrix: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, actual in enumerate(ROLE_ORDER):
        for j, predicted in enumerate(ROLE_ORDER):
            rows.append(
                {
                    "model": modelName,
                    "partition": partition,
                    "actual": actual,
                    "predicted": predicted,
                    "count": int(matrix[i, j]),
                }
            )
    return rows


def _matrixFigure(
    modelName: str,
    partition: str,
    matrix: np.ndarray,
    accuracy: float,
):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(ROLE_ORDER)))
    ax.set_yticks(np.arange(len(ROLE_ORDER)))
    ax.set_xticklabels(ROLE_ORDER)
    ax.set_yticklabels(ROLE_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    title = f"{modelName} {partition} accuracy={accuracy:.4f}"
    ax.set_title(title)
    high = float(matrix.max()) * 0.5 if int(matrix.max()) > 0 else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if float(matrix[i, j]) > high else "black"
            ax.text(
                j,
                i,
                str(int(matrix[i, j])),
                ha="center",
                va="center",
                color=color,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _writeMatrixPdf(
    path: Path,
    modelName: str,
    partition: str,
    matrix: np.ndarray,
    accuracy: float,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = _matrixFigure(modelName, partition, matrix, accuracy)
    fig.savefig(path)
    plt.close(fig)


def _writeMatrixReport(
    path: Path,
    items: list[tuple[str, str, np.ndarray, float]],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for modelName, partition, matrix, accuracy in items:
            fig = _matrixFigure(modelName, partition, matrix, accuracy)
            pdf.savefig(fig)
            plt.close(fig)


def _scoreRows(
    modelName: str,
    data: pd.DataFrame,
    predRole: np.ndarray,
    outDir: Path,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[tuple[str, str, np.ndarray, float]],
]:
    scoreRows: list[dict[str, object]] = []
    matrixRows: list[dict[str, object]] = []
    classRows: list[dict[str, object]] = []
    matrixItems: list[tuple[str, str, np.ndarray, float]] = []
    for part in ["train", "select", "holdout"]:
        mask = data["mlPartition"].to_numpy() == part
        actual = data.loc[mask, "targetRole"].to_numpy()
        guess = predRole[mask]
        matrix = confusion_matrix(actual, guess, labels=ROLE_ORDER)
        accuracy = accuracy_score(actual, guess)
        precision, recall, f1, support = precision_recall_fscore_support(
            actual,
            guess,
            labels=ROLE_ORDER,
            zero_division=0.0,
        )
        scoreRows.append(
            {
                "model": modelName,
                "partition": part,
                "rows": int(actual.shape[0]),
                "accuracy": float(accuracy),
            }
        )
        for i, role in enumerate(ROLE_ORDER):
            classRows.append(
                {
                    "model": modelName,
                    "partition": part,
                    "role": role,
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
            )
        matrixRows += _matrixRows(modelName, part, matrix)
        matrixItems.append((modelName, part, matrix, float(accuracy)))
        _writeFrame(
            outDir / "metrics" / f"confusion_matrix_{modelName}_{part}.csv",
            _matrixFrame(matrix),
        )
        _writeMatrixPdf(
            outDir / "metrics" / f"confusion_matrix_{modelName}_{part}.pdf",
            modelName,
            part,
            matrix,
            float(accuracy),
        )
    return scoreRows, matrixRows, classRows, matrixItems


########################################################################
# Posture Export
########################################################################

def _clusterPred(pred: np.ndarray) -> np.ndarray:
    out = np.full(pred.shape[0], -1, dtype=int)
    for name, cluster in ROLE_CLUSTER.items():
        out[pred == name] = int(cluster)
    return out


def _postureFrame(
    data: pd.DataFrame,
    pred: np.ndarray,
    prob: np.ndarray,
) -> pd.DataFrame:
    out = data[
        ["openMs", "closeMs", "close", "mlPartition", "targetRole"]
    ].copy()
    out["cluster"] = _clusterPred(pred)
    out["predRole"] = pred
    out["pDown"] = prob[:, 0]
    out["pNeutral"] = prob[:, 1]
    out["pStrong"] = prob[:, 2]
    out["confidence"] = np.max(prob, axis=1)
    cols = [
        "openMs",
        "closeMs",
        "close",
        "cluster",
        "predRole",
        "targetRole",
        "mlPartition",
        "pDown",
        "pNeutral",
        "pStrong",
        "confidence",
    ]
    return out[cols]


########################################################################
# Run
########################################################################

def runMl(cfg: MlConfig, outDir: Path) -> dict[str, object]:
    dataAll, features = buildDataset(cfg)
    data = dataAll[dataAll["mlPartition"] != "drop"].copy()
    train = data[data["mlPartition"] == "train"]
    xTrain = train[features].to_numpy(dtype=float)
    yTrain = _targetIds(train["targetRole"].to_numpy())
    xAll = data[features].to_numpy(dtype=float)
    scoreRows: list[dict[str, object]] = []
    matrixRows: list[dict[str, object]] = []
    classRows: list[dict[str, object]] = []
    matrixItems: list[tuple[str, str, np.ndarray, float]] = []
    modelScores: list[tuple[str, float]] = []

    _writeFrame(outDir / "dataset.csv", data)
    _writeRows(
        outDir / "feature_manifest.csv",
        ["position", "feature"],
        [
            {"position": int(i), "feature": name}
            for i, name in enumerate(features)
        ],
    )

    for name in cfg.models:
        model = buildModel(name, cfg.randomState)
        model.fit(xTrain, yTrain)
        pred = np.asarray(model.predict(xAll), dtype=int)
        predRole = _targetNames(pred)
        prob = _alignedProb(model.classes_, model.predict_proba(xAll))
        scores, matrices, classes, matrixPages = _scoreRows(
            name,
            data,
            predRole,
            outDir,
        )
        scoreRows += scores
        matrixRows += matrices
        classRows += classes
        matrixItems += matrixPages
        selectScore = [
            float(i["accuracy"])
            for i in scores
            if str(i["partition"]) == "select"
        ][0]
        modelScores.append((name, selectScore))
        _writeFrame(
            outDir / "posture" / f"{name}-posture.csv",
            _postureFrame(data, predRole, prob),
        )

    ranked = sorted(modelScores, key=lambda i: i[1], reverse=True)
    rankRows = [
        {"rank": int(i + 1), "model": name, "selectAccuracy": score}
        for i, (name, score) in enumerate(ranked)
    ]
    _writeRows(
        outDir / "model_scores.csv",
        ["model", "partition", "rows", "accuracy"],
        scoreRows,
    )
    _writeRows(
        outDir / "confusion_matrices.csv",
        ["model", "partition", "actual", "predicted", "count"],
        matrixRows,
    )
    _writeRows(
        outDir / "class_metrics.csv",
        [
            "model",
            "partition",
            "role",
            "precision",
            "recall",
            "f1",
            "support",
        ],
        classRows,
    )
    _writeRows(
        outDir / "model_rank.csv",
        ["rank", "model", "selectAccuracy"],
        rankRows,
    )
    _writeMatrixReport(outDir / "confusion_matrices.pdf", matrixItems)
    _writeJson(
        outDir / "fingerprint.json",
        {
            "schema": "gradbot-ml-posture-v1",
            "name": cfg.name,
            "clusterConfig": cfg.clusterConfig,
            "labelPath": cfg.labelPath,
            "view": cfg.view,
            "featureFamily": cfg.featureFamily,
            "targetShiftBars": cfg.targetShiftBars,
            "features": features,
            "clusterRemap": cfg.clusterRemap,
            "roleClusters": cfg.roleClusters,
            "trainFraction": cfg.trainFraction,
            "selectFraction": cfg.selectFraction,
            "models": cfg.models,
            "topK": cfg.topK,
            "rankedModels": rankRows[:cfg.topK],
        },
    )
    return {
        "outDir": str(outDir),
        "ranked": rankRows[:cfg.topK],
    }


def parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ml_posture",
        description="Train ML role classifiers and export posture CSVs.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parseArgs(argv)
    cfg = loadMlConfig(args.config)
    outDir = (
        Path(args.out)
        if str(args.out).strip()
        else ROOT_DIR / "outputs" / "ml" / cfg.name
    )
    result = runMl(cfg, outDir.resolve())
    print(f"[ml] output: {result['outDir']}")
    for row in result["ranked"]:
        print(
            f"[ml] rank {row['rank']}: {row['model']} "
            f"select_accuracy={row['selectAccuracy']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
