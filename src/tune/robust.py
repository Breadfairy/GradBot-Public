#!/usr/bin/env python3
# tune_robust.py - robust-region selection over tuner result rows.

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from tune.schema import configKeyOrder, rowIntFields, rowStrFields


METRIC_COLUMNS = {
    "preTaxEdge",
    "postTaxEdge",
    "netPctVsHodl",
    "simValue",
    "simPostTax",
    "benchValue",
    "benchPostTax",
    "trades",
    "fees",
    "tax",
    "potentialProfit",
    "potentialProfitBench",
    "netAfterTaxProfit",
    "netAfterTaxProfitBench",
    "grossEdgeVsBench",
    "netEdgeVsBench",
    "edgeVsBench",
    "sharpe",
    "sortino",
    "mdd",
    "cagr",
    "sharpe4w",
    "sortino4w",
    "sharpe13w",
    "sortino13w",
    "sharpe4wAbs",
    "sortino4wAbs",
    "sharpe13wAbs",
    "sortino13wAbs",
    "lifecycleEdgeMean",
    "lifecycleEdgeMedian",
    "lifecycleEdgeP25",
    "lifecycleEdgeMin",
    "lifecycleUnderwaterPct",
    "lifecycleUnderwaterMean",
    "lifecycleTrackingPct",
    "lifecycleEdgeMdd",
    "lifecycleEdgeScore",
    "scoreMetric",
}

IGNORE_PARAM_COLUMNS = {
    "ticker",
    "tickers",
    "days",
    "intervals",
    "DAILY_CLUSTER_PATH",
}

ROBUST_POOL_ROWS = 20000
ROBUST_CHUNK_ROWS = 250000
ROBUST_PROGRESS_ROWS = 2000000
ROBUST_DRAWDOWN_LIMIT = 0.55
ROBUST_MDD_BASE_WEIGHT = 0.35
ROBUST_MDD_EXCESS_WEIGHT = 1.25
ROBUST_LOCAL_GAP_WEIGHT = 0.15
ROBUST_TRADE_SOFT_LIMIT = 500.0
ROBUST_TRADE_WEIGHT = 0.03


def _writeCsv(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _paramColumns(df: pd.DataFrame) -> list[str]:
    ordered = ["interval"]
    ordered.extend(configKeyOrder())
    cols: list[str] = []
    for key in ordered:
        if key in IGNORE_PARAM_COLUMNS or key in METRIC_COLUMNS:
            continue
        if key in cols or key not in df.columns:
            continue
        if df[key].nunique(dropna=False) <= 1:
            continue
        cols.append(key)
    return cols


def _scoreSeries(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["scoreMetric"], errors="coerce")


def _selectionScoreSeries(df: pd.DataFrame) -> pd.Series:
    score = _scoreSeries(df).astype(float)
    if "mdd" in df.columns:
        mdd = pd.to_numeric(df["mdd"], errors="coerce").fillna(1.0)
        excess = (mdd - ROBUST_DRAWDOWN_LIMIT).clip(lower=0.0)
        drawPenalty = (
            (ROBUST_MDD_BASE_WEIGHT * mdd * 100.0)
            + (ROBUST_MDD_EXCESS_WEIGHT * excess * 100.0)
        )
        score = score - drawPenalty
    if "trades" in df.columns:
        trades = pd.to_numeric(df["trades"], errors="coerce").fillna(0.0)
        tradePenalty = (
            (trades - ROBUST_TRADE_SOFT_LIMIT).clip(lower=0.0)
            * ROBUST_TRADE_WEIGHT
        )
        score = score - tradePenalty
    return score


def _topScorePool(path: Path) -> pd.DataFrame:
    pool = pd.DataFrame()
    rowsSeen = 0
    nextProgress = ROBUST_PROGRESS_ROWS
    for chunk in pd.read_csv(path, chunksize=ROBUST_CHUNK_ROWS):
        rowsSeen += len(chunk)
        if "scoreMetric" not in chunk.columns:
            return pd.DataFrame()
        chunk["_scoreMetricNum"] = _scoreSeries(chunk)
        chunk["_selectionScoreNum"] = _selectionScoreSeries(chunk)
        chunk = chunk.dropna(subset=["_selectionScoreNum"])
        if chunk.empty:
            continue
        chunk = chunk.nlargest(
            min(ROBUST_POOL_ROWS, len(chunk)),
            "_selectionScoreNum",
        )
        pool = pd.concat([pool, chunk], ignore_index=True)
        if len(pool) > ROBUST_POOL_ROWS:
            pool = pool.nlargest(ROBUST_POOL_ROWS, "_selectionScoreNum")
        if rowsSeen >= nextProgress:
            print(
                f"[tune] robust scan rows={rowsSeen} "
                f"pool={len(pool)}",
                flush=True,
            )
            nextProgress += ROBUST_PROGRESS_ROWS
    if "_scoreMetricNum" in pool.columns:
        pool = pool.drop(columns=["_scoreMetricNum"])
    if "_selectionScoreNum" in pool.columns:
        pool = pool.drop(columns=["_selectionScoreNum"])
    return pool.reset_index(drop=True)


def _localStats(
    df: pd.DataFrame,
    params: list[str],
    scoreCol: str,
) -> pd.DataFrame:
    score = pd.to_numeric(df[scoreCol], errors="coerce").astype(float)
    out = pd.DataFrame(index=df.index)
    if not params:
        out["localMedianMean"] = score
        out["localMedianMin"] = score
        out["localP25Min"] = score
        out["localStdMax"] = 0.0
        out["localCountMin"] = 1
        return out

    medianSum = pd.Series(0.0, index=df.index)
    localCount = 0
    medianParts = []
    p25Parts = []
    stdParts = []
    countParts = []

    for key in params:
        groupCols = [i for i in params if i != key]
        if groupCols:
            grouped = df.groupby(groupCols, dropna=False)[scoreCol]
            median = grouped.transform("median").astype(float)
            p25 = grouped.transform("quantile", q=0.25).astype(float)
            std = grouped.transform("std").fillna(0.0).astype(float)
            count = grouped.transform("count").astype(float)
        else:
            median = pd.Series(score.median(), index=df.index)
            p25 = pd.Series(score.quantile(0.25), index=df.index)
            std = pd.Series(score.std(), index=df.index).fillna(0.0)
            count = pd.Series(float(len(df)), index=df.index)

        medianParts.append(median)
        p25Parts.append(p25)
        stdParts.append(std)
        countParts.append(count)
        medianSum = medianSum + median
        localCount += 1

    medianMin = pd.concat(medianParts, axis=1).min(axis=1)
    p25Min = pd.concat(p25Parts, axis=1).min(axis=1)
    stdMax = pd.concat(stdParts, axis=1).max(axis=1)
    countMin = pd.concat(countParts, axis=1).min(axis=1)
    out["localMedianMean"] = medianSum / float(localCount)
    out["localMedianMin"] = medianMin
    out["localP25Min"] = p25Min
    out["localStdMax"] = stdMax
    out["localCountMin"] = countMin
    return out


def _rowDict(row: pd.Series) -> dict[str, Any]:
    strFields = rowStrFields()
    intFields = rowIntFields()
    out: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            out[key] = ""
        elif key in strFields:
            out[key] = str(value)
        elif key in intFields:
            out[key] = int(float(value))
        elif isinstance(value, float):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def robustRowsFromResults(
    resultsCsvPath: str,
    outDir: str,
    maxRows: int = 5,
) -> list[dict[str, Any]]:
    path = Path(resultsCsvPath)
    if not path.is_file():
        return []

    df = _topScorePool(path)
    if df.empty or "scoreMetric" not in df.columns:
        return []

    df["scoreMetric"] = _scoreSeries(df)
    df["selectionScore"] = _selectionScoreSeries(df)
    df = df.dropna(subset=["selectionScore"]).reset_index(drop=True)
    if df.empty:
        return []
    params = _paramColumns(df)
    stats = _localStats(df, params, "selectionScore")
    scored = pd.concat([df, stats], axis=1)
    scoreGap = (
        scored["selectionScore"].astype(float)
        - scored["localMedianMean"].astype(float)
    ).clip(lower=0.0)
    scored["robustScore"] = (
        0.10 * scored["selectionScore"].astype(float)
        + 0.30 * scored["localMedianMean"].astype(float)
        + 0.35 * scored["localMedianMin"].astype(float)
        + 0.25 * scored["localP25Min"].astype(float)
        - 0.10 * scored["localStdMax"].astype(float)
        - ROBUST_LOCAL_GAP_WEIGHT * scoreGap
    )
    scored["robustParamCount"] = len(params)
    scored = scored.sort_values(
        ["robustScore", "selectionScore", "scoreMetric"],
        ascending=[False, False, False],
    )
    robustPath = Path(outDir) / "robust-candidates.csv"
    _writeCsv(robustPath, scored.head(int(maxRows)))
    rowPath = Path(outDir) / "robust-row.csv"
    _writeCsv(rowPath, scored.head(1))
    return [_rowDict(i) for _, i in scored.head(int(maxRows)).iterrows()]


__all__ = [
    "robustRowsFromResults",
]
