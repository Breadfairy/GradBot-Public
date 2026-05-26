#!/usr/bin/env python3

from __future__ import annotations

########################################################################
# Imports
########################################################################

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


########################################################################
# Constants
########################################################################

START_VALUE = 1000.0
DEFAULT_SEED_PCT = 55.0
DEFAULT_FEE = 0.001
TARGET_BARS = 24
DAY_MS = 86_400_000
TIME_WINDOW_DAYS = [365, 730, 1095, 1460]


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

def _readFeature(path: Path, prefix: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    keep = [
        "ticker",
        "openMs",
        "closeMs",
        "close",
        "high",
        "low",
        "partition",
        "acceptedBuy",
        "acceptedSell",
        "emaGapFastPct",
        "emaGapMidPct",
        "emaGapSlowPct",
        "emaSpreadFastMidPct",
        "emaSpreadMidSlowPct",
        "gradFastPct",
        "gradMidPct",
        "gradSlowPct",
        "ret24h",
        "ret48h",
        "rangePos48",
        "logVolumeZ168",
        "takerImbalanceZ168",
        "trendBull",
        "trendBear",
        "macroDynSigned",
        "macroBull",
        "macroBear",
        "cluster",
        "clusterConfidence",
        "clusterDistance",
        "fwdRet24h",
        "fwdRet60h",
        "fwdRet90h",
    ]
    cols = [col for col in keep if col in data.columns]
    out = data[cols].copy()
    rename = {
        col: f"{prefix}{col[0].upper()}{col[1:]}"
        for col in cols
        if col not in {
            "ticker",
            "openMs",
            "closeMs",
            "close",
            "high",
            "low",
            "partition",
            "acceptedBuy",
            "acceptedSell",
            "fwdRet24h",
            "fwdRet60h",
            "fwdRet90h",
        }
    }
    return out.rename(columns=rename)


def _alignedFrame(regimePath: Path, eventPath: Path) -> pd.DataFrame:
    regime = _readFeature(regimePath, "regime")
    event = _readFeature(eventPath, "event")
    eventCols = [
        col for col in event.columns
        if col not in {
            "ticker",
            "closeMs",
            "close",
            "high",
            "low",
            "partition",
            "acceptedBuy",
            "acceptedSell",
            "fwdRet24h",
            "fwdRet60h",
            "fwdRet90h",
        }
    ]
    data = regime.merge(event[eventCols], on="openMs", how="inner")
    data = data[
        (data["regimeCluster"] >= 0)
        & (data["eventCluster"] >= 0)
        & data["partition"].isin(["fit", "holdout"])
    ].copy()
    data = data.sort_values("openMs").reset_index(drop=True)
    return data


def _readParentFeature(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    keep = [
        "openMs",
        "closeMs",
        "partition",
        "cluster",
        "emaGapFastPct",
        "emaGapMidPct",
        "emaGapSlowPct",
        "ret24h",
        "ret48h",
        "trendBull",
    ]
    cols = [col for col in keep if col in data.columns]
    out = data[cols].copy()
    rename = {
        "openMs": "parentOpenMs",
        "closeMs": "parentCloseMs",
        "partition": "parentPartition",
        "cluster": "parentCluster",
        "emaGapFastPct": "parentEmaGapFastPct",
        "emaGapMidPct": "parentEmaGapMidPct",
        "emaGapSlowPct": "parentEmaGapSlowPct",
        "ret24h": "parentRet24h",
        "ret48h": "parentRet48h",
        "trendBull": "parentTrendBull",
    }
    out = out.rename(columns=rename)
    out = out[out["parentCluster"].astype(float) >= 0.0].copy()
    return out.sort_values("parentCloseMs").reset_index(drop=True)


def _runAge(values: pd.Series) -> pd.Series:
    run = (values != values.shift(1)).cumsum()
    return values.groupby(run).cumcount() + 1


########################################################################
# Role Inference
########################################################################

def _clusterMedians(
    data: pd.DataFrame,
    prefix: str,
    partition: str = "fit",
) -> pd.DataFrame:
    col = f"{prefix}Cluster"
    use = data[data["partition"] == partition].copy()
    return use.groupby(col).median(numeric_only=True)


def _flagMedians(
    data: pd.DataFrame,
    prefix: str,
    sideCol: str,
    partition: str = "fit",
) -> pd.Series:
    col = f"{prefix}Cluster"
    use = data[
        (data["partition"] == partition)
        & (data[sideCol].astype(float) > 0.0)
    ].copy()
    if use.empty:
        return pd.Series(dtype=float)
    grouped = use.groupby(col)["fwdRet24h"].agg(["count", "median"])
    grouped = grouped[grouped["count"] >= 8]
    if grouped.empty:
        return pd.Series(dtype=float)
    return grouped["median"]


def inferRegimeRoles(data: pd.DataFrame) -> dict[int, str]:
    stats = _clusterMedians(data, "regime")
    clusters = [int(i) for i in stats.index.tolist()]
    bullScore = (
        stats["regimeEmaGapFastPct"]
        + stats["regimeEmaGapMidPct"]
        + stats["regimeEmaGapSlowPct"]
        + stats["regimeRet24h"]
        + stats["regimeRet48h"]
        + (10.0 * stats.get("regimeTrendBull", 0.0))
    )
    shockScore = (
        stats["regimeEmaGapFastPct"]
        + stats["regimeRet24h"]
        + stats["regimeRet48h"]
        + stats["regimeGradFastPct"]
    )
    longScore = stats["fwdRet60h"] + stats["fwdRet90h"]
    ultra = int(bullScore.idxmax())
    roles = {int(i): "chop" for i in clusters}
    roles[ultra] = "ultraBull"
    remain = [i for i in clusters if int(i) != ultra]
    if remain:
        drag = int(longScore.loc[remain].idxmin())
        roles[drag] = "bearDrag"
    remain = [i for i in clusters if roles[int(i)] == "chop"]
    if remain:
        flush = int(shockScore.loc[remain].idxmin())
        roles[flush] = "flush"
    return roles


def inferEventRoles(data: pd.DataFrame) -> dict[int, str]:
    stats = _clusterMedians(data, "event")
    clusters = [int(i) for i in stats.index.tolist()]
    roles = {int(i): "neutral" for i in clusters}
    buyMedians = _flagMedians(data, "event", "acceptedBuy")
    if not buyMedians.empty:
        rebound = int(buyMedians.idxmax())
    else:
        bounceScore = stats["fwdRet24h"] - stats["eventRet24h"].clip(upper=0.0)
        rebound = int(bounceScore.idxmax())
    riskScore = stats["fwdRet60h"] + stats["fwdRet90h"]
    heatScore = (
        stats["eventEmaGapFastPct"]
        + stats["eventRet24h"]
        + stats["eventRet48h"]
    )
    roles[rebound] = "rebound"
    risk = int(riskScore.idxmin())
    if roles[risk] == "neutral":
        roles[risk] = "risk"
    overheat = int(heatScore.idxmax())
    if roles[overheat] == "neutral":
        roles[overheat] = "overheat"
    return roles


def inferParentRoles(parent: pd.DataFrame) -> dict[int, str]:
    use = parent[parent["parentPartition"] == "fit"].copy()
    stats = use.groupby("parentCluster").median(numeric_only=True)
    clusters = [int(i) for i in stats.index.tolist()]
    bullScore = (
        stats["parentEmaGapFastPct"]
        + stats["parentEmaGapMidPct"]
        + stats["parentEmaGapSlowPct"]
        + stats["parentRet24h"]
        + (10.0 * stats.get("parentTrendBull", 0.0))
    )
    bearScore = (
        stats["parentEmaGapFastPct"]
        + stats["parentEmaGapMidPct"]
        + stats["parentEmaGapSlowPct"]
        + stats["parentRet24h"]
    )
    roles = {int(i): "parentNeutral" for i in clusters}
    roles[int(bullScore.idxmax())] = "parentBull"
    remain = [i for i in clusters if roles[int(i)] == "parentNeutral"]
    if remain:
        roles[int(bearScore.loc[remain].idxmin())] = "parentBear"
    return roles


def _applyRoles(
    data: pd.DataFrame,
    regimeRoles: dict[int, str],
    eventRoles: dict[int, str],
) -> pd.DataFrame:
    out = data.copy()
    out["regimeRole"] = out["regimeCluster"].astype(int).map(regimeRoles)
    out["eventRole"] = out["eventCluster"].astype(int).map(eventRoles)
    out["regimeAge"] = _runAge(out["regimeCluster"])
    out["eventAge"] = _runAge(out["eventCluster"])
    out["prevRegimeRole"] = out["regimeRole"].shift(1).fillna("")
    out["prevRegimeAge"] = out["regimeAge"].shift(1).fillna(0).astype(int)
    out["transition"] = out["regimeRole"] != out["prevRegimeRole"]
    return out


def _alignParentRoles(
    roleData: pd.DataFrame,
    parentPath: Path | None,
) -> tuple[pd.DataFrame, dict[int, str]]:
    if parentPath is None:
        out = roleData.copy()
        out["parentCluster"] = -1
        out["parentRole"] = "none"
        out["parentOpenMs"] = np.nan
        out["parentCloseMs"] = np.nan
        return out, {}

    parent = _readParentFeature(parentPath)
    parentRoles = inferParentRoles(parent)
    parent["parentRole"] = (
        parent["parentCluster"].astype(int).map(parentRoles)
    )
    left = roleData.sort_values("closeMs").copy()
    out = pd.merge_asof(
        left,
        parent[
            [
                "parentOpenMs",
                "parentCloseMs",
                "parentCluster",
                "parentRole",
            ]
        ],
        left_on="closeMs",
        right_on="parentCloseMs",
        direction="backward",
    )
    out["parentCluster"] = out["parentCluster"].fillna(-1).astype(int)
    out["parentRole"] = out["parentRole"].fillna("none")
    return out.sort_values("openMs").reset_index(drop=True), parentRoles


def _alignPreviewRoles(
    roleData: pd.DataFrame,
    previewPath: Path | None,
    threshold: float,
) -> pd.DataFrame:
    out = roleData.copy()
    out["parentPreviewProb"] = np.nan
    out["parentPreviewRole"] = "none"
    if previewPath is None:
        return out

    preview = pd.read_csv(previewPath)
    keep = [
        "openMs",
        "parentPreviewProb",
    ]
    use = preview[keep].copy()
    use["parentPreviewRole"] = np.where(
        use["parentPreviewProb"].astype(float) >= float(threshold),
        "parentBull",
        "parentNeutral",
    )
    out = out.merge(use, on="openMs", how="left", suffixes=("", "New"))
    out["parentPreviewProb"] = out["parentPreviewProbNew"].combine_first(
        out["parentPreviewProb"]
    )
    out["parentPreviewRole"] = out["parentPreviewRoleNew"].fillna(
        out["parentPreviewRole"]
    )
    out = out.drop(columns=["parentPreviewProbNew", "parentPreviewRoleNew"])
    return out


########################################################################
# Simulation
########################################################################

def _maxDrawdown(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd = (arr / np.maximum(peak, 1e-12)) - 1.0
    return float(abs(np.nanmin(dd)) * 100.0)


def _tradeToTarget(
    quote: float,
    base: float,
    price: float,
    targetPct: float,
    fee: float,
) -> tuple[float, float, float]:
    total = quote + (base * price)
    currentValue = base * price
    targetValue = total * float(targetPct)
    diff = targetValue - currentValue
    turnover = 0.0
    if diff > 0.0:
        spend = min(diff, quote)
        base += (spend * (1.0 - fee)) / price
        quote -= spend
        turnover = spend
    elif diff < 0.0:
        sellValue = min(abs(diff), currentValue)
        qty = sellValue / price
        base -= qty
        quote += sellValue * (1.0 - fee)
        turnover = sellValue
    return quote, base, turnover


def _targetForRow(row: pd.Series, cfg: dict[str, float]) -> tuple[float, str]:
    role = str(row["regimeRole"])
    event = str(row["eventRole"])
    age = int(row["regimeAge"])
    prevRole = str(row["prevRegimeRole"])
    prevAge = int(row["prevRegimeAge"])
    buy = float(row["acceptedBuy"]) > 0.0
    sell = float(row["acceptedSell"]) > 0.0
    target = float(cfg["chopTarget"])
    reason = role

    if role == "ultraBull":
        target = float(cfg["ultraEntryTarget"])
        reason = "ultra_entry"
        if age >= int(cfg["ultraPersistAge"]):
            target = float(cfg["ultraTarget"])
            reason = "ultra_persist"
    elif role == "bearDrag":
        target = float(cfg["dragTarget"])
        reason = "bear_drag"
    elif role == "flush":
        target = float(cfg["flushTarget"])
        reason = "flush"

    if event == "rebound" and role != "bearDrag":
        target = max(target, float(cfg["reboundTarget"]))
        reason = "event_rebound"

    if event in {"risk", "overheat"} and role != "ultraBull":
        target = min(target, float(cfg["riskTarget"]))
        reason = f"event_{event}"

    if (
        prevRole == "ultraBull"
        and role != "ultraBull"
        and prevAge >= int(cfg["ultraExitAge"])
    ):
        target = min(target, float(cfg["postUltraExitTarget"]))
        reason = "post_ultra_exit"

    if buy and role != "bearDrag":
        target = max(target, float(cfg["dspBuyTarget"]))
        reason = "dsp_buy"

    if sell:
        blockSell = role == "ultraBull" and age >= int(cfg["sellBlockAge"])
        if not blockSell:
            target = min(target, float(cfg["dspSellTarget"]))
            reason = "dsp_sell"
        else:
            reason = "blocked_sell_ultra"

    return float(np.clip(target, 0.0, 1.0)), reason


def _stateMachineTargets(
    data: pd.DataFrame,
    cfg: dict[str, float],
) -> pd.DataFrame:
    out = data.copy()
    n = int(out.shape[0])
    role = out["regimeRole"].astype(str)
    event = out["eventRole"].astype(str)
    age = out["regimeAge"].astype(int)
    prevRole = out["prevRegimeRole"].astype(str)
    prevAge = out["prevRegimeAge"].astype(int)
    buy = out["acceptedBuy"].astype(float) > 0.0
    sell = out["acceptedSell"].astype(float) > 0.0
    target = np.full(n, float(cfg["chopTarget"]), dtype=float)
    reason = np.full(n, "chop", dtype=object)

    mask = role == "ultraBull"
    target[mask] = float(cfg["ultraEntryTarget"])
    reason[mask] = "ultra_entry"
    mask = (role == "ultraBull") & (age >= int(cfg["ultraPersistAge"]))
    target[mask] = float(cfg["ultraTarget"])
    reason[mask] = "ultra_persist"

    mask = role == "bearDrag"
    target[mask] = float(cfg["dragTarget"])
    reason[mask] = "bear_drag"
    mask = role == "flush"
    target[mask] = float(cfg["flushTarget"])
    reason[mask] = "flush"

    mask = (event == "rebound") & (role != "bearDrag")
    target[mask] = np.maximum(target[mask], float(cfg["reboundTarget"]))
    reason[mask] = "event_rebound"

    mask = event.isin(["risk", "overheat"]) & (role != "ultraBull")
    target[mask] = np.minimum(target[mask], float(cfg["riskTarget"]))
    reason[mask] = np.where(event[mask] == "risk", "event_risk", "event_overheat")

    mask = (
        (prevRole == "ultraBull")
        & (role != "ultraBull")
        & (prevAge >= int(cfg["ultraExitAge"]))
    )
    target[mask] = np.minimum(target[mask], float(cfg["postUltraExitTarget"]))
    reason[mask] = "post_ultra_exit"

    mask = buy & (role != "bearDrag")
    target[mask] = np.maximum(target[mask], float(cfg["dspBuyTarget"]))
    reason[mask] = "dsp_buy"

    blockSell = (role == "ultraBull") & (age >= int(cfg["sellBlockAge"]))
    mask = sell & ~blockSell
    target[mask] = np.minimum(target[mask], float(cfg["dspSellTarget"]))
    reason[mask] = "dsp_sell"
    reason[sell & blockSell] = "blocked_sell_ultra"

    out["targetPct"] = np.clip(target, 0.0, 1.0)
    out["targetReason"] = reason
    return out


def _phaseMachineTargets(
    data: pd.DataFrame,
    cfg: dict[str, object],
) -> pd.DataFrame:
    out = data.copy()
    roles = out["regimeRole"].astype(str).tolist()
    events = out["eventRole"].astype(str).tolist()
    parentRoles = out["parentRole"].astype(str).tolist()
    previewRoles = out["parentPreviewRole"].astype(str).tolist()
    buys = (out["acceptedBuy"].astype(float) > 0.0).tolist()
    sells = (out["acceptedSell"].astype(float) > 0.0).tolist()
    effRoles: list[str] = []
    phases: list[str] = []
    reasons: list[str] = []
    targets: list[float] = []
    latches: list[int] = []
    crabLeftRows: list[int] = []
    bullLatch = 0
    crabLeft = 0
    effAge = 0
    prevEff = ""

    for i in range(int(out.shape[0])):
        role = roles[i]
        event = events[i]
        parent = parentRoles[i]
        preview = previewRoles[i]
        buy = bool(buys[i])
        sell = bool(sells[i])
        parentBull = (
            bool(cfg["useParentBull"])
            and (parent == "parentBull" or preview == "parentBull")
            and role != "bearDrag"
        )
        rawUltra = role == "ultraBull" or parentBull
        hardRisk = role == "bearDrag" or parent == "parentBear"
        justExit = False
        eff = role

        if rawUltra:
            bullLatch = int(cfg["bullLatchBars"])
            eff = "ultraBull"
        elif bullLatch > 0 and role == "chop" and not hardRisk:
            bullLatch -= 1
            eff = "bullChop"
        else:
            bullLatch = 0

        if prevEff in {"ultraBull", "bullChop"} and eff not in {
            "ultraBull",
            "bullChop",
        }:
            crabLeft = int(cfg["crabBars"])
            justExit = True
        elif rawUltra:
            crabLeft = 0

        effAge = effAge + 1 if eff == prevEff else 1
        target = float(cfg["chopTarget"])
        reason = "chop"
        phase = "normal"

        if eff == "ultraBull":
            phase = "ultraRide"
            target = float(cfg["ultraEntryTarget"])
            reason = "ultra_entry"
            if effAge >= int(cfg["ultraPersistAge"]):
                target = float(cfg["ultraTarget"])
                reason = "ultra_persist"
        elif eff == "bullChop":
            phase = "bullChop"
            target = float(cfg["bullChopTarget"])
            reason = "bull_chop_latch"
        elif justExit:
            phase = "profitLock"
            target = float(cfg["lockTarget"])
            reason = "profit_lock"
        elif crabLeft > 0 and role == "chop" and not hardRisk:
            phase = "postUltraCrab"
            target = float(cfg["crabBaseTarget"])
            reason = "post_ultra_crab"
            if buy:
                target = max(target, float(cfg["crabBuyTarget"]))
                reason = "crab_dsp_buy"
            if sell:
                target = min(target, float(cfg["crabSellTarget"]))
                reason = "crab_dsp_sell"
        elif role == "bearDrag":
            phase = "bearRisk"
            target = float(cfg["dragTarget"])
            reason = "bear_drag"
        elif role == "flush":
            phase = "flush"
            target = float(cfg["flushTarget"])
            reason = "flush"

        if event == "rebound" and role != "bearDrag":
            boost = (
                float(cfg["crabBuyTarget"])
                if phase == "postUltraCrab"
                else float(cfg["reboundTarget"])
            )
            target = max(target, boost)
            reason = "event_rebound"

        if event in {"risk", "overheat"} and phase not in {
            "ultraRide",
            "bullChop",
        }:
            target = min(target, float(cfg["riskTarget"]))
            reason = f"event_{event}"

        if buy and phase not in {"postUltraCrab", "bearRisk"}:
            target = max(target, float(cfg["dspBuyTarget"]))
            reason = "dsp_buy"

        if sell:
            blockSell = phase in {"ultraRide", "bullChop"}
            if not blockSell and phase != "postUltraCrab":
                target = min(target, float(cfg["dspSellTarget"]))
                reason = "dsp_sell"
            elif blockSell:
                reason = "blocked_sell_bull_phase"

        if crabLeft > 0 and not rawUltra:
            crabLeft -= 1

        effRoles.append(eff)
        phases.append(phase)
        reasons.append(reason)
        targets.append(float(np.clip(target, 0.0, 1.0)))
        latches.append(int(bullLatch))
        crabLeftRows.append(int(crabLeft))
        prevEff = eff

    out["effectiveRegime"] = effRoles
    out["phase"] = phases
    out["targetPct"] = targets
    out["targetReason"] = reasons
    out["bullLatchBars"] = latches
    out["crabBarsLeft"] = crabLeftRows
    return out


def _lifeMachineTargets(
    data: pd.DataFrame,
    cfg: dict[str, object],
) -> pd.DataFrame:
    out = data.copy()
    roles = out["regimeRole"].astype(str).tolist()
    events = out["eventRole"].astype(str).tolist()
    parentRoles = out["parentRole"].astype(str).tolist()
    previewRoles = out["parentPreviewRole"].astype(str).tolist()
    buys = (out["acceptedBuy"].astype(float) > 0.0).tolist()
    sells = (out["acceptedSell"].astype(float) > 0.0).tolist()
    effRoles: list[str] = []
    phases: list[str] = []
    reasons: list[str] = []
    targets: list[float] = []
    missRows: list[int] = []
    crabRows: list[int] = []
    ultraAge = 0
    missCount = 0
    lockLeft = 0
    crabLeft = 0
    buyHold = 0
    sellCool = 0

    for i in range(int(out.shape[0])):
        role = roles[i]
        event = events[i]
        parent = parentRoles[i]
        preview = previewRoles[i]
        buy = bool(buys[i])
        sell = bool(sells[i])
        parentBull = (
            bool(cfg["useParentBull"])
            and (parent == "parentBull" or preview == "parentBull")
            and role != "bearDrag"
        )
        rawUltra = role == "ultraBull" or parentBull
        hardRisk = role == "bearDrag" or parent == "parentBear"
        target = float(cfg["chopTarget"])
        reason = "chop"
        phase = "normal"
        eff = role

        if hardRisk:
            phase = "bearRisk"
            eff = "bearRisk"
            target = float(cfg["dragTarget"])
            reason = "hard_risk"
            ultraAge = 0
            missCount = 0
            lockLeft = 0
            crabLeft = 0
            buyHold = 0
            sellCool = 0
        elif rawUltra:
            phase = "ultraRide"
            eff = "ultraBull"
            ultraAge += 1
            missCount = 0
            lockLeft = 0
            crabLeft = 0
            buyHold = 0
            sellCool = 0
            target = float(cfg["ultraEntryTarget"])
            reason = "ultra_ramp"
            if ultraAge >= int(cfg["rampBars"]):
                target = float(cfg["ultraTarget"])
                reason = "ultra_hold"
        elif ultraAge >= int(cfg["ultraMinAge"]):
            missCount += 1
            if missCount < int(cfg["exitConfirm"]):
                phase = "ultraWeak"
                eff = "ultraWeak"
                target = float(cfg["weakTarget"])
                reason = "confirm_exit_wait"
            else:
                phase = "profitLock"
                eff = "profitLock"
                target = float(cfg["lockTarget"])
                reason = "profit_lock"
                lockLeft = int(cfg["lockBars"]) - 1
                crabLeft = int(cfg["crabBars"])
                ultraAge = 0
                missCount = 0
        elif lockLeft > 0:
            phase = "profitLock"
            eff = "profitLock"
            target = float(cfg["lockTarget"])
            reason = "profit_lock_hold"
            lockLeft -= 1
        elif crabLeft > 0 and role == "chop":
            phase = "postUltraCrab"
            eff = "postUltraCrab"
            target = float(cfg["crabBaseTarget"])
            reason = "post_ultra_crab"
            if sell:
                sellCool = int(cfg["sellCooldownBars"])
                buyHold = 0
            if buy and sellCool == 0:
                buyHold = int(cfg["buyHoldBars"])
            if sellCool > 0:
                target = min(target, float(cfg["crabSellTarget"]))
                reason = "crab_sell_cool"
                sellCool -= 1
            elif buyHold > 0:
                target = max(target, float(cfg["crabBuyTarget"]))
                reason = "crab_buy_hold"
                buyHold -= 1
            crabLeft -= 1
        else:
            ultraAge = 0
            missCount = 0
            if role == "flush":
                phase = "flush"
                target = float(cfg["flushTarget"])
                reason = "flush"

        if event == "rebound" and phase not in {
            "bearRisk",
            "ultraRide",
            "ultraWeak",
        }:
            target = max(target, float(cfg["reboundTarget"]))
            reason = "event_rebound"

        if event in {"risk", "overheat"} and phase not in {
            "ultraRide",
            "ultraWeak",
            "profitLock",
        }:
            target = min(target, float(cfg["riskTarget"]))
            reason = f"event_{event}"

        if buy and phase not in {
            "bearRisk",
            "ultraRide",
            "ultraWeak",
            "postUltraCrab",
        }:
            target = max(target, float(cfg["dspBuyTarget"]))
            reason = "dsp_buy"

        if sell:
            blockSell = phase in {"ultraRide", "ultraWeak"}
            if not blockSell and phase != "postUltraCrab":
                target = min(target, float(cfg["dspSellTarget"]))
                reason = "dsp_sell"
            elif blockSell:
                reason = "blocked_sell_lifecycle"

        effRoles.append(eff)
        phases.append(phase)
        reasons.append(reason)
        targets.append(float(np.clip(target, 0.0, 1.0)))
        missRows.append(int(missCount))
        crabRows.append(int(crabLeft))

    out["effectiveRegime"] = effRoles
    out["phase"] = phases
    out["targetPct"] = targets
    out["targetReason"] = reasons
    out["bullLatchBars"] = missRows
    out["crabBarsLeft"] = crabRows
    return out


def _targetsForCfg(
    roleData: pd.DataFrame,
    cfg: dict[str, object],
) -> pd.DataFrame:
    if str(cfg.get("mode", "base")) == "life":
        return _lifeMachineTargets(roleData, cfg)
    if str(cfg.get("mode", "base")) == "phase":
        return _phaseMachineTargets(roleData, cfg)
    return _stateMachineTargets(roleData, cfg)


def _dspTargets(data: pd.DataFrame, cfg: dict[str, float]) -> pd.DataFrame:
    out = data.copy()
    target = float(cfg["seedPct"])
    targets: list[float] = []
    reasons: list[str] = []
    for _i, row in out.iterrows():
        reason = "hold"
        if float(row["acceptedBuy"]) > 0.0:
            target = min(1.0, target + float(cfg["dspStep"]))
            reason = "dsp_buy"
        if float(row["acceptedSell"]) > 0.0:
            target = max(0.0, target - float(cfg["dspStep"]))
            reason = "dsp_sell"
        targets.append(float(target))
        reasons.append(reason)
    out["targetPct"] = targets
    out["targetReason"] = reasons
    return out


def _staticTargets(data: pd.DataFrame, targetPct: float) -> pd.DataFrame:
    out = data.copy()
    out["targetPct"] = float(targetPct)
    out["targetReason"] = "static"
    return out


def simulateTargets(
    data: pd.DataFrame,
    partition: str,
    name: str,
    fee: float,
    rebalanceMin: float,
    keepTrades: bool = False,
) -> tuple[dict[str, object], pd.DataFrame]:
    use = data[data["partition"] == partition].copy().reset_index(drop=True)
    price0 = float(use["close"].iloc[0])
    seedPct = float(use["targetPct"].iloc[0])
    quote = START_VALUE * (1.0 - seedPct)
    base = (START_VALUE * seedPct * (1.0 - fee)) / price0
    values: list[float] = []
    exposures: list[float] = []
    trades: list[dict[str, object]] = []
    turnover = 0.0
    for i, row in use.iterrows():
        price = float(row["close"])
        total = quote + (base * price)
        currentPct = (base * price) / max(total, 1e-12)
        targetPct = float(row["targetPct"])
        if abs(targetPct - currentPct) >= float(rebalanceMin):
            oldQuote = quote
            oldBase = base
            quote, base, turn = _tradeToTarget(
                quote,
                base,
                price,
                targetPct,
                fee,
            )
            turnover += turn
            side = "BUY" if quote < oldQuote else "SELL"
            if keepTrades:
                trades.append(
                    {
                        "model": name,
                        "partition": partition,
                        "openMs": int(row["openMs"]),
                        "side": side,
                        "price": price,
                        "targetPct": targetPct,
                        "reason": str(row["targetReason"]),
                        "quoteBefore": oldQuote,
                        "baseBefore": oldBase,
                        "quoteAfter": quote,
                        "baseAfter": base,
                    }
                )
        total = quote + (base * price)
        values.append(total)
        exposures.append((base * price) / max(total, 1e-12))
    final = float(values[-1])
    startClose = float(use["close"].iloc[0])
    endClose = float(use["close"].iloc[-1])
    hodl = START_VALUE * (endClose / startClose) * (1.0 - fee)
    metrics = {
        "model": name,
        "partition": partition,
        "rows": int(use.shape[0]),
        "startClose": startClose,
        "endClose": endClose,
        "finalValue": final,
        "returnPct": ((final / START_VALUE) - 1.0) * 100.0,
        "hodlValue": hodl,
        "hodlReturnPct": ((hodl / START_VALUE) - 1.0) * 100.0,
        "vsHodlPct": ((final / max(hodl, 1e-12)) - 1.0) * 100.0,
        "maxDrawdownPct": _maxDrawdown(values),
        "avgExposurePct": float(np.mean(exposures) * 100.0),
        "minExposurePct": float(np.min(exposures) * 100.0),
        "maxExposurePct": float(np.max(exposures) * 100.0),
        "trades": int(len(trades)),
        "turnover": float(turnover),
        "score": (
            ((final / START_VALUE) - 1.0) * 100.0
            - (0.55 * _maxDrawdown(values))
            - (0.03 * len(trades))
        ),
    }
    return metrics, pd.DataFrame(trades)


def simulateTimeVals(
    data: pd.DataFrame,
    partition: str,
    name: str,
    fee: float,
    rebalanceMin: float,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    use = data[data["partition"] == partition].copy().reset_index(drop=True)
    price0 = float(use["close"].iloc[0])
    seedPct = float(use["targetPct"].iloc[0])
    quote = START_VALUE * (1.0 - seedPct)
    base = (START_VALUE * seedPct * (1.0 - fee)) / price0
    values: list[float] = []
    exposures: list[float] = []
    trades: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    turnover = 0.0
    tradeCount = 0
    for i, row in use.iterrows():
        price = float(row["close"])
        total = quote + (base * price)
        currentPct = (base * price) / max(total, 1e-12)
        targetPct = float(row["targetPct"])
        if abs(targetPct - currentPct) >= float(rebalanceMin):
            oldQuote = quote
            oldBase = base
            quote, base, turn = _tradeToTarget(
                quote,
                base,
                price,
                targetPct,
                fee,
            )
            turnover += turn
            tradeCount += 1
            side = "BUY" if quote < oldQuote else "SELL"
            trades.append(
                {
                    "model": name,
                    "partition": partition,
                    "openMs": int(row["openMs"]),
                    "side": side,
                    "price": price,
                    "targetPct": targetPct,
                    "reason": str(row["targetReason"]),
                    "quoteBefore": oldQuote,
                    "baseBefore": oldBase,
                    "quoteAfter": quote,
                    "baseAfter": base,
                }
            )
        total = quote + (base * price)
        exposure = (base * price) / max(total, 1e-12)
        hodlValue = START_VALUE * (price / price0) * (1.0 - fee)
        values.append(total)
        exposures.append(exposure)
        rows.append(
            {
                "model": name,
                "partition": partition,
                "openMs": int(row["openMs"]),
                "close": price,
                "value": float(total),
                "returnPct": ((float(total) / START_VALUE) - 1.0) * 100.0,
                "hodlValue": float(hodlValue),
                "hodlReturnPct": (
                    (float(hodlValue) / START_VALUE) - 1.0
                ) * 100.0,
                "quote": float(quote),
                "base": float(base),
                "exposurePct": float(exposure * 100.0),
                "targetPct": float(targetPct * 100.0),
                "targetReason": str(row["targetReason"]),
                "regimeCluster": int(row["regimeCluster"]),
                "eventCluster": int(row["eventCluster"]),
                "regimeRole": str(row["regimeRole"]),
                "eventRole": str(row["eventRole"]),
                "regimeAge": int(row["regimeAge"]),
                "eventAge": int(row["eventAge"]),
                "parentCluster": int(row.get("parentCluster", -1)),
                "parentRole": str(row.get("parentRole", "none")),
                "parentPreviewProb": float(
                    row.get("parentPreviewProb", np.nan)
                ),
                "parentPreviewRole": str(
                    row.get("parentPreviewRole", "none")
                ),
                "effectiveRegime": str(
                    row.get("effectiveRegime", row["regimeRole"])
                ),
                "phase": str(row.get("phase", "base")),
                "bullLatchBars": int(row.get("bullLatchBars", 0)),
                "crabBarsLeft": int(row.get("crabBarsLeft", 0)),
                "acceptedBuy": float(row["acceptedBuy"]),
                "acceptedSell": float(row["acceptedSell"]),
            }
        )
    final = float(values[-1])
    startClose = float(use["close"].iloc[0])
    endClose = float(use["close"].iloc[-1])
    hodl = START_VALUE * (endClose / startClose) * (1.0 - fee)
    drawdown = _maxDrawdown(values)
    metrics = {
        "model": name,
        "partition": partition,
        "rows": int(use.shape[0]),
        "startOpenMs": int(use["openMs"].iloc[0]),
        "endOpenMs": int(use["openMs"].iloc[-1]),
        "startClose": startClose,
        "endClose": endClose,
        "finalValue": final,
        "returnPct": ((final / START_VALUE) - 1.0) * 100.0,
        "hodlValue": hodl,
        "hodlReturnPct": ((hodl / START_VALUE) - 1.0) * 100.0,
        "vsHodlPct": ((final / max(hodl, 1e-12)) - 1.0) * 100.0,
        "maxDrawdownPct": drawdown,
        "avgExposurePct": float(np.mean(exposures) * 100.0),
        "minExposurePct": float(np.min(exposures) * 100.0),
        "maxExposurePct": float(np.max(exposures) * 100.0),
        "trades": int(tradeCount),
        "turnover": float(turnover),
        "score": (
            ((final / START_VALUE) - 1.0) * 100.0
            - (0.55 * drawdown)
            - (0.03 * tradeCount)
        ),
    }
    timeVals = pd.DataFrame(rows)
    timeVals["timeUtc"] = pd.to_datetime(
        timeVals["openMs"],
        unit="ms",
        utc=True,
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    return metrics, pd.DataFrame(trades), timeVals


def scoreTargets(
    data: pd.DataFrame,
    partition: str,
    name: str,
    fee: float,
    rebalanceMin: float,
) -> dict[str, object]:
    use = data[data["partition"] == partition].copy().reset_index(drop=True)
    close = use["close"].to_numpy(dtype=float)
    target = use["targetPct"].to_numpy(dtype=float)
    price0 = float(close[0])
    quote = START_VALUE * (1.0 - float(target[0]))
    base = (START_VALUE * float(target[0]) * (1.0 - fee)) / price0
    values: list[float] = []
    exposures: list[float] = []
    trades = 0
    turnover = 0.0
    for i in range(int(use.shape[0])):
        price = float(close[i])
        total = quote + (base * price)
        currentPct = (base * price) / max(total, 1e-12)
        targetPct = float(target[i])
        if abs(targetPct - currentPct) >= float(rebalanceMin):
            quote, base, turn = _tradeToTarget(
                quote,
                base,
                price,
                targetPct,
                fee,
            )
            turnover += turn
            trades += 1
        total = quote + (base * price)
        values.append(total)
        exposures.append((base * price) / max(total, 1e-12))
    final = float(values[-1])
    startClose = float(close[0])
    endClose = float(close[-1])
    hodl = START_VALUE * (endClose / startClose) * (1.0 - fee)
    drawdown = _maxDrawdown(values)
    return {
        "model": name,
        "partition": partition,
        "rows": int(use.shape[0]),
        "startClose": startClose,
        "endClose": endClose,
        "finalValue": final,
        "returnPct": ((final / START_VALUE) - 1.0) * 100.0,
        "hodlValue": hodl,
        "hodlReturnPct": ((hodl / START_VALUE) - 1.0) * 100.0,
        "vsHodlPct": ((final / max(hodl, 1e-12)) - 1.0) * 100.0,
        "maxDrawdownPct": drawdown,
        "avgExposurePct": float(np.mean(exposures) * 100.0),
        "minExposurePct": float(np.min(exposures) * 100.0),
        "maxExposurePct": float(np.max(exposures) * 100.0),
        "trades": int(trades),
        "turnover": float(turnover),
        "score": (
            ((final / START_VALUE) - 1.0) * 100.0
            - (0.55 * drawdown)
            - (0.03 * trades)
        ),
    }


def _windowTargets(
    targets: pd.DataFrame,
    name: str,
    startMs: int,
) -> pd.DataFrame:
    out = targets[targets["openMs"] >= int(startMs)].copy()
    out["partition"] = name
    return out


def _timeWindows(roleData: pd.DataFrame) -> list[tuple[str, int | None]]:
    endMs = int(roleData["openMs"].max())
    rows: list[tuple[str, int | None]] = [("fit", None), ("holdout", None)]
    for i in TIME_WINDOW_DAYS:
        startMs = endMs - (int(i) * DAY_MS)
        rows.append((f"tail{i}d", startMs))
    return rows


def _selectedTimeNames(
    scores: pd.DataFrame,
    includeModels: list[str],
) -> list[str]:
    bases = ["static_100", "static_seed", "dsp_step"]
    exclude = set(bases)
    names = set(bases)
    policy = scores[~scores["model"].isin(exclude)].copy()
    fit = policy[policy["partition"] == "fit"].copy()
    holdout = policy[policy["partition"] == "holdout"].copy()
    names.update(fit.sort_values("score", ascending=False)["model"].head(5))
    names.update(
        holdout.sort_values("score", ascending=False)["model"].head(5)
    )
    names.update(includeModels)
    return sorted(names)


def _targetMap(
    roleData: pd.DataFrame,
    cfgByName: dict[str, dict[str, object]],
    seedPct: float,
    selectedNames: list[str],
) -> dict[str, pd.DataFrame]:
    rows: dict[str, pd.DataFrame] = {}
    for name in selectedNames:
        if name == "static_100":
            rows[name] = _staticTargets(roleData, 1.0)
        elif name == "static_seed":
            rows[name] = _staticTargets(roleData, float(seedPct) / 100.0)
        elif name == "dsp_step":
            rows[name] = _dspTargets(
                roleData,
                {"seedPct": float(seedPct) / 100.0, "dspStep": 0.20},
            )
        else:
            rows[name] = _targetsForCfg(roleData, cfgByName[name])
    return rows


def _writeTimeVals(
    roleData: pd.DataFrame,
    scores: pd.DataFrame,
    cfgByName: dict[str, dict[str, object]],
    outDir: Path,
    seedPct: float,
    fee: float,
    rebalanceMin: float,
    includeModels: list[str],
) -> None:
    selectedNames = _selectedTimeNames(scores, includeModels)
    targetByName = _targetMap(roleData, cfgByName, seedPct, selectedNames)
    metricRows: list[dict[str, object]] = []
    tradeRows: list[pd.DataFrame] = []
    timeRows: list[pd.DataFrame] = []
    selectedRows = [{"model": name} for name in selectedNames]
    _writeFrame(outDir / "selected_time_models.csv", pd.DataFrame(selectedRows))
    for name in selectedNames:
        targets = targetByName[name]
        for partition, startMs in _timeWindows(roleData):
            windowTargets = targets
            if startMs is not None:
                windowTargets = _windowTargets(targets, partition, startMs)
            metrics, trades, timeVals = simulateTimeVals(
                windowTargets,
                partition,
                name,
                fee,
                rebalanceMin,
            )
            metricRows.append(metrics)
            if not trades.empty:
                tradeRows.append(trades)
            if not timeVals.empty:
                timeRows.append(timeVals)
    _writeFrame(outDir / "window_scores.csv", pd.DataFrame(metricRows))
    if tradeRows:
        _writeFrame(outDir / "window_trades.csv", pd.concat(tradeRows))
    if timeRows:
        _writeFrame(outDir / "timeVals.csv", pd.concat(timeRows))


########################################################################
# Sweeps
########################################################################

def _configs(seedPct: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base = {
        "mode": "base",
        "seedPct": float(seedPct) / 100.0,
        "dspStep": 0.20,
        "dspBuyTarget": 0.72,
        "dspSellTarget": 0.22,
    }
    for block in [1, 2, 3]:
        for exitAge in [4, 6, 8]:
            for ultraTarget in [0.80, 1.00]:
                for rebound in [0.75, 0.90]:
                    for chop in [0.25, 0.35]:
                        for drag in [0.00, 0.15]:
                            for postExit in [0.15, 0.30]:
                                cfg = dict(base)
                                cfg.update(
                                    {
                                        "sellBlockAge": float(block),
                                        "ultraExitAge": float(exitAge),
                                        "ultraPersistAge": 3.0,
                                        "ultraEntryTarget": 0.55,
                                        "ultraTarget": ultraTarget,
                                        "reboundTarget": rebound,
                                        "chopTarget": chop,
                                        "dragTarget": drag,
                                        "flushTarget": 0.45,
                                        "riskTarget": 0.20,
                                        "postUltraExitTarget": postExit,
                                    }
                                )
                                rows.append(cfg)
    return rows


def _phaseConfigs(seedPct: float, useParentBull: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base = {
        "mode": "phase",
        "useParentBull": 1.0 if useParentBull else 0.0,
        "seedPct": float(seedPct) / 100.0,
        "dspStep": 0.20,
        "sellBlockAge": 2.0,
        "ultraExitAge": 4.0,
        "ultraPersistAge": 3.0,
        "ultraEntryTarget": 0.55,
        "ultraTarget": 0.80,
        "reboundTarget": 0.90,
        "chopTarget": 0.25,
        "dragTarget": 0.15,
        "flushTarget": 0.45,
        "riskTarget": 0.20,
        "dspBuyTarget": 0.72,
        "dspSellTarget": 0.22,
    }
    for latch in [0, 1, 2, 3, 4]:
        for bullChop in [0.35, 0.45, 0.55]:
            for lock in [0.25, 0.35]:
                for crabBars in [40, 80]:
                    for crabBase in [0.15, 0.25]:
                        for crabBuy in [0.55, 0.72]:
                            for crabSell in [0.15, 0.22]:
                                cfg = dict(base)
                                cfg.update(
                                    {
                                        "bullLatchBars": float(latch),
                                        "bullChopTarget": bullChop,
                                        "lockTarget": lock,
                                        "crabBars": float(crabBars),
                                        "crabBaseTarget": crabBase,
                                        "crabBuyTarget": crabBuy,
                                        "crabSellTarget": crabSell,
                                    }
                                )
                                rows.append(cfg)
    return rows


def _lifeConfigs(seedPct: float, useParentBull: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base = {
        "mode": "life",
        "useParentBull": 1.0 if useParentBull else 0.0,
        "seedPct": float(seedPct) / 100.0,
        "dspStep": 0.20,
        "ultraEntryTarget": 0.55,
        "reboundTarget": 0.90,
        "chopTarget": 0.25,
        "dragTarget": 0.15,
        "flushTarget": 0.45,
        "riskTarget": 0.20,
        "dspBuyTarget": 0.72,
        "dspSellTarget": 0.22,
        "lockBars": 2.0,
        "crabBars": 40.0,
        "crabSellTarget": 0.15,
        "buyHoldBars": 4.0,
        "sellCooldownBars": 2.0,
        "rampBars": 2.0,
    }
    for exitConfirm in [2, 3, 4]:
        for minAge in [3, 5]:
            for ultraTarget in [0.80, 0.90]:
                for weakTarget in [0.35, 0.50, 0.65, 0.80]:
                    for lockTarget in [0.25, 0.35]:
                        for crabBase in [0.15, 0.25]:
                            for crabBuy in [0.55, 0.72]:
                                cfg = dict(base)
                                cfg.update(
                                    {
                                        "exitConfirm": float(exitConfirm),
                                        "ultraMinAge": float(minAge),
                                        "ultraTarget": ultraTarget,
                                        "weakTarget": weakTarget,
                                        "lockTarget": lockTarget,
                                        "crabBaseTarget": crabBase,
                                        "crabBuyTarget": crabBuy,
                                    }
                                )
                                rows.append(cfg)
    return rows


def _configName(cfg: dict[str, object]) -> str:
    if str(cfg.get("mode", "base")) == "life":
        parent = "p1" if float(cfg["useParentBull"]) > 0.0 else "p0"
        return (
            f"lm_{parent}_e{int(cfg['exitConfirm'])}_"
            f"m{int(cfg['ultraMinAge'])}_"
            f"u{int(float(cfg['ultraTarget']) * 100):03d}_"
            f"w{int(float(cfg['weakTarget']) * 100):03d}_"
            f"lk{int(float(cfg['lockTarget']) * 100):03d}_"
            f"bb{int(float(cfg['crabBaseTarget']) * 100):03d}_"
            f"by{int(float(cfg['crabBuyTarget']) * 100):03d}"
        )
    if str(cfg.get("mode", "base")) == "phase":
        parent = "p1" if float(cfg["useParentBull"]) > 0.0 else "p0"
        return (
            f"pm_{parent}_l{int(cfg['bullLatchBars'])}_"
            f"bc{int(float(cfg['bullChopTarget']) * 100):03d}_"
            f"lk{int(float(cfg['lockTarget']) * 100):03d}_"
            f"cb{int(cfg['crabBars'])}_"
            f"bb{int(float(cfg['crabBaseTarget']) * 100):03d}_"
            f"by{int(float(cfg['crabBuyTarget']) * 100):03d}_"
            f"sy{int(float(cfg['crabSellTarget']) * 100):03d}"
        )
    return (
        f"sm_b{int(cfg['sellBlockAge'])}_"
        f"x{int(cfg['ultraExitAge'])}_"
        f"u{int(cfg['ultraTarget'] * 100):03d}_"
        f"r{int(cfg['reboundTarget'] * 100):03d}_"
        f"c{int(cfg['chopTarget'] * 100):03d}_"
        f"d{int(cfg['dragTarget'] * 100):03d}_"
        f"p{int(cfg['postUltraExitTarget'] * 100):03d}"
    )


def runStateMachine(
    regimePath: Path,
    eventPath: Path,
    outDir: Path,
    seedPct: float,
    fee: float,
    rebalanceMin: float,
    writeDetails: bool = True,
    includeModels: list[str] | None = None,
    parentPath: Path | None = None,
    previewPath: Path | None = None,
    previewThreshold: float = 0.50,
) -> dict[str, object]:
    data = _alignedFrame(regimePath, eventPath)
    regimeRoles = inferRegimeRoles(data)
    eventRoles = inferEventRoles(data)
    roleData = _applyRoles(data, regimeRoles, eventRoles)
    roleData, parentRoles = _alignParentRoles(roleData, parentPath)
    roleData = _alignPreviewRoles(roleData, previewPath, previewThreshold)
    scoreRows: list[dict[str, object]] = []
    tradeRows: list[pd.DataFrame] = []
    cfgByName: dict[str, dict[str, object]] = {}

    staticRows = [
        ("static_100", _staticTargets(roleData, 1.0)),
        ("static_seed", _staticTargets(roleData, float(seedPct) / 100.0)),
        (
            "dsp_step",
            _dspTargets(
                roleData,
                {"seedPct": float(seedPct) / 100.0, "dspStep": 0.20},
            ),
        ),
    ]
    for name, targets in staticRows:
        for partition in ["fit", "holdout"]:
            metrics, trades = simulateTargets(
                targets,
                partition,
                name,
                fee,
                rebalanceMin,
                keepTrades=True,
            )
            scoreRows.append(metrics)
            if not trades.empty:
                tradeRows.append(trades)

    configs = (
        _configs(seedPct)
        + _phaseConfigs(seedPct, parentPath is not None)
        + _lifeConfigs(seedPct, parentPath is not None)
    )
    for cfg in configs:
        name = _configName(cfg)
        cfgByName[name] = cfg
        targets = _targetsForCfg(roleData, cfg)
        for partition in ["fit", "holdout"]:
            metrics = scoreTargets(
                targets,
                partition,
                name,
                fee,
                rebalanceMin,
            )
            metrics.update(cfg)
            scoreRows.append(metrics)

    scores = pd.DataFrame(scoreRows)
    roleRows = []
    for cluster, role in regimeRoles.items():
        roleRows.append(
            {"model": "regime", "cluster": int(cluster), "role": role}
        )
    for cluster, role in eventRoles.items():
        roleRows.append(
            {"model": "event", "cluster": int(cluster), "role": role}
        )
    for cluster, role in parentRoles.items():
        roleRows.append(
            {"model": "parent", "cluster": int(cluster), "role": role}
        )
    roleFrame = pd.DataFrame(roleRows)
    _writeFrame(outDir / "state_machine_scores.csv", scores)
    _writeFrame(outDir / "cluster_roles.csv", roleFrame)
    if writeDetails:
        _writeFrame(outDir / "state_rows.csv", roleData)
        if tradeRows:
            _writeFrame(
                outDir / "state_machine_trades.csv",
                pd.concat(tradeRows),
            )
        _writeTimeVals(
            roleData,
            scores,
            cfgByName,
            outDir,
            seedPct,
            fee,
            rebalanceMin,
            includeModels or [],
        )
    fitScores = scores[scores["partition"] == "fit"].copy()
    selected = fitScores.sort_values("score", ascending=False).head(25)
    names = set(selected["model"].tolist())
    holdout = scores[
        (scores["partition"] == "holdout") & scores["model"].isin(names)
    ].copy()
    _writeFrame(
        outDir / "selected_by_fit_scores.csv",
        pd.concat([selected, holdout]).sort_values(["model", "partition"]),
    )
    return {
        "outDir": str(outDir),
        "rows": int(roleData.shape[0]),
        "scores": int(scores.shape[0]),
        "regimeRoles": regimeRoles,
        "eventRoles": eventRoles,
        "parentRoles": parentRoles,
    }


########################################################################
# CLI
########################################################################

def parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cluster_state_machine",
        description="Test dual-cluster state-machine exposure policies.",
    )
    parser.add_argument("--regime-features", required=True)
    parser.add_argument("--event-features", required=True)
    parser.add_argument("--parent-features")
    parser.add_argument("--parent-preview")
    parser.add_argument("--preview-threshold", type=float, default=0.50)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed-pct", type=float, default=DEFAULT_SEED_PCT)
    parser.add_argument("--fee", type=float, default=DEFAULT_FEE)
    parser.add_argument("--rebalance-min", type=float, default=0.05)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--include-model", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parseArgs(argv)
    result = runStateMachine(
        Path(args.regime_features),
        Path(args.event_features),
        Path(args.out),
        float(args.seed_pct),
        float(args.fee),
        float(args.rebalance_min),
        not bool(args.score_only),
        list(args.include_model),
        Path(args.parent_features) if args.parent_features else None,
        Path(args.parent_preview) if args.parent_preview else None,
        float(args.preview_threshold),
    )
    print(f"[state-machine] output: {result['outDir']}")
    print(f"[state-machine] rows: {result['rows']}")
    print(f"[state-machine] scores: {result['scores']}")
    print(f"[state-machine] regime roles: {result['regimeRoles']}")
    print(f"[state-machine] event roles: {result['eventRoles']}")
    print(f"[state-machine] parent roles: {result['parentRoles']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
