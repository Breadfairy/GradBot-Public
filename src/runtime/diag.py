#!/usr/bin/env python3

########################################################################
# Imports
########################################################################

from typing import Dict, List, Tuple

import numpy as np

from runtime.gates import (
    Params,
    enforceCooldown,
    grad1ZscoreMask,
)


########################################################################
# Public Flag Interface
########################################################################

def generateFlags(
    ctx,
    signals: Dict[str, object],
    params: Params,
    startIdx: int,
    overrides: dict | None,
    macroDyn: np.ndarray | None = None,
    macroDir: np.ndarray | None = None,
    macroMom: np.ndarray | None = None,
) -> List[Tuple[int, str]]:
    diag = flagDiagnostics(
        ctx,
        signals,
        params,
        startIdx,
        overrides,
        macroDyn=macroDyn,
        macroDir=macroDir,
        macroMom=macroMom,
    )
    return diag["flags"]


def flagDiagnostics(
    ctx,
    signals: Dict[str, object],
    params: Params,
    startIdx: int,
    overrides: dict | None,
    macroDyn: np.ndarray | None = None,
    macroDir: np.ndarray | None = None,
    macroMom: np.ndarray | None = None,
) -> Dict[str, object]:
    """Vectorized BUY/SELL flag generation with macro move diagnostics."""
    n = len(ctx["closes"]) if ctx and ctx.get("closes") is not None else 0

    overrides = overrides or {}
    g1 = np.asarray(signals["g1P1"], dtype=float)
    trendCode = np.asarray(signals["trendCode"], dtype=int)
    if macroDyn is not None:
        if macroDyn.shape[0] == n:
            dyn = macroDyn
        else:
            m = min(n, macroDyn.shape[0])
            dyn = np.zeros(n, dtype=float)
            if m > 0:
                dyn[:m] = macroDyn[:m]
    else:
        dyn = np.zeros(n, dtype=float)

    if macroDir is not None:
        if macroDir.shape[0] == n:
            dirCode = np.asarray(macroDir, dtype=int)
        else:
            m = min(n, macroDir.shape[0])
            dirCode = np.zeros(n, dtype=int)
            if m > 0:
                dirCode[:m] = np.asarray(macroDir[:m], dtype=int)
    else:
        dirCode = np.zeros(n, dtype=int)

    if macroMom is not None:
        if macroMom.shape[0] == n:
            momCode = np.asarray(macroMom, dtype=int)
        else:
            m = min(n, macroMom.shape[0])
            momCode = np.zeros(n, dtype=int)
            if m > 0:
                momCode[:m] = np.asarray(macroMom[:m], dtype=int)
    else:
        momCode = np.zeros(n, dtype=int)

    idxs = np.arange(n)
    valid = idxs >= int(startIdx)
    allowBuy = trendCode == -1
    allowSell = trendCode == 1

    gradBuy = grad1ZscoreMask(ctx, allowBuy, g1, overrides, 'BUY')
    gradSell = grad1ZscoreMask(ctx, allowSell, g1, overrides, 'SELL')

    buyMask = valid & allowBuy & gradBuy
    sellMask = valid & allowSell & gradSell

    buyIdx = np.flatnonzero(buyMask)
    sellIdx = np.flatnonzero(sellMask)
    cd = max(int(params.COOLDOWN), 0)
    buyIdxF = enforceCooldown(buyIdx, cd)
    sellIdxF = enforceCooldown(sellIdx, cd)

    # Dynamic spacing since phase anchor / last BUY/SELL using macro dyn%.
    closesArr = np.asarray(ctx.get("closes"), dtype=float)

    # Phase anchors:
    # - BUY phases anchored at first allowBuy=True candle.
    # - SELL phases anchored at first allowSell=True candle.
    buyAnchorIdx = np.full(n, -1, dtype=int)
    sellAnchorIdx = np.full(n, -1, dtype=int)
    lastBuyAnchor = -1
    lastSellAnchor = -1
    prevAllowBuy = False
    prevAllowSell = False
    for i in range(n):
        if allowBuy[i] and not prevAllowBuy:
            lastBuyAnchor = i
        if allowSell[i] and not prevAllowSell:
            lastSellAnchor = i
        buyAnchorIdx[i] = lastBuyAnchor
        sellAnchorIdx[i] = lastSellAnchor
        prevAllowBuy = bool(allowBuy[i])
        prevAllowSell = bool(allowSell[i])

    buyIdxSp: List[int] = []
    sellIdxSp: List[int] = []
    lastBuy = None
    lastBuyPhase = None
    lastSell = None
    lastSellPhase = None
    buyDeltaPct = np.full(n, np.nan, dtype=float)
    sellDeltaPct = np.full(n, np.nan, dtype=float)
    buyReqPct = np.full(n, np.nan, dtype=float)
    sellReqPct = np.full(n, np.nan, dtype=float)
    buySpacePass = np.zeros(n, dtype=bool)
    sellSpacePass = np.zeros(n, dtype=bool)

    macroBull = dirCode > 0
    macroBear = dirCode < 0
    macroRev = macroBear & (momCode > 0)
    macroRoll = macroBull & (momCode < 0)

    buyMult = np.ones(n, dtype=float)
    sellMult = np.ones(n, dtype=float)
    sellRelax = max(
        0.0,
        min(100.0, float(overrides.get("MACRO_SELL_RELAX_PCT", 0.0))),
    )
    sellRelaxMult = 1.0 - (sellRelax / 100.0)

    for idx in buyIdxF:
        anchor = int(buyAnchorIdx[idx]) if 0 <= idx < n else -1
        phaseId = anchor
        if phaseId != lastBuyPhase:
            lastBuy = None
            lastBuyPhase = phaseId
        if lastBuy is not None:
            refIdx = lastBuy
        elif anchor >= 0:
            refIdx = anchor
        else:
            refIdx = -1
        if refIdx < 0:
            keep = True
        else:
            if (
                0 <= idx < closesArr.size
                and 0 <= refIdx < closesArr.size
            ):
                priceNow = closesArr[idx]
                priceRef = closesArr[refIdx]
                if priceNow > 0 and priceRef > 0:
                    deltaPct = ((priceRef / priceNow) - 1.0) * 100.0
                else:
                    deltaPct = 0.0
                dynVal = float(dyn[idx])
                if not np.isfinite(dynVal):
                    req = 0.0
                else:
                    req = max(0.0, abs(dynVal) * float(buyMult[idx]))
                buyDeltaPct[idx] = deltaPct
                buyReqPct[idx] = req
                keep = deltaPct >= req
            else:
                keep = True
            buySpacePass[idx] = keep
        if keep:
            buyIdxSp.append(idx)
            lastBuy = idx

    for idx in sellIdxF:
        anchor = int(sellAnchorIdx[idx]) if 0 <= idx < n else -1
        phaseId = anchor
        if phaseId != lastSellPhase:
            lastSell = None
            lastSellPhase = phaseId
        if lastSell is not None:
            refIdx = lastSell
        elif anchor >= 0:
            refIdx = anchor
        else:
            refIdx = -1
        if refIdx < 0:
            keep = True
        else:
            if (
                0 <= idx < closesArr.size
                and 0 <= refIdx < closesArr.size
            ):
                priceNow = closesArr[idx]
                priceRef = closesArr[refIdx]
                if priceNow > 0 and priceRef > 0:
                    deltaPct = ((priceNow / priceRef) - 1.0) * 100.0
                else:
                    deltaPct = 0.0
                dynVal = float(dyn[idx])
                if not np.isfinite(dynVal):
                    req = 0.0
                else:
                    req = max(0.0, abs(dynVal) * float(sellMult[idx]))
                    req *= sellRelaxMult
                sellDeltaPct[idx] = deltaPct
                sellReqPct[idx] = req
                keep = deltaPct >= req
            else:
                keep = True
            sellSpacePass[idx] = keep
        if keep:
            sellIdxSp.append(idx)
            lastSell = idx

    i = j = 0
    out: List[Tuple[int, str]] = []
    while i < len(sellIdxSp) or j < len(buyIdxSp):
        sVal = sellIdxSp[i] if i < len(sellIdxSp) else 10**12
        bVal = buyIdxSp[j] if j < len(buyIdxSp) else 10**12
        if sVal <= bVal:
            out.append((sVal, "SELL"))
            if bVal == sVal:
                out.append((bVal, "BUY"))
                j += 1
            i += 1
        else:
            out.append((bVal, "BUY"))
            j += 1

    return {
        "flags": out,
        "valid": valid,
        "allowBuy": allowBuy,
        "allowSell": allowSell,
        "gradBuy": gradBuy,
        "gradSell": gradSell,
        "buyMask": buyMask,
        "sellMask": sellMask,
        "buyIdx": buyIdx,
        "sellIdx": sellIdx,
        "buyIdxF": np.asarray(buyIdxF, dtype=int),
        "sellIdxF": np.asarray(sellIdxF, dtype=int),
        "buyIdxSp": np.asarray(buyIdxSp, dtype=int),
        "sellIdxSp": np.asarray(sellIdxSp, dtype=int),
        "dyn": dyn,
        "dirCode": dirCode,
        "momCode": momCode,
        "macroBull": macroBull,
        "macroBear": macroBear,
        "macroRev": macroRev,
        "macroRoll": macroRoll,
        "buyMult": buyMult,
        "sellMult": sellMult,
        "buyDeltaPct": buyDeltaPct,
        "sellDeltaPct": sellDeltaPct,
        "buyReqPct": buyReqPct,
        "sellReqPct": sellReqPct,
        "buySpacePass": buySpacePass,
        "sellSpacePass": sellSpacePass,
    }
