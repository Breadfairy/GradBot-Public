#!/usr/bin/env python3
# flags.py
# Parameterized flagger + printouts helpers (canonical exports only).

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np

from engine_shared import (
    calcSpacing,
    liveGradsAt,
    buildSignals,
    spacingState,
    zscoreSeries,
    bars_per_day,
)

# Local non-JSON tunables (presentation + snapshot window)
RED    = "\033[31m"
GREEN  = "\033[32m"
ORANGE = "\033[38;5;208m"
YELLOW = "\033[33m"
RESET  = "\033[0m"
BAR    = "=" * 50
SNAPSHOT_MIN_WIN_DEFAULT = 250

# No module-level defaults for tuneables.
# All required parameters must be provided via config overrides.


@dataclass(frozen=True)
class Params:
    # Mechanics only
    COOLDOWN: int


def paramsFromSettings(overrides: dict | None = None) -> 'Params':
    overridesDict = overrides or {}

    def _require_float(name: str) -> float:
        if name not in overridesDict:
            raise KeyError(f"missing required param: {name}")
        return float(overridesDict[name])

    def _require_int(name: str) -> int:
        if name not in overridesDict:
            raise KeyError(f"missing required param: {name}")
        return int(overridesDict[name])

    # Require core tuneables explicitly (absolute gates removed)
    cooldown = _require_int('COOLDOWN')

    return Params(
        COOLDOWN=cooldown,
    )


def _enforceCooldown(indices: np.ndarray, cooldown: int) -> List[int]:
    if indices.size == 0:
        return []
    keep: List[int] = []
    last = indices[0] - cooldown
    for idx in indices.tolist():
        if idx - last >= cooldown:
            keep.append(idx)
            last = idx
    return keep


def _grad1_window_days(overrides: dict, side: str) -> int:
    key = f'GRAD1_{side}_WIN_DAYS'
    return max(int(overrides[key]), 1)


def _grad1_zscore_min(overrides: dict, side: str) -> float:
    key = f'GRAD1_{side}_Z_MIN'
    return float(overrides[key])


def _grad1ZscoreMask(
    ctx,
    allowReg: np.ndarray,
    g1: np.ndarray,
    overrides: dict,
    side: str,
) -> np.ndarray:
    winDays = _grad1_window_days(overrides, side)
    winBars = max(int(round(winDays * bars_per_day(ctx))), 1)
    thresh = float(_grad1_zscore_min(overrides, side))
    z, valid = zscoreSeries(ctx, g1, winBars, "g1p1")
    sign = -1.0 if side == 'BUY' else 1.0
    signed = z * sign
    ready = allowReg & valid
    return ready & (signed >= thresh)


def _grad1Mask(
    ctx,
    allowReg: np.ndarray,
    g1: np.ndarray,
    overrides: dict,
    params: Params,
    side: str,
) -> np.ndarray:
    return _grad1ZscoreMask(ctx, allowReg, g1, overrides, side)


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
    """Vectorized BUY/SELL flag generation using selectable spacing gates."""
    n = len(ctx["closes"]) if ctx and ctx.get("closes") is not None else 0

    overrides = overrides or {}
    g1 = np.asarray(signals["g1P1"], dtype=float)
    g99 = np.asarray(signals["g1P3"], dtype=float)
    s12 = np.asarray(signals["s12"], dtype=float)
    s23 = np.asarray(signals["s23"], dtype=float)
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
    allowBuy = (trendCode == -1)
    allowSell = (trendCode == 1)

    buySpacing = spacingState(ctx, trendCode, allowBuy, s12, s23, overrides)
    sellSpacing = spacingState(ctx, trendCode, allowSell, s12, s23, overrides)
    legBuy = buySpacing.mask
    legSell = sellSpacing.mask
    gradBuy = _grad1Mask(ctx, allowBuy, g1, overrides, params, 'BUY')
    gradSell = _grad1Mask(ctx, allowSell, g1, overrides, params, 'SELL')

    buyMask = valid & allowBuy & legBuy & gradBuy
    sellMask = valid & allowSell & legSell & gradSell

    buyIdx = np.flatnonzero(buyMask)
    sellIdx = np.flatnonzero(sellMask)
    cd = max(int(params.COOLDOWN), 0)
    buyIdxF = _enforceCooldown(buyIdx, cd)
    sellIdxF = _enforceCooldown(sellIdx, cd)

    # Dynamic spacing since phase anchor / last BUY/SELL using macro dyn%:
    # - requirePct = abs(macroDyn[idx]) * macroMult[idx] (per-side)
    # - macroMult weights backbone + transition (MACRO_*_MULT_*).
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

    buyMultBull = float(overrides['MACRO_BUY_MULT_BULL'])
    buyMultBear = float(overrides['MACRO_BUY_MULT_BEAR'])
    buyMultRev = float(overrides['MACRO_BUY_MULT_REV'])
    buyMultRoll = float(overrides['MACRO_BUY_MULT_ROLL'])
    sellMultBull = float(overrides['MACRO_SELL_MULT_BULL'])
    sellMultBear = float(overrides['MACRO_SELL_MULT_BEAR'])
    sellMultRev = float(overrides['MACRO_SELL_MULT_REV'])
    sellMultRoll = float(overrides['MACRO_SELL_MULT_ROLL'])

    macroBull = dirCode > 0
    macroBear = dirCode < 0
    macroRev = macroBear & (momCode > 0)
    macroRoll = macroBull & (momCode < 0)

    buyMult = np.ones(n, dtype=float)
    buyMult[macroBull] *= buyMultBull
    buyMult[macroBear] *= buyMultBear
    buyMult[macroRev] *= buyMultRev
    buyMult[macroRoll] *= buyMultRoll

    sellMult = np.ones(n, dtype=float)
    sellMult[macroBull] *= sellMultBull
    sellMult[macroBear] *= sellMultBear
    sellMult[macroRev] *= sellMultRev
    sellMult[macroRoll] *= sellMultRoll

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
                keep = deltaPct >= req
            else:
                keep = True
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
                keep = deltaPct >= req
            else:
                keep = True
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
    return out


# ---------------------- Printouts (moved from printouts.py) ---------------
def utcDt(ms: int) -> datetime:
    return datetime.utcfromtimestamp(ms / 1000.0)


def printOracleHeader(klines, idx: int, side: str) -> None:
    dt = utcDt(klines[idx][0])
    label = "ORACLE SELL" if side == "SELL" else "ORACLE BUY"
    color = GREEN if side == "SELL" else RED
    print(f"{color}{BAR}\n=== {label} at {dt:%Y-%m-%d %H:%M} UTC ===")
    print(f"{BAR}{RESET}")


def printHeader(klines, idx: int, side: str) -> None:
    dt = utcDt(klines[idx][0])
    label = "SELL" if side == "SELL" else "BUY"
    color = ORANGE if side == "SELL" else YELLOW
    header = (
        f"\n{color}{BAR}\n=== {label} at {dt:%Y-%m-%d %H:%M} UTC ==="
    )
    print(header)
    print(f"{BAR}{RESET}")


def printGradients(
    ctx,
    idx: int,
    side: str,
    params: Params,
) -> None:
    p1, p2, p3 = ctx["periods"]
    g1_99, _, _, _ = liveGradsAt(ctx, p3, idx)

    win = max(p3, SNAPSHOT_MIN_WIN_DEFAULT)
    refIdx = max(0, idx - win)
    refPrice = ctx["closes"][refIdx]
    now = ctx["closes"][idx]
    pctMove = ((now - refPrice) / refPrice) * 100.0
    ath = ctx["ath"][idx]

    print("\n--- Context ---")
    print(
        f"ATH={ath:.3f} Now={now:.3f} Ref[{win}]={refPrice:.3f} "
        f"Δ%={pctMove:+.1f}%"
    )
    print(f"MA3 g1={g1_99:.3f}")

    print("\n--- Gradients (causal) ---")
    for p in (p1, p2, p3):
        g1, g2, g3, g4 = liveGradsAt(ctx, p, idx)
        print(f"P{p}: g1={g1:.3f} g2={g2:.3f} g3={g3:.3f} g4={g4:.3f}")

    s12, s23, s13 = calcSpacing(ctx, idx)
    print("\n--- MA Spacings ---")
    print(f"ma1-ma2={s12:.3f}  ma2-ma3={s23:.3f}  ma1-ma3={s13:.3f}")
