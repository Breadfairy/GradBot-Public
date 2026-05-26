#!/usr/bin/env python3
# params.py – shared parameter dataclasses and override helpers.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from tune.schema import floatKeys, intKeys

# ======================================================================
# Tune parameter structures
# ======================================================================


@dataclass(frozen=True)
class TuneParams:
    macroDynWindowDays: int
    macroDynZMin: float
    macroDynZMax: float
    macroDynPctMin: float
    macroDynPctMax: float
    macroInterval: str
    macroP1: int
    macroP3: int
    macroGradPeriod: int
    macroGradWinDays: int
    macroGradZMin: float
    macroGradZMax: float
    macroGradMultMin: float
    macroGradMultMax: float
    macroSellRelaxPct: float
    grad1BuyZscoreMin: float
    grad1SellZscoreMin: float
    grad1BuyWindowDays: int
    grad1SellWindowDays: int
    phaseBuy: int
    phaseSell: int
    finalPortionPct: float
    cooldown: int
    taxMode: str
    seedAssetPct: float
    dailyStrongSellMult: float
    dailyStrongTargetPct: float
    dailyBridgeDays: float
    dailyDownBuyMult: float
    dailyCrabAssetCapPct: float
    dailyLockTargetPct: float
    dailyLockGainPct: float
    dailyLockNearHighPct: float
    dailyLockMaxDays: int
    postUltraCoastTargetPct: float
    postUltraGivebackPct: float
    postUltraReaccumPct: float
    postUltraDoubleTopPct: float
    postUltraMaxDays: float
    postUltraLockMinAssetPct: float
    postUltraLockMaxAssetPct: float
    postUltraLockGivebackPct: float
    postUltraLockReaccumPct: float
    postUltraLockDoubleTopPct: float
    postUltraLockMaxDays: float
    peakLockCapPct: float
    peakLockUnlockGainPct: float
    peakLockReentryStepPct: float
    peakLockArmGainPct: float
    peakLockGivebackPct: float
    peakLockMaxDays: float
    peakLockEdgeDrawPct: float
    peakLockEdgeSlopeDays: float
    peakLockRequireEdgeRisk: int
    peakLockMaDays: float
    peakLockKp: float
    peakLockKi: float
    peakLockKd: float
    peakLockIntegralDecay: float
    peakLockEntryThreshold: float
    peakLockExitThreshold: float
    peakLockConfirmBars: int
    peakLockReleaseTargetPct: float
    peakLockUltraGraceDays: float


# ======================================================================
# Override helpers
# ======================================================================


def overridesFromDict(dct: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize overrides to scalars where applicable."""
    out: Dict[str, Any] = {}
    intSet = intKeys()
    floatSet = floatKeys()
    for k, v in dct.items():
        val = _pickScalar(v, v)
        if k in intSet:
            out[k] = int(val)
        elif k in floatSet:
            out[k] = float(val)
        else:
            out[k] = val
    return out


def _pickScalar(spec, default):
    if spec is None:
        return default
    if isinstance(spec, list):
        return _pickScalar(spec[0] if spec else default, default)
    if isinstance(spec, dict) and 'range' in spec:
        rng = spec.get('range')
        if isinstance(rng, list) and rng:
            return _pickScalar(rng[0], default)
    if (
        isinstance(spec, dict)
        and all(k in spec for k in ('start', 'stop', 'step'))
    ):
        return _pickScalar(spec.get('start'), default)
    return spec
