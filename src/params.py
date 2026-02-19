#!/usr/bin/env python3
# params.py – shared parameter dataclasses and override helpers.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

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
    macroP2: int
    macroP3: int
    macroGradWinDays: int
    macroGradZMin: float
    macroGradZMax: float
    macroGradMultMin: float
    macroGradMultMax: float
    grad1BuyZscoreMin: float
    grad1SellZscoreMin: float
    grad1BuyWindowDays: int
    grad1SellWindowDays: int
    phaseBuy: int
    phaseSell: int
    finalPortionPct: float
    cooldown: int
    taxMode: str
    annualIncomeBase: float
    profitSweepInterval: str
    profitSweepShare: float
    spacingZscoreMin12: float
    spacingZscoreMin23: float
    spacingWindowDays12: int
    spacingWindowDays23: int
    spacingEnergyModel: str
    spacingEnergyWinDays: int
    spacingEnergyMin12: float
    spacingEnergyMin23: float


# ======================================================================
# Override helpers
# ======================================================================


def overridesFromDict(dct: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize overrides to scalars where applicable."""
    out: Dict[str, Any] = {}
    intKeys = {
        'COOLDOWN',
        'CHART_CHUNK_SIZE',
        'SPACING_WIN_DAYS_12',
        'SPACING_WIN_DAYS_23',
        'GRAD1_BUY_WIN_DAYS',
        'GRAD1_SELL_WIN_DAYS',
        'MICRO_NRG_WIN_DAYS',
        'MACRO_NRG_WIN_DAYS',
        'MACRO_GRAD_WIN_DAYS',
        'MACRO_P1',
        'MACRO_P2',
        'MACRO_P3',
        'PHASE_BUY_PORTIONS',
        'PHASE_SELL_PORTIONS',
    }
    floatKeys = {
        'SPACING_Z_MIN_12', 'SPACING_Z_MIN_23',
        'MICRO_NRG_MIN_12', 'MICRO_NRG_MIN_23',
        'FINAL_PORTION_PCT',
        'GRAD1_BUY_Z_MIN', 'GRAD1_SELL_Z_MIN',
        'MACRO_NRG_Z_MIN', 'MACRO_NRG_Z_MAX',
        'MACRO_DYN_PCT_MIN', 'MACRO_DYN_PCT_MAX',
        'MACRO_GRAD_Z_MIN', 'MACRO_GRAD_Z_MAX',
        'MACRO_MULT_GRAD_MIN', 'MACRO_MULT_GRAD_MAX',
        'MACRO_BUY_MULT_BULL', 'MACRO_BUY_MULT_BEAR',
        'MACRO_BUY_MULT_REV', 'MACRO_BUY_MULT_ROLL',
        'MACRO_SELL_MULT_BULL', 'MACRO_SELL_MULT_BEAR',
        'MACRO_SELL_MULT_REV', 'MACRO_SELL_MULT_ROLL',
        'WALLET_SEED_QUOTE',
        'WALLET_FEE_RATE',
        'QUOTE_TO_AUD_RATE',
        'PROFIT_SWEEP_SHARE',
        'ANNUAL_INCOME_BASE',
    }
    for k, v in dct.items():
        val = _pickScalar(v, v)
        if k in intKeys:
            out[k] = int(val)
        elif k in floatKeys:
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
