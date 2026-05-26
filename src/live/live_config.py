#!/usr/bin/env python3
# live_config.py - live profile loading and scalar normalization.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from repo_paths import rootPath


def loadJson(path: Path) -> dict:
    # Load a JSON profile from disk.
    with open(path) as fh:
        return json.load(fh)


def pickScalar(spec: Any, defaultVal: Any) -> Any:
    # Return the first scalar from list/range-like profile specs.
    if spec is None:
        return defaultVal
    if isinstance(spec, list):
        if spec:
            return pickScalar(spec[0], defaultVal)
        return defaultVal
    if isinstance(spec, dict) and 'range' in spec:
        rng = spec.get('range')
        if isinstance(rng, list) and rng:
            return pickScalar(rng[0], defaultVal)
    if (
        isinstance(spec, dict)
        and all(key in spec for key in ('start', 'stop', 'step'))
    ):
        return pickScalar(spec.get('start'), defaultVal)
    return spec


def scalarValue(spec: Any, defaultVal: Any) -> Any:
    # Return the first scalar value from tuner-style specs.
    return pickScalar(spec, defaultVal)


def boolValue(spec: Any) -> bool:
    # Convert profile value into a strict bool.
    raw = scalarValue(spec, False)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        txt = raw.strip().lower()
        return txt in ('1', 'true', 'yes', 'on')
    return bool(raw)


def intervalsFromConfig(config: Dict[str, Any]) -> List[str]:
    # Normalize profile intervals into a string list.
    ivals = config['intervals']
    if isinstance(ivals, str):
        return [item.strip() for item in ivals.split(',') if item.strip()]
    return [str(item).strip() for item in ivals]


def configPath(config: Dict[str, Any], profilePath: Path) -> Path:
    # Resolve config.ini path from profile key.
    cfgPath = Path(str(config['config_path']))
    if cfgPath.is_absolute():
        return cfgPath
    return rootPath(cfgPath).resolve()


def overridesFromDict(rawData: Dict[str, Any]) -> Dict[str, Any]:
    # Normalize profile overrides to scalar int/float/string values.
    intKeys = {
        'COOLDOWN',
        'CHART_CHUNK_SIZE',
        'GRAD1_BUY_WIN_DAYS',
        'GRAD1_SELL_WIN_DAYS',
        'MACRO_NRG_WIN_DAYS',
        'MACRO_GRAD_WIN_DAYS',
        'MACRO_P1',
        'MACRO_GRAD_PERIOD',
        'MACRO_P3',
        'PHASE_BUY_PORTIONS',
        'PHASE_SELL_PORTIONS',
        'ULTRA_EXIT_HOLD_DAYS',
        'HOLDOUT_START_MIN_PCT',
        'HOLDOUT_START_MAX_PCT',
        'HOLDOUT_START_STEP_PCT',
        'PEAK_LOCK_CONFIRM_BARS',
        'PEAK_LOCK_REQUIRE_EDGE_RISK',
    }
    floatKeys = {
        'FINAL_PORTION_PCT',
        'GRAD1_BUY_Z_MIN',
        'GRAD1_SELL_Z_MIN',
        'MACRO_NRG_Z_MIN',
        'MACRO_NRG_Z_MAX',
        'MACRO_DYN_PCT_MIN',
        'MACRO_DYN_PCT_MAX',
        'MACRO_GRAD_Z_MIN',
        'MACRO_GRAD_Z_MAX',
        'MACRO_MULT_GRAD_MIN',
        'MACRO_MULT_GRAD_MAX',
        'MACRO_SELL_RELAX_PCT',
        'WALLET_SEED_QUOTE',
        'WALLET_FEE_RATE',
        'QUOTE_TO_AUD_RATE',
        'ULTRA_SELL_MULT',
        'ULTRA_BRIDGE_DAYS',
        'DAILY_DOWN_BUY_MULT',
        'CRAB_ASSET_CAP_PCT',
        'ULTRA_EXIT_DEPTH',
        'ULTRA_GAIN_MIN_PCT',
        'ULTRA_GAIN_MAX_PCT',
        'ULTRA_EXPOSURE_TARGET',
        'PEAK_LOCK_CAP_PCT',
        'PEAK_LOCK_UNLOCK_GAIN_PCT',
        'PEAK_LOCK_REENTRY_STEP_PCT',
        'PEAK_LOCK_ARM_GAIN_PCT',
        'PEAK_LOCK_GIVEBACK_PCT',
        'PEAK_LOCK_MAX_DAYS',
        'PEAK_LOCK_EDGE_DRAW_PCT',
        'PEAK_LOCK_EDGE_SLOPE_DAYS',
        'PEAK_LOCK_MA_DAYS',
        'PEAK_LOCK_KP',
        'PEAK_LOCK_KI',
        'PEAK_LOCK_KD',
        'PEAK_LOCK_INTEGRAL_DECAY',
        'PEAK_LOCK_ENTRY_THRESHOLD',
        'PEAK_LOCK_EXIT_THRESHOLD',
        'PEAK_LOCK_RELEASE_TARGET_PCT',
        'PEAK_LOCK_ULTRA_GRACE_DAYS',
    }
    outData: Dict[str, Any] = {}
    for key, rawVal in rawData.items():
        val = pickScalar(rawVal, rawVal)
        if key in intKeys:
            outData[key] = int(val)
        elif key in floatKeys:
            outData[key] = float(val)
        else:
            outData[key] = val
    return outData
