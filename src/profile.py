#!/usr/bin/env python3
# profile.py – profile loading helpers (validation + overrides).

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from params import overridesFromDict


def loadJson(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)

LEGACY_PROFILE_KEYS = {
    "HOLDOUT_DAYS": (
        "Use primer_days/tuner_days/holdout_days instead."
    ),
    "days": (
        "Use primer_days/tuner_days/holdout_days instead."
    ),
    "WALLET_TAX_RATE": (
        "Remove WALLET_TAX_RATE; CGT tax rate is derived from "
        "ANNUAL_INCOME_BASE via AU brackets."
    ),
}


def ensureFinalPortionPct(cfg: dict) -> dict:
    if 'FINAL_PORTION_PCT' not in cfg:
        raise SystemExit("profile missing required key: FINAL_PORTION_PCT")
    return cfg


def scalarValue(spec: Any, default: Any | None = None) -> Any:
    """Return the first scalar-like value from lists/range-like dicts."""
    if spec is None:
        return default
    if isinstance(spec, list):
        return scalarValue(spec[0] if spec else default, default)
    if isinstance(spec, dict):
        if 'range' in spec and isinstance(spec['range'], list) and spec['range']:
            return scalarValue(spec['range'][0], default)
        if all(k in spec for k in ('start', 'stop', 'step')):
            return scalarValue(spec['start'], default)
    return spec


def requireKeys(obj: dict, keys: Iterable[str], context: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        msg = (
            f"missing required keys for {context}: "
            + ", ".join(sorted(missing))
            + "\nSee scripts/PARAMS.md (Required keys)."
        )
        raise SystemExit(msg)

def rejectLegacyKeys(obj: dict, context: str) -> None:
    present = [k for k in sorted(LEGACY_PROFILE_KEYS) if k in obj]
    if present:
        hints: List[str] = []
        for k in present:
            hint = LEGACY_PROFILE_KEYS.get(k)
            if hint and hint not in hints:
                hints.append(hint)
        extra = ("\n" + "\n".join(hints)) if hints else ""
        raise SystemExit(
            f"legacy keys not allowed for {context}: "
            + ", ".join(present)
            + extra
        )


def overrides(dct: dict) -> dict:
    if not isinstance(dct, dict):
        return {}
    # Delegate to shared overrides normaliser so tuner/backtest
    # paths treat scalar-like specs consistently.
    return overridesFromDict(dct)


__all__ = [
    "loadJson",
    "ensureFinalPortionPct",
    "resolveProfilePath",
    "scalarValue",
    "requireKeys",
    "overrides",
    "windowParts",
    "intervalsFromConfig",
    "validate",
]


def resolveProfilePath(profileInput: str, profilesDir: Path) -> Path:
    candidate = Path(profileInput)
    if "/" not in profileInput:
        short = profilesDir / profileInput
        if short.is_file():
            candidate = short
    return candidate.resolve()


def intervalsFromConfig(config: Dict[str, object]) -> List[str]:
    """Normalise config['intervals'] into a list of strings."""
    iv = config["intervals"]
    if isinstance(iv, str):
        return [s.strip() for s in iv.split(",") if s.strip()]
    return [str(x) for x in iv]


def windowParts(cfg: Dict[str, Any]) -> tuple[int, int, int, int]:
    rejectLegacyKeys(cfg, context="windowParts")
    windowKeys = ("primer_days", "tuner_days", "holdout_days")
    missing = [k for k in windowKeys if k not in cfg]
    if missing:
        raise SystemExit(
            "profile missing required window keys: "
            + ", ".join(sorted(missing))
        )
    primerDays = int(scalarValue(cfg["primer_days"], 0))
    tunerDays = int(scalarValue(cfg["tuner_days"], 0))
    holdoutDays = int(scalarValue(cfg["holdout_days"], 0))
    totalDays = primerDays + tunerDays + holdoutDays
    return primerDays, tunerDays, holdoutDays, totalDays


def _requireTickers(dct: dict) -> List[str]:
    """Normalise ticker/tickers config into a tickers list.

    - Accepts 'tickers' (list) only.
    - Ensures at least one non-empty entry.
    - Writes dct['tickers'] = [T1, ...] and dct['ticker'] = T1.
    """
    raw = dct.get('tickers')
    if isinstance(raw, list) and raw:
        tickers = [
            str(t).strip().upper() for t in raw if str(t).strip()
        ]
    else:
        raise SystemExit("profile missing non-empty 'tickers' list")
    if not tickers:
        raise SystemExit(
            "profile requires at least one ticker in 'tickers'"
        )
    dct['tickers'] = tickers
    dct['ticker'] = tickers[0]
    return tickers


def validate(dct: dict, kind: str) -> None:
    """Validate config/overrides for a mode: 'backtest' or 'tuner'.

    - backtest: validate overrides dict (post profile extraction)
    - tuner: validate full profile config used by axesFromConfig
    """
    k = kind.strip().lower()
    if k not in ("backtest", "tuner"):
        raise SystemExit(f"unknown validation kind: {kind}")

    if k == "backtest":
        rejectLegacyKeys(dct, context="backtest overrides")
        required = [
            'SUMMARY_LABEL',
            'CHART_CHUNK_SIZE',
            'WALLET_SEED_QUOTE', 'WALLET_FEE_RATE', 'QUOTE_TO_AUD_RATE',
            'GRAD1_BUY_Z_MIN', 'GRAD1_SELL_Z_MIN',
            'GRAD1_BUY_WIN_DAYS', 'GRAD1_SELL_WIN_DAYS',
            'SPACING_Z_MIN_12', 'SPACING_Z_MIN_23',
            'SPACING_WIN_DAYS_12', 'SPACING_WIN_DAYS_23',
            'MICRO_NRG_MODEL', 'MICRO_NRG_WIN_DAYS',
            'MICRO_NRG_MIN_12', 'MICRO_NRG_MIN_23',
            'MACRO_INTERVAL',
            'MACRO_NRG_WIN_DAYS', 'MACRO_NRG_Z_MIN', 'MACRO_NRG_Z_MAX',
            'MACRO_DYN_PCT_MIN', 'MACRO_DYN_PCT_MAX',
            'MACRO_P1', 'MACRO_P2', 'MACRO_P3',
            'MACRO_GRAD_WIN_DAYS', 'MACRO_GRAD_Z_MIN', 'MACRO_GRAD_Z_MAX',
            'MACRO_MULT_GRAD_MIN', 'MACRO_MULT_GRAD_MAX',
            'MACRO_BUY_MULT_BULL', 'MACRO_BUY_MULT_BEAR',
            'MACRO_BUY_MULT_REV', 'MACRO_BUY_MULT_ROLL',
            'MACRO_SELL_MULT_BULL', 'MACRO_SELL_MULT_BEAR',
            'MACRO_SELL_MULT_REV', 'MACRO_SELL_MULT_ROLL',
            'PHASE_BUY_PORTIONS', 'PHASE_SELL_PORTIONS',
            'FINAL_PORTION_PCT', 'COOLDOWN',
            'TAX_MODE', 'ANNUAL_INCOME_BASE',
            'PROFIT_SWEEP_INTERVAL', 'PROFIT_SWEEP_SHARE',
        ]
        requireKeys(dct, required, context='backtest overrides')
        return

    # tuner
    rejectLegacyKeys(dct, context="tuner profile")
    _requireTickers(dct)
    baseKeys = ['primer_days', 'tuner_days', 'holdout_days']
    required = baseKeys + [
        'intervals', 'p1', 'p2', 'p3',
        'SUMMARY_LABEL',
        'CHART_CHUNK_SIZE',
        'WALLET_SEED_QUOTE', 'WALLET_FEE_RATE', 'QUOTE_TO_AUD_RATE',
        'GRAD1_BUY_Z_MIN', 'GRAD1_SELL_Z_MIN',
        'GRAD1_BUY_WIN_DAYS', 'GRAD1_SELL_WIN_DAYS',
        'SPACING_Z_MIN_12', 'SPACING_Z_MIN_23',
        'SPACING_WIN_DAYS_12', 'SPACING_WIN_DAYS_23',
        'MICRO_NRG_MODEL', 'MICRO_NRG_WIN_DAYS',
        'MICRO_NRG_MIN_12', 'MICRO_NRG_MIN_23',
        'MACRO_INTERVAL',
        'MACRO_NRG_WIN_DAYS', 'MACRO_NRG_Z_MIN', 'MACRO_NRG_Z_MAX',
        'MACRO_DYN_PCT_MIN', 'MACRO_DYN_PCT_MAX',
        'MACRO_P1', 'MACRO_P2', 'MACRO_P3',
        'MACRO_GRAD_WIN_DAYS', 'MACRO_GRAD_Z_MIN', 'MACRO_GRAD_Z_MAX',
        'MACRO_MULT_GRAD_MIN', 'MACRO_MULT_GRAD_MAX',
        'MACRO_BUY_MULT_BULL', 'MACRO_BUY_MULT_BEAR',
        'MACRO_BUY_MULT_REV', 'MACRO_BUY_MULT_ROLL',
        'MACRO_SELL_MULT_BULL', 'MACRO_SELL_MULT_BEAR',
        'MACRO_SELL_MULT_REV', 'MACRO_SELL_MULT_ROLL',
        'PHASE_BUY_PORTIONS', 'PHASE_SELL_PORTIONS', 'FINAL_PORTION_PCT',
        'COOLDOWN',
        'TAX_MODE', 'ANNUAL_INCOME_BASE',
        'PROFIT_SWEEP_INTERVAL', 'PROFIT_SWEEP_SHARE',
        'out',
    ]
    requireKeys(dct, required, context='tuner profile')
