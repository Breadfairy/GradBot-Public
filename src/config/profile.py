#!/usr/bin/env python3
# profile.py – profile loading helpers (validation + overrides).

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from config.params import overridesFromDict
from tune.schema import requiredKeys


HOLDOUT_START_KEYS = {
    "HOLDOUT_START_MIN_PCT",
    "HOLDOUT_START_MAX_PCT",
    "HOLDOUT_START_STEP_PCT",
}

HOLDOUT_START_DEFAULTS = {
    "HOLDOUT_START_MIN_PCT": 0,
    "HOLDOUT_START_MAX_PCT": 20,
    "HOLDOUT_START_STEP_PCT": 5,
}


def loadJson(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)

LEGACY_PROFILE_KEYS = {
    "HOLDOUT_DAYS": (
        "Use primer_days/training_days/tuner_days/holdout_days instead."
    ),
    "days": (
        "Use primer_days/training_days/tuner_days/holdout_days instead."
    ),
    "WALLET_TAX_RATE": (
        "Remove WALLET_TAX_RATE; tax handling is derived by TAX_MODE."
    ),
    "MACRO_P2": (
        "Rename MACRO_P2 to MACRO_GRAD_PERIOD."
    ),
    "SPACING_Z_MIN_12": (
        "Remove spacing and micro-energy keys; those gates were deleted."
    ),
    "SPACING_Z_MIN_23": (
        "Remove spacing and micro-energy keys; those gates were deleted."
    ),
    "SPACING_WIN_DAYS_12": (
        "Remove spacing and micro-energy keys; those gates were deleted."
    ),
    "SPACING_WIN_DAYS_23": (
        "Remove spacing and micro-energy keys; those gates were deleted."
    ),
    "MICRO_NRG_MODEL": (
        "Remove spacing and micro-energy keys; those gates were deleted."
    ),
    "MICRO_NRG_WIN_DAYS": (
        "Remove spacing and micro-energy keys; those gates were deleted."
    ),
    "MICRO_NRG_MIN_12": (
        "Remove spacing and micro-energy keys; those gates were deleted."
    ),
    "MICRO_NRG_MIN_23": (
        "Remove spacing and micro-energy keys; those gates were deleted."
    ),
    "SUMMARY_LABEL": (
        "Remove SUMMARY_LABEL; summary label overrides were deleted."
    ),
    "ANNUAL_INCOME_BASE": (
        "Remove ANNUAL_INCOME_BASE; income mode now uses a fixed base."
    ),
    "PROFIT_SWEEP_INTERVAL": (
        "Remove PROFIT_SWEEP_INTERVAL; profit sweep was deleted."
    ),
    "PROFIT_SWEEP_SHARE": (
        "Remove PROFIT_SWEEP_SHARE; profit sweep was deleted."
    ),
    "GATE_CLUSTER_ENABLE": (
        "Remove 1h cluster keys; daily posture uses DAILY_* keys only."
    ),
    "CLUSTER_ENABLE": (
        "Remove 1h cluster keys; daily posture uses DAILY_* keys only."
    ),
    "CLUSTER_FEATURE_SET": (
        "Remove 1h cluster keys; daily posture uses DAILY_* keys only."
    ),
    "CLUSTER_K": (
        "Remove 1h cluster keys; daily posture uses DAILY_* keys only."
    ),
    "CLUSTER_FIT_SCOPE": (
        "Remove 1h cluster keys; daily posture uses DAILY_* keys only."
    ),
    "CLUSTER_POLICY_MODE": (
        "Remove 1h cluster keys; daily posture uses DAILY_* keys only."
    ),
    "CLUSTER_POLICY_TARGET": (
        "Remove 1h cluster keys; daily posture uses DAILY_* keys only."
    ),
    "ML_CLUSTER_MODEL": (
        "Remove 1h cluster keys; daily posture uses DAILY_* keys only."
    ),
    "GATE_TREND_ENABLE": (
        "Gate toggles are frozen on; remove GATE_* keys."
    ),
    "GATE_GRAD1_BUY_ENABLE": (
        "Gate toggles are frozen on; remove GATE_* keys."
    ),
    "GATE_GRAD1_SELL_ENABLE": (
        "Gate toggles are frozen on; remove GATE_* keys."
    ),
    "GATE_COOLDOWN_ENABLE": (
        "Gate toggles are frozen on; remove GATE_* keys."
    ),
    "GATE_MACRO_MOVE_ENABLE": (
        "Gate toggles are frozen on; remove GATE_* keys."
    ),
    "DEFENSE_ENABLE": (
        "Local-HODL defense is frozen off; remove DEFENSE_* keys."
    ),
    "DEFENSE_RESET_DAYS": (
        "Local-HODL defense is frozen off; remove DEFENSE_* keys."
    ),
    "DEFENSE_ENTER_PCT": (
        "Local-HODL defense is frozen off; remove DEFENSE_* keys."
    ),
    "DEFENSE_EXIT_PCT": (
        "Local-HODL defense is frozen off; remove DEFENSE_* keys."
    ),
    "DEFENSE_SELL_MULT": (
        "Local-HODL defense is frozen off; remove DEFENSE_* keys."
    ),
    "DEFENSE_MAX_DAYS": (
        "Local-HODL defense is frozen off; remove DEFENSE_* keys."
    ),
    "DAILY_CLUSTER_ENABLE": (
        "Daily posture is enabled by DAILY_CLUSTER_PATH; remove enable keys."
    ),
    "DAILY_STRONG_CLUSTER": (
        "Daily cluster identity is frozen in code; remove identity keys."
    ),
    "DAILY_DOWN_CLUSTERS": (
        "Daily cluster identity is frozen in code; remove identity keys."
    ),
    "DAILY_DOWN_MASK": (
        "Daily cluster identity is frozen in code; remove identity keys."
    ),
    "DAILY_STRONG_SELL_MULT": (
        "Use ULTRA_SELL_MULT for confirmed ultraBull sell blocking."
    ),
    "DAILY_STRONG_TARGET_PCT": (
        "Use ULTRA_EXPOSURE_TARGET for confirmed ultraBull exposure."
    ),
    "DAILY_LOCK_TARGET_PCT": (
        "Use ULTRA_EXIT_DEPTH for dynamic post-ultra exits."
    ),
    "DAILY_LOCK_GAIN_PCT": (
        "Use ULTRA_GAIN_MIN_PCT for dynamic post-ultra exits."
    ),
    "DAILY_LOCK_NEAR_HIGH_PCT": (
        "Use ULTRA_GAIN_MAX_PCT for dynamic post-ultra exits."
    ),
    "DAILY_LOCK_MIN_STRONG_DAYS": (
        "Ultra exits now score entry-to-exit gain; remove this key."
    ),
    "DAILY_LOCK_MAX_DAYS": (
        "Use ULTRA_EXIT_HOLD_DAYS for dynamic post-ultra exits."
    ),
    "DAILY_LOCK_ENABLE": (
        "Daily lock exit mode is frozen on; remove lock toggle keys."
    ),
    "DAILY_LOCK_MODE": (
        "Daily lock exit mode is frozen on; remove lock mode keys."
    ),
    "DAILY_LOCK_MODE_CODE": (
        "Daily lock exit mode is frozen on; remove lock mode keys."
    ),
    "DAILY_LOCK_PULLBACK_PCT": (
        "Daily pullback lock mode was removed; remove pullback keys."
    ),
    "DAILY_LOCK_QUALIFIED_EXIT": (
        "Daily qualified-exit branch was removed; remove branch keys."
    ),
    "DAILY_LOCK_EXIT_NEAR_HIGH_MULT": (
        "Daily qualified-exit branch was removed; remove branch keys."
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
        if (
            'range' in spec
            and isinstance(spec['range'], list)
            and spec['range']
        ):
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
    "holdoutStartParts",
    "profileWindows",
    "profile_windows",
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
    scoped = profilesDir / profileInput
    searchDirs = (
        ("user",),
        ("user", "results"),
        ("codex",),
        ("codex", "results"),
    )
    if not candidate.is_absolute() and scoped.is_file():
        candidate = scoped
    elif "/" not in profileInput:
        short = profilesDir / profileInput
        if short.is_file():
            candidate = short
        else:
            for i in searchDirs:
                nested = profilesDir.joinpath(*i) / profileInput
                if nested.is_file():
                    candidate = nested
                    break
    return candidate.resolve()


def intervalsFromConfig(config: Dict[str, object]) -> List[str]:
    """Normalise config['intervals'] into a list of strings."""
    iv = config["intervals"]
    if isinstance(iv, str):
        return [s.strip() for s in iv.split(",") if s.strip()]
    return [str(x) for x in iv]


def windowParts(cfg: Dict[str, Any]) -> tuple[int, int, int, int, int]:
    rejectLegacyKeys(cfg, context="windowParts")
    windowKeys = (
        "primer_days",
        "training_days",
        "tuner_days",
        "holdout_days",
    )
    missing = [k for k in windowKeys if k not in cfg]
    if missing:
        raise SystemExit(
            "profile missing required window keys: "
            + ", ".join(sorted(missing))
        )
    primerDays = int(scalarValue(cfg["primer_days"], 0))
    trainingDays = int(scalarValue(cfg["training_days"], 0))
    tunerDays = int(scalarValue(cfg["tuner_days"], 0))
    holdoutDays = int(scalarValue(cfg["holdout_days"], 0))
    totalDays = primerDays + trainingDays + tunerDays + holdoutDays
    return primerDays, trainingDays, tunerDays, holdoutDays, totalDays


def profileWindows(cfg: Dict[str, Any]) -> tuple[int, int, int, int, int]:
    return windowParts(cfg)


profile_windows = profileWindows


def holdoutStartParts(
    cfg: Dict[str, Any],
    startMinPct: int | None = None,
    startMaxPct: int | None = None,
    startStepPct: int | None = None,
) -> tuple[int, int, int]:
    minPct = int(
        scalarValue(
            cfg.get("HOLDOUT_START_MIN_PCT"),
            HOLDOUT_START_DEFAULTS["HOLDOUT_START_MIN_PCT"],
        )
    )
    maxPct = int(
        scalarValue(
            cfg.get("HOLDOUT_START_MAX_PCT"),
            HOLDOUT_START_DEFAULTS["HOLDOUT_START_MAX_PCT"],
        )
    )
    stepPct = int(
        scalarValue(
            cfg.get("HOLDOUT_START_STEP_PCT"),
            HOLDOUT_START_DEFAULTS["HOLDOUT_START_STEP_PCT"],
        )
    )
    if startMinPct is not None:
        minPct = int(startMinPct)
    if startMaxPct is not None:
        maxPct = int(startMaxPct)
    if startStepPct is not None:
        stepPct = int(startStepPct)
    return minPct, maxPct, stepPct


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
        requireKeys(dct, requiredKeys("backtest"), context='backtest overrides')
        return

    # tuner
    rejectLegacyKeys(dct, context="tuner profile")
    _requireTickers(dct)
    requireKeys(dct, requiredKeys("tuner"), context='tuner profile')
