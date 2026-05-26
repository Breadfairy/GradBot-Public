#!/usr/bin/env python3

########################################################################
# Imports
########################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


########################################################################
# Schema Records
########################################################################

@dataclass(frozen=True)
class ParamSpec:
    key: str
    kind: str
    backtest: bool = False
    tuner: bool = False
    axisName: str | None = None
    caster: Callable[[Any], Any] | None = None
    rowKind: str | None = None
    default: Any | None = None


WINDOW_KEYS = [
    "primer_days",
    "training_days",
    "tuner_days",
    "holdout_days",
]

ENTRY_KEYS = [
    "ticker",
    "tickers",
    "intervals",
    "p1",
    "p2",
    "p3",
]

PARAM_SPECS = [
    ParamSpec("DAILY_CLUSTER_PATH", "str", False, False),
    ParamSpec("DAILY_CLUSTER_MODEL_PATH", "str", False, False),
    ParamSpec(
        "ULTRA_SELL_MULT",
        "float",
        False,
        False,
        "dailyStrongSellMultValues",
        float,
        default=0.0,
    ),
    ParamSpec(
        "ULTRA_EXPOSURE_TARGET",
        "float",
        False,
        False,
        "dailyStrongTargetPctValues",
        float,
        default=0.0,
    ),
    ParamSpec(
        "ULTRA_BRIDGE_DAYS",
        "float",
        False,
        False,
        "dailyBridgeDaysValues",
        float,
        default=0.0,
    ),
    ParamSpec(
        "DAILY_DOWN_BUY_MULT",
        "float",
        False,
        False,
        "dailyDownBuyMultValues",
        float,
        default=0.4,
    ),
    ParamSpec(
        "CRAB_ASSET_CAP_PCT",
        "float",
        False,
        False,
        "dailyCrabAssetCapPctValues",
        float,
        default=1.0,
    ),
    ParamSpec(
        "ULTRA_EXIT_DEPTH",
        "float",
        False,
        False,
        "dailyLockTargetPctValues",
        float,
        default=0.0,
    ),
    ParamSpec(
        "ULTRA_GAIN_MIN_PCT",
        "float",
        False,
        False,
        "dailyLockGainPctValues",
        float,
        default=5.0,
    ),
    ParamSpec(
        "ULTRA_GAIN_MAX_PCT",
        "float",
        False,
        False,
        "dailyLockNearHighPctValues",
        float,
        default=35.0,
    ),
    ParamSpec(
        "ULTRA_EXIT_HOLD_DAYS",
        "int",
        False,
        False,
        "dailyLockMaxDaysValues",
        int,
        "int",
        60,
    ),
    ParamSpec(
        "POST_ULTRA_COAST_TARGET_PCT",
        "float",
        False,
        False,
        "postUltraCoastTargetPctValues",
        float,
        default=1.0,
    ),
    ParamSpec(
        "POST_ULTRA_GIVEBACK_PCT",
        "float",
        False,
        False,
        "postUltraGivebackPctValues",
        float,
        default=0.0,
    ),
    ParamSpec(
        "POST_ULTRA_REACCUM_PCT",
        "float",
        False,
        False,
        "postUltraReaccumPctValues",
        float,
        default=0.0,
    ),
    ParamSpec(
        "POST_ULTRA_DOUBLE_TOP_PCT",
        "float",
        False,
        False,
        "postUltraDoubleTopPctValues",
        float,
        default=0.0,
    ),
    ParamSpec(
        "POST_ULTRA_MAX_DAYS",
        "float",
        False,
        False,
        "postUltraMaxDaysValues",
        float,
        default=0.0,
    ),
    ParamSpec(
        "POST_ULTRA_LOCK_MIN_ASSET_PCT",
        "float",
        False,
        False,
        "postUltraLockMinAssetPctValues",
        float,
        default=0.0,
    ),
    ParamSpec(
        "POST_ULTRA_LOCK_MAX_ASSET_PCT",
        "float",
        False,
        False,
        "postUltraLockMaxAssetPctValues",
        float,
        default=1.0,
    ),
    ParamSpec(
        "POST_ULTRA_LOCK_GIVEBACK_PCT",
        "float",
        False,
        False,
        "postUltraLockGivebackPctValues",
        float,
        default=0.0,
    ),
    ParamSpec(
        "POST_ULTRA_LOCK_REACCUM_PCT",
        "float",
        False,
        False,
        "postUltraLockReaccumPctValues",
        float,
        default=0.0,
    ),
    ParamSpec(
        "POST_ULTRA_LOCK_DOUBLE_TOP_PCT",
        "float",
        False,
        False,
        "postUltraLockDoubleTopPctValues",
        float,
        default=0.0,
    ),
    ParamSpec(
        "POST_ULTRA_LOCK_MAX_DAYS",
        "float",
        False,
        False,
        "postUltraLockMaxDaysValues",
        float,
        default=0.0,
    ),
    ParamSpec("CHART_CHUNK_SIZE", "int", True, True),
    ParamSpec("WALLET_SEED_QUOTE", "float", True, True),
    ParamSpec(
        "WALLET_SEED_ASSET_PCT",
        "float",
        False,
        False,
        "walletSeedAssetPctValues",
        float,
        default=1.0,
    ),
    ParamSpec("WALLET_FEE_RATE", "float", True, True),
    ParamSpec("QUOTE_TO_AUD_RATE", "float", True, True),
    ParamSpec("HOLDOUT_START_MIN_PCT", "int", False, False),
    ParamSpec("HOLDOUT_START_MAX_PCT", "int", False, False),
    ParamSpec("HOLDOUT_START_STEP_PCT", "int", False, False),
    ParamSpec(
        "GRAD1_BUY_Z_MIN",
        "float",
        True,
        True,
        "grad1BuyZscoreValues",
        float,
    ),
    ParamSpec(
        "GRAD1_SELL_Z_MIN",
        "float",
        True,
        True,
        "grad1SellZscoreValues",
        float,
    ),
    ParamSpec(
        "GRAD1_BUY_WIN_DAYS",
        "int",
        True,
        True,
        "grad1BuyWindowValues",
        int,
        "int",
    ),
    ParamSpec(
        "GRAD1_SELL_WIN_DAYS",
        "int",
        True,
        True,
        "grad1SellWindowValues",
        int,
        "int",
    ),
    ParamSpec(
        "PHASE_BUY_PORTIONS",
        "int",
        True,
        True,
        "phaseBuyValues",
        int,
        "int",
    ),
    ParamSpec(
        "PHASE_SELL_PORTIONS",
        "int",
        True,
        True,
        "phaseSellValues",
        int,
        "int",
    ),
    ParamSpec(
        "FINAL_PORTION_PCT",
        "float",
        True,
        True,
        "finalPortionValues",
        float,
    ),
    ParamSpec(
        "COOLDOWN",
        "int",
        True,
        True,
        "cooldownValues",
        int,
        "int",
    ),
    ParamSpec(
        "TAX_MODE",
        "str",
        True,
        True,
        "taxModeValues",
        str,
        "str",
    ),
    ParamSpec(
        "MACRO_INTERVAL",
        "str",
        True,
        True,
        "macroIntervalValues",
        str,
        "str",
    ),
    ParamSpec(
        "MACRO_NRG_WIN_DAYS",
        "int",
        True,
        True,
        "macroDynWinValues",
        int,
        "int",
    ),
    ParamSpec(
        "MACRO_NRG_Z_MIN",
        "float",
        True,
        True,
        "macroDynZMinValues",
        float,
    ),
    ParamSpec(
        "MACRO_NRG_Z_MAX",
        "float",
        True,
        True,
        "macroDynZMaxValues",
        float,
    ),
    ParamSpec(
        "MACRO_DYN_PCT_MIN",
        "float",
        True,
        True,
        "macroDynPctMinValues",
        float,
    ),
    ParamSpec(
        "MACRO_DYN_PCT_MAX",
        "float",
        True,
        True,
        "macroDynPctMaxValues",
        float,
    ),
    ParamSpec(
        "MACRO_P1",
        "int",
        True,
        True,
        "macroP1Values",
        int,
        "int",
    ),
    ParamSpec(
        "MACRO_P3",
        "int",
        True,
        True,
        "macroP3Values",
        int,
        "int",
    ),
    ParamSpec(
        "MACRO_GRAD_PERIOD",
        "int",
        True,
        True,
        "macroGradPeriodValues",
        int,
        "int",
    ),
    ParamSpec(
        "MACRO_GRAD_WIN_DAYS",
        "int",
        True,
        True,
        "macroGradWinValues",
        int,
        "int",
    ),
    ParamSpec(
        "MACRO_GRAD_Z_MIN",
        "float",
        True,
        True,
        "macroGradZMinValues",
        float,
    ),
    ParamSpec(
        "MACRO_GRAD_Z_MAX",
        "float",
        True,
        True,
        "macroGradZMaxValues",
        float,
    ),
    ParamSpec(
        "MACRO_MULT_GRAD_MIN",
        "float",
        True,
        True,
        "macroGradMultMinValues",
        float,
    ),
    ParamSpec(
        "MACRO_MULT_GRAD_MAX",
        "float",
        True,
        True,
        "macroGradMultMaxValues",
        float,
    ),
    ParamSpec(
        "MACRO_SELL_RELAX_PCT",
        "float",
        False,
        False,
        "macroSellRelaxPctValues",
        float,
        default=0.0,
    ),
    ParamSpec(
        "PEAK_LOCK_CAP_PCT",
        "float",
        False,
        False,
        "peakLockCapPctValues",
        float,
        default=1.0,
    ),
    ParamSpec(
        "PEAK_LOCK_UNLOCK_GAIN_PCT",
        "float",
        False,
        False,
        "peakLockUnlockGainPctValues",
        float,
        default=25.0,
    ),
    ParamSpec(
        "PEAK_LOCK_REENTRY_STEP_PCT",
        "float",
        False,
        False,
        "peakLockReentryStepPctValues",
        float,
        default=0.15,
    ),
    ParamSpec(
        "PEAK_LOCK_ARM_GAIN_PCT",
        "float",
        False,
        False,
        "peakLockArmGainPctValues",
        float,
        default=15.0,
    ),
    ParamSpec(
        "PEAK_LOCK_GIVEBACK_PCT",
        "float",
        False,
        False,
        "peakLockGivebackPctValues",
        float,
        default=4.0,
    ),
    ParamSpec(
        "PEAK_LOCK_MAX_DAYS",
        "float",
        False,
        False,
        "peakLockMaxDaysValues",
        float,
        default=120.0,
    ),
    ParamSpec(
        "PEAK_LOCK_EDGE_DRAW_PCT",
        "float",
        False,
        False,
        "peakLockEdgeDrawPctValues",
        float,
        default=5.0,
    ),
    ParamSpec(
        "PEAK_LOCK_EDGE_SLOPE_DAYS",
        "float",
        False,
        False,
        "peakLockEdgeSlopeDaysValues",
        float,
        default=7.0,
    ),
    ParamSpec(
        "PEAK_LOCK_REQUIRE_EDGE_RISK",
        "int",
        False,
        False,
        "peakLockRequireEdgeRiskValues",
        int,
        "int",
        1,
    ),
    ParamSpec(
        "PEAK_LOCK_MA_DAYS",
        "float",
        False,
        False,
        "peakLockMaDaysValues",
        float,
        default=30.0,
    ),
    ParamSpec(
        "PEAK_LOCK_KP",
        "float",
        False,
        False,
        "peakLockKpValues",
        float,
        default=6.0,
    ),
    ParamSpec(
        "PEAK_LOCK_KI",
        "float",
        False,
        False,
        "peakLockKiValues",
        float,
        default=0.0,
    ),
    ParamSpec(
        "PEAK_LOCK_KD",
        "float",
        False,
        False,
        "peakLockKdValues",
        float,
        default=0.0,
    ),
    ParamSpec(
        "PEAK_LOCK_INTEGRAL_DECAY",
        "float",
        False,
        False,
        "peakLockIntegralDecayValues",
        float,
        default=0.985,
    ),
    ParamSpec(
        "PEAK_LOCK_ENTRY_THRESHOLD",
        "float",
        False,
        False,
        "peakLockEntryThresholdValues",
        float,
        default=0.25,
    ),
    ParamSpec(
        "PEAK_LOCK_EXIT_THRESHOLD",
        "float",
        False,
        False,
        "peakLockExitThresholdValues",
        float,
        default=0.05,
    ),
    ParamSpec(
        "PEAK_LOCK_CONFIRM_BARS",
        "int",
        False,
        False,
        "peakLockConfirmBarsValues",
        int,
        "int",
        6,
    ),
    ParamSpec(
        "PEAK_LOCK_RELEASE_TARGET_PCT",
        "float",
        False,
        False,
        "peakLockReleaseTargetPctValues",
        float,
        default=0.0,
    ),
    ParamSpec(
        "PEAK_LOCK_ULTRA_GRACE_DAYS",
        "float",
        False,
        False,
        "peakLockUltraGraceDaysValues",
        float,
        default=0.0,
    ),
]

INTERVAL_AXIS_SPECS = [
    ("p1Values", "p1", int, None),
    ("p2Values", "p2", int, None),
    ("p3Values", "p3", int, None),
]

AXIS_SPECS = [
    *INTERVAL_AXIS_SPECS,
    *[
        (s.axisName, s.key, s.caster, s.default)
        for s in PARAM_SPECS
        if s.axisName is not None
    ],
]

HOST_AXIS_NAME_MAP = {
    "grad1BuyZMin": "grad1BuyZscoreValues",
    "grad1SellZMin": "grad1SellZscoreValues",
    "grad1BuyWinDays": "grad1BuyWindowValues",
    "grad1SellWinDays": "grad1SellWindowValues",
    "phaseBuy": "phaseBuyValues",
    "phaseSell": "phaseSellValues",
    "finalPortionPct": "finalPortionValues",
    "cooldown": "cooldownValues",
    "taxMode": "taxModeValues",
    "seedAssetPct": "walletSeedAssetPctValues",
    "dailyStrongSellMult": "dailyStrongSellMultValues",
    "dailyStrongTargetPct": "dailyStrongTargetPctValues",
    "dailyBridgeDays": "dailyBridgeDaysValues",
    "dailyDownBuyMult": "dailyDownBuyMultValues",
    "dailyCrabAssetCapPct": "dailyCrabAssetCapPctValues",
    "dailyLockTargetPct": "dailyLockTargetPctValues",
    "dailyLockGainPct": "dailyLockGainPctValues",
    "dailyLockNearHighPct": "dailyLockNearHighPctValues",
    "dailyLockMaxDays": "dailyLockMaxDaysValues",
    "postUltraCoastTargetPct": "postUltraCoastTargetPctValues",
    "postUltraGivebackPct": "postUltraGivebackPctValues",
    "postUltraReaccumPct": "postUltraReaccumPctValues",
    "postUltraDoubleTopPct": "postUltraDoubleTopPctValues",
    "postUltraMaxDays": "postUltraMaxDaysValues",
    "postUltraLockMinAssetPct": "postUltraLockMinAssetPctValues",
    "postUltraLockMaxAssetPct": "postUltraLockMaxAssetPctValues",
    "postUltraLockGivebackPct": "postUltraLockGivebackPctValues",
    "postUltraLockReaccumPct": "postUltraLockReaccumPctValues",
    "postUltraLockDoubleTopPct": "postUltraLockDoubleTopPctValues",
    "postUltraLockMaxDays": "postUltraLockMaxDaysValues",
    "macroSellRelaxPct": "macroSellRelaxPctValues",
    "peakLockCapPct": "peakLockCapPctValues",
    "peakLockUnlockGainPct": "peakLockUnlockGainPctValues",
    "peakLockReentryStepPct": "peakLockReentryStepPctValues",
    "peakLockArmGainPct": "peakLockArmGainPctValues",
    "peakLockGivebackPct": "peakLockGivebackPctValues",
    "peakLockMaxDays": "peakLockMaxDaysValues",
    "peakLockEdgeDrawPct": "peakLockEdgeDrawPctValues",
    "peakLockEdgeSlopeDays": "peakLockEdgeSlopeDaysValues",
    "peakLockRequireEdgeRisk": "peakLockRequireEdgeRiskValues",
    "peakLockMaDays": "peakLockMaDaysValues",
    "peakLockKp": "peakLockKpValues",
    "peakLockKi": "peakLockKiValues",
    "peakLockKd": "peakLockKdValues",
    "peakLockIntegralDecay": "peakLockIntegralDecayValues",
    "peakLockEntryThreshold": "peakLockEntryThresholdValues",
    "peakLockExitThreshold": "peakLockExitThresholdValues",
    "peakLockConfirmBars": "peakLockConfirmBarsValues",
    "peakLockReleaseTargetPct": "peakLockReleaseTargetPctValues",
    "peakLockUltraGraceDays": "peakLockUltraGraceDaysValues",
}

CONFIG_KEY_ORDER = [
    "ticker",
    "tickers",
    "primer_days",
    "training_days",
    "tuner_days",
    "holdout_days",
    "intervals",
    "p1",
    "p2",
    "p3",
    "GRAD1_BUY_Z_MIN",
    "GRAD1_SELL_Z_MIN",
    "GRAD1_BUY_WIN_DAYS",
    "GRAD1_SELL_WIN_DAYS",
    "PHASE_BUY_PORTIONS",
    "MACRO_INTERVAL",
    "MACRO_P1",
    "MACRO_P3",
    "MACRO_NRG_WIN_DAYS",
    "MACRO_NRG_Z_MIN",
    "MACRO_NRG_Z_MAX",
    "MACRO_DYN_PCT_MIN",
    "MACRO_DYN_PCT_MAX",
    "MACRO_GRAD_PERIOD",
    "MACRO_GRAD_WIN_DAYS",
    "MACRO_GRAD_Z_MIN",
    "MACRO_GRAD_Z_MAX",
    "MACRO_MULT_GRAD_MIN",
    "MACRO_MULT_GRAD_MAX",
    "MACRO_SELL_RELAX_PCT",
    "DAILY_CLUSTER_PATH",
    "DAILY_CLUSTER_MODEL_PATH",
    "ULTRA_SELL_MULT",
    "ULTRA_EXPOSURE_TARGET",
    "ULTRA_BRIDGE_DAYS",
    "DAILY_DOWN_BUY_MULT",
    "CRAB_ASSET_CAP_PCT",
    "ULTRA_EXIT_DEPTH",
    "ULTRA_GAIN_MIN_PCT",
    "ULTRA_GAIN_MAX_PCT",
    "ULTRA_EXIT_HOLD_DAYS",
    "POST_ULTRA_COAST_TARGET_PCT",
    "POST_ULTRA_GIVEBACK_PCT",
    "POST_ULTRA_REACCUM_PCT",
    "POST_ULTRA_DOUBLE_TOP_PCT",
    "POST_ULTRA_MAX_DAYS",
    "POST_ULTRA_LOCK_MIN_ASSET_PCT",
    "POST_ULTRA_LOCK_MAX_ASSET_PCT",
    "POST_ULTRA_LOCK_GIVEBACK_PCT",
    "POST_ULTRA_LOCK_REACCUM_PCT",
    "POST_ULTRA_LOCK_DOUBLE_TOP_PCT",
    "POST_ULTRA_LOCK_MAX_DAYS",
    "PEAK_LOCK_CAP_PCT",
    "PEAK_LOCK_UNLOCK_GAIN_PCT",
    "PEAK_LOCK_REENTRY_STEP_PCT",
    "PEAK_LOCK_ARM_GAIN_PCT",
    "PEAK_LOCK_GIVEBACK_PCT",
    "PEAK_LOCK_MAX_DAYS",
    "PEAK_LOCK_EDGE_DRAW_PCT",
    "PEAK_LOCK_EDGE_SLOPE_DAYS",
    "PEAK_LOCK_REQUIRE_EDGE_RISK",
    "PEAK_LOCK_MA_DAYS",
    "PEAK_LOCK_KP",
    "PEAK_LOCK_KI",
    "PEAK_LOCK_KD",
    "PEAK_LOCK_INTEGRAL_DECAY",
    "PEAK_LOCK_ENTRY_THRESHOLD",
    "PEAK_LOCK_EXIT_THRESHOLD",
    "PEAK_LOCK_CONFIRM_BARS",
    "PEAK_LOCK_RELEASE_TARGET_PCT",
    "PEAK_LOCK_ULTRA_GRACE_DAYS",
    "PHASE_SELL_PORTIONS",
    "FINAL_PORTION_PCT",
    "COOLDOWN",
    "WALLET_SEED_QUOTE",
    "WALLET_SEED_ASSET_PCT",
    "WALLET_FEE_RATE",
    "QUOTE_TO_AUD_RATE",
    "TAX_MODE",
    "CHART_CHUNK_SIZE",
    "CHARTS_TIMEVAL",
    "CHARTS_TRADES",
    "HOLDOUT_START_MIN_PCT",
    "HOLDOUT_START_MAX_PCT",
    "HOLDOUT_START_STEP_PCT",
]

ROW_STR_FIELDS = {
    "ticker",
    "interval",
    "DAILY_CLUSTER_PATH",
    *[s.key for s in PARAM_SPECS if s.rowKind == "str"],
}

ROW_INT_FIELDS = {
    "days",
    "p1",
    "p2",
    "p3",
    "trades",
    *[s.key for s in PARAM_SPECS if s.rowKind == "int"],
}

INT_KEYS = {
    s.key for s in PARAM_SPECS if s.kind == "int"
}

FLOAT_KEYS = {
    s.key for s in PARAM_SPECS if s.kind == "float"
}

BACKTEST_REQUIRED_KEYS = [
    s.key for s in PARAM_SPECS if s.backtest
]

TUNER_REQUIRED_KEYS = [
    *WINDOW_KEYS,
    "intervals",
    "p1",
    "p2",
    "p3",
    *[s.key for s in PARAM_SPECS if s.tuner],
]


########################################################################
# Public Helpers
########################################################################

def requiredKeys(kind: str) -> list[str]:
    k = kind.strip().lower()
    if k == "backtest":
        return list(BACKTEST_REQUIRED_KEYS)
    if k == "tuner":
        return list(TUNER_REQUIRED_KEYS)
    raise SystemExit(f"unknown validation kind: {kind}")


def intKeys() -> set[str]:
    return set(INT_KEYS)


def floatKeys() -> set[str]:
    return set(FLOAT_KEYS)


def rowStrFields() -> set[str]:
    return set(ROW_STR_FIELDS)


def rowIntFields() -> set[str]:
    return set(ROW_INT_FIELDS)


def hostAxisNameMap() -> dict[str, str]:
    return dict(HOST_AXIS_NAME_MAP)


def configKeyOrder() -> list[str]:
    return list(CONFIG_KEY_ORDER)


__all__ = [
    "AXIS_SPECS",
    "CONFIG_KEY_ORDER",
    "HOST_AXIS_NAME_MAP",
    "ParamSpec",
    "configKeyOrder",
    "floatKeys",
    "hostAxisNameMap",
    "intKeys",
    "requiredKeys",
    "rowIntFields",
    "rowStrFields",
]
