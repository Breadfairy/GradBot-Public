#!/usr/bin/env python3

from __future__ import annotations


########################################################################
# C-compatible feature registry
########################################################################

RUNTIME_FEATURES = (
    ("emaGapFastPct", 0),
    ("emaGapMidPct", 1),
    ("emaGapSlowPct", 2),
    ("emaSpreadFastMidPct", 3),
    ("emaSpreadMidSlowPct", 4),
    ("emaSpreadFastSlowPct", 5),
    ("gradFastPct", 6),
    ("gradMidPct", 7),
    ("gradSlowPct", 8),
    ("trendCode", 9),
    ("distHigh24Pct", 10),
    ("distLow24Pct", 11),
    ("range24Pct", 12),
    ("realVol24", 13),
    ("ret1h", 14),
    ("ret3h", 15),
    ("ret6h", 16),
    ("ret12h", 17),
    ("ret24h", 18),
    ("trendEfficiency24", 19),
    ("bodyPct", 20),
    ("bodyAbsPct", 21),
    ("upperWickPct", 22),
    ("lowerWickPct", 23),
    ("bodyAbsMean24", 24),
    ("rangeMean24", 25),
    ("logVolumeZ168", 26),
    ("macroDynSigned", 27),
    ("macroDynMag", 28),
    ("macroDir", 29),
    ("macroMom", 30),
    ("macroBull", 31),
    ("macroBear", 32),
    ("macroRev", 33),
    ("macroRoll", 34),
    ("ret2h", 35),
    ("ret4h", 36),
    ("ret8h", 37),
    ("ret48h", 38),
    ("realVol12", 39),
    ("realVol48", 40),
    ("distHigh48Pct", 41),
    ("distLow48Pct", 42),
    ("range48Pct", 43),
    ("rangePos24", 44),
    ("rangePos48", 45),
    ("ageHigh24", 46),
    ("ageLow24", 47),
    ("ageHigh48", 48),
    ("ageLow48", 49),
    ("bodyAbsMean12", 50),
    ("bodyAbsMean48", 51),
    ("rangeMean12", 52),
    ("rangeMean48", 53),
    ("logQuoteZ168", 54),
    ("logTradesZ168", 55),
    ("takerBaseRatio", 56),
    ("takerQuoteRatio", 57),
    ("takerImbalance", 58),
    ("takerImbalanceZ168", 59),
)

FEATURE_IDS_BY_NAME = dict(RUNTIME_FEATURES)
RUNTIME_FEATURE_NAMES = tuple(
    name for name, featureId in RUNTIME_FEATURES
    if int(featureId) < 35
)


########################################################################
# Feature families
########################################################################

PRICE_STATE_FEATURES = (
    "distHigh24Pct",
    "distLow24Pct",
    "range24Pct",
    "realVol24",
    "ret1h",
    "ret3h",
    "ret6h",
    "ret12h",
    "ret24h",
    "trendEfficiency24",
    "bodyPct",
    "bodyAbsPct",
    "upperWickPct",
    "lowerWickPct",
    "bodyAbsMean24",
    "rangeMean24",
    "logVolumeZ168",
)

RAW_MARKET_EXPANDED_FEATURES = (
    "distHigh24Pct",
    "distLow24Pct",
    "range24Pct",
    "distHigh48Pct",
    "distLow48Pct",
    "range48Pct",
    "rangePos24",
    "rangePos48",
    "ageHigh24",
    "ageLow24",
    "ageHigh48",
    "ageLow48",
    "realVol12",
    "realVol24",
    "realVol48",
    "ret1h",
    "ret2h",
    "ret3h",
    "ret4h",
    "ret6h",
    "ret8h",
    "ret12h",
    "ret24h",
    "ret48h",
    "trendEfficiency24",
    "bodyPct",
    "bodyAbsPct",
    "upperWickPct",
    "lowerWickPct",
    "bodyAbsMean12",
    "bodyAbsMean24",
    "bodyAbsMean48",
    "rangeMean12",
    "rangeMean24",
    "rangeMean48",
    "logVolumeZ168",
    "logQuoteZ168",
    "logTradesZ168",
    "takerBaseRatio",
    "takerQuoteRatio",
    "takerImbalance",
    "takerImbalanceZ168",
)

CAPITULATION_STATE_FEATURES = (
    "ret1h",
    "ret2h",
    "ret3h",
    "ret4h",
    "ret6h",
    "ret8h",
    "ret12h",
    "ret24h",
    "ret48h",
    "distHigh24Pct",
    "distLow24Pct",
    "range24Pct",
    "distHigh48Pct",
    "distLow48Pct",
    "range48Pct",
    "rangePos24",
    "rangePos48",
    "bodyPct",
    "bodyAbsPct",
    "lowerWickPct",
    "rangeMean12",
    "rangeMean24",
    "rangeMean48",
    "logVolumeZ168",
    "logQuoteZ168",
    "logTradesZ168",
    "takerImbalance",
    "takerImbalanceZ168",
    "emaGapFastPct",
    "emaGapMidPct",
    "emaGapSlowPct",
    "gradFastPct",
    "gradMidPct",
    "gradSlowPct",
)

EMA_STATE_FEATURES = (
    "emaGapFastPct",
    "emaGapMidPct",
    "emaGapSlowPct",
    "emaSpreadFastMidPct",
    "emaSpreadMidSlowPct",
    "emaSpreadFastSlowPct",
    "gradFastPct",
    "gradMidPct",
    "gradSlowPct",
)

MACRO_STATE_FEATURES = (
    "macroDynSigned",
    "macroDynMag",
    "macroDir",
    "macroMom",
    "macroBull",
    "macroBear",
    "macroRev",
    "macroRoll",
)

FEATURE_FAMILIES = {
    "none": (),
    "price_state": PRICE_STATE_FEATURES,
    "ema_state": EMA_STATE_FEATURES,
    "macro_state": MACRO_STATE_FEATURES,
    "runtime_mixed": RUNTIME_FEATURE_NAMES,
    "reduced_no_gate_inputs": PRICE_STATE_FEATURES + EMA_STATE_FEATURES,
    "raw_market_expanded": RAW_MARKET_EXPANDED_FEATURES,
    "raw_market_ema_expanded": (
        RAW_MARKET_EXPANDED_FEATURES + EMA_STATE_FEATURES
    ),
    "capitulation_state": CAPITULATION_STATE_FEATURES,
}


def familyNames() -> list[str]:
    return list(FEATURE_FAMILIES.keys())


def featureNames(featureFamily: str) -> list[str]:
    return list(FEATURE_FAMILIES[str(featureFamily)])


def featureIds(features: list[str]) -> list[int]:
    return [int(FEATURE_IDS_BY_NAME[name]) for name in features]


def manifestRows(featureFamily: str, features: list[str]) -> list[dict]:
    rows: list[dict] = []
    for i, name in enumerate(features):
        rows.append(
            {
                "featureFamily": featureFamily,
                "position": int(i),
                "featureId": int(FEATURE_IDS_BY_NAME.get(name, -1)),
                "feature": name,
            }
        )
    return rows
