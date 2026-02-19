#!/usr/bin/env python3
# tune_axes.py – tuning axis expansion and parameter products.

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, Iterator, List, Tuple

from params import TuneParams


def expandRange(
    start: float,
    stop: float,
    step: float,
    caster: Callable[[float], Any],
) -> List[Any]:
    s = float(start)
    e = float(stop)
    d = float(step)
    if d == 0:
        raise ValueError("range step cannot be 0")
    values: List[Any] = []
    cur = s
    if d > 0:
        while cur <= e + 1e-12:
            values.append(caster(cur))
            cur += d
    else:
        while cur >= e - 1e-12:
            values.append(caster(cur))
            cur += d
    return values


def expandSpec(spec: Any, caster: Callable[[float], Any]) -> List[Any]:
    if isinstance(spec, list):
        return [caster(value) for value in spec]
    if isinstance(spec, dict):
        if "range" in spec:
            arr = spec["range"]
            return expandRange(arr[0], arr[1], arr[2], caster)
        return expandRange(spec["start"], spec["stop"], spec["step"], caster)
    return [caster(spec)]


def axesFromConfig(config: Dict[str, Any]) -> Dict[str, List[Any]]:
    def getAxisRequired(key: str, caster):
        if key not in config:
            raise KeyError(f"tune config missing required key: {key}")
        return expandSpec(config[key], caster)

    axes = {
        "p1Values": expandSpec(config["p1"], int),
        "p2Values": expandSpec(config["p2"], int),
        "p3Values": expandSpec(config["p3"], int),
        "grad1BuyZscoreValues": expandSpec(config["GRAD1_BUY_Z_MIN"], float),
        "grad1SellZscoreValues": expandSpec(config["GRAD1_SELL_Z_MIN"], float),
        "grad1BuyWindowValues": expandSpec(config["GRAD1_BUY_WIN_DAYS"], int),
        "grad1SellWindowValues": expandSpec(config["GRAD1_SELL_WIN_DAYS"], int),
        "phaseBuyValues": expandSpec(config["PHASE_BUY_PORTIONS"], int),
        "phaseSellValues": expandSpec(config["PHASE_SELL_PORTIONS"], int),
        "finalPortionValues": expandSpec(config["FINAL_PORTION_PCT"], float),
        "cooldownValues": expandSpec(config["COOLDOWN"], int),
        "taxModeValues": getAxisRequired("TAX_MODE", str),
        "annualIncomeBaseValues": getAxisRequired("ANNUAL_INCOME_BASE", float),
        "profitSweepIntervalValues": getAxisRequired(
            "PROFIT_SWEEP_INTERVAL", str
        ),
        "profitSweepShareValues": getAxisRequired("PROFIT_SWEEP_SHARE", float),
    }

    axes["spacingZscore12Values"] = expandSpec(config["SPACING_Z_MIN_12"], float)
    axes["spacingZscore23Values"] = expandSpec(config["SPACING_Z_MIN_23"], float)
    axes["spacingWindow12Values"] = expandSpec(
        config["SPACING_WIN_DAYS_12"], int
    )
    axes["spacingWindow23Values"] = expandSpec(
        config["SPACING_WIN_DAYS_23"], int
    )
    axes["spacingEnergyModelValues"] = expandSpec(config["MICRO_NRG_MODEL"], str)
    axes["spacingEnergyWinValues"] = expandSpec(config["MICRO_NRG_WIN_DAYS"], int)
    axes["spacingEnergyMin12Values"] = expandSpec(config["MICRO_NRG_MIN_12"], float)
    axes["spacingEnergyMin23Values"] = expandSpec(config["MICRO_NRG_MIN_23"], float)

    axes["macroIntervalValues"] = getAxisRequired("MACRO_INTERVAL", str)
    axes["macroDynWinValues"] = getAxisRequired("MACRO_NRG_WIN_DAYS", int)
    axes["macroDynZMinValues"] = getAxisRequired("MACRO_NRG_Z_MIN", float)
    axes["macroDynZMaxValues"] = getAxisRequired("MACRO_NRG_Z_MAX", float)
    axes["macroDynPctMinValues"] = getAxisRequired("MACRO_DYN_PCT_MIN", float)
    axes["macroDynPctMaxValues"] = getAxisRequired("MACRO_DYN_PCT_MAX", float)
    axes["macroGradWinValues"] = getAxisRequired("MACRO_GRAD_WIN_DAYS", int)
    axes["macroGradZMinValues"] = getAxisRequired("MACRO_GRAD_Z_MIN", float)
    axes["macroGradZMaxValues"] = getAxisRequired("MACRO_GRAD_Z_MAX", float)
    axes["macroGradMultMinValues"] = getAxisRequired(
        "MACRO_MULT_GRAD_MIN", float
    )
    axes["macroGradMultMaxValues"] = getAxisRequired(
        "MACRO_MULT_GRAD_MAX", float
    )

    axes["macroP1Values"] = getAxisRequired("MACRO_P1", int)
    axes["macroP2Values"] = getAxisRequired("MACRO_P2", int)
    axes["macroP3Values"] = getAxisRequired("MACRO_P3", int)
    return axes


def buildIntervalGroups(
    intervals: List[str],
    p1Values: List[int],
    p2Values: List[int],
    p3Values: List[int],
) -> List[Tuple[str, int, int, int]]:
    return list(itertools.product(intervals, p1Values, p2Values, p3Values))


def spacingVariants(
    axes: Dict[str, List[Any]],
) -> List[Tuple[float, float, int, int]]:
    return [
        (float(z12), float(z23), int(w12), int(w23))
        for z12, z23, w12, w23 in itertools.product(
            axes["spacingZscore12Values"],
            axes["spacingZscore23Values"],
            axes["spacingWindow12Values"],
            axes["spacingWindow23Values"],
        )
    ]


def gradientVariants(
    axes: Dict[str, List[Any]],
) -> List[Tuple[float, int, float, int]]:
    return [
        (float(bz), int(bw), float(sz), int(sw))
        for bz, bw, sz, sw in itertools.product(
            axes["grad1BuyZscoreValues"],
            axes["grad1BuyWindowValues"],
            axes["grad1SellZscoreValues"],
            axes["grad1SellWindowValues"],
        )
    ]


def buildParamProduct(
    axes: Dict[str, List[Any]],
    gradVariantsList: List[Tuple[float, int, float, int]],
    spacingVariantsList: List[Tuple[float, float, int, int]],
) -> Iterator[TuneParams]:
    for (
        macroInterval,
        macroDynWin,
        macroDynZMin,
        macroDynZMax,
        macroDynPctMin,
        macroDynPctMax,
        macroP1,
        macroP2,
        macroP3,
        macroGradWinDays,
        macroGradZMin,
        macroGradZMax,
        macroGradMultMin,
        macroGradMultMax,
        phaseBuy,
        phaseSell,
        finalPortionPct,
        cooldown,
        taxMode,
        annualIncomeBase,
        profitSweepInterval,
        profitSweepShare,
        spacingEnergyModel,
        spacingEnergyWinDays,
        spacingEnergyMin12,
        spacingEnergyMin23,
        (gradBuyZ, gradBuyWin, gradSellZ, gradSellWin),
        (
            spacingZscoreMin12,
            spacingZscoreMin23,
            spacingWindowDays12,
            spacingWindowDays23,
        ),
    ) in itertools.product(
        axes["macroIntervalValues"],
        axes["macroDynWinValues"],
        axes["macroDynZMinValues"],
        axes["macroDynZMaxValues"],
        axes["macroDynPctMinValues"],
        axes["macroDynPctMaxValues"],
        axes["macroP1Values"],
        axes["macroP2Values"],
        axes["macroP3Values"],
        axes["macroGradWinValues"],
        axes["macroGradZMinValues"],
        axes["macroGradZMaxValues"],
        axes["macroGradMultMinValues"],
        axes["macroGradMultMaxValues"],
        axes["phaseBuyValues"],
        axes["phaseSellValues"],
        axes["finalPortionValues"],
        axes["cooldownValues"],
        axes["taxModeValues"],
        axes["annualIncomeBaseValues"],
        axes["profitSweepIntervalValues"],
        axes["profitSweepShareValues"],
        axes["spacingEnergyModelValues"],
        axes["spacingEnergyWinValues"],
        axes["spacingEnergyMin12Values"],
        axes["spacingEnergyMin23Values"],
        gradVariantsList,
        spacingVariantsList,
    ):
        yield TuneParams(
            macroDynWindowDays=macroDynWin,
            macroDynZMin=macroDynZMin,
            macroDynZMax=macroDynZMax,
            macroDynPctMin=macroDynPctMin,
            macroDynPctMax=macroDynPctMax,
            macroInterval=macroInterval,
            macroP1=macroP1,
            macroP2=macroP2,
            macroP3=macroP3,
            macroGradWinDays=macroGradWinDays,
            macroGradZMin=macroGradZMin,
            macroGradZMax=macroGradZMax,
            macroGradMultMin=macroGradMultMin,
            macroGradMultMax=macroGradMultMax,
            grad1BuyZscoreMin=gradBuyZ,
            grad1SellZscoreMin=gradSellZ,
            grad1BuyWindowDays=gradBuyWin,
            grad1SellWindowDays=gradSellWin,
            phaseBuy=phaseBuy,
            phaseSell=phaseSell,
            finalPortionPct=finalPortionPct,
            cooldown=cooldown,
            taxMode=taxMode,
            annualIncomeBase=annualIncomeBase,
            profitSweepInterval=profitSweepInterval,
            profitSweepShare=profitSweepShare,
            spacingZscoreMin12=spacingZscoreMin12,
            spacingZscoreMin23=spacingZscoreMin23,
            spacingWindowDays12=spacingWindowDays12,
            spacingWindowDays23=spacingWindowDays23,
            spacingEnergyModel=spacingEnergyModel,
            spacingEnergyWinDays=spacingEnergyWinDays,
            spacingEnergyMin12=spacingEnergyMin12,
            spacingEnergyMin23=spacingEnergyMin23,
        )


__all__ = [
    "axesFromConfig",
    "buildIntervalGroups",
    "buildParamProduct",
    "expandRange",
    "expandSpec",
    "gradientVariants",
    "spacingVariants",
]
