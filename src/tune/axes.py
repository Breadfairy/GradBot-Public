#!/usr/bin/env python3
# tune_axes.py – tuning axis expansion and parameter products.

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, Iterator, List, Tuple

from config.params import TuneParams
from tune.schema import AXIS_SPECS


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
    axes = {}
    for axisName, key, caster, default in AXIS_SPECS:
        if key in config:
            axes[axisName] = expandSpec(config[key], caster)
        elif default is not None:
            axes[axisName] = expandSpec(default, caster)
        else:
            raise KeyError(f"tune config missing required key: {key}")
    return axes


def buildIntervalGroups(
    intervals: List[str],
    p1Values: List[int],
    p2Values: List[int],
    p3Values: List[int],
) -> List[Tuple[str, int, int, int]]:
    return list(itertools.product(intervals, p1Values, p2Values, p3Values))


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
) -> Iterator[TuneParams]:
    for (
        macroInterval,
        macroDynWin,
        macroDynZMin,
        macroDynZMax,
        macroDynPctMin,
        macroDynPctMax,
        macroP1,
        macroP3,
        macroGradPeriod,
        macroGradWinDays,
        macroGradZMin,
        macroGradZMax,
        macroGradMultMin,
        macroGradMultMax,
        macroSellRelaxPct,
        phaseBuy,
        phaseSell,
        finalPortionPct,
        cooldown,
        taxMode,
        seedAssetPct,
        dailyStrongSellMult,
        dailyStrongTargetPct,
        dailyBridgeDays,
        dailyDownBuyMult,
        dailyCrabAssetCapPct,
        dailyLockTargetPct,
        dailyLockGainPct,
        dailyLockNearHighPct,
        dailyLockMaxDays,
        postUltraCoastTargetPct,
        postUltraGivebackPct,
        postUltraReaccumPct,
        postUltraDoubleTopPct,
        postUltraMaxDays,
        postUltraLockMinAssetPct,
        postUltraLockMaxAssetPct,
        postUltraLockGivebackPct,
        postUltraLockReaccumPct,
        postUltraLockDoubleTopPct,
        postUltraLockMaxDays,
        peakLockCapPct,
        peakLockUnlockGainPct,
        peakLockReentryStepPct,
        peakLockArmGainPct,
        peakLockGivebackPct,
        peakLockMaxDays,
        peakLockEdgeDrawPct,
        peakLockEdgeSlopeDays,
        peakLockRequireEdgeRisk,
        peakLockMaDays,
        peakLockKp,
        peakLockKi,
        peakLockKd,
        peakLockIntegralDecay,
        peakLockEntryThreshold,
        peakLockExitThreshold,
        peakLockConfirmBars,
        peakLockReleaseTargetPct,
        peakLockUltraGraceDays,
        (gradBuyZ, gradBuyWin, gradSellZ, gradSellWin),
    ) in itertools.product(
        axes["macroIntervalValues"],
        axes["macroDynWinValues"],
        axes["macroDynZMinValues"],
        axes["macroDynZMaxValues"],
        axes["macroDynPctMinValues"],
        axes["macroDynPctMaxValues"],
        axes["macroP1Values"],
        axes["macroP3Values"],
        axes["macroGradPeriodValues"],
        axes["macroGradWinValues"],
        axes["macroGradZMinValues"],
        axes["macroGradZMaxValues"],
        axes["macroGradMultMinValues"],
        axes["macroGradMultMaxValues"],
        axes["macroSellRelaxPctValues"],
        axes["phaseBuyValues"],
        axes["phaseSellValues"],
        axes["finalPortionValues"],
        axes["cooldownValues"],
        axes["taxModeValues"],
        axes["walletSeedAssetPctValues"],
        axes["dailyStrongSellMultValues"],
        axes["dailyStrongTargetPctValues"],
        axes["dailyBridgeDaysValues"],
        axes["dailyDownBuyMultValues"],
        axes["dailyCrabAssetCapPctValues"],
        axes["dailyLockTargetPctValues"],
        axes["dailyLockGainPctValues"],
        axes["dailyLockNearHighPctValues"],
        axes["dailyLockMaxDaysValues"],
        axes["postUltraCoastTargetPctValues"],
        axes["postUltraGivebackPctValues"],
        axes["postUltraReaccumPctValues"],
        axes["postUltraDoubleTopPctValues"],
        axes["postUltraMaxDaysValues"],
        axes["postUltraLockMinAssetPctValues"],
        axes["postUltraLockMaxAssetPctValues"],
        axes["postUltraLockGivebackPctValues"],
        axes["postUltraLockReaccumPctValues"],
        axes["postUltraLockDoubleTopPctValues"],
        axes["postUltraLockMaxDaysValues"],
        axes["peakLockCapPctValues"],
        axes["peakLockUnlockGainPctValues"],
        axes["peakLockReentryStepPctValues"],
        axes["peakLockArmGainPctValues"],
        axes["peakLockGivebackPctValues"],
        axes["peakLockMaxDaysValues"],
        axes["peakLockEdgeDrawPctValues"],
        axes["peakLockEdgeSlopeDaysValues"],
        axes["peakLockRequireEdgeRiskValues"],
        axes["peakLockMaDaysValues"],
        axes["peakLockKpValues"],
        axes["peakLockKiValues"],
        axes["peakLockKdValues"],
        axes["peakLockIntegralDecayValues"],
        axes["peakLockEntryThresholdValues"],
        axes["peakLockExitThresholdValues"],
        axes["peakLockConfirmBarsValues"],
        axes["peakLockReleaseTargetPctValues"],
        axes["peakLockUltraGraceDaysValues"],
        gradVariantsList,
    ):
        yield TuneParams(
            macroDynWindowDays=macroDynWin,
            macroDynZMin=macroDynZMin,
            macroDynZMax=macroDynZMax,
            macroDynPctMin=macroDynPctMin,
            macroDynPctMax=macroDynPctMax,
            macroInterval=macroInterval,
            macroP1=macroP1,
            macroP3=macroP3,
            macroGradPeriod=macroGradPeriod,
            macroGradWinDays=macroGradWinDays,
            macroGradZMin=macroGradZMin,
            macroGradZMax=macroGradZMax,
            macroGradMultMin=macroGradMultMin,
            macroGradMultMax=macroGradMultMax,
            macroSellRelaxPct=macroSellRelaxPct,
            grad1BuyZscoreMin=gradBuyZ,
            grad1SellZscoreMin=gradSellZ,
            grad1BuyWindowDays=gradBuyWin,
            grad1SellWindowDays=gradSellWin,
            phaseBuy=phaseBuy,
            phaseSell=phaseSell,
            finalPortionPct=finalPortionPct,
            cooldown=cooldown,
            taxMode=taxMode,
            seedAssetPct=seedAssetPct,
            dailyStrongSellMult=dailyStrongSellMult,
            dailyStrongTargetPct=dailyStrongTargetPct,
            dailyBridgeDays=dailyBridgeDays,
            dailyDownBuyMult=dailyDownBuyMult,
            dailyCrabAssetCapPct=dailyCrabAssetCapPct,
            dailyLockTargetPct=dailyLockTargetPct,
            dailyLockGainPct=dailyLockGainPct,
            dailyLockNearHighPct=dailyLockNearHighPct,
            dailyLockMaxDays=dailyLockMaxDays,
            postUltraCoastTargetPct=postUltraCoastTargetPct,
            postUltraGivebackPct=postUltraGivebackPct,
            postUltraReaccumPct=postUltraReaccumPct,
            postUltraDoubleTopPct=postUltraDoubleTopPct,
            postUltraMaxDays=postUltraMaxDays,
            postUltraLockMinAssetPct=postUltraLockMinAssetPct,
            postUltraLockMaxAssetPct=postUltraLockMaxAssetPct,
            postUltraLockGivebackPct=postUltraLockGivebackPct,
            postUltraLockReaccumPct=postUltraLockReaccumPct,
            postUltraLockDoubleTopPct=postUltraLockDoubleTopPct,
            postUltraLockMaxDays=postUltraLockMaxDays,
            peakLockCapPct=peakLockCapPct,
            peakLockUnlockGainPct=peakLockUnlockGainPct,
            peakLockReentryStepPct=peakLockReentryStepPct,
            peakLockArmGainPct=peakLockArmGainPct,
            peakLockGivebackPct=peakLockGivebackPct,
            peakLockMaxDays=peakLockMaxDays,
            peakLockEdgeDrawPct=peakLockEdgeDrawPct,
            peakLockEdgeSlopeDays=peakLockEdgeSlopeDays,
            peakLockRequireEdgeRisk=peakLockRequireEdgeRisk,
            peakLockMaDays=peakLockMaDays,
            peakLockKp=peakLockKp,
            peakLockKi=peakLockKi,
            peakLockKd=peakLockKd,
            peakLockIntegralDecay=peakLockIntegralDecay,
            peakLockEntryThreshold=peakLockEntryThreshold,
            peakLockExitThreshold=peakLockExitThreshold,
            peakLockConfirmBars=peakLockConfirmBars,
            peakLockReleaseTargetPct=peakLockReleaseTargetPct,
            peakLockUltraGraceDays=peakLockUltraGraceDays,
        )


__all__ = [
    "axesFromConfig",
    "buildIntervalGroups",
    "buildParamProduct",
    "expandRange",
    "expandSpec",
    "gradientVariants",
]
