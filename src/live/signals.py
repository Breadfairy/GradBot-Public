#!/usr/bin/env python3
# signals.py – phase sizing, signal evaluation, and order execution.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from live.dashboard import RuntimeCfg
from live.daily_posture import dailyPostureForIndex, markLockState
from engine.shared import bars_per_day
from live.execution import (
    executeMarketBuy,
    executeMarketSell,
    executePaperReplayBuy,
    executePaperReplaySell,
    modeLabel,
    walletFreeBalance,
)
from live.live_engine import (
    decisionContext,
    evaluate,
)
from runtime.gates import enforceCooldown, grad1ZscoreMask
from strategy.execution import (
    buySpend,
    buySpendToTargetCap,
    calcBuyScale,
    calcSellScale,
    dailyLockQty,
    floorSellValueCap,
    phaseBuyValue,
    phaseSellValue,
    sellQty,
)
from strategy.supervisor import (
    PeakLockState,
    armPeakLock,
    evaluatePeakLock,
    peakLockConfigFromOverrides,
    peakLockStateFromBenchQty,
    recordPeakCappedBuy,
    recordPeakLock,
    stepActivePeakLock,
    stepPeakPid,
    stepPeakStrong,
    warmPeakLockState,
)
from strategy.signals import fitSignalArray as fitMacroArray
from strategy.signals import regimeAnchors
from strategy.signals import trendLabel


PERTH_TZ = timezone(timedelta(hours=8))


@dataclass
class PhaseState:
    # Persist phase sizing state across closed-candle evaluations.
    phaseSide: Optional[str]
    phaseBaseValue: float
    phaseLastPrice: Optional[float]
    phasePortionsRemaining: Optional[float]
    lastTrendLabel: Optional[str]
    peakState: Optional[PeakLockState] = None


def makeState() -> PhaseState:
    # Build a fresh phase state.
    return PhaseState(
        phaseSide=None,
        phaseBaseValue=0.0,
        phaseLastPrice=None,
        phasePortionsRemaining=0.0,
        lastTrendLabel=None,
        peakState=None,
    )


def latestFlags(flags: list, barIndex: int) -> list:
    # Return all labels generated at one candle index.
    return [label for idx, label in flags if int(idx) == int(barIndex)]


def postureRowsFor(
    runCfg: RuntimeCfg,
    microRows: list,
    macroRows: list,
    postureRows: list,
) -> list:
    # Select the posture feed matching the configured posture interval.
    if runCfg.postureInterval == runCfg.interval:
        return list(microRows)
    if runCfg.postureInterval == runCfg.macroInterval:
        return list(macroRows)
    return postureRows


def accountContext(dash, price: float) -> dict[str, float]:
    # Snapshot wallet/benchmark values before the latest decision mutates them.
    quoteTotal = float(dash.quoteTotal or 0.0)
    baseTotal = float(dash.baseTotal or 0.0)
    strategyValue = quoteTotal + (baseTotal * float(price))
    hodlValue = 0.0
    edgeValue = 0.0
    if dash.seeded and dash.hodlQty is not None:
        hodlValue = float(dash.hodlQty) * float(price)
        edgeValue = strategyValue - hodlValue
    return {
        'quote_total': quoteTotal,
        'base_total': baseTotal,
        'strategy_value': strategyValue,
        'hodl_value': hodlValue,
        'edge_value': edgeValue,
    }


def _resetPhase(state: PhaseState) -> None:
    state.phaseSide = None
    state.phaseLastPrice = None
    state.phasePortionsRemaining = 0.0


def _walletValue(
    quoteFree: float,
    baseFree: float,
    price: float,
) -> float:
    return float(quoteFree) + (float(baseFree) * float(price))


def _ensurePeakState(
    state: PhaseState,
    runCfg: RuntimeCfg,
    dash,
    ctx: dict,
    barIndex: int,
    barsDay: float,
):
    cfg = peakLockConfigFromOverrides(runCfg.overrides, barsDay)
    firstPrice = float(ctx['closes'][0])
    benchQty = float(dash.hodlQty or 0.0)
    if not cfg.enabled or not bool(dash.seeded) or benchQty <= 0.0:
        return cfg, None
    if state.peakState is None or float(state.peakState.benchQty) <= 0.0:
        state.peakState = peakLockStateFromBenchQty(firstPrice, benchQty)
        warmPeakLockState(
            state.peakState,
            cfg,
            ctx['closes'],
            int(barIndex),
        )
    return cfg, state.peakState


def _stepPeakBeforeLocks(
    peakState,
    peakCfg,
    posture: dict,
    postureState: dict,
    closePrice: float,
) -> tuple[bool, bool, float]:
    strongEntry = False
    graceActive = False
    givebackPct = 0.0
    entryPrice = 0.0
    peakPrice = 0.0
    ultraGainPct = 0.0
    if (
        peakState is None
        or not peakCfg.enabled
        or bool(posture.get('cloudActive', False))
        or not bool(posture.get('pidEnabled', True))
    ):
        return strongEntry, graceActive, givebackPct
    stepPeakPid(peakState, peakCfg, closePrice)
    strongEntry, graceActive = stepPeakStrong(
        peakState,
        peakCfg,
        bool(posture.get('strong', False)),
    )
    entryPrice = float(postureState.get('ultraEntryPrice', 0.0))
    peakPrice = float(postureState.get('ultraPeakPrice', 0.0))
    ultraGainPct = (
        ((peakPrice / entryPrice) - 1.0) * 100.0
        if entryPrice > 0.0 else 0.0
    )
    givebackPct = (
        ((peakPrice / closePrice) - 1.0) * 100.0
        if peakPrice > 0.0 else 0.0
    )
    armPeakLock(
        peakState,
        peakCfg,
        bool(posture.get('strong', False)),
        ultraGainPct,
    )
    return strongEntry, graceActive, givebackPct


def _strongTargetPct(
    overrides: dict,
    peakState,
) -> float:
    targetPct = max(0.0, min(1.0, float(
        overrides.get('ULTRA_EXPOSURE_TARGET', 0.0)
    )))
    if (
        peakState is not None
        and peakState.active
        and peakState.cap < 1.0 - 1e-9
    ):
        baseCap = float(overrides.get('PEAK_LOCK_CAP_PCT', 1.0))
        if peakState.cap <= baseCap + 1e-9:
            return 0.0
        targetPct = min(targetPct, float(peakState.cap))
    return targetPct


def _strongSellFloorPct(
    overrides: dict,
    posture: dict,
    peakState,
) -> float | None:
    if bool(posture.get('cloudActive', False)):
        return max(0.0, min(1.0, float(
            posture.get('cloudMinAssetPct', 0.0)
        )))
    if not bool(posture.get('strong', False)):
        return None
    floorPct = max(0.0, min(1.0, float(
        overrides.get('ULTRA_EXPOSURE_TARGET', 0.0)
    )))
    if (
        peakState is not None
        and peakState.active
        and peakState.cap < 1.0 - 1e-9
    ):
        floorPct = min(floorPct, float(peakState.cap))
    return floorPct


def _coastBuyCapPct(
    posture: dict,
    buyCapPct: float | None,
) -> float | None:
    if not bool(posture.get('coastActive', False)):
        return buyCapPct
    coastTarget = max(0.0, min(1.0, float(
        posture.get('cloudMaxAssetPct', posture.get('coastTarget', 1.0))
    )))
    if buyCapPct is None:
        return coastTarget
    return min(float(buyCapPct), coastTarget)


def tradeAuditContext(decision: dict) -> dict[str, object]:
    # Keep trade rows self-explaining without copying full candle/account data.
    keys = [
        'signal_open_ms',
        'signal_close_ms',
        'daily_cluster',
        'daily_posture',
        'daily_force_lock',
        'macro_dyn_signed',
        'macro_dir',
        'macro_mom',
        'trend_code',
        'trend_label',
        'grad1',
        'buy_z',
        'sell_z',
        'accepted_buy',
        'accepted_sell',
        'final_action',
        'decision_reason',
    ]
    return {key: decision.get(key, '') for key in keys}


def _fmtStatus(isOpen: bool) -> str:
    return 'open' if bool(isOpen) else 'closed'


def _fmtPct(value: float) -> str:
    return f"{float(value):.2f}%"


def _fmtPrice(value: float) -> str:
    return f"${float(value):.2f}"


def _fmtAsset(value: float, asset: str) -> str:
    return f"{float(value):.2f} {asset}"


def _macroText(value: int) -> str:
    if int(value) > 0:
        return 'bull'
    if int(value) < 0:
        return 'bear'
    return 'flat'


def _phaseCopy(state: PhaseState) -> PhaseState:
    return PhaseState(
        phaseSide=state.phaseSide,
        phaseBaseValue=state.phaseBaseValue,
        phaseLastPrice=state.phaseLastPrice,
        phasePortionsRemaining=state.phasePortionsRemaining,
        lastTrendLabel=state.lastTrendLabel,
        peakState=state.peakState,
    )


def _candleRow(candle: dict) -> list:
    closeVal = str(candle['c'])
    return [
        int(candle['t']),
        str(candle.get('o', closeVal)),
        str(candle.get('h', closeVal)),
        str(candle.get('l', closeVal)),
        closeVal,
        str(candle.get('v', 0.0)),
        int(candle['T']),
        str(candle.get('q', 0.0)),
        int(candle.get('n', 0)),
        str(candle.get('V', 0.0)),
        str(candle.get('Q', 0.0)),
        str(candle.get('B', 0.0)),
    ]


def _previewRows(rows: list, candle: dict) -> list:
    if not candle or 'o' not in candle:
        return list(rows)

    out = list(rows)
    row = _candleRow(candle)
    openMs = int(row[0])
    lastOpen = int(out[-1][0])
    if openMs == lastOpen:
        out[-1] = row
    elif openMs > lastOpen:
        out.append(row)
    return out


def _sideMoveState(
    pack,
    idx: int,
    side: str,
    filteredIdx: list[int],
    overrides: dict,
) -> dict[str, object]:
    n = len(pack.ctx['closes'])
    closes = np.asarray(pack.ctx['closes'], dtype=float)
    trendCode = np.asarray(pack.signals['trendCode'], dtype=int)
    allowReg = trendCode == (-1 if side == 'BUY' else 1)
    anchors = regimeAnchors(allowReg)
    dyn = fitMacroArray(pack.macroDyn, n, float)
    sellRelax = max(
        0.0,
        min(100.0, float(overrides.get('MACRO_SELL_RELAX_PCT', 0.0))),
    )
    sellRelaxMult = 1.0 - (sellRelax / 100.0)
    lastIdx = None
    lastPhase = None

    for i in filteredIdx:
        anchor = int(anchors[int(i)]) if 0 <= int(i) < n else -1
        if anchor != lastPhase:
            lastIdx = None
            lastPhase = anchor

        refIdx = lastIdx if lastIdx is not None else anchor
        movePct = 0.0
        reqPct = 0.0
        if refIdx >= 0 and 0 <= int(i) < closes.size:
            nowPrice = float(closes[int(i)])
            refPrice = float(closes[int(refIdx)])
            if nowPrice > 0.0 and refPrice > 0.0:
                if side == 'BUY':
                    movePct = ((refPrice / nowPrice) - 1.0) * 100.0
                else:
                    movePct = ((nowPrice / refPrice) - 1.0) * 100.0
            dynVal = float(dyn[int(i)])
            reqPct = 0.0 if not np.isfinite(dynVal) else abs(dynVal)
            if side == 'SELL':
                reqPct *= sellRelaxMult
        keep = movePct >= reqPct
        if int(i) == int(idx):
            return {
                'active': True,
                'open': bool(keep),
                'movePct': movePct,
                'reqPct': reqPct,
            }
        if keep:
            lastIdx = int(i)

    return {
        'active': False,
        'open': False,
        'movePct': 0.0,
        'reqPct': 0.0,
    }


def _sideGateLines(
    pack,
    decision: dict,
    overrides: dict,
    side: str,
) -> list[str]:
    idx = int(decision['bar_index'])
    g1 = np.asarray(pack.signals['g1P1'], dtype=float)
    trendCode = np.asarray(pack.signals['trendCode'], dtype=int)
    allowReg = trendCode == (-1 if side == 'BUY' else 1)
    rawGrad = grad1ZscoreMask(pack.ctx, allowReg, g1, overrides, side)
    rawMask = (np.arange(len(rawGrad)) >= int(pack.startIdx)) & rawGrad
    rawIdx = np.flatnonzero(rawMask)
    cooldown = max(int(overrides['COOLDOWN']), 0)
    filteredIdx = enforceCooldown(rawIdx, cooldown)
    rawNow = int(idx) in rawIdx.tolist()
    coolNow = int(idx) in filteredIdx
    moveState = _sideMoveState(pack, idx, side, filteredIdx, overrides)
    trendName = 'BEAR' if side == 'BUY' else 'BULL'
    trendOpen = bool(decision[f"allow_{side.lower()}"])
    zVal = float(decision[f"{side.lower()}_z"])
    zMin = float(overrides[f'GRAD1_{side}_Z_MIN'])
    zReady = bool(decision[f"{side.lower()}_z_valid"])
    zOpen = bool(zReady and zVal >= zMin)
    flagOpen = bool(decision[f"accepted_{side.lower()}"])
    coolText = 'n/a'
    moveText = 'n/a'

    if rawNow:
        coolText = _fmtStatus(coolNow)
    if bool(moveState['active']):
        moveText = (
            f"{_fmtStatus(bool(moveState['open']))} "
            f"{_fmtPct(float(moveState['movePct']))} "
            f"(req {_fmtPct(float(moveState['reqPct']))})"
        )

    return [
        f"- {trendName} trend  : {_fmtStatus(trendOpen)}",
        f"- grad1 z     : {_fmtStatus(zOpen)} {zVal:.2f} (min {zMin:.2f})",
        f"- cooldown    : {coolText}",
        f"- macro move  : {moveText}",
        f"- flag        : {_fmtStatus(flagOpen)}",
    ]


def _minNotional(symbolMeta) -> float:
    return max(float(getattr(symbolMeta, 'minNotional', 0.0)), 0.0)


def _notionalLine(value: float, minVal: float) -> str:
    status = _fmtStatus(float(value) >= float(minVal))
    op = '>=' if float(value) >= float(minVal) else '<'
    return f"{status} {_fmtPrice(value)} {op} {_fmtPrice(minVal)}"


def _postureMult(side: str, posture: dict, overrides: dict) -> float:
    if side == 'BUY' and bool(posture.get('down', False)):
        return float(overrides.get('DAILY_DOWN_BUY_MULT', 0.4))
    if (
        side == 'SELL'
        and bool(posture.get('strong', False))
        and not bool(posture.get('cloudActive', False))
    ):
        return float(overrides.get('ULTRA_SELL_MULT', 0.0))
    return 1.0


def _walletNoSignalLines(decision: dict) -> list[str]:
    posture = str(decision.get('daily_posture', 'unknown'))
    return [
        f"- daily lock  : off ({posture})",
        "- buy funds   : closed (no BUY flag)",
        "- sell funds  : closed (no SELL flag)",
        "- phase size  : closed (no plan)",
        "- min notional: closed (no plan)",
    ]


def _walletLockLines(
    symbolMeta,
    runCfg: RuntimeCfg,
    dash,
    closePrice: float,
    posture: dict,
) -> tuple[list[str], str, str]:
    postureName = str(posture.get('label', 'unknown'))
    quoteFree = float(dash.quoteTotal or 0.0)
    baseFree = float(dash.baseTotal or 0.0)
    qty = dailyLockQty(
        quoteFree,
        baseFree,
        closePrice,
        float(posture.get('exitTarget', 1.0)),
        float(runCfg.overrides.get('WALLET_FEE_RATE', 0.0)),
    )
    notional = float(qty) * float(closePrice)
    minVal = _minNotional(symbolMeta)
    action = 'LOCK_SELL preview' if qty > 0.0 and notional >= minVal else 'HOLD'
    reason = 'daily_force_lock'
    return [
        f"- daily lock  : ON ({postureName})",
        "- buy funds   : closed (daily lock)",
        "- sell funds  : closed (daily lock)",
        "- phase size  : closed (daily lock)",
        "- posture mult: closed (daily lock)",
        "- final cap   : closed (daily lock)",
        "- min notional: closed (daily lock)",
    ], action, reason


def _walletPlanLines(
    symbolMeta,
    runCfg: RuntimeCfg,
    dash,
    phaseState: PhaseState,
    side: str,
    trendCode: int,
    closePrice: float,
    posture: dict,
    lockActive: bool,
) -> tuple[list[str], str, str]:
    quoteFree = float(dash.quoteTotal or 0.0)
    baseFree = float(dash.baseTotal or 0.0)
    phaseCopy = _phaseCopy(phaseState)
    mult = _postureMult(side, posture, runCfg.overrides)
    buyCapPct = _coastBuyCapPct(posture, None) if side == 'BUY' else None
    plan = nextOrder(
        phaseCopy,
        side,
        trendCode,
        closePrice,
        quoteFree,
        baseFree,
        int(runCfg.overrides['PHASE_BUY_PORTIONS']),
        int(runCfg.overrides['PHASE_SELL_PORTIONS']),
        float(runCfg.overrides['FINAL_PORTION_PCT']),
        runCfg.overrides,
        posture,
        lockActive,
        buyCapPct,
    )
    fundsLine = ''
    sizeLine = '- phase size  : closed (no plan)'
    finalLine = '- final cap   : closed (no plan)'
    minLine = '- min notional: closed (no plan)'
    action = 'HOLD'
    reason = f"{side.lower()}_plan_rejected"

    if side == 'BUY':
        fundsOpen = quoteFree > 0.0
        fundsLine = (
            f"- buy funds   : {_fmtStatus(fundsOpen)} "
            f"{_fmtPrice(quoteFree)} available"
        )
    else:
        fundsOpen = baseFree > 0.0
        fundsLine = (
            f"- sell funds  : {_fmtStatus(fundsOpen)} "
            f"{_fmtAsset(baseFree, symbolMeta.baseAsset)} available"
        )

    if plan is not None:
        if side == 'BUY':
            value = float(plan['quoteQty'])
            sizeLine = f"- phase size  : open {_fmtPrice(value)} planned"
        else:
            qty = float(plan['qty'])
            value = qty * float(closePrice)
            sizeLine = (
                f"- phase size  : open "
                f"{_fmtAsset(qty, symbolMeta.baseAsset)} planned"
            )
        remaining = phaseCopy.phasePortionsRemaining
        if remaining is None:
            finalLine = "- final cap   : open uncapped"
        else:
            finalLine = f"- final cap   : open {float(remaining):.1f} left"
        minVal = _minNotional(symbolMeta)
        minLine = f"- min notional: {_notionalLine(value, minVal)}"
        if value >= minVal:
            action = f"{side} preview"
            reason = 'flag_ready'
        else:
            reason = f"{side.lower()}_order_rejected"

    postureStatus = 'open' if mult > 0.0 else 'closed'
    lines = [
        f"- daily lock  : off ({posture.get('label', 'unknown')})",
        fundsLine,
        sizeLine,
        f"- posture mult: {postureStatus} {mult:.2f}x",
        finalLine,
        minLine,
    ]
    return lines, action, reason


def _walletPreviewLines(
    symbolMeta,
    runCfg: RuntimeCfg,
    dash,
    phaseState: PhaseState,
    decision: dict,
    labels: list[str],
    posture: dict,
    lockActive: bool,
) -> tuple[list[str], str, str]:
    closePrice = float(decision['close'])
    posture['label'] = str(decision.get('daily_posture', 'unknown'))
    if bool(posture.get('forceLock', False)):
        return _walletLockLines(
            symbolMeta,
            runCfg,
            dash,
            closePrice,
            posture,
        )
    if not labels:
        return _walletNoSignalLines(decision), 'HOLD', str(
            decision.get('decision_reason', 'hold')
        )
    side = str(labels[0])
    trendCode = int(decision.get('trend_code', 0))
    return _walletPlanLines(
        symbolMeta,
        runCfg,
        dash,
        phaseState,
        side,
        trendCode,
        closePrice,
        posture,
        lockActive,
    )


def gatePreviewLines(
    symbolMeta,
    runCfg: RuntimeCfg,
    dash,
    microRows: list,
    macroRows: list,
    postureRows: list,
    phaseState: PhaseState,
    candle: dict,
) -> list[str]:
    # Build side-effect-free gate/window lines for the active dashboard pane.
    previewMicro = _previewRows(microRows, candle)
    previewMacro = (
        list(previewMicro)
        if runCfg.macroInterval == runCfg.interval else list(macroRows)
    )
    previewPosture = postureRowsFor(
        runCfg,
        previewMicro,
        previewMacro,
        postureRows,
    )
    pack = evaluate(
        previewMicro,
        previewMacro,
        runCfg.interval,
        runCfg.periods,
        runCfg.primerDays,
        runCfg.overrides,
    )
    idx = len(previewMicro) - 1
    closePrice = float(pack.ctx['closes'][idx])
    barsDay = max(bars_per_day(pack.ctx), 1.0)
    posture, postureState = dailyPostureForIndex(
        pack.ctx,
        runCfg.overrides,
        idx,
        barsDay,
        previewPosture,
    )
    decision = decisionContext(
        pack,
        previewMicro,
        idx,
        runCfg.overrides,
        posture,
        bool(dash.tradingEnabled),
        bool(dash.seeded),
    )
    decision.update(accountContext(dash, closePrice))
    labels = [
        str(label) for flagIdx, label in pack.flags if int(flagIdx) == idx
    ]
    lockActive = bool(postureState.get('lockActive', False))
    walletLines, action, reason = _walletPreviewLines(
        symbolMeta,
        runCfg,
        dash,
        phaseState,
        decision,
        labels,
        posture,
        lockActive,
    )
    openTs = datetime.fromtimestamp(
        int(previewMicro[-1][0]) / 1000.0,
        tz=PERTH_TZ,
    )
    dynReq = abs(float(decision.get('macro_dyn_signed', 0.0)))
    return [
        'GATES - LIVE PREVIEW',
        f"candle        : {openTs:%Y-%m-%d %H:%M} GMT+8",
        f"mode          : {modeLabel(runCfg)} preview only",
        '',
        'MICRO BUY',
        *_sideGateLines(pack, decision, runCfg.overrides, 'BUY'),
        '',
        'MICRO SELL',
        *_sideGateLines(pack, decision, runCfg.overrides, 'SELL'),
        '',
        'MACRO',
        f"- dyn req     : {_fmtPct(dynReq)}",
        f"- direction   : {_macroText(int(decision.get('macro_dir', 0)))}",
        '',
        'WALLET',
        *walletLines,
        '',
        'DECISION',
        f"- action      : {action}",
        f"- reason      : {reason}",
    ]


def nextOrder(
    state: PhaseState,
    flagLabel: str,
    trendCode: int,
    price: float,
    quoteFree: float,
    baseFree: float,
    phaseBuyPortions: int,
    phaseSellPortions: int,
    finalPortionPct: float,
    overrides: dict,
    posture: dict,
    lockActive: bool,
    buyCapPct: float | None = None,
    sellFloorPct: float | None = None,
) -> dict | None:
    # Build a live BUY/SELL order plan from phase state + latest flag.
    currentTrend = trendLabel(trendCode)
    newBearRegime = currentTrend == 'BEAR' and state.lastTrendLabel != 'BEAR'
    newBullRegime = currentTrend == 'BULL' and state.lastTrendLabel != 'BULL'
    state.lastTrendLabel = currentTrend

    if flagLabel == 'BUY':
        if currentTrend == 'BEAR' and quoteFree > 0:
            if state.phaseSide != 'BUY' or newBearRegime:
                state.phaseSide = 'BUY'
                state.phaseLastPrice = None
                state.phaseBaseValue = phaseBuyValue(
                    quoteFree,
                    phaseBuyPortions,
                )
                if finalPortionPct >= 1.0 - 1e-9:
                    state.phasePortionsRemaining = None
                else:
                    if state.phaseBaseValue > 0:
                        state.phasePortionsRemaining = float(phaseBuyPortions)
                    else:
                        state.phasePortionsRemaining = 0.0

            scale, _pct = calcBuyScale(state.phaseLastPrice, price)
            maxSpendQuote = None
            if bool(posture.get('down', False)):
                scale *= float(overrides.get('DAILY_DOWN_BUY_MULT', 0.4))
                crabCap = max(0.0, min(1.0, float(
                    overrides.get('CRAB_ASSET_CAP_PCT', 1.0)
                )))
                if crabCap < 1.0 - 1e-9:
                    maxSpendQuote = buySpendToTargetCap(
                        quoteFree,
                        baseFree,
                        price,
                        crabCap,
                        float(overrides.get('WALLET_FEE_RATE', 0.0)),
                    )
            buyCapPct = _coastBuyCapPct(posture, buyCapPct)
            if buyCapPct is not None and float(buyCapPct) < 1.0 - 1e-9:
                capTarget = float(buyCapPct)
                if bool(posture.get('down', False)):
                    capTarget = min(float(capTarget), float(crabCap))
                capSpendQuote = buySpendToTargetCap(
                    quoteFree,
                    baseFree,
                    price,
                    capTarget,
                    float(overrides.get('WALLET_FEE_RATE', 0.0)),
                )
                if maxSpendQuote is None:
                    maxSpendQuote = capSpendQuote
                else:
                    maxSpendQuote = min(maxSpendQuote, capSpendQuote)
            spend, portionUsed = buySpend(
                quoteFree,
                state.phaseBaseValue,
                scale,
                state.phasePortionsRemaining,
                finalPortionPct,
                maxSpendQuote,
            )
            if spend > 0:
                prevRemaining = state.phasePortionsRemaining
                state.phaseLastPrice = price
                if prevRemaining is not None and finalPortionPct < 1.0 - 1e-9:
                    nextRem = float(prevRemaining) - float(portionUsed)
                    state.phasePortionsRemaining = max(0.0, nextRem)
                return {
                    'side': 'BUY',
                    'quoteQty': spend,
                }
        return None

    if flagLabel == 'SELL':
        if currentTrend == 'BULL' and baseFree > 0:
            if state.phaseSide != 'SELL' or newBullRegime:
                state.phaseSide = 'SELL'
                state.phaseLastPrice = None
                state.phaseBaseValue = phaseSellValue(
                    baseFree,
                    price,
                    phaseSellPortions,
                )
                if finalPortionPct >= 1.0 - 1e-9:
                    state.phasePortionsRemaining = None
                else:
                    if state.phaseBaseValue > 0:
                        state.phasePortionsRemaining = float(phaseSellPortions)
                    else:
                        state.phasePortionsRemaining = 0.0

            scale, _pct = calcSellScale(state.phaseLastPrice, price)
            if (
                bool(posture.get('strong', False))
                and not bool(posture.get('cloudActive', False))
            ):
                scale *= float(overrides.get('ULTRA_SELL_MULT', 0.0))
            maxSellValue = None
            floorPct = 0.0
            if sellFloorPct is not None:
                floorPct = max(floorPct, float(sellFloorPct))
            if bool(lockActive):
                floorPct = max(floorPct, float(
                    posture.get('lockTarget', 1.0)
                ))
            if floorPct > 0.0:
                maxSellValue = floorSellValueCap(
                    quoteFree,
                    baseFree,
                    price,
                    floorPct,
                    float(overrides.get('WALLET_FEE_RATE', 0.0)),
                )
            qty, portionUsed = sellQty(
                baseFree,
                price,
                state.phaseBaseValue,
                scale,
                state.phasePortionsRemaining,
                finalPortionPct,
                maxSellValue,
            )
            if qty > 0:
                prevRemaining = state.phasePortionsRemaining
                state.phaseLastPrice = price
                if prevRemaining is not None and finalPortionPct < 1.0 - 1e-9:
                    nextRem = float(prevRemaining) - float(portionUsed)
                    state.phasePortionsRemaining = max(0.0, nextRem)
                return {
                    'side': 'SELL',
                    'qty': qty,
                }
        return None

    return None


async def processSignals(
    client,
    symbolMeta,
    runCfg: RuntimeCfg,
    dash,
    microRows: list,
    macroRows: list,
    postureRows: list,
    phaseState: PhaseState,
) -> list[dict]:
    # Evaluate latest closed candle and place any triggered spot orders.
    pack = evaluate(
        microRows,
        macroRows,
        runCfg.interval,
        runCfg.periods,
        runCfg.primerDays,
        runCfg.overrides,
    )
    barIndex = len(microRows) - 1
    closePrice = float(pack.ctx['closes'][barIndex])
    barCloseMs = int(microRows[-1][6])
    barsDay = max(bars_per_day(pack.ctx), 1.0)
    posture, postureState = dailyPostureForIndex(
        pack.ctx,
        runCfg.overrides,
        barIndex,
        barsDay,
        postureRows,
    )
    lockActive = bool(postureState.get('lockActive', False))
    cloudActive = bool(posture.get('cloudActive', False))
    events: list[dict] = []
    decision = decisionContext(
        pack,
        microRows,
        barIndex,
        runCfg.overrides,
        posture,
        bool(dash.tradingEnabled),
        bool(dash.seeded),
    )
    decision.update(accountContext(dash, closePrice))
    dash.currentDailyCluster = int(decision.get('daily_cluster', -1))
    dash.currentPosture = str(decision.get('daily_posture', 'unknown'))

    if barIndex < pack.startIdx:
        if dash.logger is not None:
            dash.logger.logDecision(decision)
        return events

    if not bool(dash.tradingEnabled):
        if dash.logger is not None:
            dash.logger.logDecision(decision)
        return events

    peakCfg, peakState = _ensurePeakState(
        phaseState,
        runCfg,
        dash,
        pack.ctx,
        barIndex,
        barsDay,
    )
    strongEntry, peakGraceActive, peakGivebackPct = _stepPeakBeforeLocks(
        peakState,
        peakCfg,
        posture,
        postureState,
        closePrice,
    )

    if bool(posture.get('forceLock', False)):
        quoteFree = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.quoteAsset,
        )
        baseFree = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.baseAsset,
        )
        qty = dailyLockQty(
            quoteFree,
            baseFree,
            closePrice,
            float(posture.get('exitTarget', 1.0)),
            float(runCfg.overrides.get('WALLET_FEE_RATE', 0.0)),
        )
        if qty > 0.0:
            event = await executeMarketSell(
                client,
                symbolMeta,
                runCfg,
                dash,
                runCfg.symbol,
                qty,
                closePrice,
                'daily_posture_lock',
                barCloseMs,
            )
            if event is not None:
                events.append(event)
                decision['final_action'] = 'LOCK_SELL'
                decision['decision_reason'] = 'daily_force_lock'
                _resetPhase(phaseState)
        markLockState(postureState, barIndex)
        lockActive = True

    if peakState is not None and peakCfg.enabled and not cloudActive:
        quoteFree = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.quoteAsset,
        )
        baseFree = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.baseAsset,
        )
        peakDecision = evaluatePeakLock(
            peakState,
            peakCfg,
            closePrice,
            _walletValue(quoteFree, baseFree, closePrice),
            peakGivebackPct,
            strongEntry,
            peakGraceActive,
        )
        if peakDecision.canLock:
            qty = dailyLockQty(
                quoteFree,
                baseFree,
                closePrice,
                peakCfg.capPct,
                float(runCfg.overrides.get('WALLET_FEE_RATE', 0.0)),
            )
            if qty > 0.0:
                event = await executeMarketSell(
                    client,
                    symbolMeta,
                    runCfg,
                    dash,
                    runCfg.symbol,
                    qty,
                    closePrice,
                    'peak_lock',
                    barCloseMs,
                )
                if event is not None:
                    events.append(event)
                    recordPeakLock(peakState, peakCfg, barIndex)
                    decision['final_action'] = 'PEAK_LOCK_SELL'
                    decision['decision_reason'] = 'peak_lock'
                    _resetPhase(phaseState)
        stepActivePeakLock(peakState, peakCfg, barIndex)

    crabCap = max(0.0, min(1.0, float(
        runCfg.overrides.get('CRAB_ASSET_CAP_PCT', 1.0)
    )))
    if bool(posture.get('downEntry', False)) and crabCap < 1.0 - 1e-9:
        quoteFree = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.quoteAsset,
        )
        baseFree = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.baseAsset,
        )
        qty = dailyLockQty(
            quoteFree,
            baseFree,
            closePrice,
            crabCap,
            float(runCfg.overrides.get('WALLET_FEE_RATE', 0.0)),
        )
        if qty > 0.0:
            event = await executeMarketSell(
                client,
                symbolMeta,
                runCfg,
                dash,
                runCfg.symbol,
                qty,
                closePrice,
                'daily_crab_cap',
                barCloseMs,
            )
            if event is not None:
                events.append(event)
                decision['final_action'] = 'CRAB_CAP_SELL'
                decision['decision_reason'] = 'daily_crab_cap'
                _resetPhase(phaseState)

    targetPct = _strongTargetPct(runCfg.overrides, peakState)
    if (
        bool(posture.get('rawStrong', False))
        and not bool(lockActive)
        and not cloudActive
        and bool(posture.get('clusterEnabled', True))
        and targetPct > 0.0
    ):
        quoteFree = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.quoteAsset,
        )
        baseFree = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.baseAsset,
        )
        spend = buySpendToTargetCap(
            quoteFree,
            baseFree,
            closePrice,
            targetPct,
            float(runCfg.overrides.get('WALLET_FEE_RATE', 0.0)),
        )
        if spend > 0.0:
            event = await executeMarketBuy(
                client,
                symbolMeta,
                runCfg,
                dash,
                runCfg.symbol,
                spend,
                closePrice,
                'daily_strong_target_buy',
                barCloseMs,
            )
            if event is not None:
                events.append(event)
                decision['final_action'] = 'TARGET_BUY'
                decision['decision_reason'] = 'daily_strong_target_buy'
                _resetPhase(phaseState)

    labels = latestFlags(pack.flags, barIndex)
    trendCode = int(
        np.asarray(pack.signals['trendCode'], dtype=int)[barIndex]
    )
    if not labels:
        decision['order_count'] = len(events)
        if dash.logger is not None:
            dash.logger.logDecision(decision)
        for event in events:
            event.update(tradeAuditContext(decision))
        return events

    actions: list[str] = [
        str(decision.get('final_action', ''))
        if str(decision.get('final_action', '')) != 'HOLD' else ''
    ]
    for label in labels:
        phaseBefore = PhaseState(
            phaseSide=phaseState.phaseSide,
            phaseBaseValue=phaseState.phaseBaseValue,
            phaseLastPrice=phaseState.phaseLastPrice,
            phasePortionsRemaining=phaseState.phasePortionsRemaining,
            lastTrendLabel=phaseState.lastTrendLabel,
        )
        quoteFree = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.quoteAsset,
        )
        baseFree = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.baseAsset,
        )
        buyCapPct = None
        sellFloorPct = _strongSellFloorPct(
            runCfg.overrides,
            posture,
            peakState,
        )
        if (
            label == 'BUY'
            and not cloudActive
            and peakState is not None
            and peakState.active
            and peakState.cap < 1.0 - 1e-9
        ):
            baseCap = float(runCfg.overrides.get('PEAK_LOCK_CAP_PCT', 1.0))
            if peakState.cap > baseCap + 1e-9:
                buyCapPct = float(peakState.cap)
            else:
                buyCapPct = 0.0
        if label == 'BUY':
            buyCapPct = _coastBuyCapPct(posture, buyCapPct)
        plan = nextOrder(
            phaseState,
            label,
            trendCode,
            closePrice,
            quoteFree,
            baseFree,
            int(runCfg.overrides['PHASE_BUY_PORTIONS']),
            int(runCfg.overrides['PHASE_SELL_PORTIONS']),
            float(runCfg.overrides['FINAL_PORTION_PCT']),
            runCfg.overrides,
            posture,
            lockActive,
            buyCapPct,
            sellFloorPct,
        )
        if plan is None:
            if not events:
                decision['decision_reason'] = (
                    f"{label.lower()}_plan_rejected"
                )
            continue

        if plan['side'] == 'BUY':
            event = await executeMarketBuy(
                client,
                symbolMeta,
                runCfg,
                dash,
                runCfg.symbol,
                float(plan['quoteQty']),
                closePrice,
                label,
                barCloseMs,
            )
        else:
            event = await executeMarketSell(
                client,
                symbolMeta,
                runCfg,
                dash,
                runCfg.symbol,
                float(plan['qty']),
                closePrice,
                label,
                barCloseMs,
            )
        if event is None:
            phaseState.phaseSide = phaseBefore.phaseSide
            phaseState.phaseBaseValue = phaseBefore.phaseBaseValue
            phaseState.phaseLastPrice = phaseBefore.phaseLastPrice
            phaseState.phasePortionsRemaining = (
                phaseBefore.phasePortionsRemaining
            )
            phaseState.lastTrendLabel = phaseBefore.lastTrendLabel
            if not events:
                decision['decision_reason'] = (
                    f"{label.lower()}_order_rejected"
                )
            continue
        if not event.get('side'):
            event['side'] = str(plan['side'])
        if (
            plan['side'] == 'BUY'
            and buyCapPct is not None
            and buyCapPct > 0.0
            and not cloudActive
            and peakState is not None
            and peakState.active
        ):
            recordPeakCappedBuy(peakState)
        events.append(event)
        actions.append(str(plan['side']).upper())

    actions = [txt for txt in actions if txt]
    if actions:
        decision['final_action'] = '|'.join(actions)
        decision['decision_reason'] = 'order_sent'
    decision['order_count'] = len(events)
    if dash.logger is not None:
        dash.logger.logDecision(decision)
    for event in events:
        event.update(tradeAuditContext(decision))
    return events


async def processPaperReplaySignals(
    client,
    symbolMeta,
    runCfg: RuntimeCfg,
    dash,
    microRows: list,
    macroRows: list,
    postureRows: list,
    phaseState: PhaseState,
) -> list[dict]:
    # Evaluate one missed paper candle and simulate historical fills.
    pack = evaluate(
        microRows,
        macroRows,
        runCfg.interval,
        runCfg.periods,
        runCfg.primerDays,
        runCfg.overrides,
    )
    barIndex = len(microRows) - 1
    closePrice = float(pack.ctx['closes'][barIndex])
    barCloseMs = int(microRows[-1][6])
    barsDay = max(bars_per_day(pack.ctx), 1.0)
    posture, postureState = dailyPostureForIndex(
        pack.ctx,
        runCfg.overrides,
        barIndex,
        barsDay,
        postureRows,
    )
    lockActive = bool(postureState.get('lockActive', False))
    cloudActive = bool(posture.get('cloudActive', False))
    events: list[dict] = []
    decision = decisionContext(
        pack,
        microRows,
        barIndex,
        runCfg.overrides,
        posture,
        bool(dash.tradingEnabled),
        bool(dash.seeded),
    )
    decision.update(accountContext(dash, closePrice))
    dash.currentDailyCluster = int(decision.get('daily_cluster', -1))
    dash.currentPosture = str(decision.get('daily_posture', 'unknown'))

    if barIndex < pack.startIdx:
        if dash.logger is not None:
            dash.logger.logDecision(decision)
        return events

    if not bool(dash.tradingEnabled):
        if dash.logger is not None:
            dash.logger.logDecision(decision)
        return events

    peakCfg, peakState = _ensurePeakState(
        phaseState,
        runCfg,
        dash,
        pack.ctx,
        barIndex,
        barsDay,
    )
    strongEntry, peakGraceActive, peakGivebackPct = _stepPeakBeforeLocks(
        peakState,
        peakCfg,
        posture,
        postureState,
        closePrice,
    )

    if bool(posture.get('forceLock', False)):
        quoteFree = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.quoteAsset,
        )
        baseFree = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.baseAsset,
        )
        qty = dailyLockQty(
            quoteFree,
            baseFree,
            closePrice,
            float(posture.get('exitTarget', 1.0)),
            float(runCfg.overrides.get('WALLET_FEE_RATE', 0.0)),
        )
        if qty > 0.0:
            event = executePaperReplaySell(
                symbolMeta,
                runCfg,
                dash,
                qty,
                closePrice,
                'daily_posture_lock',
                barCloseMs,
            )
            if event is not None:
                events.append(event)
                decision['final_action'] = 'LOCK_SELL'
                decision['decision_reason'] = 'daily_force_lock'
                _resetPhase(phaseState)
        markLockState(postureState, barIndex)
        lockActive = True

    if peakState is not None and peakCfg.enabled and not cloudActive:
        quoteFree = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.quoteAsset,
        )
        baseFree = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.baseAsset,
        )
        peakDecision = evaluatePeakLock(
            peakState,
            peakCfg,
            closePrice,
            _walletValue(quoteFree, baseFree, closePrice),
            peakGivebackPct,
            strongEntry,
            peakGraceActive,
        )
        if peakDecision.canLock:
            qty = dailyLockQty(
                quoteFree,
                baseFree,
                closePrice,
                peakCfg.capPct,
                float(runCfg.overrides.get('WALLET_FEE_RATE', 0.0)),
            )
            if qty > 0.0:
                event = executePaperReplaySell(
                    symbolMeta,
                    runCfg,
                    dash,
                    qty,
                    closePrice,
                    'peak_lock',
                    barCloseMs,
                )
                if event is not None:
                    events.append(event)
                    recordPeakLock(peakState, peakCfg, barIndex)
                    decision['final_action'] = 'PEAK_LOCK_SELL'
                    decision['decision_reason'] = 'peak_lock'
                    _resetPhase(phaseState)
        stepActivePeakLock(peakState, peakCfg, barIndex)

    crabCap = max(0.0, min(1.0, float(
        runCfg.overrides.get('CRAB_ASSET_CAP_PCT', 1.0)
    )))
    if bool(posture.get('downEntry', False)) and crabCap < 1.0 - 1e-9:
        quoteFree = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.quoteAsset,
        )
        baseFree = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.baseAsset,
        )
        qty = dailyLockQty(
            quoteFree,
            baseFree,
            closePrice,
            crabCap,
            float(runCfg.overrides.get('WALLET_FEE_RATE', 0.0)),
        )
        if qty > 0.0:
            event = executePaperReplaySell(
                symbolMeta,
                runCfg,
                dash,
                qty,
                closePrice,
                'daily_crab_cap',
                barCloseMs,
            )
            if event is not None:
                events.append(event)
                decision['final_action'] = 'CRAB_CAP_SELL'
                decision['decision_reason'] = 'daily_crab_cap'
                _resetPhase(phaseState)

    targetPct = _strongTargetPct(runCfg.overrides, peakState)
    if (
        bool(posture.get('rawStrong', False))
        and not bool(lockActive)
        and not cloudActive
        and bool(posture.get('clusterEnabled', True))
        and targetPct > 0.0
    ):
        quoteFree = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.quoteAsset,
        )
        baseFree = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.baseAsset,
        )
        spend = buySpendToTargetCap(
            quoteFree,
            baseFree,
            closePrice,
            targetPct,
            float(runCfg.overrides.get('WALLET_FEE_RATE', 0.0)),
        )
        if spend > 0.0:
            event = executePaperReplayBuy(
                symbolMeta,
                runCfg,
                dash,
                spend,
                closePrice,
                'daily_strong_target_buy',
                barCloseMs,
            )
            if event is not None:
                events.append(event)
                decision['final_action'] = 'TARGET_BUY'
                decision['decision_reason'] = 'daily_strong_target_buy'
                _resetPhase(phaseState)

    labels = latestFlags(pack.flags, barIndex)
    trendCode = int(
        np.asarray(pack.signals['trendCode'], dtype=int)[barIndex]
    )
    if not labels:
        decision['order_count'] = len(events)
        if dash.logger is not None:
            dash.logger.logDecision(decision)
        for event in events:
            event.update(tradeAuditContext(decision))
        return events

    actions: list[str] = [
        str(decision.get('final_action', ''))
        if str(decision.get('final_action', '')) != 'HOLD' else ''
    ]
    for label in labels:
        phaseBefore = PhaseState(
            phaseSide=phaseState.phaseSide,
            phaseBaseValue=phaseState.phaseBaseValue,
            phaseLastPrice=phaseState.phaseLastPrice,
            phasePortionsRemaining=phaseState.phasePortionsRemaining,
            lastTrendLabel=phaseState.lastTrendLabel,
        )
        quoteFree = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.quoteAsset,
        )
        baseFree = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.baseAsset,
        )
        buyCapPct = None
        sellFloorPct = _strongSellFloorPct(
            runCfg.overrides,
            posture,
            peakState,
        )
        if (
            label == 'BUY'
            and not cloudActive
            and peakState is not None
            and peakState.active
            and peakState.cap < 1.0 - 1e-9
        ):
            baseCap = float(runCfg.overrides.get('PEAK_LOCK_CAP_PCT', 1.0))
            if peakState.cap > baseCap + 1e-9:
                buyCapPct = float(peakState.cap)
            else:
                buyCapPct = 0.0
        if label == 'BUY':
            buyCapPct = _coastBuyCapPct(posture, buyCapPct)
        plan = nextOrder(
            phaseState,
            label,
            trendCode,
            closePrice,
            quoteFree,
            baseFree,
            int(runCfg.overrides['PHASE_BUY_PORTIONS']),
            int(runCfg.overrides['PHASE_SELL_PORTIONS']),
            float(runCfg.overrides['FINAL_PORTION_PCT']),
            runCfg.overrides,
            posture,
            lockActive,
            buyCapPct,
            sellFloorPct,
        )
        if plan is None:
            if not events:
                decision['decision_reason'] = f"{label.lower()}_plan_rejected"
            continue

        if plan['side'] == 'BUY':
            event = executePaperReplayBuy(
                symbolMeta,
                runCfg,
                dash,
                float(plan['quoteQty']),
                closePrice,
                label,
                barCloseMs,
            )
        else:
            event = executePaperReplaySell(
                symbolMeta,
                runCfg,
                dash,
                float(plan['qty']),
                closePrice,
                label,
                barCloseMs,
            )
        if event is None:
            phaseState.phaseSide = phaseBefore.phaseSide
            phaseState.phaseBaseValue = phaseBefore.phaseBaseValue
            phaseState.phaseLastPrice = phaseBefore.phaseLastPrice
            phaseState.phasePortionsRemaining = (
                phaseBefore.phasePortionsRemaining
            )
            phaseState.lastTrendLabel = phaseBefore.lastTrendLabel
            if not events:
                decision['decision_reason'] = (
                    f"{label.lower()}_order_rejected"
                )
            continue
        if not event.get('side'):
            event['side'] = str(plan['side'])
        if (
            plan['side'] == 'BUY'
            and buyCapPct is not None
            and buyCapPct > 0.0
            and not cloudActive
            and peakState is not None
            and peakState.active
        ):
            recordPeakCappedBuy(peakState)
        events.append(event)
        actions.append(str(plan['side']).upper())

    actions = [txt for txt in actions if txt]
    if actions:
        decision['final_action'] = '|'.join(actions)
        decision['decision_reason'] = 'paper_replay_order'
    decision['order_count'] = len(events)
    if dash.logger is not None:
        dash.logger.logDecision(decision)
    for event in events:
        event.update(tradeAuditContext(decision))
    return events
