#!/usr/bin/env python3
# live_engine.py – context/signal/flag evaluation for live candles.

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from engine.dynamics import alignMacroDyn, macroDynFromContext
from engine.shared import bars_per_day, buildContext, buildSignals
from engine.shared import zscoreSeries
from runtime.diag import generateFlags
from runtime.gates import paramsFromSettings
from strategy.posture import DAILY_STRONG_CLUSTER, dailyDownNow
from strategy.signals import trendLabel


@dataclass(frozen=True)
class EvalPack:
    # Bundle live evaluation outputs for one kline snapshot.
    ctx: Dict[str, Any]
    signals: Dict[str, object]
    flags: List[Tuple[int, str]]
    startIdx: int
    macroDyn: np.ndarray | None
    macroDir: np.ndarray | None
    macroMom: np.ndarray | None


def candleTimes(rows: list) -> list:
    # Convert kline open times to UTC datetimes.
    return [
        datetime.fromtimestamp(
            int(row[0]) / 1000.0,
            tz=timezone.utc,
        )
        for row in rows
    ]


def candleCloseTimes(rows: list) -> list:
    # Convert kline close times to UTC datetimes.
    return [
        datetime.fromtimestamp(
            int(row[6]) / 1000.0,
            tz=timezone.utc,
        )
        for row in rows
    ]


def startIndex(ctx: dict, periods: list[int], primerDays: int) -> int:
    # Compute first tradable index using MA warmup + primer days.
    idx = max(periods) * 2
    if primerDays > 0:
        idx += int(round(float(primerDays) * bars_per_day(ctx)))
    return idx


def macroArrays(
    microTimes: list,
    macroRows: list,
    periods: list[int],
    overrides: dict,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    # Build macro dynamic/dir/mom arrays aligned to micro timestamps.
    macroInterval = str(overrides['MACRO_INTERVAL']).strip()
    if not macroInterval:
        return None, None, None

    periodsMacro = list(periods)
    macroP1 = int(overrides['MACRO_P1'])
    macroGradPeriod = int(overrides['MACRO_GRAD_PERIOD'])
    macroP3 = int(overrides['MACRO_P3'])
    if macroP1 > 0 and len(periodsMacro) >= 1:
        periodsMacro[0] = macroP1
    if macroGradPeriod > 0 and len(periodsMacro) >= 2:
        periodsMacro[1] = macroGradPeriod
    if macroP3 > 0:
        if len(periodsMacro) >= 3:
            periodsMacro[2] = macroP3
        else:
            periodsMacro.append(macroP3)

    ctxMacro = buildContext(macroRows, periodsMacro)
    ctxMacro['intervalStr'] = macroInterval

    dynMacro = macroDynFromContext(
        ctxMacro,
        float(overrides['MACRO_NRG_WIN_DAYS']),
        float(overrides['MACRO_NRG_Z_MIN']),
        float(overrides['MACRO_NRG_Z_MAX']),
        float(overrides['MACRO_DYN_PCT_MAX']),
        float(overrides['MACRO_DYN_PCT_MIN']),
        float(overrides['MACRO_GRAD_WIN_DAYS']),
        float(overrides['MACRO_GRAD_Z_MIN']),
        float(overrides['MACRO_GRAD_Z_MAX']),
        float(overrides['MACRO_MULT_GRAD_MIN']),
        float(overrides['MACRO_MULT_GRAD_MAX']),
    )

    mas = ctxMacro['mas']
    m1 = np.asarray(mas[0], dtype=float)
    m2 = np.asarray(mas[1], dtype=float)
    m3 = np.asarray(mas[2], dtype=float)

    macroDir = np.zeros_like(m1, dtype=int)
    macroDir[m1 > m3] = 1
    macroDir[m1 < m3] = -1

    macroMom = np.zeros_like(m1, dtype=int)
    macroMom[m1 > m2] = 1
    macroMom[m1 < m2] = -1

    macroTimes = candleCloseTimes(macroRows)
    dyn = alignMacroDyn(macroTimes, dynMacro, microTimes)
    dirAligned = alignMacroDyn(
        macroTimes,
        macroDir.astype(float),
        microTimes,
    ).astype(int)
    momAligned = alignMacroDyn(
        macroTimes,
        macroMom.astype(float),
        microTimes,
    ).astype(int)
    return dyn, dirAligned, momAligned


def evaluate(
    microRows: list,
    macroRows: list,
    interval: str,
    periods: list[int],
    primerDays: int,
    overrides: dict,
) -> EvalPack:
    # Build ctx/signals/flags for current snapshot.
    ctx = buildContext(microRows, periods)
    ctx['intervalStr'] = interval
    sig = buildSignals(ctx, [])
    idx0 = startIndex(ctx, periods, primerDays)
    microTimes = candleTimes(microRows)
    dyn, macroDir, macroMom = macroArrays(
        microTimes,
        macroRows,
        periods,
        overrides,
    )
    params = paramsFromSettings(overrides)
    idxFlags = generateFlags(
        ctx,
        sig,
        params,
        idx0,
        overrides,
        dyn,
        macroDir,
        macroMom,
    )
    return EvalPack(
        ctx=ctx,
        signals=sig,
        flags=idxFlags,
        startIdx=idx0,
        macroDyn=dyn,
        macroDir=macroDir,
        macroMom=macroMom,
    )


########################################################################
# Decision Logging Helpers
########################################################################

def postureText(cluster: int) -> str:
    # Map the live daily cluster into the dashboard/post-run posture label.
    if int(cluster) == DAILY_STRONG_CLUSTER:
        return 'up'
    if dailyDownNow(int(cluster)):
        return 'down'
    if int(cluster) >= 0:
        return 'neutral'
    return 'unknown'


def _arrayVal(arr: np.ndarray | None, index: int, default: float) -> float:
    if arr is None:
        return float(default)
    values = np.asarray(arr)
    if int(index) < 0 or int(index) >= values.size:
        return float(default)
    return float(values[int(index)])


def _signedGradZ(
    ctx: dict,
    g1: np.ndarray,
    overrides: dict,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    winDays = max(int(overrides[f'GRAD1_{side}_WIN_DAYS']), 1)
    winBars = max(int(round(winDays * bars_per_day(ctx))), 1)
    zVals, valid = zscoreSeries(ctx, g1, winBars, 'g1p1')
    sign = -1.0 if side == 'BUY' else 1.0
    return zVals * sign, valid


def decisionContext(
    pack: EvalPack,
    microRows: list,
    barIndex: int,
    overrides: dict,
    posture: dict,
    tradingEnabled: bool,
    seeded: bool,
) -> dict[str, object]:
    # Build one flat audit row from the same EvalPack used for trading.
    row = microRows[int(barIndex)]
    g1 = np.asarray(pack.signals['g1P1'], dtype=float)
    trendArr = np.asarray(pack.signals['trendCode'], dtype=int)
    trendCode = int(trendArr[int(barIndex)])
    buyZ, buyValid = _signedGradZ(pack.ctx, g1, overrides, 'BUY')
    sellZ, sellValid = _signedGradZ(pack.ctx, g1, overrides, 'SELL')
    labels = [label for idx, label in pack.flags if int(idx) == barIndex]
    valid = int(barIndex) >= int(pack.startIdx)
    allowBuy = trendCode == -1
    allowSell = trendCode == 1
    acceptedBuy = 'BUY' in labels
    acceptedSell = 'SELL' in labels
    cluster = int(posture.get('cluster', -1))
    reason = 'hold'
    if not valid:
        reason = 'warmup'
    elif not bool(tradingEnabled):
        reason = 'paused'
    elif bool(posture.get('forceLock', False)):
        reason = 'daily_force_lock'
    elif acceptedBuy or acceptedSell:
        reason = 'flag_ready'
    elif not allowBuy and not allowSell:
        reason = 'trend_neutral'
    elif allowBuy and not bool(buyValid[int(barIndex)]):
        reason = 'buy_z_not_ready'
    elif allowSell and not bool(sellValid[int(barIndex)]):
        reason = 'sell_z_not_ready'
    elif allowBuy:
        reason = 'buy_gate_rejected'
    elif allowSell:
        reason = 'sell_gate_rejected'

    return {
        'signal_open_ms': int(row[0]),
        'signal_close_ms': int(row[6]),
        'open': float(row[1]),
        'high': float(row[2]),
        'low': float(row[3]),
        'close': float(row[4]),
        'volume': float(row[5]),
        'bar_index': int(barIndex),
        'start_index': int(pack.startIdx),
        'trading_enabled': bool(tradingEnabled),
        'seeded': bool(seeded),
        'daily_cluster': cluster,
        'daily_posture': postureText(cluster),
        'daily_strong': bool(posture.get('strong', False)),
        'daily_down': bool(posture.get('down', False)),
        'daily_late': bool(posture.get('late', False)),
        'daily_force_lock': bool(posture.get('forceLock', False)),
        'macro_dyn_signed': _arrayVal(pack.macroDyn, barIndex, 0.0),
        'macro_dir': int(_arrayVal(pack.macroDir, barIndex, 0.0)),
        'macro_mom': int(_arrayVal(pack.macroMom, barIndex, 0.0)),
        'trend_code': trendCode,
        'trend_label': trendLabel(trendCode),
        'grad1': float(g1[int(barIndex)]),
        'buy_z': float(buyZ[int(barIndex)]),
        'sell_z': float(sellZ[int(barIndex)]),
        'buy_z_valid': bool(buyValid[int(barIndex)]),
        'sell_z_valid': bool(sellValid[int(barIndex)]),
        'allow_buy': bool(allowBuy),
        'allow_sell': bool(allowSell),
        'accepted_buy': bool(acceptedBuy),
        'accepted_sell': bool(acceptedSell),
        'flag_labels': '|'.join(labels),
        'final_action': 'HOLD',
        'decision_reason': reason,
    }
