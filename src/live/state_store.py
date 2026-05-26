#!/usr/bin/env python3
# state_store.py - rolling live runtime state persistence.

from __future__ import annotations

import csv
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

from strategy.supervisor import PeakLockState


STATE_FIELDS = [
    'saved_at_utc',
    'saved_at_ms',
    'mode',
    'symbol',
    'interval',
    'macro_interval',
    'last_processed_open_ms',
    'last_processed_close_ms',
    'last_closed_price',
    'trading_enabled',
    'seeded',
    'seed_quote',
    'hodl_qty',
    'hodl_entry_price',
    'quote_total',
    'base_total',
    'trade_count',
    'phase_side',
    'phase_base_value',
    'phase_last_price',
    'phase_portions_remaining',
    'last_trend_label',
    'current_daily_cluster',
    'current_posture',
    'last_trade_time_ms',
    'last_trade_flag',
    'last_trade_side',
    'last_order_id',
    'peak_enabled',
    'peak_ma',
    'peak_bench_qty',
    'peak_integral',
    'peak_prev_err',
    'peak_long',
    'peak_bear_count',
    'peak_strong_grace_bars',
    'peak_strong_releases',
    'peak_prev_strong',
    'peak_active',
    'peak_start',
    'peak_cap',
    'peak_edge_start',
    'peak_edge_now',
    'peak_edge_peak',
    'peak_lock_gain',
    'peak_lock_gain_max',
    'peak_locks',
    'peak_capped_buys',
    'peak_lock_hours',
    'peak_unlock_steps',
    'peak_armed',
    'peak_edge_vals',
]


def utcNowMs() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000.0)


def utcNowText() -> str:
    return datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def statePath(outPath: Path) -> Path:
    base = outPath if outPath.suffix else outPath.with_suffix('.csv')
    return base.with_name(f"{base.stem}_state{base.suffix}")


def _boolText(value: object) -> str:
    return '1' if bool(value) else '0'


def _noneText(value: object) -> str:
    return '' if value is None else str(value)


def _floatVal(row: dict[str, str], key: str, default: float) -> float:
    raw = str(row.get(key, '')).strip()
    return float(raw) if raw else float(default)


def _intVal(row: dict[str, str], key: str, default: int) -> int:
    raw = str(row.get(key, '')).strip()
    return int(float(raw)) if raw else int(default)


def _boolVal(row: dict[str, str], key: str, default: bool) -> bool:
    raw = str(row.get(key, '')).strip().lower()
    if not raw:
        return bool(default)
    return raw in ('1', 'true', 'yes', 'on')


def _edgeValsText(value: object) -> str:
    vals = getattr(value, 'edgeVals', [])
    return '|'.join(str(float(i)) for i in vals)


def _edgeVals(raw: str) -> list[float]:
    text = str(raw).strip()
    if not text:
        return []
    return [float(i) for i in text.split('|') if str(i).strip()]


def _capturePeakState(value: object) -> dict[str, object]:
    state = value
    if state is None:
        return {
            'peak_enabled': '0',
            'peak_edge_vals': '',
        }
    return {
        'peak_enabled': '1',
        'peak_ma': float(state.ma),
        'peak_bench_qty': float(state.benchQty),
        'peak_integral': float(state.integral),
        'peak_prev_err': float(state.prevErr),
        'peak_long': _boolText(state.long),
        'peak_bear_count': int(state.bearCount),
        'peak_strong_grace_bars': int(state.strongGraceBars),
        'peak_strong_releases': int(state.strongReleases),
        'peak_prev_strong': _boolText(state.prevStrong),
        'peak_active': _boolText(state.active),
        'peak_start': int(state.start),
        'peak_cap': float(state.cap),
        'peak_edge_start': float(state.edgeStart),
        'peak_edge_now': float(state.edgeNow),
        'peak_edge_peak': float(state.edgePeak),
        'peak_lock_gain': float(state.lockGain),
        'peak_lock_gain_max': float(state.lockGainMax),
        'peak_locks': int(state.locks),
        'peak_capped_buys': int(state.cappedBuys),
        'peak_lock_hours': int(state.lockHours),
        'peak_unlock_steps': int(state.unlockSteps),
        'peak_armed': _boolText(state.armed),
        'peak_edge_vals': _edgeValsText(state),
    }


def _restorePeakState(row: dict[str, str]) -> PeakLockState | None:
    if not _boolVal(row, 'peak_enabled', False):
        return None
    state = PeakLockState(
        ma=_floatVal(row, 'peak_ma', 0.0),
        benchQty=_floatVal(row, 'peak_bench_qty', 0.0),
    )
    state.integral = _floatVal(row, 'peak_integral', 0.0)
    state.prevErr = _floatVal(row, 'peak_prev_err', 0.0)
    state.long = _boolVal(row, 'peak_long', False)
    state.bearCount = _intVal(row, 'peak_bear_count', 0)
    state.strongGraceBars = _intVal(row, 'peak_strong_grace_bars', 0)
    state.strongReleases = _intVal(row, 'peak_strong_releases', 0)
    state.prevStrong = _boolVal(row, 'peak_prev_strong', False)
    state.active = _boolVal(row, 'peak_active', False)
    state.start = _intVal(row, 'peak_start', -1)
    state.cap = _floatVal(row, 'peak_cap', 1.0)
    state.edgeStart = _floatVal(row, 'peak_edge_start', 0.0)
    state.edgeNow = _floatVal(row, 'peak_edge_now', 0.0)
    state.edgePeak = _floatVal(row, 'peak_edge_peak', 0.0)
    state.lockGain = _floatVal(row, 'peak_lock_gain', 0.0)
    state.lockGainMax = _floatVal(row, 'peak_lock_gain_max', 0.0)
    state.locks = _intVal(row, 'peak_locks', 0)
    state.cappedBuys = _intVal(row, 'peak_capped_buys', 0)
    state.lockHours = _intVal(row, 'peak_lock_hours', 0)
    state.unlockSteps = _intVal(row, 'peak_unlock_steps', 0)
    state.armed = _boolVal(row, 'peak_armed', False)
    state.edgeVals = _edgeVals(str(row.get('peak_edge_vals', '')))
    return state


def loadState(path: Path) -> dict[str, str] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open(newline='') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    return rows[-1] if rows else None


def stateMatches(row: dict[str, str] | None, runCfg: Any) -> bool:
    if row is None:
        return False
    mode = 'PAPER' if runCfg.paperTrading else (
        'LIVE-DRY' if runCfg.dryRun else 'LIVE'
    )
    return (
        str(row.get('symbol', '')).upper() == str(runCfg.symbol).upper()
        and str(row.get('interval', '')) == str(runCfg.interval)
        and str(row.get('mode', '')) == mode
    )


def captureState(
    runCfg: Any,
    dash: Any,
    phaseState: Any,
    lastOpenMs: int,
    lastCloseMs: int,
) -> dict[str, object]:
    trade = dash.lastTrade or {}
    peakRow = _capturePeakState(getattr(phaseState, 'peakState', None))
    row = {
        'saved_at_utc': utcNowText(),
        'saved_at_ms': utcNowMs(),
        'mode': 'PAPER' if runCfg.paperTrading else (
            'LIVE-DRY' if runCfg.dryRun else 'LIVE'
        ),
        'symbol': runCfg.symbol,
        'interval': runCfg.interval,
        'macro_interval': runCfg.macroInterval,
        'last_processed_open_ms': int(lastOpenMs or 0),
        'last_processed_close_ms': int(lastCloseMs or 0),
        'last_closed_price': float(dash.lastClosedPrice or 0.0),
        'trading_enabled': _boolText(dash.tradingEnabled),
        'seeded': _boolText(dash.seeded),
        'seed_quote': float(dash.seedQuote or 0.0),
        'hodl_qty': _noneText(dash.hodlQty),
        'hodl_entry_price': _noneText(dash.hodlEntryPrice),
        'quote_total': float(dash.quoteTotal or 0.0),
        'base_total': float(dash.baseTotal or 0.0),
        'trade_count': int(dash.tradeCount or 0),
        'phase_side': _noneText(phaseState.phaseSide),
        'phase_base_value': float(phaseState.phaseBaseValue or 0.0),
        'phase_last_price': _noneText(phaseState.phaseLastPrice),
        'phase_portions_remaining': _noneText(
            phaseState.phasePortionsRemaining
        ),
        'last_trend_label': _noneText(phaseState.lastTrendLabel),
        'current_daily_cluster': int(dash.currentDailyCluster or -1),
        'current_posture': str(dash.currentPosture or ''),
        'last_trade_time_ms': int(trade.get('timeMs', 0) or 0),
        'last_trade_flag': str(trade.get('flag', '')),
        'last_trade_side': str(trade.get('side', '')),
        'last_order_id': str(trade.get('orderId', '')),
    }
    row.update(peakRow)
    return row


def saveState(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmpPath = path.with_name(f"{path.name}.tmp")
    with tmpPath.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=STATE_FIELDS)
        writer.writeheader()
        writer.writerow({key: row.get(key, '') for key in STATE_FIELDS})
    os.replace(tmpPath, path)


def saveRuntimeState(
    path: Path,
    runCfg: Any,
    dash: Any,
    phaseState: Any,
    lastOpenMs: int,
    lastCloseMs: int,
) -> None:
    row = captureState(runCfg, dash, phaseState, lastOpenMs, lastCloseMs)
    saveState(path, row)


def restoreDashboardState(dash: Any, runCfg: Any, row: dict[str, str]) -> None:
    dash.tradingEnabled = _boolVal(row, 'trading_enabled', dash.tradingEnabled)
    dash.seeded = _boolVal(row, 'seeded', dash.seeded)
    dash.seedQuote = _floatVal(row, 'seed_quote', dash.seedQuote)
    dash.tradeCount = _intVal(row, 'trade_count', dash.tradeCount)
    dash.lastClosedPrice = _floatVal(
        row,
        'last_closed_price',
        dash.lastClosedPrice,
    )
    dash.currentDailyCluster = _intVal(
        row,
        'current_daily_cluster',
        dash.currentDailyCluster,
    )
    posture = str(row.get('current_posture', '')).strip()
    if posture:
        dash.currentPosture = posture
    hodlQty = str(row.get('hodl_qty', '')).strip()
    hodlEntry = str(row.get('hodl_entry_price', '')).strip()
    dash.hodlQty = float(hodlQty) if hodlQty else None
    dash.hodlEntryPrice = float(hodlEntry) if hodlEntry else None
    if bool(runCfg.paperTrading):
        dash.quoteTotal = _floatVal(row, 'quote_total', dash.quoteTotal)
        dash.baseTotal = _floatVal(row, 'base_total', dash.baseTotal)


def restorePhaseState(phaseState: Any, row: dict[str, str]) -> None:
    phaseSide = str(row.get('phase_side', '')).strip()
    phaseLast = str(row.get('phase_last_price', '')).strip()
    phaseRem = str(row.get('phase_portions_remaining', '')).strip()
    trend = str(row.get('last_trend_label', '')).strip()
    phaseState.phaseSide = phaseSide or None
    phaseState.phaseBaseValue = _floatVal(row, 'phase_base_value', 0.0)
    phaseState.phaseLastPrice = float(phaseLast) if phaseLast else None
    phaseState.phasePortionsRemaining = (
        float(phaseRem) if phaseRem else None
    )
    phaseState.lastTrendLabel = trend or None
    phaseState.peakState = _restorePeakState(row)
