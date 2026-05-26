#!/usr/bin/env python3
# session_logger.py - persistent CSV logging for live sessions.

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
from typing import Any

from live.execution import modeLabel
from repo_paths import LIVE_PROFILE_PATH, ROOT_DIR, livePath


def _utcNowText() -> str:
    # Return current UTC timestamp in ISO-like text format.
    return datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _csvPath(outPath: Path) -> Path:
    # Normalize log output path to a CSV path.
    if outPath.suffix:
        return outPath
    return outPath.with_suffix('.csv')


def _tradePath(outPath: Path) -> Path:
    # Derive the companion trade-log CSV path from snapshot path.
    base = _csvPath(outPath)
    return base.with_name(f"{base.stem}_trades{base.suffix}")


def _eventPath(outPath: Path) -> Path:
    # Derive the companion runtime-event CSV path from snapshot path.
    base = _csvPath(outPath)
    return base.with_name(f"{base.stem}_events{base.suffix}")


def _decisionPath(outPath: Path) -> Path:
    # Derive the companion closed-candle decision CSV path.
    base = _csvPath(outPath)
    return base.with_name(f"{base.stem}_decisions{base.suffix}")


def _fileHash(path: Path) -> str:
    if not path.exists():
        return ''
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _runtimePath(rawPath: str) -> Path:
    path = Path(str(rawPath).strip())
    if path.is_absolute():
        return path
    return livePath(path)


def _modelHash(overrides: dict) -> str:
    modelPath = str(overrides.get('DAILY_CLUSTER_MODEL_PATH', '')).strip()
    labelPath = str(overrides.get('DAILY_CLUSTER_PATH', '')).strip()
    rawPath = modelPath if modelPath else labelPath
    if not rawPath:
        return ''
    return _fileHash(_runtimePath(rawPath))


def _codeVersion() -> str:
    headPath = ROOT_DIR / '.git' / 'HEAD'
    if not headPath.exists():
        return ''
    headText = headPath.read_text().strip()
    if headText.startswith('ref: '):
        refPath = ROOT_DIR / '.git' / headText[5:].strip()
        if refPath.exists():
            return refPath.read_text().strip()[:12]
    return headText[:12]


def _appendRow(
    path: Path,
    fieldnames: list[str],
    row: dict[str, object],
) -> None:
    # Append a CSV row and create parent directories and header as needed.
    path.parent.mkdir(parents=True, exist_ok=True)
    needHeader = (not path.exists()) or path.stat().st_size == 0
    if not needHeader:
        with path.open(newline='') as fh:
            reader = csv.DictReader(fh)
            oldFields = list(reader.fieldnames or [])
            oldRows = list(reader)
        mergedFields = list(oldFields)
        for key in fieldnames:
            if key not in mergedFields:
                mergedFields.append(key)
        if mergedFields != oldFields:
            tmpPath = path.with_name(f"{path.name}.tmp")
            with tmpPath.open('w', newline='') as fh:
                writer = csv.DictWriter(fh, fieldnames=mergedFields)
                writer.writeheader()
                writer.writerows(oldRows)
            os.replace(tmpPath, path)
        fieldnames = mergedFields
    with path.open('a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if needHeader:
            writer.writeheader()
        writer.writerow(row)


@dataclass
class SessionLogger:
    # Persist session snapshots and trades for later performance review.
    snapshotPath: Path
    tradePath: Path
    eventPath: Path
    decisionPath: Path
    sessionId: str
    summaryLabel: str
    symbol: str
    interval: str
    mode: str
    quoteAsset: str
    baseAsset: str
    profileHash: str
    modelHash: str
    codeVersion: str
    lastSnapshotKey: tuple[object, ...] | None = None

    @classmethod
    def fromRuntime(
        cls,
        outPath: Path,
        runCfg: Any,
        symbolMeta,
        sessionId: str,
    ) -> 'SessionLogger':
        # Build logger from runtime config and symbol metadata.
        summaryLabel = str(runCfg.overrides.get('SUMMARY_LABEL', 'session'))
        return cls(
            snapshotPath=_csvPath(outPath),
            tradePath=_tradePath(outPath),
            eventPath=_eventPath(outPath),
            decisionPath=_decisionPath(outPath),
            sessionId=str(sessionId),
            summaryLabel=summaryLabel,
            symbol=str(runCfg.symbol),
            interval=str(runCfg.interval),
            mode=modeLabel(runCfg),
            quoteAsset=str(symbolMeta.quoteAsset),
            baseAsset=str(symbolMeta.baseAsset),
            profileHash=_fileHash(
                Path(getattr(runCfg, 'profilePath', LIVE_PROFILE_PATH))
            ),
            modelHash=_modelHash(runCfg.overrides),
            codeVersion=_codeVersion(),
        )

    def _baseRow(self) -> dict[str, object]:
        return {
            'logged_at_utc': _utcNowText(),
            'session_id': self.sessionId,
            'summary_label': self.summaryLabel,
            'mode': self.mode,
            'symbol': self.symbol,
            'interval': self.interval,
            'quote_asset': self.quoteAsset,
            'base_asset': self.baseAsset,
            'profile_hash': self.profileHash,
            'model_hash': self.modelHash,
            'code_version': self.codeVersion,
        }

    def logTrade(self, event: dict) -> None:
        # Append one normalized trade event to the trade CSV.
        row = {
            **self._baseRow(),
            'time_ms': int(event.get('timeMs', 0)),
            'trade_no': int(event.get('tradeNo', 0)),
            'flag': str(event.get('flag', '')),
            'side': str(event.get('side', '')),
            'qty': float(event.get('qty', 0.0)),
            'quote_qty': float(event.get('quoteQty', 0.0)),
            'price': float(event.get('price', 0.0)),
            'status': str(event.get('status', '')),
            'indicator': str(event.get('indicator', '')),
            'fee_text': str(event.get('feeText', '')),
            'order_id': str(event.get('orderId', '')),
            'signal_price': float(event.get('signalPrice', 0.0)),
            'quote_time_ms': int(event.get('quoteTimeMs', 0)),
            'quote_delay_ms': int(event.get('quoteDelayMs', 0)),
            'best_bid': float(event.get('bestBid', 0.0)),
            'best_ask': float(event.get('bestAsk', 0.0)),
            'bid_qty': float(event.get('bidQty', 0.0)),
            'ask_qty': float(event.get('askQty', 0.0)),
            'synthetic_fill_price': float(
                event.get('syntheticFillPrice', 0.0)
            ),
            'synthetic_adverse_bps': float(
                event.get('syntheticAdverseBps', 0.0)
            ),
            'adverse_bps': float(event.get('adverseBps', 0.0)),
            'signal_open_ms': int(event.get('signal_open_ms', 0)),
            'signal_close_ms': int(event.get('signal_close_ms', 0)),
            'daily_cluster': int(event.get('daily_cluster', -1)),
            'daily_posture': str(event.get('daily_posture', '')),
            'daily_force_lock': bool(event.get('daily_force_lock', False)),
            'macro_dyn_signed': float(event.get('macro_dyn_signed', 0.0)),
            'macro_dir': int(event.get('macro_dir', 0)),
            'macro_mom': int(event.get('macro_mom', 0)),
            'trend_code': int(event.get('trend_code', 0)),
            'trend_label': str(event.get('trend_label', '')),
            'grad1': float(event.get('grad1', 0.0)),
            'buy_z': float(event.get('buy_z', 0.0)),
            'sell_z': float(event.get('sell_z', 0.0)),
            'accepted_buy': bool(event.get('accepted_buy', False)),
            'accepted_sell': bool(event.get('accepted_sell', False)),
            'final_action': str(event.get('final_action', '')),
            'decision_reason': str(event.get('decision_reason', '')),
        }
        fieldnames = list(row.keys())
        _appendRow(self.tradePath, fieldnames, row)

    def logEvent(self, level: str, message: str) -> None:
        # Append one runtime event for post-run diagnosis.
        row = {
            **self._baseRow(),
            'level': str(level),
            'message': str(message),
        }
        fieldnames = list(row.keys())
        _appendRow(self.eventPath, fieldnames, row)

    def logDecision(self, decision: dict) -> None:
        # Append one closed-candle decision/audit row.
        row = {
            **self._baseRow(),
            'signal_open_ms': int(decision.get('signal_open_ms', 0)),
            'signal_close_ms': int(decision.get('signal_close_ms', 0)),
            'open': float(decision.get('open', 0.0)),
            'high': float(decision.get('high', 0.0)),
            'low': float(decision.get('low', 0.0)),
            'close': float(decision.get('close', 0.0)),
            'volume': float(decision.get('volume', 0.0)),
            'bar_index': int(decision.get('bar_index', 0)),
            'start_index': int(decision.get('start_index', 0)),
            'trading_enabled': bool(decision.get('trading_enabled', False)),
            'seeded': bool(decision.get('seeded', False)),
            'quote_total': float(decision.get('quote_total', 0.0)),
            'base_total': float(decision.get('base_total', 0.0)),
            'strategy_value': float(decision.get('strategy_value', 0.0)),
            'hodl_value': float(decision.get('hodl_value', 0.0)),
            'edge_value': float(decision.get('edge_value', 0.0)),
            'daily_cluster': int(decision.get('daily_cluster', -1)),
            'daily_posture': str(decision.get('daily_posture', '')),
            'daily_strong': bool(decision.get('daily_strong', False)),
            'daily_down': bool(decision.get('daily_down', False)),
            'daily_late': bool(decision.get('daily_late', False)),
            'daily_force_lock': bool(decision.get('daily_force_lock', False)),
            'macro_dyn_signed': float(
                decision.get('macro_dyn_signed', 0.0)
            ),
            'macro_dir': int(decision.get('macro_dir', 0)),
            'macro_mom': int(decision.get('macro_mom', 0)),
            'trend_code': int(decision.get('trend_code', 0)),
            'trend_label': str(decision.get('trend_label', '')),
            'grad1': float(decision.get('grad1', 0.0)),
            'buy_z': float(decision.get('buy_z', 0.0)),
            'sell_z': float(decision.get('sell_z', 0.0)),
            'buy_z_valid': bool(decision.get('buy_z_valid', False)),
            'sell_z_valid': bool(decision.get('sell_z_valid', False)),
            'allow_buy': bool(decision.get('allow_buy', False)),
            'allow_sell': bool(decision.get('allow_sell', False)),
            'accepted_buy': bool(decision.get('accepted_buy', False)),
            'accepted_sell': bool(decision.get('accepted_sell', False)),
            'flag_labels': str(decision.get('flag_labels', '')),
            'final_action': str(decision.get('final_action', '')),
            'decision_reason': str(decision.get('decision_reason', '')),
            'order_count': int(decision.get('order_count', 0)),
        }
        fieldnames = list(row.keys())
        _appendRow(self.decisionPath, fieldnames, row)

    def logSnapshot(self, snapshot: dict) -> None:
        # Append a dashboard snapshot when state changed materially.
        key = (
            snapshot.get('candleTimeMs'),
            snapshot.get('quoteTotal'),
            snapshot.get('baseTotal'),
            snapshot.get('tradeCount'),
            snapshot.get('tradingEnabled'),
            snapshot.get('seeded'),
            snapshot.get('currentDailyCluster'),
            snapshot.get('currentPosture'),
            snapshot.get('lastCommand'),
        )
        if key == self.lastSnapshotKey:
            return
        self.lastSnapshotKey = key

        row = {
            **self._baseRow(),
            'candle_time_ms': int(snapshot.get('candleTimeMs', 0)),
            'price': float(snapshot.get('price', 0.0)),
            'seeded': bool(snapshot.get('seeded', False)),
            'trading_enabled': bool(snapshot.get('tradingEnabled', False)),
            'quote_total': float(snapshot.get('quoteTotal', 0.0)),
            'base_total': float(snapshot.get('baseTotal', 0.0)),
            'strategy_value': float(snapshot.get('strategyValue', 0.0)),
            'hodl_value': float(snapshot.get('hodlValue', 0.0)),
            'edge_value': float(snapshot.get('edgeValue', 0.0)),
            'strategy_pct': float(snapshot.get('strategyPct', 0.0)),
            'hodl_pct': float(snapshot.get('hodlPct', 0.0)),
            'edge_pct': float(snapshot.get('edgePct', 0.0)),
            'trade_count': int(snapshot.get('tradeCount', 0)),
            'elapsed_text': str(snapshot.get('elapsedText', '')),
            'current_daily_cluster': int(
                snapshot.get('currentDailyCluster', -1)
            ),
            'current_posture': str(snapshot.get('currentPosture', '')),
            'last_command': str(snapshot.get('lastCommand', '')),
        }
        fieldnames = list(row.keys())
        _appendRow(self.snapshotPath, fieldnames, row)
