#!/usr/bin/env python3
# session_runtime.py - live session directory lifecycle.

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from repo_paths import LIVE_ACTIVE_PATH, LIVE_SESSION_DIR


SESSION_DIR = LIVE_SESSION_DIR
ACTIVE_PATH = LIVE_ACTIVE_PATH


def utcNowText() -> str:
    return datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sessionId() -> str:
    return datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def _modeText(runCfg: Any) -> str:
    if bool(runCfg.paperTrading):
        return 'PAPER'
    if bool(runCfg.dryRun):
        return 'LIVE-DRY'
    return 'LIVE'


def _readJson(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open() as fh:
        return json.load(fh)


def _writeJson(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmpPath = path.with_name(f"{path.name}.tmp")
    with tmpPath.open('w') as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write('\n')
    os.replace(tmpPath, path)


def _baseCsvName(outPath: Path) -> str:
    path = outPath if outPath.suffix else outPath.with_suffix('.csv')
    return path.name


def _sessionMeta(runCfg: Any, sid: str, path: Path) -> dict[str, Any]:
    return {
        'session_id': sid,
        'status': 'active',
        'started_at_utc': utcNowText(),
        'closed_at_utc': '',
        'symbol': str(runCfg.symbol),
        'interval': str(runCfg.interval),
        'mode': _modeText(runCfg),
        'session_dir': str(path),
    }


def _activeMeta(runCfg: Any) -> dict[str, Any] | None:
    meta = _readJson(ACTIVE_PATH)
    if meta is None:
        return None
    sessionPath = Path(str(meta.get('session_dir', '')))
    sessionMeta = _readJson(sessionPath / 'session.json')
    if sessionMeta is None:
        return None
    if str(sessionMeta.get('status', '')) != 'active':
        return None
    if str(sessionMeta.get('symbol', '')).upper() != str(runCfg.symbol):
        return None
    if str(sessionMeta.get('interval', '')) != str(runCfg.interval):
        return None
    if str(sessionMeta.get('mode', '')) != _modeText(runCfg):
        return None
    return sessionMeta


def openSession(runCfg: Any) -> tuple[Any, Path, str, bool]:
    # Resume one unclosed session, otherwise create a fresh session directory.
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    old = _activeMeta(runCfg)
    resumed = old is not None
    if resumed:
        sid = str(old['session_id'])
        sessionPath = Path(str(old['session_dir']))
    else:
        sid = sessionId()
        sessionPath = SESSION_DIR / sid
        suffix = 1
        while sessionPath.exists():
            sid = f"{sessionId()}-{suffix:02d}"
            sessionPath = SESSION_DIR / sid
            suffix += 1
        sessionPath.mkdir(parents=True, exist_ok=False)
        meta = _sessionMeta(runCfg, sid, sessionPath)
        _writeJson(sessionPath / 'session.json', meta)
        _writeJson(ACTIVE_PATH, meta)

    outPath = sessionPath / _baseCsvName(runCfg.outPath)
    nextCfg = replace(runCfg, outPath=outPath, sessionId=sid)
    return nextCfg, sessionPath, sid, resumed


def closeSession(sessionPath: Path) -> None:
    meta = _readJson(sessionPath / 'session.json')
    if meta is None:
        return
    meta['status'] = 'closed'
    meta['closed_at_utc'] = utcNowText()
    _writeJson(sessionPath / 'session.json', meta)
    _writeJson(ACTIVE_PATH, meta)
