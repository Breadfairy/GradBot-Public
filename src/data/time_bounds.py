#!/usr/bin/env python3
# time_bounds.py - user-facing date parsing for anchored historical runs.

from __future__ import annotations

from datetime import datetime, timedelta, timezone


DATE_FMT = "%Y-%m-%d"


def parseAnchorDate(anchorDate: str) -> int:
    raw = str(anchorDate).strip()
    try:
        dt = datetime.strptime(raw, DATE_FMT).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise SystemExit(
            f"invalid anchor date '{raw}'; expected YYYY-MM-DD"
        ) from exc
    endOfDay = dt + timedelta(days=1) - timedelta(milliseconds=1)
    return int(endOfDay.timestamp() * 1000)


def resolveAnchorMs(
    anchorMs: int | None = None,
    anchorDate: str | None = None,
) -> int | None:
    if anchorMs is not None and anchorDate is not None:
        raise SystemExit(
            "use only one of anchorMs or anchorDate"
        )
    if anchorDate is not None:
        return parseAnchorDate(anchorDate)
    if anchorMs is not None:
        return int(anchorMs)
    return None

