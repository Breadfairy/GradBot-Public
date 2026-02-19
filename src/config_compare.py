#!/usr/bin/env python3
# config_compare.py – helpers for stable config comparisons.

from __future__ import annotations

from typing import Any

import numpy as np


def normalizeForCompare(value: Any, places: int = 6) -> Any:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        rounded = round(float(value), int(places))
        if rounded.is_integer():
            return int(rounded)
        return rounded
    if isinstance(value, list):
        return [normalizeForCompare(v, places) for v in value]
    if isinstance(value, tuple):
        return tuple(normalizeForCompare(v, places) for v in value)
    if isinstance(value, dict):
        return {k: normalizeForCompare(v, places) for k, v in value.items()}
    return value


def configsEqual(a: dict, b: dict, places: int = 6) -> bool:
    return normalizeForCompare(a, places) == normalizeForCompare(b, places)


__all__ = [
    "normalizeForCompare",
    "configsEqual",
]
