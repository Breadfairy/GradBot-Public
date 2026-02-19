#!/usr/bin/env python3
import os
import sys
from typing import Any, List

import profile


def _pickScalar(v: Any) -> Any:
    return profile.scalarValue(v, None)


def probeCache() -> int:
    prof = sys.argv[1]
    root = sys.argv[2]
    cfg = profile.loadJson(prof)
    sys.path.insert(0, os.path.join(root, 'src'))
    from binance_io import loadCachedKlines  # type: ignore

    tickers = profile._requireTickers(cfg)
    ticker = str(tickers[0]).upper()
    intervals: List[str] = profile.intervalsFromConfig(cfg)
    _primer, _tuner, _holdout, days = profile.windowParts(cfg)
    p1 = int(_pickScalar(cfg['p1']))
    p2 = int(_pickScalar(cfg['p2']))
    p3 = int(_pickScalar(cfg['p3']))
    minCandles = max(p1, p2, p3) * 2 + 1
    for iv in intervals:
        kl = loadCachedKlines(ticker, iv, days, minCandles=minCandles)
        if kl:
            print(
                f"[tune] Using cached klines for {iv} "
                f"(no Binance call needed)."
            )
        else:
            print(f"[tune] No cached klines for {iv}; will fetch.")
    return 0


if __name__ == '__main__':
    raise SystemExit(probeCache())
