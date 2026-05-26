#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pandas as pd

from clustering.cluster_config import ClusterConfig
from engine import core
from config import profile
from data.klines_io import loadCachedKlines
from runtime.diag import flagDiagnostics
from tune.trace import Trace


def loadFrame(
    ticker: str,
    interval: str,
    days: int,
    anchorMs: int | None,
) -> pd.DataFrame:
    rows = loadCachedKlines(
        ticker,
        interval,
        days,
        minCandles=None,
        anchorMs=anchorMs,
    )
    cols = [
        "openMs",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "closeMs",
        "quoteVolume",
        "trades",
        "takerBuyBaseVolume",
        "takerBuyQuoteVolume",
        "ignore",
    ]
    frame = pd.DataFrame(rows, columns=cols)
    for name in cols:
        frame[name] = pd.to_numeric(frame[name])
    return frame


def baseFrame(cfg: ClusterConfig) -> pd.DataFrame:
    frame = loadFrame(cfg.ticker, cfg.interval, cfg.days, cfg.anchorMs)
    frame.insert(0, "ticker", cfg.ticker)
    return frame


def featureColumns(frame: pd.DataFrame) -> list[str]:
    meta = {
        "ticker",
        "openMs",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "closeMs",
        "quoteVolume",
        "trades",
        "takerBuyBaseVolume",
        "takerBuyQuoteVolume",
        "ignore",
        "macroOpenMs",
        "cluster",
        "partition",
    }
    return [
        name for name in frame.columns
        if name not in meta and not name.startswith("fwdRet")
    ]


def addForwardReturns(
    frame: pd.DataFrame,
    forwardBars: list[int],
) -> pd.DataFrame:
    out = frame.copy()
    close = out["close"].astype(float)
    for bars in forwardBars:
        future = close.shift(-int(bars))
        out[f"fwdRet{int(bars)}h"] = ((future / close) - 1.0) * 100.0
    return out


def _safePct(num: pd.Series, den: pd.Series) -> pd.Series:
    denSafe = den.where(den != 0.0, np.nan)
    return (num / denSafe) * 100.0


def _retPct(values: pd.Series, bars: int) -> pd.Series:
    prev = values.shift(int(bars))
    return ((values / prev) - 1.0) * 100.0


def _rollingZ(values: pd.Series, window: int) -> pd.Series:
    roll = values.rolling(int(window), min_periods=int(window))
    mean = roll.mean()
    std = roll.std(ddof=0)
    return (values - mean) / std.where(std > 1e-12, np.nan)


def _priorRollingZ(values: pd.Series, window: int) -> pd.Series:
    prior = values.shift(1)
    roll = prior.rolling(int(window), min_periods=int(window))
    mean = roll.mean()
    std = roll.std(ddof=0)
    return (values - mean) / std.where(std > 1e-12, np.nan)


def _rangePos(value: pd.Series, low: pd.Series, high: pd.Series) -> pd.Series:
    span = high - low
    return (value - low) / span.where(span > 1e-12, np.nan)


def _rollingAge(values: pd.Series, window: int, high: bool) -> pd.Series:
    def _age(arr: np.ndarray) -> float:
        index = np.argmax(arr) if high else np.argmin(arr)
        return float(arr.shape[0] - 1 - int(index))

    return values.rolling(
        int(window),
        min_periods=int(window),
    ).apply(_age, raw=True)


def _ema(values: pd.Series, period: int) -> pd.Series:
    arr = core.emaLpf(values.to_numpy(dtype=float), int(period))
    return pd.Series(arr, index=values.index)


def _gradPct(values: pd.Series) -> pd.Series:
    prev = values.shift(1)
    return ((values - prev) / values.where(values != 0.0, np.nan)) * 100.0


def _rsi(close: pd.Series, window: int) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0.0)
    loss = (-diff).clip(lower=0.0)
    avgGain = gain.rolling(int(window), min_periods=int(window)).mean()
    avgLoss = loss.rolling(int(window), min_periods=int(window)).mean()
    rs = avgGain / avgLoss.where(avgLoss > 1e-12, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _trendCode(m1: pd.Series, m2: pd.Series, m3: pd.Series) -> pd.Series:
    code = core.trendCodes(
        m1.to_numpy(dtype=float),
        m2.to_numpy(dtype=float),
        m3.to_numpy(dtype=float),
    )
    return pd.Series(code.astype(float), index=m1.index)


def _barsPerDay(interval: str) -> float:
    return core.barsPerDayFromInterval(str(interval))


def _klinesFromFrame(frame: pd.DataFrame) -> list[list[object]]:
    cols = [
        "openMs",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "closeMs",
        "quoteVolume",
        "trades",
        "takerBuyBaseVolume",
        "takerBuyQuoteVolume",
        "ignore",
    ]
    return frame[cols].values.tolist()


def _engineOverrides(cfg: ClusterConfig) -> dict[str, object]:
    out = dict(cfg.engine)
    out["WALLET_SEED_QUOTE"] = 10000.0
    out["WALLET_FEE_RATE"] = 0.001
    out["QUOTE_TO_AUD_RATE"] = 1.0
    out["FINAL_PORTION_PCT"] = 0.5
    out["TAX_MODE"] = "income"
    return out


def _runtimeBlock(cfg: ClusterConfig, frame: pd.DataFrame) -> pd.DataFrame:
    periods = [cfg.periods.fast, cfg.periods.mid, cfg.periods.slow]
    overrides = profile.overrides(_engineOverrides(cfg))
    bt = Trace(
        cfg.ticker,
        _klinesFromFrame(frame),
        cfg.interval,
        periods,
        days=cfg.days,
        showCharts=False,
        showPrints=False,
        showSummary=False,
        overrides=overrides,
    )
    ctx = bt._ensureContext()
    ts = bt._timestamps()
    params, signals = bt._flagParamsAndSignals(ctx)
    macro = bt._macroDynArray(ts, overrides)
    macroDyn, macroDir, macroMom = (
        macro if macro is not None else (None, None, None)
    )
    diag = flagDiagnostics(
        ctx,
        signals,
        params,
        0,
        overrides,
        macroDyn,
        macroDir,
        macroMom,
    )
    n = int(frame.shape[0])
    acceptedBuy = np.zeros(n, dtype=float)
    acceptedSell = np.zeros(n, dtype=float)
    for i, side in diag["flags"]:
        if side == "BUY":
            acceptedBuy[int(i)] = 1.0
        elif side == "SELL":
            acceptedSell[int(i)] = 1.0
    return pd.DataFrame(
        {
            "allowBuy": np.asarray(diag["allowBuy"], dtype=float),
            "allowSell": np.asarray(diag["allowSell"], dtype=float),
            "buyDeltaPct": np.asarray(diag["buyDeltaPct"], dtype=float),
            "sellDeltaPct": np.asarray(diag["sellDeltaPct"], dtype=float),
            "buyReqPct": np.asarray(diag["buyReqPct"], dtype=float),
            "sellReqPct": np.asarray(diag["sellReqPct"], dtype=float),
            "macroDynSigned": np.asarray(diag["dyn"], dtype=float),
            "macroDynMag": np.abs(np.asarray(diag["dyn"], dtype=float)),
            "macroDir": np.asarray(diag["dirCode"], dtype=float),
            "macroMom": np.asarray(diag["momCode"], dtype=float),
            "macroBull": np.asarray(diag["macroBull"], dtype=float),
            "macroBear": np.asarray(diag["macroBear"], dtype=float),
            "macroRev": np.asarray(diag["macroRev"], dtype=float),
            "macroRoll": np.asarray(diag["macroRoll"], dtype=float),
            "acceptedBuy": acceptedBuy,
            "acceptedSell": acceptedSell,
        },
        index=frame.index,
    )


def engineFeatures(cfg: ClusterConfig) -> pd.DataFrame:
    frame = baseFrame(cfg)
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    openPx = frame["open"].astype(float)
    p1 = cfg.periods.fast
    p2 = cfg.periods.mid
    p3 = cfg.periods.slow
    ema1 = _ema(close, p1)
    ema2 = _ema(close, p2)
    ema3 = _ema(close, p3)
    trend = _trendCode(ema1, ema2, ema3)
    logRet = np.log(close / close.shift(1))
    window = cfg.windowBars
    barsDay = _barsPerDay(cfg.interval)
    buyWin = max(
        int(round(float(cfg.engine["GRAD1_BUY_WIN_DAYS"]) * barsDay)),
        1,
    )
    sellWin = max(
        int(round(float(cfg.engine["GRAD1_SELL_WIN_DAYS"]) * barsDay)),
        1,
    )
    highRoll = high.rolling(window, min_periods=window).max()
    lowRoll = low.rolling(window, min_periods=window).min()
    highRoll48 = high.rolling(window * 2, min_periods=window * 2).max()
    lowRoll48 = low.rolling(window * 2, min_periods=window * 2).min()
    g1p1 = _gradPct(ema1)
    g1p2 = _gradPct(ema2)
    g1p3 = _gradPct(ema3)
    runtime = _runtimeBlock(cfg, frame)

    out = frame.copy()
    out["emaGapFastPct"] = _safePct(close - ema1, close)
    out["emaGapMidPct"] = _safePct(close - ema2, close)
    out["emaGapSlowPct"] = _safePct(close - ema3, close)
    out["emaSpreadFastMidPct"] = _safePct(ema1 - ema2, close)
    out["emaSpreadMidSlowPct"] = _safePct(ema2 - ema3, close)
    out["emaSpreadFastSlowPct"] = _safePct(ema1 - ema3, close)
    out["gradFastPct"] = g1p1
    out["gradMidPct"] = g1p2
    out["gradSlowPct"] = g1p3
    out[f"gradFastBuyZ_{buyWin}"] = (
        -_rollingZ(g1p1, buyWin)
    ).clip(-10.0, 10.0)
    out[f"gradFastSellZ_{sellWin}"] = _rollingZ(
        g1p1,
        sellWin,
    ).clip(-10.0, 10.0)
    out["trendCode"] = trend
    out["trendBull"] = (trend == 1.0).astype(float)
    out["trendBear"] = (trend == -1.0).astype(float)
    out["trendHalfBull"] = (trend == 2.0).astype(float)
    out["trendHalfBear"] = (trend == -2.0).astype(float)
    out["distHigh24Pct"] = _safePct(highRoll - close, close)
    out["distLow24Pct"] = _safePct(close - lowRoll, close)
    out["range24Pct"] = _safePct(highRoll - lowRoll, close)
    out["distHigh48Pct"] = _safePct(highRoll48 - close, close)
    out["distLow48Pct"] = _safePct(close - lowRoll48, close)
    out["range48Pct"] = _safePct(highRoll48 - lowRoll48, close)
    out["rangePos24"] = _rangePos(close, lowRoll, highRoll)
    out["rangePos48"] = _rangePos(close, lowRoll48, highRoll48)
    out["ageHigh24"] = _rollingAge(high, window, True)
    out["ageLow24"] = _rollingAge(low, window, False)
    out["ageHigh48"] = _rollingAge(high, window * 2, True)
    out["ageLow48"] = _rollingAge(low, window * 2, False)
    out["realVol12"] = logRet.rolling(
        max(int(window / 2), 1),
        min_periods=max(int(window / 2), 1),
    ).std(ddof=0)
    out["realVol24"] = logRet.rolling(window, min_periods=window).std(ddof=0)
    out["realVol48"] = logRet.rolling(
        window * 2,
        min_periods=window * 2,
    ).std(ddof=0)
    out["ret1h"] = _retPct(close, 1)
    out["ret2h"] = _retPct(close, 2)
    out["ret3h"] = _retPct(close, 3)
    out["ret4h"] = _retPct(close, 4)
    out["ret6h"] = _retPct(close, 6)
    out["ret8h"] = _retPct(close, 8)
    out["ret12h"] = _retPct(close, 12)
    out["ret24h"] = _retPct(close, 24)
    out["ret48h"] = _retPct(close, 48)
    sumAbs = out["ret1h"].abs().rolling(window, min_periods=window).sum()
    out["trendEfficiency24"] = out["ret24h"].abs() / sumAbs.where(
        sumAbs > 1e-12,
        np.nan,
    )
    candleRange = high - low
    candleBody = close - openPx
    upperWick = high - pd.concat([openPx, close], axis=1).max(axis=1)
    lowerWick = pd.concat([openPx, close], axis=1).min(axis=1) - low
    out["bodyPct"] = _safePct(candleBody, close)
    out["bodyAbsPct"] = _safePct(candleBody.abs(), close)
    out["upperWickPct"] = _safePct(upperWick, close)
    out["lowerWickPct"] = _safePct(lowerWick, close)
    out["bodyAbsMean12"] = out["bodyAbsPct"].rolling(
        max(int(window / 2), 1),
        min_periods=max(int(window / 2), 1),
    ).mean()
    out["bodyAbsMean24"] = out["bodyAbsPct"].rolling(
        window,
        min_periods=window,
    ).mean()
    out["bodyAbsMean48"] = out["bodyAbsPct"].rolling(
        window * 2,
        min_periods=window * 2,
    ).mean()
    rangePct = _safePct(candleRange, close)
    out["rangeMean12"] = rangePct.rolling(
        max(int(window / 2), 1),
        min_periods=max(int(window / 2), 1),
    ).mean()
    out["rangeMean24"] = rangePct.rolling(window, min_periods=window).mean()
    out["rangeMean48"] = rangePct.rolling(
        window * 2,
        min_periods=window * 2,
    ).mean()
    logVolume = np.log1p(out["volume"].astype(float))
    logQuote = np.log1p(out["quoteVolume"].astype(float))
    logTrades = np.log1p(out["trades"].astype(float))
    takerBase = out["takerBuyBaseVolume"].astype(float)
    takerQuote = out["takerBuyQuoteVolume"].astype(float)
    volume = out["volume"].astype(float)
    quoteVolume = out["quoteVolume"].astype(float)
    out["takerBaseRatio"] = takerBase / volume.where(volume > 0.0, np.nan)
    out["takerQuoteRatio"] = takerQuote / quoteVolume.where(
        quoteVolume > 0.0,
        np.nan,
    )
    out["takerImbalance"] = (out["takerBaseRatio"] - 0.5) * 2.0
    out[f"logVolumeZ{cfg.explore.volumeZBars}"] = _priorRollingZ(
        logVolume,
        cfg.explore.volumeZBars,
    ).clip(-10.0, 10.0)
    out[f"logQuoteZ{cfg.explore.volumeZBars}"] = _priorRollingZ(
        logQuote,
        cfg.explore.volumeZBars,
    ).clip(-10.0, 10.0)
    out[f"logTradesZ{cfg.explore.volumeZBars}"] = _priorRollingZ(
        logTrades,
        cfg.explore.volumeZBars,
    ).clip(-10.0, 10.0)
    out[f"takerImbalanceZ{cfg.explore.volumeZBars}"] = _priorRollingZ(
        out["takerImbalance"],
        cfg.explore.volumeZBars,
    ).clip(-10.0, 10.0)
    for name in runtime.columns:
        out[name] = runtime[name]
    return out


def _maDerivatives(
    out: pd.DataFrame,
    close: pd.Series,
    period: int,
    name: str,
    clipVal: float,
) -> None:
    ma = _ema(close, period)
    out[f"{name}MaGapPct"] = _safePct(close - ma, close)
    d1 = (np.log(ma / ma.shift(1)) * 100.0).clip(-clipVal, clipVal)
    d2 = (d1 - d1.shift(1)).clip(-clipVal, clipVal)
    d3 = (d2 - d2.shift(1)).clip(-clipVal, clipVal)
    d4 = (d3 - d3.shift(1)).clip(-clipVal, clipVal)
    out[f"{name}MaD1"] = d1
    out[f"{name}MaD2"] = d2
    out[f"{name}MaD3"] = d3
    out[f"{name}MaD4"] = d4


def exploreFeatures(cfg: ClusterConfig) -> pd.DataFrame:
    out = baseFrame(cfg)
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    openPx = out["open"].astype(float)
    window = cfg.windowBars
    logRet = np.log(close / close.shift(1))

    for bars in cfg.explore.returnBars:
        out[f"ret{int(bars)}h"] = _retPct(close, int(bars))

    sumAbs = out["ret1h"].abs().rolling(window, min_periods=window).sum()
    out["trendEfficiency24"] = out["ret24h"].abs() / sumAbs.where(
        sumAbs > 1e-12,
        np.nan,
    )
    out["realVol24"] = logRet.rolling(window, min_periods=window).std(ddof=0)
    high24 = high.rolling(window, min_periods=window).max()
    low24 = low.rolling(window, min_periods=window).min()
    out["range24Pct"] = _safePct(high24 - low24, close)
    out["drawdownHigh24Pct"] = _safePct(high24 - close, close)
    out["distanceLow24Pct"] = _safePct(close - low24, close)

    candleRange = high - low
    candleBody = close - openPx
    upperWick = high - pd.concat([openPx, close], axis=1).max(axis=1)
    lowerWick = pd.concat([openPx, close], axis=1).min(axis=1) - low
    out["bodyPct"] = _safePct(candleBody, close)
    out["bodyAbsPct"] = _safePct(candleBody.abs(), close)
    out["upperWickPct"] = _safePct(upperWick, close)
    out["lowerWickPct"] = _safePct(lowerWick, close)
    out["bodyAbsMean24"] = out["bodyAbsPct"].rolling(
        window,
        min_periods=window,
    ).mean()
    rangePct = _safePct(candleRange, close)
    out["rangeMean24"] = rangePct.rolling(window, min_periods=window).mean()

    fast = _ema(close, cfg.periods.fast)
    slow = _ema(close, cfg.periods.slow)
    out["emaFastSlowGapPct"] = _safePct(fast - slow, close)
    slopeBars = cfg.explore.emaSlopeBars
    out[f"emaFastSlope{slopeBars}h"] = _retPct(fast, slopeBars)
    if cfg.explore.includeRsi:
        out[f"rsi{cfg.explore.rsiBars}"] = _rsi(close, cfg.explore.rsiBars)

    logVolume = np.log1p(out["volume"].astype(float))
    out[f"logVolumeZ{cfg.explore.volumeZBars}"] = _priorRollingZ(
        logVolume,
        cfg.explore.volumeZBars,
    ).clip(-10.0, 10.0)

    _maDerivatives(
        out,
        close,
        cfg.periods.fast,
        "fast",
        cfg.explore.derivativeClip,
    )
    _maDerivatives(
        out,
        close,
        cfg.periods.mid,
        "mid",
        cfg.explore.derivativeClip,
    )
    _maDerivatives(
        out,
        close,
        cfg.periods.slow,
        "slow",
        cfg.explore.derivativeClip,
    )
    return out
