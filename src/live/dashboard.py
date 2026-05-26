#!/usr/bin/env python3
# dashboard.py – runtime config, dashboard rendering, and stdin commands.

from __future__ import annotations

import asyncio
import csv
import curses
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import re
import textwrap
from typing import Dict, List

from engine.core import barsPerDayFromInterval
from live.execution import (
    cancelOpenOrder,
    executeMarketBuy,
    executeMarketSell,
    modeLabel,
    walletFreeBalance,
    walletTotalBalance,
)
from live import live_config
from live.session_logger import SessionLogger
from repo_paths import LIVE_PROFILE_PATH, liveOutputPath, liveProfilePath


PERTH_TZ = timezone(timedelta(hours=8))
BALANCE_REFRESH_SECONDS = 5.0
UI_WIDTH = 80
STREAM_STALE_SECONDS = 90.0
QUOTE_STALE_SECONDS = 30.0
CANDLE_STALE_MULT = 2.5
OPEN_ORDER_DONE_MS = 60 * 1000
TIP_SECONDS = 10
TIPS = [
    "enter 'buy 55%' or 'sell $400'",
    "enter 'cls ord 4' to cancel open order 4",
]
STAGE_QUIET = False
STAGE_MESSAGES: List[str] = []


def uiWrap(text: str) -> list[str]:
    # Wrap runtime lines so terminal prints stay inside the UI width.
    return textwrap.wrap(
        str(text),
        width=UI_WIDTH,
        break_long_words=False,
        break_on_hyphens=False,
    ) or ['']


@dataclass(frozen=True)
class RuntimeCfg:
    # Hold runtime inputs resolved from profile.
    symbol: str
    interval: str
    macroInterval: str
    postureInterval: str
    periods: List[int]
    primerDays: int
    totalDays: int
    overrides: Dict[str, object]
    paperTrading: bool
    dryRun: bool
    outPath: Path
    profilePath: Path
    sessionId: str = ''
    warmStartWeeks: int = 0
    warmStartAssetPct: float = 0.0


@dataclass
class DashboardState:
    # Hold live dashboard state for strategy and benchmark tracking.
    startUtc: datetime
    seedQuote: float
    quoteTotal: float
    baseTotal: float
    lastBalanceSync: float
    tradeCount: int
    seeded: bool = False
    hodlQty: float | None = None
    hodlEntryPrice: float | None = None
    lastTrade: dict | None = None
    lastOrder: dict | None = None
    tradingEnabled: bool = True
    lastCommand: str = 'none'
    lastStreamMs: int = 0
    lastClosedMs: int = 0
    lastClosedPrice: float = 0.0
    lastQuoteMs: int = 0
    currentDailyCluster: int = -1
    currentPosture: str = 'unknown'
    eventHistory: List[str] = field(default_factory=list)
    openOrders: List[dict] = field(default_factory=list)
    nextOpenOrderNo: int = 1
    viewMode: str = 'main'
    gateLines: List[str] = field(default_factory=list)
    volume24hQuote: float = 0.0
    volume24hMs: int = 0
    logger: SessionLogger | None = None


def printStage(message: str) -> None:
    # Print a startup/runtime stage line with Perth timestamp.
    global STAGE_MESSAGES
    nowTs = datetime.now(tz=PERTH_TZ)
    line = f"[{nowTs:%Y-%m-%d %H:%M:%S} GMT+8] {message}"
    wrapped = uiWrap(line)
    if STAGE_QUIET:
        STAGE_MESSAGES = wrapped + STAGE_MESSAGES
        STAGE_MESSAGES = STAGE_MESSAGES[:20]
        return
    for chunk in wrapped:
        print(chunk, flush=True)


def setStageQuiet(enabled: bool) -> None:
    # Route runtime stage lines into the curses UI instead of stdout.
    global STAGE_QUIET
    STAGE_QUIET = bool(enabled)


def commandMenuText() -> str:
    # Return supported stdin commands.
    return (
        '<quit> <seed> <pause> <resume> <buy> <sell> '
        '<cls ord N> <gates> <back>'
    )


def parseWarmStartArgs(args: list[str]) -> tuple[int, float]:
    # Parse optional paper warm-start args: ./run_live 1w 55.
    if not args:
        return 0, 0.0
    if len(args) != 2:
        raise ValueError('usage: ./run_live [1w|2w|3w|4w asset_pct]')

    period = str(args[0]).strip().lower()
    pctText = str(args[1]).strip()
    if period not in ('1w', '2w', '3w', '4w'):
        raise ValueError('warm-start period must be 1w, 2w, 3w, or 4w')
    if not re.fullmatch(r'\d+(?:\.\d+)?', pctText):
        raise ValueError('warm-start asset percent must be 0..100')

    pct = float(pctText)
    if pct < 0.0 or pct > 100.0:
        raise ValueError('warm-start asset percent must be 0..100')
    return int(period[:-1]), pct


def applyWarmStartArgs(runCfg: RuntimeCfg, args: list[str]) -> RuntimeCfg:
    # Return runtime config with optional paper warm-start settings applied.
    weeks, pct = parseWarmStartArgs(args)
    if weeks <= 0:
        return runCfg
    if not bool(runCfg.paperTrading):
        raise ValueError('warm-start args are only supported in PAPER mode')
    return replace(
        runCfg,
        totalDays=int(runCfg.totalDays) + (int(weeks) * 7),
        warmStartWeeks=int(weeks),
        warmStartAssetPct=float(pct),
    )


def utcNowMs() -> int:
    # Return current UTC epoch milliseconds.
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000.0)


def profilePath() -> Path:
    # Resolve the live profile path, allowing an explicit environment override.
    return liveProfilePath(os.environ.get('LIVE_PROFILE', LIVE_PROFILE_PATH))


def loadRuntime() -> tuple[RuntimeCfg, Path]:
    # Load runtime config and normalized overrides from profile.
    pPath = profilePath()
    cfg = live_config.loadJson(pPath)
    ticker = str(cfg['tickers'][0]).strip().upper()
    interval = live_config.intervalsFromConfig(cfg)[0]
    paperTrading = live_config.boolValue(cfg.get('PAPER_TRADING', False))
    dryRun = live_config.boolValue(cfg.get('LIVE_DRY_RUN', False))
    p1 = int(live_config.scalarValue(cfg['p1'], 0))
    p2 = int(live_config.scalarValue(cfg['p2'], 0))
    p3 = int(live_config.scalarValue(cfg['p3'], 0))
    primer = int(live_config.scalarValue(cfg['primer_days'], 0))
    days = int(live_config.scalarValue(cfg['history_days'], 0))

    baseFields = {
        '_comment',
        'tickers',
        'intervals',
        'p1',
        'p2',
        'p3',
        'primer_days',
        'history_days',
        'out',
        'config_path',
        'PAPER_TRADING',
        'LIVE_DRY_RUN',
    }
    overridesRaw = {
        key: val for key, val in cfg.items() if key not in baseFields
    }
    overrides = live_config.overridesFromDict(overridesRaw)
    macroInterval = str(overrides['MACRO_INTERVAL']).strip()
    postureInterval = str(
        overrides.get('DAILY_CLUSTER_INTERVAL', macroInterval)
    ).strip()

    runCfg = RuntimeCfg(
        symbol=ticker,
        interval=interval,
        macroInterval=macroInterval,
        postureInterval=postureInterval,
        periods=[p1, p2, p3],
        primerDays=primer,
        totalDays=days,
        overrides=overrides,
        paperTrading=paperTrading,
        dryRun=(dryRun and not paperTrading),
        outPath=liveOutputPath(str(cfg.get('out', 'results.csv'))).resolve(),
        profilePath=pPath,
    )
    return runCfg, live_config.configPath(cfg, pPath)


def maxBars(days: int, interval: str) -> int:
    # Convert profile day window into row count cap.
    bpd = barsPerDayFromInterval(interval)
    return int(round(float(days) * float(bpd))) + 8


def trimRows(rows: list, keepRows: int) -> list:
    # Trim kline list to trailing keepRows entries.
    if len(rows) <= keepRows:
        return rows
    return rows[-keepRows:]


def candleFromRow(row: list, interval: str) -> dict:
    # Convert a historical kline row into a candle-like dict.
    return {
        't': int(row[0]),
        'T': int(row[6]),
        'c': str(row[4]),
        'i': str(interval),
        'x': True,
    }


def parseAmountSpec(spec: str) -> tuple[str, float] | None:
    # Parse command amount spec as quote dollars or percent.
    txt = str(spec).strip()
    numberPattern = r'\d+(?:\.\d+)?'
    if not txt:
        return None
    if txt.endswith('%'):
        raw = txt[:-1].strip()
        if re.fullmatch(numberPattern, raw):
            return 'pct', float(raw)
        return None
    if txt.startswith('$'):
        raw = txt[1:].strip()
        if re.fullmatch(numberPattern, raw):
            return 'quote', float(raw)
        return None
    if re.fullmatch(numberPattern, txt):
        return 'quote', float(txt)
    return None


def tradeAmountText(event: dict, baseAsset: str) -> str:
    # Render trade amount text for history lines.
    quoteQty = float(event.get('quoteQty', 0.0))
    qty = float(event.get('qty', 0.0))
    return f"{qty:.2f} {baseAsset} (${quoteQty:.2f})"


def quoteVolumeText(value: float) -> str:
    # Format 24h quote volume with stable M/B units for the top line.
    val = max(float(value), 0.0)
    if val >= 1_000_000_000.0:
        return f"${val / 1_000_000_000.0:.2f}B"
    return f"${val / 1_000_000.0:.2f}M"


def timePerth(msVal: int) -> datetime:
    # Convert millisecond epoch into Perth timezone datetime.
    return datetime.fromtimestamp(int(msVal) / 1000.0, tz=PERTH_TZ)


def clipHistoryLine(line: str) -> str:
    # Keep visible history rows inside the fixed dashboard width.
    text = str(line)
    if len(text) <= UI_WIDTH:
        return text
    return f"{text[:UI_WIDTH - 3]}..."


def compactFeeText(raw: object) -> str:
    text = str(raw or '').strip()
    parts = text.split(maxsplit=1)
    if not parts:
        return 'n/a'
    try:
        amount = float(parts[0])
    except ValueError:
        return text
    asset = f" {parts[1]}" if len(parts) > 1 else ''
    return f"{amount:.2f}{asset}"


def historyLine(event: dict, baseAsset: str) -> str:
    # Build one compact trade history line.
    ts = timePerth(int(event['timeMs']))
    tradeNo = int(event['tradeNo'])
    side = str(event.get('side', event.get('flag', ''))).upper()
    quoteQty = float(event.get('quoteQty', 0.0))
    qty = float(event.get('qty', 0.0))
    price = float(event.get('price', 0.0))
    slip = ''
    if 'adverseBps' in event:
        slip = f" bps={float(event.get('adverseBps', 0.0)):+.2f}"
    return clipHistoryLine(
        f"{ts:%H:%M:%S} - {ts:%Y-%m-%d} TRADE: #{tradeNo} "
        f"{side} {qty:.2f} {baseAsset} at ${price:.2f} "
        f"(${quoteQty:.2f}){slip}"
    )


def isFilledStatus(status: object) -> bool:
    return str(status).upper() in ('FILLED', 'DRY_RUN')


def isDoneStatus(status: object) -> bool:
    return str(status).upper() in ('FILLED', 'DRY_RUN', 'CANCELED')


def isFilledEvent(event: dict) -> bool:
    return isFilledStatus(event.get('status', ''))


def openOrderLine(order: dict) -> str:
    no = int(order.get('openNo', 0))
    side = str(order.get('side', order.get('flag', ''))).upper()
    status = str(order.get('status', ''))
    baseAsset = str(order.get('baseAsset', 'BASE'))
    amount = tradeAmountText(order, baseAsset)
    return clipHistoryLine(f"ORD {no}: {side} {amount} {status}")


def trackOpenOrder(dash: DashboardState, event: dict) -> None:
    if isFilledEvent(event):
        return
    order = dict(event)
    order['openNo'] = int(dash.nextOpenOrderNo)
    order['openedAtMs'] = utcNowMs()
    order['filledAtMs'] = 0
    dash.nextOpenOrderNo += 1
    dash.openOrders.append(order)


def updateOpenOrder(dash: DashboardState, orderId: str, status: str) -> None:
    nowMs = utcNowMs()
    for order in dash.openOrders:
        if str(order.get('orderId', '')) == str(orderId):
            order['status'] = str(status)
            if isDoneStatus(status) and int(order.get('filledAtMs', 0)) <= 0:
                order['filledAtMs'] = nowMs


def updateOpenOrderEvent(order: dict, raw: dict) -> None:
    order['status'] = str(raw.get('status', order.get('status', '')))
    order['qty'] = float(raw.get('executedQty', order.get('qty', 0.0)) or 0.0)
    order['quoteQty'] = float(
        raw.get('cummulativeQuoteQty', order.get('quoteQty', 0.0)) or 0.0
    )
    if isDoneStatus(order.get('status', '')):
        order['filledAtMs'] = int(order.get('filledAtMs', 0) or utcNowMs())


async def refreshOpenOrders(
    client,
    symbolMeta,
    runCfg: RuntimeCfg,
    dash: DashboardState,
) -> None:
    pruneOpenOrders(dash)
    if modeLabel(runCfg) != 'LIVE':
        return
    for order in list(dash.openOrders):
        if isDoneStatus(order.get('status', '')):
            continue
        raw = await client.get_order(
            symbol=runCfg.symbol,
            orderId=str(order.get('orderId', '')),
        )
        wasDone = isDoneStatus(order.get('status', ''))
        updateOpenOrderEvent(order, raw)
        nowDone = isDoneStatus(order.get('status', ''))
        if nowDone and not wasDone and isFilledStatus(order.get('status', '')):
            dash.tradeCount += 1
            order['tradeNo'] = dash.tradeCount
            dash.lastTrade = order
            dash.lastOrder = order
            line = historyLine(order, symbolMeta.baseAsset)
            dash.eventHistory.insert(0, line)
            dash.eventHistory = dash.eventHistory[:80]
            if dash.logger is not None:
                dash.logger.logTrade(order)


def pruneOpenOrders(dash: DashboardState) -> None:
    nowMs = utcNowMs()
    kept = []
    for order in dash.openOrders:
        filledAt = int(order.get('filledAtMs', 0) or 0)
        if filledAt > 0 and nowMs - filledAt >= OPEN_ORDER_DONE_MS:
            continue
        kept.append(order)
    dash.openOrders = kept


def clearOpenOrder(dash: DashboardState, orderNo: int) -> dict | None:
    for idx, order in enumerate(dash.openOrders):
        if int(order.get('openNo', 0)) == int(orderNo):
            return dash.openOrders.pop(idx)
    return None


def findOpenOrder(dash: DashboardState, orderNo: int) -> dict | None:
    for order in dash.openOrders:
        if int(order.get('openNo', 0)) == int(orderNo):
            return order
    return None


def tipText() -> str:
    idx = (utcNowMs() // (TIP_SECONDS * 1000)) % len(TIPS)
    return f"TIPS             : {TIPS[int(idx)]}"


def _tradeEventFromCsv(row: dict[str, str]) -> dict[str, object]:
    return {
        'timeMs': int(float(row.get('time_ms', 0) or 0)),
        'tradeNo': int(float(row.get('trade_no', 0) or 0)),
        'flag': str(row.get('flag', '')),
        'side': str(row.get('side', '')),
        'qty': float(row.get('qty', 0.0) or 0.0),
        'quoteQty': float(row.get('quote_qty', 0.0) or 0.0),
        'price': float(row.get('price', 0.0) or 0.0),
        'status': str(row.get('status', '')),
        'feeText': str(row.get('fee_text', '')),
        'adverseBps': float(row.get('adverse_bps', 0.0) or 0.0),
    }


def _historyItemsFromTrades(
    path: Path,
    baseAsset: str,
) -> list[tuple[int, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline='') as fh:
        rows = list(csv.DictReader(fh))
    return [
        (
            int(float(row.get('time_ms', 0) or 0)),
            historyLine(_tradeEventFromCsv(row), baseAsset),
        )
        for row in rows
    ]


def loadRecentHistory(
    dash: DashboardState,
    baseAsset: str,
    limit: int,
    cutoffMs: int = 0,
) -> None:
    if dash.logger is None:
        return
    items = []
    items += _historyItemsFromTrades(dash.logger.tradePath, baseAsset)
    items = sorted(items, key=lambda item: item[0])
    if int(cutoffMs) > 0:
        items = [item for item in items if int(item[0]) <= int(cutoffMs)]
    seen = set()
    unique = []
    for msVal, line in items:
        key = (int(msVal) // 1000, str(line))
        if key not in seen:
            unique.append((msVal, line))
            seen.add(key)
    dash.eventHistory = [line for _ms, line in unique[-int(limit):]][::-1]


def recordTradeEvent(dash: DashboardState, event: dict, baseAsset: str) -> None:
    # Store trade event, update counters, and keep newest trades first.
    event['baseAsset'] = baseAsset
    dash.lastOrder = event
    if isFilledEvent(event):
        dash.tradeCount += 1
        event['tradeNo'] = dash.tradeCount
        dash.lastTrade = event
        dash.eventHistory.insert(0, historyLine(event, baseAsset))
        if len(dash.eventHistory) > 80:
            dash.eventHistory = dash.eventHistory[:80]
    else:
        event['tradeNo'] = 0
    trackOpenOrder(dash, event)
    if dash.logger is not None:
        dash.logger.logTrade(event)


def recordHistory(dash: DashboardState, line: str) -> None:
    # Store one command/runtime event in the unified dashboard history.
    if dash.logger is not None:
        dash.logger.logEvent('info', line)


async def refreshPrimaryPrice(
    client,
    symbol: str,
    candle: dict,
    dash: DashboardState | None = None,
) -> None:
    # Refresh displayed price from live symbol ticker endpoint.
    ticker = await client.get_symbol_ticker(symbol=symbol)
    nowMs = utcNowMs()
    candle['c'] = str(ticker['price'])
    if dash is None:
        return
    if nowMs - int(dash.volume24hMs) < 60 * 1000:
        return
    stats = await client.get_ticker(symbol=symbol)
    dash.volume24hQuote = float(stats.get('quoteVolume', 0.0) or 0.0)
    dash.volume24hMs = nowMs


def pctChange(startVal: float, nowVal: float) -> float:
    # Return percentage change from start to now value.
    if startVal <= 0.0:
        return 0.0
    return ((nowVal / startVal) - 1.0) * 100.0


def ageSeconds(nowMs: int, thenMs: int) -> float:
    # Return positive elapsed seconds between two epoch millisecond values.
    if thenMs <= 0:
        return 10**9
    return max((int(nowMs) - int(thenMs)) / 1000.0, 0.0)


def ageText(seconds: float) -> str:
    # Format heartbeat ages compactly for the 80-column dashboard.
    val = float(seconds)
    if val >= 3600.0:
        return f"{val / 3600.0:.1f}h"
    if val >= 60.0:
        return f"{val / 60.0:.1f}m"
    return f"{val:.0f}s"


def statusText(age: float, limit: float) -> str:
    # Return OK/STALE text for one runtime heartbeat.
    label = 'OK' if float(age) <= float(limit) else 'STALE'
    return f"{label} {ageText(age)}"


def intervalSeconds(interval: str) -> float:
    # Convert an interval string to seconds using the existing bars/day parser.
    bpd = max(float(barsPerDayFromInterval(interval)), 1e-9)
    return 86400.0 / bpd


def elapsedText(startUtc: datetime) -> str:
    # Format runtime age as a stable dashboard string.
    elapsed = datetime.now(tz=timezone.utc) - startUtc
    totalSeconds = max(int(elapsed.total_seconds()), 0)
    days = totalSeconds // 86400
    hours = (totalSeconds % 86400) // 3600
    minutes = (totalSeconds % 3600) // 60
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def dashboardSnapshot(
    runCfg: RuntimeCfg,
    symbolMeta,
    dash: DashboardState,
    candle: dict,
) -> dict:
    # Build one dashboard snapshot dict shared by plain and curses renderers.
    nowMs = utcNowMs()
    closeMs = int(dash.lastClosedMs or candle.get('T', candle['t']))
    ts = timePerth(closeMs)
    lastPrice = float(candle['c'])
    lastClosePrice = float(dash.lastClosedPrice or lastPrice)
    delta1hPct = pctChange(lastClosePrice, lastPrice)
    strategyVal = dash.quoteTotal + (dash.baseTotal * lastPrice)
    candleLimit = intervalSeconds(runCfg.interval) * CANDLE_STALE_MULT
    streamAge = ageSeconds(nowMs, dash.lastStreamMs)
    candleAge = ageSeconds(nowMs, dash.lastClosedMs)
    quoteAge = ageSeconds(nowMs, dash.lastQuoteMs)

    if dash.seeded and dash.hodlQty is not None and dash.seedQuote > 0.0:
        hodlVal = float(dash.hodlQty) * lastPrice
        edgeVal = strategyVal - hodlVal
        strategyPct = pctChange(dash.seedQuote, strategyVal)
        hodlPct = pctChange(dash.seedQuote, hodlVal)
        edgePct = pctChange(hodlVal, strategyVal)
    else:
        hodlVal = 0.0
        edgeVal = 0.0
        strategyPct = 0.0
        hodlPct = 0.0
        edgePct = 0.0

    _ = runCfg
    _ = symbolMeta
    return {
        'candleTimeMs': closeMs,
        'time': ts,
        'price': lastPrice,
        'delta1hPct': delta1hPct,
        'volume24hQuote': float(dash.volume24hQuote),
        'seeded': dash.seeded,
        'tradingEnabled': dash.tradingEnabled,
        'quoteTotal': dash.quoteTotal,
        'baseTotal': dash.baseTotal,
        'strategyValue': strategyVal,
        'hodlValue': hodlVal,
        'edgeValue': edgeVal,
        'strategyPct': strategyPct,
        'hodlPct': hodlPct,
        'edgePct': edgePct,
        'tradeCount': dash.tradeCount,
        'elapsedText': elapsedText(dash.startUtc),
        'currentDailyCluster': dash.currentDailyCluster,
        'currentPosture': dash.currentPosture,
        'lastCommand': dash.lastCommand,
        'streamStatus': statusText(streamAge, STREAM_STALE_SECONDS),
        'candleStatus': statusText(candleAge, candleLimit),
        'quoteStatus': statusText(quoteAge, QUOTE_STALE_SECONDS),
    }


def dashboardLines(
    runCfg: RuntimeCfg,
    symbolMeta,
    dash: DashboardState,
    candle: dict,
) -> tuple[list[str], dict]:
    # Build display lines for dashboard renderers.
    snap = dashboardSnapshot(runCfg, symbolMeta, dash, candle)
    nowTs = datetime.now(tz=PERTH_TZ)
    ts = snap['time']
    divider = '-' * UI_WIDTH
    lines = [
        (
            f"{runCfg.symbol:<10}: ${float(snap['price']):.2f} "
            f"({float(snap['delta1hPct']):+.2f}% 1h), "
            f"vol: {quoteVolumeText(float(snap['volume24hQuote']))} (24h)"
        ),
        f"time       : {nowTs:%Y-%m-%d - %H:%M:%S} GMT+8",
        (
            f"last close : {ts:%Y-%m-%d %H:%M:%S} GMT+8 "
            f"({runCfg.interval})"
        ),
        (
            f"heartbeat  : stream {snap['streamStatus']} | "
            f"candle {snap['candleStatus']} | quote {snap['quoteStatus']}"
        ),
        divider,
    ]
    if dash.seeded:
        lines += [
            (
                f"strategy value   : ${float(snap['strategyValue']):.2f} "
                f"({float(snap['strategyPct']):+.2f}%)"
            ),
            (
                f"hodl value       : ${float(snap['hodlValue']):.2f} "
                f"({float(snap['hodlPct']):+.2f}%)"
            ),
            (
                f"strategy vs hodl : ${float(snap['edgeValue']):+.2f} "
                f"({float(snap['edgePct']):+.2f}%)"
            ),
        ]
    else:
        lines += [
            "strategy value   : (n/a)",
            "hodl value       : (n/a)",
            "strategy vs hodl : (n/a)",
        ]
    lines += [
        divider,
        (
            f"balances         : {symbolMeta.quoteAsset} "
            f"{dash.quoteTotal:.2f}"
        ),
        f"                 : {symbolMeta.baseAsset} {dash.baseTotal:.2f}",
        f"execution mode   : {modeLabel(runCfg)}",
        f"auto trading     : {'ON' if dash.tradingEnabled else 'PAUSED'}",
        f"current posture  : {dash.currentPosture}",
        f"trade count      : {dash.tradeCount}",
        f"time elapsed     : {snap['elapsedText']}",
        tipText(),
    ]
    if not dash.seeded:
        lines.append(
            'HELLO!           : allocate funds, then `seed` the benchmark'
        )

    return lines, snap


def recordDashboardSnapshot(dash: DashboardState, snapshot: dict) -> None:
    if dash.logger is not None:
        dash.logger.logSnapshot(snapshot)


class SplitDashboardUi:
    # Curses split-pane dashboard with stable command input.
    def __init__(self) -> None:
        self.screen = None
        self.inputBuffer = ''
        self.inputDirty = False
        self.enabled = False
        self.commandRow = 0

    def start(self) -> None:
        self.screen = curses.initscr()
        self.enabled = True
        curses.noecho()
        curses.cbreak()
        if curses.has_colors():
            try:
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_WHITE, -1)
            except curses.error:
                pass
        self.screen.keypad(True)
        self.screen.nodelay(True)
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        setStageQuiet(True)

    def stop(self) -> None:
        if not self.enabled or self.screen is None:
            return
        try:
            setStageQuiet(False)
            self.screen.nodelay(False)
            self.screen.keypad(False)
            curses.nocbreak()
            curses.echo()
            curses.endwin()
        finally:
            self.enabled = False
            self.screen = None

    def pollCommands(self) -> list[str]:
        if not self.enabled or self.screen is None:
            return []
        out: list[str] = []
        while True:
            try:
                ch = self.screen.getch()
            except curses.error:
                break
            if ch == -1:
                break
            if ch in (10, 13, curses.KEY_ENTER):
                cmd = self.inputBuffer.strip().lower()
                self.inputBuffer = ''
                self.inputDirty = True
                if cmd:
                    out.append(cmd)
            elif ch in (3,):
                self.inputBuffer = ''
                self.inputDirty = True
                out.append('quit')
            elif ch in (27, 21):
                self.inputBuffer = ''
                self.inputDirty = True
            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                self.inputBuffer = self.inputBuffer[:-1]
                self.inputDirty = True
            elif 32 <= ch <= 126:
                self.inputBuffer += chr(ch)
                self.inputDirty = True
        return out

    def consumeInputDirty(self) -> bool:
        dirty = bool(self.inputDirty)
        self.inputDirty = False
        return dirty

    def _add(
        self,
        y: int,
        x: int,
        text: str,
        width: int,
        attr: int = 0,
    ) -> None:
        if self.screen is None or y < 0 or width <= 0:
            return
        maxY, _maxX = self.screen.getmaxyx()
        if y >= maxY:
            return
        try:
            self.screen.addstr(y, x, str(text)[:max(width, 0)], attr)
        except curses.error:
            pass

    def _titleAttr(self) -> int:
        attr = curses.A_BOLD
        if curses.has_colors():
            attr |= curses.color_pair(1)
        return attr

    def _commandRows(self, width: int) -> list[str]:
        label = 'COMMANDS |'
        prompt = f"{label} {commandMenuText()} : {self.inputBuffer}"
        if len(prompt) <= width:
            return [prompt]

        row1 = label
        row2 = ' ' * len(label)
        tokens = commandMenuText().split()
        for token in tokens:
            candidate = f"{row1} {token}"
            if len(candidate) <= width:
                row1 = candidate
            else:
                row2 = f"{row2} {token}".rstrip()
        row2 = f"{row2} : {self.inputBuffer}"
        return [row1, row2[:width]]

    def _drawGatePane(
        self,
        y: int,
        x: int,
        width: int,
        bottom: int,
        dash: DashboardState,
    ) -> None:
        row = y
        lines = dash.gateLines or [
            'GATES - LIVE PREVIEW',
            'waiting for live candle',
        ]
        for line in lines:
            if row >= bottom:
                break
            attr = self._titleAttr() if str(line).startswith('GATES') else 0
            self._add(row, x, line, width, attr)
            row += 1

    def _drawCommandPane(
        self,
        y: int,
        x: int,
        width: int,
        height: int,
        dash: DashboardState,
    ) -> None:
        bottom = y + height
        row = y
        commandRows = self._commandRows(width)
        pruneOpenOrders(dash)
        for commandRow in commandRows:
            if row >= bottom:
                break
            self._add(row, x, commandRow, width, self._titleAttr())
            row += 1
        self._add(row, x, '-' * width, width)
        row += 1
        if dash.viewMode == 'gates':
            self._drawGatePane(row, x, width, bottom, dash)
            return
        self._add(row, x, 'OPEN ORDERS', width, self._titleAttr())
        row += 1
        if dash.openOrders:
            for order in dash.openOrders:
                if row >= bottom:
                    break
                self._add(row, x, openOrderLine(order), width)
                row += 1
        else:
            self._add(row, x, 'none', width)
            row += 1
        if row < bottom:
            self._add(row, x, '-' * width, width)
            row += 1
        if row >= bottom:
            return
        self._add(row, x, 'HISTORY', width, self._titleAttr())
        row += 1
        for line in dash.eventHistory[:max(0, bottom - row)]:
            self._add(row, x, line, width)
            row += 1
            if row >= bottom:
                break

    def _drawDashboardPane(
        self,
        y: int,
        x: int,
        width: int,
        height: int,
        lines: list[str],
    ) -> None:
        for row, line in enumerate(lines[:height]):
            attr = self._titleAttr() if row == 0 else 0
            self._add(y + row, x, line, width, attr)

    def _layout(self) -> tuple[int, int, int, int, int]:
        if self.screen is None:
            return 0, 0, 0, 0, 0
        height, width = self.screen.getmaxyx()
        contentWidth = max(min(width, UI_WIDTH), 1)
        return height, width, contentWidth, 0, self.commandRow

    def _moveCursor(self, contentWidth: int, commandTop: int) -> None:
        if self.screen is None:
            return
        rows = self._commandRows(contentWidth)
        cursorRow = commandTop + len(rows) - 1
        cursorX = min(len(rows[-1]), contentWidth - 1)
        try:
            self.screen.move(cursorRow, cursorX)
        except curses.error:
            pass

    def renderCommandPane(self, dash: DashboardState) -> None:
        if not self.enabled or self.screen is None:
            return
        height, _width, contentWidth, _unused, commandTop = self._layout()
        commandHeight = max(height - commandTop, 1)
        for row in range(commandTop, height):
            self._add(row, 0, ' ' * contentWidth, contentWidth)
        self._drawCommandPane(
            commandTop,
            0,
            contentWidth,
            commandHeight,
            dash,
        )
        self._moveCursor(contentWidth, commandTop)
        self.screen.refresh()

    def render(
        self,
        runCfg: RuntimeCfg,
        symbolMeta,
        dash: DashboardState,
        candle: dict,
    ) -> None:
        if not self.enabled or self.screen is None:
            return
        lines, snapshot = dashboardLines(runCfg, symbolMeta, dash, candle)
        height, _width, contentWidth, _unused, _commandTop = self._layout()
        commandTop = min(len(lines) + 1, max(height - 1, 0))
        self.commandRow = commandTop
        dashHeight = max(commandTop - 1, 1)
        commandHeight = max(height - commandTop, 1)
        self.screen.erase()
        self._drawDashboardPane(0, 0, contentWidth, dashHeight, lines)
        self._drawCommandPane(commandTop, 0, contentWidth, commandHeight, dash)
        self._moveCursor(contentWidth, commandTop)
        self.screen.refresh()
        recordDashboardSnapshot(dash, snapshot)


async def syncBalances(
    client,
    symbolMeta,
    runCfg: RuntimeCfg,
    dash: DashboardState,
    nowMono: float,
    force: bool,
) -> None:
    # Refresh wallet balances on cadence or when forced after a trade.
    stale = (nowMono - dash.lastBalanceSync) >= BALANCE_REFRESH_SECONDS
    if not force and not stale:
        pruneOpenOrders(dash)
        return
    await refreshOpenOrders(client, symbolMeta, runCfg, dash)
    dash.quoteTotal = await walletTotalBalance(
        client,
        symbolMeta,
        runCfg,
        dash,
        symbolMeta.quoteAsset,
    )
    dash.baseTotal = await walletTotalBalance(
        client,
        symbolMeta,
        runCfg,
        dash,
        symbolMeta.baseAsset,
    )
    dash.lastBalanceSync = nowMono


async def makeDashboard(
    client,
    symbolMeta,
    runCfg: RuntimeCfg,
) -> DashboardState:
    # Build initial dashboard state from live balances or paper wallet.
    nowMono = asyncio.get_running_loop().time()
    nowMs = utcNowMs()
    if runCfg.paperTrading:
        startQuote = float(runCfg.overrides.get('WALLET_SEED_QUOTE', 0.0))
        seedTotal = max(startQuote, 0.0)
        dash = DashboardState(
            startUtc=datetime.now(tz=timezone.utc),
            seedQuote=0.0,
            quoteTotal=seedTotal,
            baseTotal=0.0,
            lastBalanceSync=nowMono,
            tradeCount=0,
            lastStreamMs=nowMs,
            lastClosedMs=nowMs,
            lastQuoteMs=nowMs,
        )
    else:
        quoteTotal = await walletTotalBalance(
            client,
            symbolMeta,
            runCfg,
            None,
            symbolMeta.quoteAsset,
        )
        baseTotal = await walletTotalBalance(
            client,
            symbolMeta,
            runCfg,
            None,
            symbolMeta.baseAsset,
        )
        dash = DashboardState(
            startUtc=datetime.now(tz=timezone.utc),
            seedQuote=0.0,
            quoteTotal=quoteTotal,
            baseTotal=baseTotal,
            lastBalanceSync=nowMono,
            tradeCount=0,
            tradingEnabled=False,
            lastStreamMs=nowMs,
            lastClosedMs=nowMs,
            lastQuoteMs=nowMs,
        )

    dash.logger = SessionLogger.fromRuntime(
        runCfg.outPath,
        runCfg,
        symbolMeta,
        runCfg.sessionId,
    )
    return dash


def commandResult(
    dash: DashboardState,
    detail: str,
    shouldQuit: bool = False,
    forceSync: bool = False,
) -> tuple[bool, bool]:
    # Store command status text and return loop control booleans.
    dash.lastCommand = detail
    recordHistory(dash, f"COMMAND: {detail}")
    printStage(detail)
    return shouldQuit, forceSync


async def handleCommand(
    cmd: str,
    client,
    symbolMeta,
    runCfg: RuntimeCfg,
    dash: DashboardState,
    latestPrimaryCandle: dict,
) -> tuple[bool, bool]:
    # Execute one stdin command; return quit flag and force-sync flag.
    parts = str(cmd).strip().split(maxsplit=1)
    action = parts[0] if parts else ''
    arg = parts[1] if len(parts) > 1 else ''

    if action == 'pause':
        dash.tradingEnabled = False
        return commandResult(dash, 'pause -> auto trading paused')

    if action == 'resume':
        dash.tradingEnabled = True
        return commandResult(dash, 'resume -> auto trading resumed')

    if action == 'gates':
        dash.viewMode = 'gates'
        return commandResult(dash, 'gates -> live preview')

    if action == 'back':
        dash.viewMode = 'main'
        return commandResult(dash, 'back -> orders/history')

    if action == 'cls':
        bits = arg.split()
        valid = len(bits) == 2 and bits[0] == 'ord'
        if not valid or not bits[1].isdigit():
            return commandResult(dash, 'cls ord -> use cls ord 4')
        orderNo = int(bits[1])
        order = findOpenOrder(dash, orderNo)
        if order is None:
            return commandResult(dash, f"cls ord {orderNo} -> not found")
        result = await cancelOpenOrder(
            client,
            runCfg,
            runCfg.symbol,
            str(order.get('orderId', '')),
        )
        status = str(result.get('status', 'CANCELED'))
        clearOpenOrder(dash, orderNo)
        return commandResult(
            dash,
            f"cls ord {orderNo} -> {status}",
            forceSync=True,
        )

    if action == 'seed':
        lastPrice = float(latestPrimaryCandle['c'])
        strategyVal = dash.quoteTotal + (dash.baseTotal * lastPrice)
        if lastPrice <= 0.0 or strategyVal <= 0.0:
            return commandResult(
                dash,
                'seed -> skipped, invalid price or strategy value',
            )
        dash.seedQuote = strategyVal
        dash.hodlQty = strategyVal / lastPrice
        dash.hodlEntryPrice = lastPrice
        dash.seeded = True
        detail = (
            f"seed -> set ${strategyVal:.2f}, "
            f"hodl {dash.hodlQty:.2f} {symbolMeta.baseAsset}"
        )
        return commandResult(dash, detail, forceSync=True)

    if action == 'buy':
        spec = parseAmountSpec(arg)
        if spec is None:
            return commandResult(dash, 'buy -> use buy $100 or buy 25%')
        mode, value = spec
        if value <= 0.0:
            return commandResult(dash, 'buy -> amount must be > 0')
        quoteFree = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.quoteAsset,
        )
        if mode == 'pct':
            if value > 100.0:
                return commandResult(
                    dash,
                    'buy -> percent exceeds quote holdings',
                )
            quoteQty = quoteFree * (value / 100.0)
        else:
            quoteQty = float(value)
        if quoteQty > quoteFree:
            return commandResult(dash, 'buy -> amount exceeds quote holdings')
        if quoteQty <= 0.0:
            return commandResult(dash, 'buy -> no free quote balance')
        lastPrice = float(latestPrimaryCandle['c'])
        nowMs = int(datetime.now(tz=timezone.utc).timestamp() * 1000.0)
        event = await executeMarketBuy(
            client,
            symbolMeta,
            runCfg,
            dash,
            runCfg.symbol,
            quoteQty,
            lastPrice,
            'MANUAL_BUY',
            nowMs,
        )
        if event is None:
            return commandResult(dash, 'buy -> order rejected')
        recordTradeEvent(dash, event, symbolMeta.baseAsset)
        detail = (
            f"buy -> {event['status']} [{event['indicator']}] "
            f"{tradeAmountText(event, symbolMeta.baseAsset)}"
        )
        return commandResult(dash, detail, forceSync=True)

    if action == 'sell':
        spec = parseAmountSpec(arg)
        if spec is None:
            return commandResult(dash, 'sell -> use sell $100 or sell 25%')
        mode, value = spec
        if value <= 0.0:
            return commandResult(dash, 'sell -> amount must be > 0')
        lastPrice = float(latestPrimaryCandle['c'])
        baseFree = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.baseAsset,
        )
        if mode == 'pct':
            if value > 100.0:
                return commandResult(
                    dash,
                    'sell -> percent exceeds base holdings',
                )
            sellQty = baseFree * (value / 100.0)
        else:
            quoteValue = float(value)
            maxQuote = baseFree * lastPrice
            if quoteValue > maxQuote:
                return commandResult(
                    dash,
                    'sell -> amount exceeds base holdings',
                )
            sellQty = quoteValue / lastPrice if lastPrice > 0.0 else 0.0
        if sellQty <= 0.0:
            return commandResult(dash, 'sell -> no free base balance')
        nowMs = int(datetime.now(tz=timezone.utc).timestamp() * 1000.0)
        event = await executeMarketSell(
            client,
            symbolMeta,
            runCfg,
            dash,
            runCfg.symbol,
            float(sellQty),
            lastPrice,
            'MANUAL_SELL',
            nowMs,
        )
        if event is None:
            return commandResult(dash, 'sell -> below lot-size minimum')
        recordTradeEvent(
            dash,
            event,
            symbolMeta.baseAsset,
        )
        detail = (
            f"sell -> {event['status']} [{event['indicator']}] "
            f"{tradeAmountText(event, symbolMeta.baseAsset)}"
        )
        return commandResult(dash, detail, forceSync=True)

    if action == 'quit':
        return commandResult(dash, 'quit -> shutting down', shouldQuit=True)

    detail = f"unknown command: {cmd} (use buy/sell, gates, or back)"
    return commandResult(dash, detail)
