#!/usr/bin/env python3
# execution.py - live/paper execution helpers and wallet access.

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from live.binance_live import (
    cancelOrder,
    clampSellQty,
    fetchBookQuote,
    freeBalance,
    orderEvent,
    placeMarketBuy,
    placeMarketSell,
)


def isPaperTrading(runCfg: Any) -> bool:
    # Return True when runtime is configured for paper execution.
    return bool(getattr(runCfg, 'paperTrading', False))


def isDryRun(runCfg: Any) -> bool:
    # Return True when live runtime should simulate orders only.
    return bool(getattr(runCfg, 'dryRun', False))


def modeLabel(runCfg: Any) -> str:
    # Return a compact execution mode label for UI/logging.
    if isPaperTrading(runCfg):
        return 'PAPER'
    if isDryRun(runCfg):
        return 'LIVE-DRY'
    return 'LIVE'


async def cancelOpenOrder(
    client,
    runCfg: Any,
    symbol: str,
    orderId: str,
) -> dict:
    # Cancel or locally clear one pending runtime order.
    if isPaperTrading(runCfg) or isDryRun(runCfg):
        return {
            'orderId': str(orderId),
            'status': 'CANCELED',
        }
    return await cancelOrder(client, symbol, str(orderId))


async def walletFreeBalance(
    client,
    symbolMeta,
    runCfg: Any,
    dash,
    asset: str,
) -> float:
    # Read free balance from live account or paper wallet.
    if isPaperTrading(runCfg):
        if asset == symbolMeta.quoteAsset:
            return float(dash.quoteTotal)
        if asset == symbolMeta.baseAsset:
            return float(dash.baseTotal)
        return 0.0
    return await freeBalance(client, asset)


async def walletTotalBalance(
    client,
    symbolMeta,
    runCfg: Any,
    dash,
    asset: str,
) -> float:
    # Read total balance from live account or paper wallet.
    if isPaperTrading(runCfg):
        if asset == symbolMeta.quoteAsset:
            return float(dash.quoteTotal)
        if asset == symbolMeta.baseAsset:
            return float(dash.baseTotal)
        return 0.0
    bal = await client.get_asset_balance(asset=asset)
    return float(bal['free']) + float(bal['locked'])


def _paperOrderId(side: str) -> str:
    # Generate a unique paper order id for log correlation.
    stamp = int(datetime.now(tz=timezone.utc).timestamp() * 1000.0)
    return f"paper-{side.lower()}-{stamp}"


def _paperReplayOrderId(side: str, timeMs: int) -> str:
    # Generate a stable paper replay id from the signal close timestamp.
    return f"paper-replay-{side.lower()}-{int(timeMs)}"


def _dryRunOrderId(side: str) -> str:
    # Generate a unique dry-run order id for log correlation.
    stamp = int(datetime.now(tz=timezone.utc).timestamp() * 1000.0)
    return f"dry-run-{side.lower()}-{stamp}"


def _paperFeeRate(runCfg: Any) -> float:
    # Read configured paper fee rate from runtime overrides.
    raw = getattr(runCfg, 'overrides', {}).get('WALLET_FEE_RATE', 0.0)
    return max(float(raw), 0.0)


def _minNotional(symbolMeta) -> float:
    # Read Binance minimum order notional from symbol metadata.
    return max(float(getattr(symbolMeta, 'minNotional', 0.0)), 0.0)


def _meetsMinNotional(symbolMeta, quoteQty: float) -> bool:
    # Return True when quote value satisfies exchange minimum notional.
    minVal = _minNotional(symbolMeta)
    if minVal <= 0.0:
        return True
    return float(quoteQty) >= minVal


def _quoteFillPrice(side: str, quote: dict, fallbackPrice: float) -> float:
    # Map top-of-book quote to the price a marketable order would touch.
    sideTxt = str(side).upper()
    if sideTxt == 'BUY':
        return float(quote['bestAsk'])
    if sideTxt == 'SELL':
        return float(quote['bestBid'])
    return float(fallbackPrice)


def _adverseBps(side: str, signalPrice: float, fillPrice: float) -> float:
    # Measure fill drag versus the model's candle-close signal price.
    sideTxt = str(side).upper()
    signal = float(signalPrice)
    fill = float(fillPrice)
    if signal <= 0.0 or fill <= 0.0:
        return 0.0
    if sideTxt == 'BUY':
        return ((fill / signal) - 1.0) * 10000.0
    if sideTxt == 'SELL':
        return ((signal / fill) - 1.0) * 10000.0
    return 0.0


def _attachSlippage(
    event: dict,
    quote: dict,
    signalPrice: float,
    signalTimeMs: int,
) -> None:
    # Attach quote-paper execution fields for dashboard and CSV review.
    side = str(event.get('side', '')).upper()
    syntheticFill = _quoteFillPrice(side, quote, signalPrice)
    quoteMs = int(quote['quoteTimeMs'])
    event['signalPrice'] = float(signalPrice)
    event['quoteTimeMs'] = quoteMs
    event['quoteDelayMs'] = quoteMs - int(signalTimeMs)
    event['bestBid'] = float(quote['bestBid'])
    event['bestAsk'] = float(quote['bestAsk'])
    event['bidQty'] = float(quote['bidQty'])
    event['askQty'] = float(quote['askQty'])
    event['syntheticFillPrice'] = syntheticFill
    event['syntheticAdverseBps'] = _adverseBps(
        side,
        float(signalPrice),
        syntheticFill,
    )
    event['adverseBps'] = _adverseBps(
        side,
        float(signalPrice),
        float(event.get('price', 0.0)),
    )


def _replayQuote(price: float, timeMs: int) -> dict:
    # Build a zero-spread historical quote for paper replay fills.
    fillPrice = float(price)
    return {
        'quoteTimeMs': int(timeMs),
        'bestBid': fillPrice,
        'bestAsk': fillPrice,
        'bidQty': 0.0,
        'askQty': 0.0,
    }


def _paperBuyOrder(
    symbolMeta,
    quoteQty: float,
    price: float,
    feeRate: float,
) -> dict | None:
    # Build a Binance-like filled BUY order payload for paper mode.
    if quoteQty <= 0.0 or price <= 0.0:
        return None
    grossQty = float(quoteQty) / float(price)
    if grossQty <= 0.0:
        return None
    feeQty = grossQty * float(feeRate)
    return {
        'status': 'FILLED',
        'side': 'BUY',
        'executedQty': grossQty,
        'cummulativeQuoteQty': float(quoteQty),
        'orderId': _paperOrderId('BUY'),
        'fills': [
            {
                'commissionAsset': symbolMeta.baseAsset,
                'commission': feeQty,
            }
        ],
    }


def _dryRunBuyOrder(
    symbolMeta,
    quoteQty: float,
    price: float,
) -> dict | None:
    # Build a Binance-like BUY payload without placing an exchange order.
    if quoteQty <= 0.0 or price <= 0.0:
        return None
    grossQty = float(quoteQty) / float(price)
    if grossQty <= 0.0:
        return None
    _ = symbolMeta
    return {
        'status': 'DRY_RUN',
        'side': 'BUY',
        'executedQty': grossQty,
        'cummulativeQuoteQty': float(quoteQty),
        'orderId': _dryRunOrderId('BUY'),
        'fills': [],
    }


def _paperSellOrder(
    symbolMeta,
    qty: float,
    price: float,
    feeRate: float,
) -> dict | None:
    # Build a Binance-like filled SELL order payload for paper mode.
    if qty <= 0.0 or price <= 0.0:
        return None
    grossQuoteQty = float(qty) * float(price)
    if grossQuoteQty <= 0.0:
        return None
    feeQuote = grossQuoteQty * float(feeRate)
    return {
        'status': 'FILLED',
        'side': 'SELL',
        'executedQty': float(qty),
        'cummulativeQuoteQty': grossQuoteQty,
        'orderId': _paperOrderId('SELL'),
        'fills': [
            {
                'commissionAsset': symbolMeta.quoteAsset,
                'commission': feeQuote,
            }
        ],
    }


def _dryRunSellOrder(
    symbolMeta,
    qty: float,
    price: float,
) -> dict | None:
    # Build a Binance-like SELL payload without placing an exchange order.
    if qty <= 0.0 or price <= 0.0:
        return None
    grossQuoteQty = float(qty) * float(price)
    if grossQuoteQty <= 0.0:
        return None
    _ = symbolMeta
    return {
        'status': 'DRY_RUN',
        'side': 'SELL',
        'executedQty': float(qty),
        'cummulativeQuoteQty': grossQuoteQty,
        'orderId': _dryRunOrderId('SELL'),
        'fills': [],
    }


def _applyPaperBuy(dash, order: dict) -> None:
    # Apply filled paper BUY balances to the dashboard wallet.
    quoteQty = float(order.get('cummulativeQuoteQty', 0.0))
    execQty = float(order.get('executedQty', 0.0))
    fills = order.get('fills', [])
    feeQty = 0.0
    if isinstance(fills, list):
        for fill in fills:
            feeQty += float(fill.get('commission', 0.0))
    dash.quoteTotal = max(float(dash.quoteTotal) - quoteQty, 0.0)
    dash.baseTotal = float(dash.baseTotal) + max(execQty - feeQty, 0.0)


def _applyPaperSell(dash, order: dict) -> None:
    # Apply filled paper SELL balances to the dashboard wallet.
    qty = float(order.get('executedQty', 0.0))
    grossQuote = float(order.get('cummulativeQuoteQty', 0.0))
    fills = order.get('fills', [])
    feeQuote = 0.0
    if isinstance(fills, list):
        for fill in fills:
            feeQuote += float(fill.get('commission', 0.0))
    dash.baseTotal = max(float(dash.baseTotal) - qty, 0.0)
    dash.quoteTotal = float(dash.quoteTotal) + max(grossQuote - feeQuote, 0.0)


async def executeMarketBuy(
    client,
    symbolMeta,
    runCfg: Any,
    dash,
    symbol: str,
    quoteQty: float,
    fallbackPrice: float,
    flag: str,
    timeMs: int,
) -> dict | None:
    # Execute or simulate one market BUY and return a normalized event.
    quote = await fetchBookQuote(client, symbol)
    fillPrice = _quoteFillPrice('BUY', quote, fallbackPrice)
    if isPaperTrading(runCfg):
        available = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.quoteAsset,
        )
        spend = min(float(quoteQty), float(available))
        order = _paperBuyOrder(
            symbolMeta,
            spend,
            fillPrice,
            _paperFeeRate(runCfg),
        )
        if order is None:
            return None
        if not _meetsMinNotional(symbolMeta, spend):
            return None
        _applyPaperBuy(dash, order)
    elif isDryRun(runCfg):
        spend = float(quoteQty)
        if not _meetsMinNotional(symbolMeta, spend):
            return None
        order = _dryRunBuyOrder(symbolMeta, spend, fillPrice)
        if order is None:
            return None
    else:
        spend = float(quoteQty)
        if not _meetsMinNotional(symbolMeta, spend):
            return None
        order = await placeMarketBuy(client, symbol, spend)

    event = orderEvent(
        order,
        flag=flag,
        fallbackPrice=float(fallbackPrice),
        timeMs=int(timeMs),
    )
    if not event.get('side'):
        event['side'] = 'BUY'
    if isDryRun(runCfg):
        event['indicator'] = 'dry-run'
    _attachSlippage(event, quote, float(fallbackPrice), int(timeMs))
    return event


async def executeMarketSell(
    client,
    symbolMeta,
    runCfg: Any,
    dash,
    symbol: str,
    qty: float,
    fallbackPrice: float,
    flag: str,
    timeMs: int,
) -> dict | None:
    # Execute or simulate one market SELL and return a normalized event.
    quote = await fetchBookQuote(client, symbol)
    fillPrice = _quoteFillPrice('SELL', quote, fallbackPrice)
    if isPaperTrading(runCfg):
        available = await walletFreeBalance(
            client,
            symbolMeta,
            runCfg,
            dash,
            symbolMeta.baseAsset,
        )
        sellQty = min(float(qty), float(available))
        order = _paperSellOrder(
            symbolMeta,
            sellQty,
            fillPrice,
            _paperFeeRate(runCfg),
        )
        if order is None:
            return None
        if not _meetsMinNotional(symbolMeta, sellQty * fillPrice):
            return None
        _applyPaperSell(dash, order)
    elif isDryRun(runCfg):
        sellQty = clampSellQty(float(qty), symbolMeta)
        if not _meetsMinNotional(symbolMeta, sellQty * fillPrice):
            return None
        order = _dryRunSellOrder(symbolMeta, sellQty, fillPrice)
        if order is None:
            return None
    else:
        sellQty = clampSellQty(float(qty), symbolMeta)
        if not _meetsMinNotional(symbolMeta, sellQty * fillPrice):
            return None
        order = await placeMarketSell(
            client,
            symbol,
            sellQty,
            symbolMeta,
        )
        if order is None:
            return None

    event = orderEvent(
        order,
        flag=flag,
        fallbackPrice=float(fallbackPrice),
        timeMs=int(timeMs),
    )
    if not event.get('side'):
        event['side'] = 'SELL'
    if isDryRun(runCfg):
        event['indicator'] = 'dry-run'
    _attachSlippage(event, quote, float(fallbackPrice), int(timeMs))
    return event


def executePaperReplayBuy(
    symbolMeta,
    runCfg: Any,
    dash,
    quoteQty: float,
    fallbackPrice: float,
    flag: str,
    timeMs: int,
) -> dict | None:
    # Simulate one missed paper BUY at the historical signal close price.
    fillPrice = float(fallbackPrice)
    available = float(dash.quoteTotal)
    spend = min(float(quoteQty), available)
    order = _paperBuyOrder(
        symbolMeta,
        spend,
        fillPrice,
        _paperFeeRate(runCfg),
    )
    if order is None:
        return None
    if not _meetsMinNotional(symbolMeta, spend):
        return None
    order['orderId'] = _paperReplayOrderId('BUY', int(timeMs))
    _applyPaperBuy(dash, order)

    event = orderEvent(
        order,
        flag=flag,
        fallbackPrice=fillPrice,
        timeMs=int(timeMs),
    )
    event['indicator'] = 'paper-replay'
    _attachSlippage(
        event,
        _replayQuote(fillPrice, int(timeMs)),
        fillPrice,
        int(timeMs),
    )
    return event


def executePaperReplaySell(
    symbolMeta,
    runCfg: Any,
    dash,
    qty: float,
    fallbackPrice: float,
    flag: str,
    timeMs: int,
) -> dict | None:
    # Simulate one missed paper SELL at the historical signal close price.
    fillPrice = float(fallbackPrice)
    available = float(dash.baseTotal)
    sellQty = min(float(qty), available)
    order = _paperSellOrder(
        symbolMeta,
        sellQty,
        fillPrice,
        _paperFeeRate(runCfg),
    )
    if order is None:
        return None
    if not _meetsMinNotional(symbolMeta, sellQty * fillPrice):
        return None
    order['orderId'] = _paperReplayOrderId('SELL', int(timeMs))
    _applyPaperSell(dash, order)

    event = orderEvent(
        order,
        flag=flag,
        fallbackPrice=fillPrice,
        timeMs=int(timeMs),
    )
    event['indicator'] = 'paper-replay'
    _attachSlippage(
        event,
        _replayQuote(fillPrice, int(timeMs)),
        fillPrice,
        int(timeMs),
    )
    return event
