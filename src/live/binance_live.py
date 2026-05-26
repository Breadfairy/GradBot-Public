#!/usr/bin/env python3
# binance_live.py – python-binance helpers for live trading runtime.

from __future__ import annotations

import configparser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import ssl
from typing import Dict

import aiohttp
from binance import AsyncClient
from binance.helpers import round_step_size
import certifi


@dataclass(frozen=True)
class ApiCreds:
    # Hold API key/secret loaded from config.ini.
    apiKey: str
    apiSecret: str


@dataclass(frozen=True)
class SymbolMeta:
    # Hold symbol assets and lot-size constraints.
    symbol: str
    baseAsset: str
    quoteAsset: str
    stepSize: float
    minQty: float
    minNotional: float


def loadCreds(configPath: Path) -> ApiCreds:
    # Load Binance API credentials from INI path.
    parser = configparser.ConfigParser()
    parser.read(configPath)
    if 'binance' not in parser:
        raise ValueError(
            f"missing [binance] section in credentials file: {configPath}"
        )
    apiKey = parser['binance']['api_key']
    apiSecret = parser['binance']['api_secret']
    return ApiCreds(apiKey=apiKey, apiSecret=apiSecret)


async def makeClient(creds: ApiCreds | None = None) -> AsyncClient:
    # Create async Binance client with optional credentials.
    sslCtx = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=sslCtx)
    sessionParams = {
        'connector': connector,
    }
    kwargs = {
        'session_params': sessionParams,
    }
    if creds is not None:
        kwargs['api_key'] = creds.apiKey
        kwargs['api_secret'] = creds.apiSecret
    return await AsyncClient.create(**kwargs)


def normalizeRow(row: list) -> list:
    # Normalize one raw kline row into expected types.
    outRow = list(row)
    outRow[0] = int(float(outRow[0]))
    outRow[6] = int(float(outRow[6]))
    outRow[8] = int(float(outRow[8]))
    return outRow


async def fetchHistory(
    client: AsyncClient,
    symbol: str,
    interval: str,
    days: int,
) -> list:
    # Fetch historical klines and keep only fully closed candles.
    startStr = f"{int(days)} day ago UTC"
    rows = await client.get_historical_klines(symbol, interval, startStr)
    nowMs = int(datetime.now(tz=timezone.utc).timestamp() * 1000.0)
    outRows = [normalizeRow(r) for r in rows]
    return [r for r in outRows if int(r[6]) <= nowMs]


async def fetchSymbolMeta(client: AsyncClient, symbol: str) -> SymbolMeta:
    # Load symbol metadata and lot-size filters.
    info = await client.get_symbol_info(symbol)
    filters = list(info['filters'])
    lot = [flt for flt in filters if flt['filterType'] == 'LOT_SIZE'][0]
    notional = [
        flt for flt in filters
        if flt['filterType'] in ('MIN_NOTIONAL', 'NOTIONAL')
    ]
    stepSize = float(lot['stepSize'])
    minQty = float(lot['minQty'])
    minNotional = 0.0
    if notional:
        minNotional = float(notional[0].get('minNotional', 0.0))
    return SymbolMeta(
        symbol=symbol,
        baseAsset=str(info['baseAsset']),
        quoteAsset=str(info['quoteAsset']),
        stepSize=stepSize,
        minQty=minQty,
        minNotional=minNotional,
    )


async def fetchBookQuote(client: AsyncClient, symbol: str) -> dict:
    # Capture executable top-of-book prices for slippage measurement.
    quoteMs = int(datetime.now(tz=timezone.utc).timestamp() * 1000.0)
    ticker = await client.get_orderbook_ticker(symbol=symbol)
    return {
        'quoteTimeMs': quoteMs,
        'bestBid': float(ticker['bidPrice']),
        'bestAsk': float(ticker['askPrice']),
        'bidQty': float(ticker['bidQty']),
        'askQty': float(ticker['askQty']),
    }


async def freeBalance(client: AsyncClient, asset: str) -> float:
    # Fetch free balance for one asset symbol.
    bal = await client.get_asset_balance(asset=asset)
    return float(bal['free'])


def mergeClosedCandle(rows: list, candle: dict) -> list:
    # Upsert a closed websocket candle into kline list.
    row = [
        int(candle['t']),
        candle['o'],
        candle['h'],
        candle['l'],
        candle['c'],
        candle['v'],
        int(candle['T']),
        candle['q'],
        int(candle['n']),
        candle['V'],
        candle['Q'],
        candle['B'],
    ]
    if not rows:
        rows.append(row)
        return rows

    openTime = int(candle['t'])
    lastOpen = int(rows[-1][0])
    if openTime == lastOpen:
        rows[-1] = row
    elif openTime > lastOpen:
        rows.append(row)
    return rows


def clampSellQty(qty: float, meta: SymbolMeta) -> float:
    # Clamp sell qty to Binance LOT_SIZE step and minQty.
    rounded = float(round_step_size(qty, meta.stepSize))
    if rounded < meta.minQty:
        return 0.0
    return rounded


async def placeMarketBuy(
    client: AsyncClient,
    symbol: str,
    quoteQty: float,
) -> dict:
    # Send spot market BUY by quote amount.
    qty = round(float(quoteQty), 8)
    return await client.order_market_buy(
        symbol=symbol,
        quoteOrderQty=str(qty),
    )


async def placeMarketSell(
    client: AsyncClient,
    symbol: str,
    qty: float,
    meta: SymbolMeta,
) -> dict | None:
    # Send spot market SELL by base quantity.
    clamped = clampSellQty(qty, meta)
    if clamped <= 0.0:
        return None
    return await client.order_market_sell(
        symbol=symbol,
        quantity=str(clamped),
    )


async def cancelOrder(
    client: AsyncClient,
    symbol: str,
    orderId: str,
) -> dict:
    # Cancel one live Binance order by id.
    return await client.cancel_order(symbol=symbol, orderId=str(orderId))


def orderPrice(order: dict, fallbackPrice: float) -> float:
    # Return average fill price from order response.
    execQty = float(order.get('executedQty', 0.0))
    quoteQty = float(order.get('cummulativeQuoteQty', 0.0))
    if execQty > 0:
        return quoteQty / execQty
    return float(fallbackPrice)


def orderFeeText(order: dict) -> str:
    # Build a fee summary text from Binance order fills.
    fills = order.get('fills')
    if not isinstance(fills, list) or not fills:
        return 'n/a'
    byAsset: Dict[str, float] = {}
    for fill in fills:
        asset = str(fill.get('commissionAsset', '')).strip()
        fee = float(fill.get('commission', 0.0))
        byAsset[asset] = byAsset.get(asset, 0.0) + fee
    chunks = []
    for asset in sorted(byAsset):
        chunks.append(f"{byAsset[asset]:.10f} {asset}".strip())
    return ' + '.join(chunks)


def orderIndicator(status: str) -> str:
    # Map Binance order status to fulfilled/pending indicator.
    return 'fulfilled' if str(status).upper() == 'FILLED' else 'pending'


def orderEvent(
    order: dict,
    flag: str,
    fallbackPrice: float,
    timeMs: int,
) -> dict:
    # Build a normalized order event from Binance response.
    status = str(order.get('status', ''))
    return {
        'timeMs': int(timeMs),
        'flag': str(flag),
        'side': str(order.get('side', '')),
        'qty': float(order.get('executedQty', 0.0)),
        'quoteQty': float(order.get('cummulativeQuoteQty', 0.0)),
        'price': float(orderPrice(order, fallbackPrice)),
        'orderId': order.get('orderId', ''),
        'status': status,
        'indicator': orderIndicator(status),
        'feeText': orderFeeText(order),
    }
