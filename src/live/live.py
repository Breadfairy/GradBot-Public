#!/usr/bin/env python3
# live.py - live runtime orchestrator.

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import sys

import aiohttp
from binance import BinanceSocketManager
from binance.exceptions import (
    BinanceAPIException,
    BinanceOrderException,
    BinanceRequestException,
    BinanceWebsocketClosed,
    BinanceWebsocketQueueOverflow,
    BinanceWebsocketUnableToConnect,
)

from live.binance_live import (
    fetchHistory,
    fetchSymbolMeta,
    loadCreds,
    makeClient,
    mergeClosedCandle,
)
from live.dashboard import (
    applyWarmStartArgs,
    candleFromRow,
    handleCommand,
    loadRecentHistory,
    loadRuntime,
    makeDashboard,
    maxBars,
    printStage,
    recordHistory,
    recordTradeEvent,
    refreshPrimaryPrice,
    SplitDashboardUi,
    syncBalances,
    trimRows,
    utcNowMs,
)
from live.execution import modeLabel
from live.daily_posture import (
    dailyPostureEnabled,
    dailyPostureForIndex,
    dailyPostureWarmupRows,
)
from live.live_engine import bars_per_day, evaluate
from live.live_engine import postureText
from live.session_runtime import closeSession, openSession
from live.signals import (
    gatePreviewLines,
    makeState,
    processPaperReplaySignals,
    processSignals,
)
from live.state_store import (
    loadState,
    restoreDashboardState,
    restorePhaseState,
    saveRuntimeState,
    stateMatches,
    statePath,
)


DISPLAY_REFRESH_SECONDS = 1.0
SOCKET_POLL_SECONDS = 0.1
DAY_MS = 24 * 60 * 60 * 1000
RECONNECT_DELAYS = (5.0, 15.0, 30.0, 60.0, 120.0, 300.0)
RECONNECT_ERRORS = (
    aiohttp.ClientError,
    asyncio.TimeoutError,
    ConnectionError,
    OSError,
    BinanceAPIException,
    BinanceOrderException,
    BinanceRequestException,
    BinanceWebsocketClosed,
    BinanceWebsocketQueueOverflow,
    BinanceWebsocketUnableToConnect,
)


def refreshGatePreview(
    dash,
    symbolMeta,
    runCfg,
    microRows: list,
    macroRows: list,
    postureRows: list,
    phaseState,
    candle: dict | None,
) -> None:
    # Update the optional gates pane without mutating trading state.
    if dash.viewMode != 'gates' or candle is None:
        return
    dash.gateLines = gatePreviewLines(
        symbolMeta,
        runCfg,
        dash,
        microRows,
        macroRows,
        postureRows,
        phaseState,
        candle,
    )


def activePostureRows(
    runCfg,
    microRows: list,
    macroRows: list,
    postureRows: list,
) -> list:
    # Select the posture feed matching the configured posture interval.
    if runCfg.postureInterval == runCfg.interval:
        return list(microRows)
    if runCfg.postureInterval == runCfg.macroInterval:
        return list(macroRows)
    return postureRows


def warmStartEnabled(runCfg) -> bool:
    return int(getattr(runCfg, 'warmStartWeeks', 0)) > 0


def warmStartSeedIndex(runCfg, microRows: list, macroRows: list) -> int:
    # Return the last closed micro candle at or before the backdate target.
    weeks = int(runCfg.warmStartWeeks)
    targetMs = int(microRows[-1][6]) - (weeks * 7 * DAY_MS)
    indexes = [
        i for i, r in enumerate(microRows)
        if int(r[6]) <= int(targetMs)
    ]
    if not indexes:
        raise ValueError('not enough fetched history for warm-start seed')

    seedIdx = int(indexes[-1])
    seedMicro = microRows[:seedIdx + 1]
    seedMacro = (
        list(seedMicro)
        if runCfg.macroInterval == runCfg.interval else macroRows
    )
    pack = evaluate(
        seedMicro,
        seedMacro,
        runCfg.interval,
        runCfg.periods,
        runCfg.primerDays,
        runCfg.overrides,
    )
    if seedIdx < int(pack.startIdx):
        raise ValueError(
            'warm-start seed is before runtime warmup is ready'
        )
    return seedIdx


def applyWarmStartSeed(dash, runCfg, microRows: list, macroRows: list) -> int:
    # Seed the paper wallet at a historical candle split.
    seedIdx = warmStartSeedIndex(runCfg, microRows, macroRows)
    row = microRows[int(seedIdx)]
    closeMs = int(row[6])
    price = float(row[4])
    seedTotal = max(float(runCfg.overrides['WALLET_SEED_QUOTE']), 0.0)
    assetFrac = max(float(runCfg.warmStartAssetPct), 0.0) / 100.0
    assetValue = seedTotal * assetFrac
    quoteValue = seedTotal - assetValue

    dash.startUtc = datetime.fromtimestamp(closeMs / 1000.0, tz=timezone.utc)
    dash.seedQuote = seedTotal
    dash.quoteTotal = quoteValue
    dash.baseTotal = assetValue / price if price > 0.0 else 0.0
    dash.hodlQty = seedTotal / price if price > 0.0 else 0.0
    dash.hodlEntryPrice = price
    dash.seeded = True
    dash.lastClosedMs = closeMs
    dash.lastClosedPrice = price
    return closeMs


def warmStartDetail(runCfg, seedCloseMs: int) -> str:
    seedTs = datetime.fromtimestamp(
        int(seedCloseMs) / 1000.0,
        tz=timezone.utc,
    )
    return (
        f"warm-start -> {int(runCfg.warmStartWeeks)}w, "
        f"asset={float(runCfg.warmStartAssetPct):.2f}%, "
        f"seedClose={seedTs:%Y-%m-%dT%H:%M:%SZ}"
    )


async def closeClient(client) -> None:
    # Close a Binance client if it has been created.
    if client is not None:
        try:
            await client.close_connection()
        except RECONNECT_ERRORS:
            pass


async def makeRuntimeClient(runCfg, iniPath):
    # Create public paper client or authenticated live client.
    if runCfg.paperTrading:
        printStage('creating public binance market-data session')
        return await makeClient()

    printStage(f"loading api credentials from {iniPath}")
    creds = loadCreds(iniPath)
    printStage('creating authenticated binance client session')
    return await makeClient(creds)


async def loadMarketState(client, runCfg):
    # Fetch symbol metadata and fresh trailing history.
    printStage('fetching symbol metadata')
    symbolMeta = await fetchSymbolMeta(client, runCfg.symbol)

    printStage(
        f"fetching micro klines: {runCfg.symbol} "
        f"{runCfg.interval} days={runCfg.totalDays}"
    )
    microRows = await fetchHistory(
        client,
        runCfg.symbol,
        runCfg.interval,
        runCfg.totalDays,
    )
    printStage(f"micro klines loaded: rows={len(microRows)}")

    if runCfg.macroInterval == runCfg.interval:
        macroRows = list(microRows)
    else:
        printStage(
            f"fetching macro klines: {runCfg.symbol} "
            f"{runCfg.macroInterval} days={runCfg.totalDays}"
        )
        macroRows = await fetchHistory(
            client,
            runCfg.symbol,
            runCfg.macroInterval,
            runCfg.totalDays,
        )
        printStage(f"macro klines loaded: rows={len(macroRows)}")

    if not dailyPostureEnabled(runCfg.overrides):
        postureRows = list(macroRows)
    elif runCfg.postureInterval == runCfg.interval:
        postureRows = list(microRows)
    elif runCfg.postureInterval == runCfg.macroInterval:
        postureRows = list(macroRows)
    else:
        printStage(
            f"fetching posture klines: {runCfg.symbol} "
            f"{runCfg.postureInterval} days={runCfg.totalDays}"
        )
        postureRows = await fetchHistory(
            client,
            runCfg.symbol,
            runCfg.postureInterval,
            runCfg.totalDays,
        )
        printStage(f"posture klines loaded: rows={len(postureRows)}")

    return symbolMeta, microRows, macroRows, postureRows


def streamIdsFor(runCfg) -> list[str]:
    # Build websocket stream ids for micro/macro/posture klines.
    intervals = [runCfg.interval, runCfg.macroInterval]
    if dailyPostureEnabled(runCfg.overrides):
        intervals.append(runCfg.postureInterval)
    streamIds = []
    for i in intervals:
        streamId = f"{runCfg.symbol.lower()}@kline_{i}"
        if streamId not in streamIds:
            streamIds.append(streamId)
    return streamIds


async def connectWithRetry(runCfg, iniPath, dash=None, ui=None):
    # Keep trying until Binance client and fresh candle context are loaded.
    delayIndex = 0
    while True:
        client = None
        try:
            client = await makeRuntimeClient(runCfg, iniPath)
            symbolMeta, microRows, macroRows, postureRows = (
                await loadMarketState(client, runCfg)
            )
            return client, symbolMeta, microRows, macroRows, postureRows
        except RECONNECT_ERRORS as exc:
            await closeClient(client)
            delay = RECONNECT_DELAYS[
                min(delayIndex, len(RECONNECT_DELAYS) - 1)
            ]
            detail = (
                f"connect failed: {type(exc).__name__}; "
                f"retry in {delay:.0f}s"
            )
            if dash is None:
                printStage(detail)
            else:
                recordHistory(dash, detail)
                if ui is not None:
                    ui.renderCommandPane(dash)
            await asyncio.sleep(delay)
            delayIndex += 1


def launchReadiness(
    runCfg,
    microRows: list,
    macroRows: list,
    postureRows: list,
) -> str:
    # Verify startup has enough closed candles for immediate live evaluation.
    pack = evaluate(
        microRows,
        macroRows,
        runCfg.interval,
        runCfg.periods,
        runCfg.primerDays,
        runCfg.overrides,
    )
    barIndex = len(microRows) - 1
    macroNeed = max(
        int(runCfg.overrides.get('MACRO_NRG_WIN_DAYS', 1)),
        int(runCfg.overrides.get('MACRO_GRAD_WIN_DAYS', 1)),
        int(runCfg.overrides.get('MACRO_P3', 1)),
        168,
    )
    if barIndex < pack.startIdx:
        raise RuntimeError(
            f"not enough micro warmup: rows={len(microRows)} "
            f"startIdx={pack.startIdx}"
        )
    if len(macroRows) < macroNeed:
        raise RuntimeError(
            f"not enough daily/macro warmup: rows={len(macroRows)} "
            f"required={macroNeed}"
        )
    if dailyPostureEnabled(runCfg.overrides):
        postureNeed = dailyPostureWarmupRows(runCfg.overrides)
        if len(postureRows) < postureNeed:
            raise RuntimeError(
                f"not enough posture warmup: rows={len(postureRows)} "
                f"required={postureNeed}"
            )

    barsDay = max(bars_per_day(pack.ctx), 1.0)
    posture, _state = dailyPostureForIndex(
        pack.ctx,
        runCfg.overrides,
        barIndex,
        barsDay,
        postureRows,
    )
    if dailyPostureEnabled(runCfg.overrides) and int(posture['cluster']) < 0:
        raise RuntimeError(
            "posture model not ready for latest closed candle"
        )
    return (
        "warmup ready: "
        f"microRows={len(microRows)} startIdx={pack.startIdx} "
        f"macroRows={len(macroRows)} "
        f"postureRows={len(postureRows)} "
        f"postureCluster={posture['cluster']}"
    )


def hydratePosture(
    dash,
    runCfg,
    microRows: list,
    macroRows: list,
    postureRows: list,
) -> None:
    # Push the latest causal daily posture into dashboard state at launch.
    pack = evaluate(
        microRows,
        macroRows,
        runCfg.interval,
        runCfg.periods,
        runCfg.primerDays,
        runCfg.overrides,
    )
    barIndex = len(microRows) - 1
    barsDay = max(bars_per_day(pack.ctx), 1.0)
    posture, _state = dailyPostureForIndex(
        pack.ctx,
        runCfg.overrides,
        barIndex,
        barsDay,
        postureRows,
    )
    cluster = int(posture.get('cluster', -1))
    dash.currentDailyCluster = cluster
    dash.currentPosture = postureText(cluster)


def missedSignalSummary(
    runCfg,
    microRows: list,
    macroRows: list,
    savedCloseMs: int,
) -> str:
    # Summarize closed-candle signals since saved state without trading them.
    if int(savedCloseMs) <= 0:
        return ''
    pack = evaluate(
        microRows,
        macroRows,
        runCfg.interval,
        runCfg.periods,
        runCfg.primerDays,
        runCfg.overrides,
    )
    missed = [
        f"{label}@{int(microRows[int(idx)][6])}"
        for idx, label in pack.flags
        if int(microRows[int(idx)][6]) > int(savedCloseMs)
    ]
    missedBars = sum(
        1 for r in microRows if int(r[6]) > int(savedCloseMs)
    )
    if missedBars <= 0:
        return ''
    labels = ','.join(missed[:8]) if missed else 'none'
    if len(missed) > 8:
        labels = f"{labels},..."
    return (
        f"state catch-up -> missedBars={missedBars}, "
        f"missedSignals={len(missed)} [{labels}], no retro orders"
    )


def missedBarIndexes(microRows: list, savedCloseMs: int) -> list[int]:
    # Return closed-candle indexes after the saved runtime checkpoint.
    if int(savedCloseMs) <= 0:
        return []
    return [
        i for i, r in enumerate(microRows)
        if int(r[6]) > int(savedCloseMs)
    ]


def savedProcessedCloseMs(stateFile, fallbackCloseMs: int) -> int:
    # Prefer durable state over in-memory candle markers for reconnect replay.
    row = loadState(stateFile)
    if row is None:
        return int(fallbackCloseMs)
    raw = row.get('last_processed_close_ms', '')
    if not str(raw).strip():
        return int(fallbackCloseMs)
    return int(float(raw))


async def replayPaperMissedBars(
    client,
    symbolMeta,
    runCfg,
    dash,
    microRows: list,
    macroRows: list,
    postureRows: list,
    phaseState,
    savedCloseMs: int,
    stateFile,
    keepMicro: int,
) -> tuple[int, int, int]:
    # Replay missed paper candles through the normal wallet/log pipeline.
    indexes = missedBarIndexes(microRows, savedCloseMs)
    replayedBars = 0
    replayedTrades = 0
    lastOpenMs = 0
    replayMicroRows = []
    replayMacroRows = []
    replayPostureRows = []
    events = []
    row = None

    if not bool(runCfg.paperTrading):
        return replayedBars, replayedTrades, lastOpenMs

    for i in indexes:
        row = microRows[int(i)]
        replayMicroRows = trimRows(
            microRows[:int(i) + 1],
            int(keepMicro),
        )
        replayMacroRows = (
            list(replayMicroRows)
            if runCfg.macroInterval == runCfg.interval else macroRows
        )
        replayPostureRows = activePostureRows(
            runCfg,
            replayMicroRows,
            replayMacroRows,
            postureRows,
        )
        dash.lastClosedMs = int(row[6])
        dash.lastClosedPrice = float(row[4])
        events = await processPaperReplaySignals(
            client,
            symbolMeta,
            runCfg,
            dash,
            replayMicroRows,
            replayMacroRows,
            replayPostureRows,
            phaseState,
        )
        for event in events:
            recordTradeEvent(dash, event, symbolMeta.baseAsset)
        replayedBars += 1
        replayedTrades += len(events)
        lastOpenMs = int(row[0])
        saveRuntimeState(
            stateFile,
            runCfg,
            dash,
            phaseState,
            lastOpenMs,
            dash.lastClosedMs,
        )

    return replayedBars, replayedTrades, lastOpenMs


async def run() -> None:
    # Start client, preload context, then run live dashboard + order loop.
    printStage('loading runtime profile')
    try:
        runCfg, iniPath = loadRuntime()
        runCfg = applyWarmStartArgs(runCfg, sys.argv[1:])
    except ValueError as exc:
        printStage(f"startup error: {exc}")
        raise SystemExit(2)
    runCfg, sessionDir, sessionId, resumed = openSession(runCfg)
    if warmStartEnabled(runCfg) and resumed:
        printStage(
            'startup error: warm-start requires no active session; '
            'close the active session first'
        )
        raise SystemExit(2)
    stateFile = statePath(runCfg.outPath)
    client = None
    ui = None
    cleanQuit = False
    printStage(
        f"runtime: symbol={runCfg.symbol} interval={runCfg.interval} "
        f"macro={runCfg.macroInterval} posture={runCfg.postureInterval} "
        f"mode={modeLabel(runCfg)}"
    )
    printStage(
        f"session: id={sessionId} "
        f"{'resuming active' if resumed else 'fresh'} "
        f"dir={sessionDir}"
    )

    try:
        client, symbolMeta, microRows, macroRows, postureRows = (
            await connectWithRetry(runCfg, iniPath)
        )

        phaseState = makeState()
        keepMicro = maxBars(runCfg.totalDays, runCfg.interval)
        keepMacro = maxBars(runCfg.totalDays, runCfg.macroInterval)
        keepPosture = maxBars(runCfg.totalDays, runCfg.postureInterval)

        printStage(
            'preparing dashboard state '
            '(paper wallet or live account balances)'
        )
        dash = await makeDashboard(
            client,
            symbolMeta,
            runCfg,
        )
        savedState = loadState(stateFile)
        savedCloseMs = 0
        historyCutoffMs = 0
        if warmStartEnabled(runCfg):
            savedCloseMs = applyWarmStartSeed(
                dash,
                runCfg,
                microRows,
                macroRows,
            )
            historyCutoffMs = int(savedCloseMs)
            detail = warmStartDetail(runCfg, savedCloseMs)
            recordHistory(dash, detail)
            printStage(detail)
        if resumed and stateMatches(savedState, runCfg):
            restoreDashboardState(dash, runCfg, savedState)
            restorePhaseState(phaseState, savedState)
            savedCloseMs = int(
                float(savedState.get('last_processed_close_ms', 0) or 0)
            )
            historyCutoffMs = int(
                float(savedState.get('saved_at_ms', 0) or 0)
            )
            printStage(
                f"restored rolling state: {stateFile} "
                f"lastCloseMs={savedCloseMs}"
            )
        loadRecentHistory(dash, symbolMeta.baseAsset, 20, historyCutoffMs)
        if runCfg.paperTrading:
            printStage(
                f"dashboard ready: local paper wallet quote="
                f"{dash.quoteTotal:.6f}, base={dash.baseTotal:.10f}, "
                f"seed pending, logs={runCfg.outPath}"
            )
        else:
            printStage(
                'dashboard ready: live balances loaded, '
                'auto trading paused, seed pending'
            )

        lastMicroOpen = int(microRows[-1][0])
        dash.lastClosedMs = int(microRows[-1][6])
        dash.lastClosedPrice = float(microRows[-1][4])
        postureRows = activePostureRows(
            runCfg,
            microRows,
            macroRows,
            postureRows,
        )
        printStage(
            launchReadiness(runCfg, microRows, macroRows, postureRows)
        )
        hydratePosture(dash, runCfg, microRows, macroRows, postureRows)
        streamIds = streamIdsFor(runCfg)
        baseAsset = symbolMeta.baseAsset

        printStage(
            f"dashboard refresh interval: {DISPLAY_REFRESH_SECONDS:.0f}s"
        )
        printStage('dashboard ready: command line enabled')

        latestPrimaryCandle = candleFromRow(
            microRows[-1],
            runCfg.interval,
        )
        loop = asyncio.get_running_loop()
        lastRender = loop.time()
        ui = SplitDashboardUi()
        ui.start()
        refreshGatePreview(
            dash,
            symbolMeta,
            runCfg,
            microRows,
            macroRows,
            postureRows,
            phaseState,
            latestPrimaryCandle,
        )
        ui.render(runCfg, symbolMeta, dash, latestPrimaryCandle)
        if runCfg.paperTrading:
            if warmStartEnabled(runCfg):
                printStage(
                    'warm-start history replay: full gates + daily ML'
                )
            replayedBars, replayedTrades, _lastReplayOpen = (
                await replayPaperMissedBars(
                    client,
                    symbolMeta,
                    runCfg,
                    dash,
                    microRows,
                    macroRows,
                    postureRows,
                    phaseState,
                    savedCloseMs,
                    stateFile,
                    keepMicro,
                )
            )
            if replayedBars > 0:
                replayLabel = (
                    'warm-start history'
                    if warmStartEnabled(runCfg) else 'paper catch-up'
                )
                detail = (
                    f"{replayLabel} -> "
                    f"replayedBars={replayedBars}, "
                    f"replayTrades={replayedTrades}, "
                    "engine=full-gates+daily-ml"
                )
                recordHistory(dash, detail)
                refreshGatePreview(
                    dash,
                    symbolMeta,
                    runCfg,
                    microRows,
                    macroRows,
                    postureRows,
                    phaseState,
                    latestPrimaryCandle,
                )
                ui.render(runCfg, symbolMeta, dash, latestPrimaryCandle)
        else:
            detail = missedSignalSummary(
                runCfg,
                microRows,
                macroRows,
                savedCloseMs,
            )
            if detail:
                recordHistory(dash, detail)
                refreshGatePreview(
                    dash,
                    symbolMeta,
                    runCfg,
                    microRows,
                    macroRows,
                    postureRows,
                    phaseState,
                    latestPrimaryCandle,
                )
                ui.renderCommandPane(dash)
        saveRuntimeState(
            stateFile,
            runCfg,
            dash,
            phaseState,
            lastMicroOpen,
            dash.lastClosedMs,
        )

        while True:
            quitRequested = False
            socketMgr = BinanceSocketManager(client)
            printStage(f"opening websocket streams: {','.join(streamIds)}")
            try:
                async with socketMgr.multiplex_socket(streamIds) as stream:
                    while True:
                        forceSync = False
                        try:
                            msg = await asyncio.wait_for(
                                stream.recv(),
                                timeout=SOCKET_POLL_SECONDS,
                            )
                        except asyncio.TimeoutError:
                            msg = None

                        if msg is not None:
                            data = msg.get('data', {})
                            if data.get('e') == 'kline':
                                dash.lastStreamMs = utcNowMs()
                                candle = data['k']
                                intervalValue = str(candle['i'])

                                if (
                                    intervalValue == runCfg.macroInterval
                                    and intervalValue != runCfg.interval
                                ):
                                    if bool(candle['x']):
                                        macroRows = mergeClosedCandle(
                                            macroRows,
                                            candle,
                                        )
                                        macroRows = trimRows(
                                            macroRows,
                                            keepMacro,
                                        )
                                        if (
                                            runCfg.postureInterval
                                            == runCfg.macroInterval
                                        ):
                                            postureRows = list(macroRows)

                                if (
                                    intervalValue == runCfg.postureInterval
                                    and intervalValue != runCfg.interval
                                    and intervalValue != runCfg.macroInterval
                                ):
                                    if bool(candle['x']):
                                        postureRows = mergeClosedCandle(
                                            postureRows,
                                            candle,
                                        )
                                        postureRows = trimRows(
                                            postureRows,
                                            keepPosture,
                                        )

                                if intervalValue == runCfg.interval:
                                    latestPrimaryCandle = candle
                                    if bool(candle['x']):
                                        dash.lastClosedMs = int(candle['T'])
                                        dash.lastClosedPrice = float(
                                            candle['c']
                                        )
                                        microRows = mergeClosedCandle(
                                            microRows,
                                            candle,
                                        )
                                        microRows = trimRows(
                                            microRows,
                                            keepMicro,
                                        )
                                        if (
                                            runCfg.postureInterval
                                            == runCfg.interval
                                        ):
                                            postureRows = list(microRows)
                                        newOpen = int(candle['t'])
                                        if (
                                            runCfg.macroInterval
                                            == runCfg.interval
                                        ):
                                            macroRows = list(microRows)
                                        if newOpen > lastMicroOpen:
                                            lastMicroOpen = newOpen
                                            if (
                                                runCfg.macroInterval
                                                != runCfg.interval
                                            ):
                                                macroRows = trimRows(
                                                    macroRows,
                                                    keepMacro,
                                                )
                                            postureRows = activePostureRows(
                                                runCfg,
                                                microRows,
                                                macroRows,
                                                postureRows,
                                            )
                                            events = await processSignals(
                                                client,
                                                symbolMeta,
                                                runCfg,
                                                dash,
                                                microRows,
                                                macroRows,
                                                postureRows,
                                                phaseState,
                                            )
                                            if events:
                                                for event in events:
                                                    recordTradeEvent(
                                                        dash,
                                                        event,
                                                        baseAsset,
                                                    )
                                                forceSync = True
                                            saveRuntimeState(
                                                stateFile,
                                                runCfg,
                                                dash,
                                                phaseState,
                                                newOpen,
                                                dash.lastClosedMs,
                                            )

                        nowMono = loop.time()
                        await syncBalances(
                            client,
                            symbolMeta,
                            runCfg,
                            dash,
                            nowMono,
                            force=forceSync,
                        )
                        commands = ui.pollCommands()
                        commandPaneDirty = ui.consumeInputDirty()
                        for cmd in commands:
                            shouldQuit, cmdForce = await handleCommand(
                                cmd,
                                client,
                                symbolMeta,
                                runCfg,
                                dash,
                                latestPrimaryCandle,
                            )
                            forceSync = forceSync or cmdForce
                            commandPaneDirty = True
                            if shouldQuit:
                                quitRequested = True
                                break
                        if commands:
                            saveRuntimeState(
                                stateFile,
                                runCfg,
                                dash,
                                phaseState,
                                lastMicroOpen,
                                dash.lastClosedMs,
                            )
                        if quitRequested:
                            cleanQuit = True
                            break
                        if commandPaneDirty:
                            refreshGatePreview(
                                dash,
                                symbolMeta,
                                runCfg,
                                microRows,
                                macroRows,
                                postureRows,
                                phaseState,
                                latestPrimaryCandle,
                            )
                            ui.renderCommandPane(dash)
                        if forceSync:
                            nowMono = loop.time()
                            await syncBalances(
                                client,
                                symbolMeta,
                                runCfg,
                                dash,
                                nowMono,
                                force=True,
                            )
                        due = (
                            nowMono - lastRender
                        ) >= DISPLAY_REFRESH_SECONDS
                        if latestPrimaryCandle is not None and (
                            due or forceSync
                        ):
                            await refreshPrimaryPrice(
                                client,
                                runCfg.symbol,
                                latestPrimaryCandle,
                                dash,
                            )
                            dash.lastQuoteMs = utcNowMs()
                            refreshGatePreview(
                                dash,
                                symbolMeta,
                                runCfg,
                                microRows,
                                macroRows,
                                postureRows,
                                phaseState,
                                latestPrimaryCandle,
                            )
                            ui.render(
                                runCfg,
                                symbolMeta,
                                dash,
                                latestPrimaryCandle,
                            )
                            lastRender = nowMono
            except RECONNECT_ERRORS as exc:
                wasTrading = bool(dash.tradingEnabled)
                oldClosedMs = savedProcessedCloseMs(
                    stateFile,
                    int(dash.lastClosedMs),
                )
                dash.tradingEnabled = False
                dash.lastStreamMs = 0
                dash.lastQuoteMs = 0
                detail = (
                    f"runtime error: {type(exc).__name__}; "
                    "auto trading paused"
                )
                recordHistory(dash, detail)
                refreshGatePreview(
                    dash,
                    symbolMeta,
                    runCfg,
                    microRows,
                    macroRows,
                    postureRows,
                    phaseState,
                    latestPrimaryCandle,
                )
                ui.render(runCfg, symbolMeta, dash, latestPrimaryCandle)
                await closeClient(client)
                client, symbolMeta, microRows, macroRows, postureRows = (
                    await connectWithRetry(runCfg, iniPath, dash, ui)
                )
                baseAsset = symbolMeta.baseAsset
                keepMicro = maxBars(runCfg.totalDays, runCfg.interval)
                keepMacro = maxBars(runCfg.totalDays, runCfg.macroInterval)
                keepPosture = maxBars(runCfg.totalDays, runCfg.postureInterval)
                postureRows = activePostureRows(
                    runCfg,
                    microRows,
                    macroRows,
                    postureRows,
                )
                printStage(
                    launchReadiness(
                        runCfg,
                        microRows,
                        macroRows,
                        postureRows,
                    )
                )
                hydratePosture(
                    dash,
                    runCfg,
                    microRows,
                    macroRows,
                    postureRows,
                )
                lastMicroOpen = int(microRows[-1][0])
                missedBars = sum(
                    1 for r in microRows if int(r[6]) > oldClosedMs
                )
                dash.lastClosedMs = int(microRows[-1][6])
                dash.lastClosedPrice = float(microRows[-1][4])
                dash.lastStreamMs = utcNowMs()
                latestPrimaryCandle = candleFromRow(
                    microRows[-1],
                    runCfg.interval,
                )
                dash.tradingEnabled = wasTrading
                replayedBars = 0
                replayedTrades = 0
                if runCfg.paperTrading:
                    replayedBars, replayedTrades, _lastReplayOpen = (
                        await replayPaperMissedBars(
                            client,
                            symbolMeta,
                            runCfg,
                            dash,
                            microRows,
                            macroRows,
                            postureRows,
                            phaseState,
                            oldClosedMs,
                            stateFile,
                            keepMicro,
                        )
                    )
                nowMono = loop.time()
                await syncBalances(
                    client,
                    symbolMeta,
                    runCfg,
                    dash,
                    nowMono,
                    force=True,
                )
                if runCfg.paperTrading:
                    detail = (
                        'reconnected -> auto trading '
                        f"{'resumed' if wasTrading else 'paused'}, "
                        f"missed={missedBars}, "
                        f"replayedBars={replayedBars}, "
                        f"replayTrades={replayedTrades}"
                    )
                else:
                    detail = (
                        'reconnected -> auto trading '
                        f"{'resumed' if wasTrading else 'paused'}, "
                        f"phase kept, missed={missedBars} no retro orders"
                    )
                recordHistory(dash, detail)
                if not runCfg.paperTrading:
                    missedDetail = missedSignalSummary(
                        runCfg,
                        microRows,
                        macroRows,
                        oldClosedMs,
                    )
                    if missedDetail:
                        recordHistory(dash, missedDetail)
                saveRuntimeState(
                    stateFile,
                    runCfg,
                    dash,
                    phaseState,
                    lastMicroOpen,
                    dash.lastClosedMs,
                )
                refreshGatePreview(
                    dash,
                    symbolMeta,
                    runCfg,
                    microRows,
                    macroRows,
                    postureRows,
                    phaseState,
                    latestPrimaryCandle,
                )
                ui.render(runCfg, symbolMeta, dash, latestPrimaryCandle)
                lastRender = nowMono
                continue

            if quitRequested:
                cleanQuit = True
                break
    finally:
        if cleanQuit:
            closeSession(sessionDir)
        if ui is not None:
            ui.stop()
        if client is not None:
            printStage('closing binance client session')
            await closeClient(client)


if __name__ == '__main__':
    asyncio.run(run())
