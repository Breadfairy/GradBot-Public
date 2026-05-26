#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include "engine.h"

#define TUNE_BATCH_LANES 32
#define DAILY_STRONG_CLUSTER 2
#define DAILY_DOWN_MASK 9

typedef struct{
    double* meanA;
    double* stdA;
    double* meanB;
    double* stdB;
} FlagScratch;

typedef struct{
    unsigned char* buyFlags;
    unsigned char* sellFlags;
    double* curveSim;
    double* retsSim;
    double* edgeRets;
    double* edgeVals;
} RowScratch;

typedef struct{
    double* cs;
    double* cs2;
    double* csNeg;
    double* cs2Neg;
    double* csCntNeg;
    double* sharpeVals;
    double* sortinoVals;
} MetricScratch;

typedef struct{
    int* windows;
    double** series;
    int count;
    int cap;
} ZSeriesCache;

typedef struct{
    ZSeriesCache grad;
} ZCacheSet;

typedef struct{
    double barsPerDay;
    double* ema1;
    double* ema2;
    double* ema3;
    double* g1p1;
    int* trend;
} MicroDerived;

typedef struct{
    double* dyn;
    int* dir;
    int* mom;
} MacroDerived;

typedef struct{
    double qty;
    double costPerUnit;
    int64_t ts;
    int index;
} LotState;

typedef struct{
    double quoteBalance;
    double baseBalance;
    double feeRate;
    double taxRate;
    int discountDays;
    double discountRate;
    int taxMode;
    double annualIncomeBase;
    int taxYearStartMonth;
    double feesPaidQuote;
    double realizedGain;
    double taxLiability;
    double baseIncomeTax;
    int tradeCount;
    int buyTrades;
    int sellTrades;
    LotState* lots;
    int lotCount;
    int lotCap;
} WalletState;

typedef struct{
    int side;
    double baseValue;
    double lastPrice;
    int hasLastPrice;
    double portionsRemaining;
    int hasInfiniteRemaining;
    double finalPortionPct;
} PhaseState;

typedef struct{
    const MicroSoa* micro;
    int* allowBuy;
    int* allowSell;
    int* buyAnchors;
    int* sellAnchors;
    FlagScratch scratch;
    ZCacheSet zCache;
    MicroDerived microDerived;
    MacroDerived macroDerived;
} PreparedDatasetState;

typedef struct{
    RowScratch rowScratch;
    MetricScratch metricScratch;
} EvalLaneState;

typedef struct{
    int cap;
    int n;
    int* cooldown;
    int* lastTrendCode;
    int* lastSignalSellIndex;
    int* buyLastCd;
    int* sellLastCd;
    int* buyLastAccepted;
    int* sellLastAccepted;
    int* buyLastPhase;
    int* sellLastPhase;
    int* buyCounts;
    int* sellCounts;
    double* benchQty;
    double* curveSim;
    int* dailyStrongDays;
    int* dailyLockActive;
    int* dailyLockStart;
    int* dailyPrevStrong;
    int* dailyPrevDown;
    int* dailyEpisodeLocked;
    int* dailyBridgeBars;
    int* dailyLockReleaseOnStrong;
    int* dailyCoastActive;
    int* dailyCoastStart;
    int* peakLong;
    int* peakBearCount;
    int* peakStrongGraceBars;
    int* peakStrongReleases;
    int* peakPrevStrong;
    int* peakActive;
    int* peakStart;
    int* peakLocks;
    int* peakCappedBuys;
    int* peakLockHours;
    int* peakUnlockSteps;
    int* peakArmed;
    double* dailyUltraEntryPrice;
    double* dailyUltraPeakPrice;
    double* dailyLockTargetPct;
    double* dailyLockHoldDays;
    double* dailyCoastTargetPct;
    double* dailyCoastMinAssetPct;
    double* dailyCoastMaxAssetPct;
    double* peakMa;
    double* peakIntegral;
    double* peakPrevErr;
    double* peakCap;
    double* peakEdgeStart;
    double* peakEdgeNow;
    double* peakEdgePeak;
    double* peakLockGain;
    double* peakLockGainMax;
    double* peakEdgeVals;
    const double** gradBuyZ;
    const double** gradSellZ;
    WalletState* wallets;
    PhaseState* phases;
    unsigned char* buyFlags;
    unsigned char* sellFlags;
    EvalRow* rows;
} BatchLaneState;

typedef struct{
    const MicroSoa* micro;
    PreparedDatasetState* prepared;
    EvalLaneState lane;
    BatchLaneState batch;
    int ownsPrepared;
} EvalSession;

typedef struct{
    TuneParams param;
} BatchLaneParam;

typedef struct{
    BatchParamsSoa params;
    double grad1BuyZMin[TUNE_BATCH_LANES];
    double grad1SellZMin[TUNE_BATCH_LANES];
    int grad1BuyWinDays[TUNE_BATCH_LANES];
    int grad1SellWinDays[TUNE_BATCH_LANES];
    int phaseBuy[TUNE_BATCH_LANES];
    int phaseSell[TUNE_BATCH_LANES];
    double finalPortionPct[TUNE_BATCH_LANES];
    int cooldown[TUNE_BATCH_LANES];
    double feeRate[TUNE_BATCH_LANES];
    double seedQuote[TUNE_BATCH_LANES];
    double seedAssetPct[TUNE_BATCH_LANES];
    int taxMode[TUNE_BATCH_LANES];
    double annualIncomeBase[TUNE_BATCH_LANES];
    double dailyStrongSellMult[TUNE_BATCH_LANES];
    double dailyStrongTargetPct[TUNE_BATCH_LANES];
    double dailyBridgeDays[TUNE_BATCH_LANES];
    double dailyDownBuyMult[TUNE_BATCH_LANES];
    double dailyCrabAssetCapPct[TUNE_BATCH_LANES];
    double dailyLockTargetPct[TUNE_BATCH_LANES];
    double dailyLockGainPct[TUNE_BATCH_LANES];
    double dailyLockNearHighPct[TUNE_BATCH_LANES];
    int dailyLockMaxDays[TUNE_BATCH_LANES];
    double postUltraCoastTargetPct[TUNE_BATCH_LANES];
    double postUltraGivebackPct[TUNE_BATCH_LANES];
    double postUltraReaccumPct[TUNE_BATCH_LANES];
    double postUltraDoubleTopPct[TUNE_BATCH_LANES];
    double postUltraMaxDays[TUNE_BATCH_LANES];
    double postUltraLockMinAssetPct[TUNE_BATCH_LANES];
    double postUltraLockMaxAssetPct[TUNE_BATCH_LANES];
    double postUltraLockGivebackPct[TUNE_BATCH_LANES];
    double postUltraLockReaccumPct[TUNE_BATCH_LANES];
    double postUltraLockDoubleTopPct[TUNE_BATCH_LANES];
    double postUltraLockMaxDays[TUNE_BATCH_LANES];
    double macroSellRelaxPct[TUNE_BATCH_LANES];
    double peakLockCapPct[TUNE_BATCH_LANES];
    double peakLockUnlockGainPct[TUNE_BATCH_LANES];
    double peakLockReentryStepPct[TUNE_BATCH_LANES];
    double peakLockArmGainPct[TUNE_BATCH_LANES];
    double peakLockGivebackPct[TUNE_BATCH_LANES];
    double peakLockMaxDays[TUNE_BATCH_LANES];
    double peakLockEdgeDrawPct[TUNE_BATCH_LANES];
    double peakLockEdgeSlopeDays[TUNE_BATCH_LANES];
    int peakLockRequireEdgeRisk[TUNE_BATCH_LANES];
    double peakLockMaDays[TUNE_BATCH_LANES];
    double peakLockKp[TUNE_BATCH_LANES];
    double peakLockKi[TUNE_BATCH_LANES];
    double peakLockKd[TUNE_BATCH_LANES];
    double peakLockIntegralDecay[TUNE_BATCH_LANES];
    double peakLockEntryThreshold[TUNE_BATCH_LANES];
    double peakLockExitThreshold[TUNE_BATCH_LANES];
    int peakLockConfirmBars[TUNE_BATCH_LANES];
    double peakLockReleaseTargetPct[TUNE_BATCH_LANES];
    double peakLockUltraGraceDays[TUNE_BATCH_LANES];
} BatchParamChunk;

void* createEvalSession(
    const MicroSoa* micro,
    const MacroSoa* macro
);

void* createPreparedDataset(
    const MicroSoa* micro,
    const MacroSoa* macro
);

void* createEvalSessionPrepared(void* preparedHandle);

void destroyPreparedDataset(void* preparedHandle);

void destroyEvalSession(void* sessionHandle);

/* Return wall-clock seconds with sub-second resolution. */
static double nowSecs(void) {
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + ((double)tv.tv_usec / 1000000.0);
}

/* Clamp a scalar into a fixed range. */
static double clipVal(double value, double low, double high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

/* Return NaN without depending on caller literals. */
static double nanVal(void) {
    return NAN;
}

/* Match the Python CSV metric rounding. */
static double roundMetric(double value) {
    if (!isfinite(value)) {
        return value;
    }
    return round(value * 1000000.0) / 1000000.0;
}

/* Map one tax enum onto its CSV label. */
static const char* taxName(int code) {
    if (code == TAX_INCOME) {
        return "income";
    }
    return "cgt";
}

/* Progressive AU income tax used by the Python reference path. */
static double calcIncomeTax(double income) {
    double taxable = income;

    if (taxable < 0.0) {
        taxable = 0.0;
    }
    if (taxable < 18200.0) {
        return 0.0;
    }
    if (taxable < 45000.0) {
        return (taxable - 18200.0) * 0.16;
    }
    if (taxable < 135000.0) {
        return 4288.0 + ((taxable - 45000.0) * 0.30);
    }
    if (taxable < 190000.0) {
        return 31288.0 + ((taxable - 135000.0) * 0.37);
    }
    return 51638.0 + ((taxable - 190000.0) * 0.45);
}

/* Initialize the compact tuning-time wallet state. */
static int initWallet(
    WalletState* wallet,
    double feeRate,
    double taxRate,
    int taxMode,
    double annualIncomeBase
) {
    wallet->quoteBalance = 0.0;
    wallet->baseBalance = 0.0;
    wallet->feeRate = feeRate;
    wallet->taxRate = taxRate;
    wallet->discountDays = 365;
    wallet->discountRate = 0.50;
    wallet->taxMode = taxMode;
    wallet->annualIncomeBase = annualIncomeBase;
    wallet->taxYearStartMonth = 7;
    wallet->feesPaidQuote = 0.0;
    wallet->realizedGain = 0.0;
    wallet->taxLiability = 0.0;
    wallet->baseIncomeTax = (
        taxMode == TAX_INCOME
        ? calcIncomeTax(annualIncomeBase)
        : 0.0
    );
    wallet->tradeCount = 0;
    wallet->buyTrades = 0;
    wallet->sellTrades = 0;
    wallet->lotCap = 0;
    wallet->lotCount = 0;
    wallet->lots = NULL;

    return 0;
}

/* Free the compact tuning-time wallet state. */
static void freeWallet(WalletState* wallet) {
    free(wallet->lots);
}

/* Apply a full-balance or capped buy to the wallet state. */
static int applyBuy(
    WalletState* wallet,
    int index,
    int64_t tsMs,
    double price,
    double* spendQuote
) {
    double spend;
    double fee;
    double netQuote;
    double qty;

    spend = (
        spendQuote == NULL
        ? wallet->quoteBalance
        : *spendQuote
    );
    if (spend > wallet->quoteBalance) {
        spend = wallet->quoteBalance;
    }
    if (spend <= 0.0 || price <= 0.0) {
        return 0;
    }

    fee = spend * wallet->feeRate;
    netQuote = spend - fee;
    if (netQuote <= 0.0) {
        return 0;
    }
    qty = netQuote / price;
    if (qty <= 0.0) {
        return 0;
    }

    wallet->quoteBalance -= spend;
    wallet->baseBalance += qty;
    wallet->feesPaidQuote += fee;

    wallet->tradeCount += 1;
    wallet->buyTrades += 1;
    (void)tsMs;
    (void)index;
    return 1;
}

/* Apply a full-balance or capped sell to the wallet state. */
static int applySell(
    WalletState* wallet,
    int index,
    int64_t tsMs,
    double price,
    double* qtyIn
) {
    double sellQty;
    double gross;
    double fee;
    double netProceeds;

    sellQty = qtyIn == NULL ? wallet->baseBalance : *qtyIn;
    if (sellQty > wallet->baseBalance) {
        sellQty = wallet->baseBalance;
    }
    if (sellQty <= 0.0 || price <= 0.0) {
        return 0;
    }

    gross = sellQty * price;
    fee = gross * wallet->feeRate;
    netProceeds = gross - fee;

    wallet->baseBalance -= sellQty;
    wallet->quoteBalance += netProceeds;
    wallet->feesPaidQuote += fee;

    wallet->tradeCount += 1;
    wallet->sellTrades += 1;
    (void)tsMs;
    (void)index;
    return 1;
}

/* Return the current portfolio value at one price point. */
static double portfolioValue(const WalletState* wallet, double price) {
    return wallet->quoteBalance + (wallet->baseBalance * price);
}

/* Compute the base phase value when entering a BUY phase. */
static double enterBuyPhase(const WalletState* wallet, int portions) {
    if (portions <= 0) {
        return 0.0;
    }
    return wallet->quoteBalance / (double)portions;
}

/* Compute the base phase value when entering a SELL phase. */
static double enterSellPhase(
    const WalletState* wallet,
    double price,
    int portions
) {
    if (portions <= 0) {
        return 0.0;
    }
    return (wallet->baseBalance * price) / (double)portions;
}

/* BUY phase scaling mirrors the Python helper. */
static double calcBuyScale(int hasLastPrice, double lastPrice, double price) {
    double pctChange;
    double scale;

    if (!hasLastPrice || lastPrice <= 0.0) {
        return 1.0;
    }
    pctChange = (price - lastPrice) / lastPrice;
    scale = 1.0 - pctChange;
    if (scale < 0.0) {
        scale = 0.0;
    }
    return scale;
}

/* SELL phase scaling mirrors the Python helper. */
static double calcSellScale(int hasLastPrice, double lastPrice, double price) {
    double pctChange;
    double scale;

    if (!hasLastPrice || lastPrice <= 0.0) {
        return 1.0;
    }
    pctChange = (price - lastPrice) / lastPrice;
    scale = 1.0 + pctChange;
    if (scale < 0.0) {
        scale = 0.0;
    }
    return scale;
}

/* Apply final-portion logic to a requested portion amount. */
static double cappedPortions(
    double requested,
    int hasInfiniteRemaining,
    double remaining,
    double finalPortionPct
) {
    double fpct = clipVal(finalPortionPct, 0.0, 1.0);
    double use = requested;
    double head;
    double finalRem;
    double finalReq;
    double maxFinalUse;
    double finalUse;

    if (use <= 0.0) {
        return 0.0;
    }
    if (!hasInfiniteRemaining) {
        if (use > remaining) {
            use = remaining;
        }
        if (fpct < 1.0) {
            if (remaining <= 1.0 + 1e-9) {
                if (use > remaining * fpct) {
                    use = remaining * fpct;
                }
            }
            else {
                head = remaining - 1.0;
                if (head < 0.0) {
                    head = 0.0;
                }
                if (use > head + 1e-9) {
                    finalRem = remaining - head;
                    if (finalRem < 0.0) {
                        finalRem = 0.0;
                    }
                    finalReq = use - head;
                    maxFinalUse = finalRem * fpct;
                    finalUse = finalReq;
                    if (finalUse > maxFinalUse) {
                        finalUse = maxFinalUse;
                    }
                    use = head + finalUse;
                }
            }
        }
    }
    return use;
}

/* Execute a scaled BUY trade against the phase state. */
static int applyScaledBuy(
    WalletState* wallet,
    PhaseState* phase,
    int index,
    int64_t tsMs,
    double price,
    double scale,
    double maxSpendValue
) {
    double requestedValue;
    double reqPortions;
    double usePortions;
    double spend;
    int traded;

    if (phase->baseValue <= 0.0 || scale <= 0.0) {
        return 0;
    }
    requestedValue = phase->baseValue * scale;
    if (requestedValue <= 0.0) {
        return 0;
    }
    reqPortions = requestedValue / phase->baseValue;
    usePortions = cappedPortions(
        reqPortions,
        phase->hasInfiniteRemaining,
        phase->portionsRemaining,
        phase->finalPortionPct
    );
    spend = phase->baseValue * usePortions;
    if (maxSpendValue >= 0.0 && spend > maxSpendValue) {
        spend = maxSpendValue;
    }
    if (spend > wallet->quoteBalance) {
        spend = wallet->quoteBalance;
    }
    if (spend <= 0.0) {
        return 0;
    }
    traded = applyBuy(wallet, index, tsMs, price, &spend);
    if (!traded) {
        return 0;
    }
    phase->lastPrice = price;
    phase->hasLastPrice = 1;
    if (!phase->hasInfiniteRemaining) {
        phase->portionsRemaining -= usePortions;
        if (phase->portionsRemaining < 0.0) {
            phase->portionsRemaining = 0.0;
        }
    }
    return 1;
}

/* Return the quote spend needed to reach one asset allocation target. */
static double buySpendToTargetPct(
    const WalletState* wallet,
    double price,
    double targetPct
) {
    double targetLocal = clipVal(targetPct, 0.0, 1.0);
    double assetValue = wallet->baseBalance * price;
    double totalValue = assetValue + wallet->quoteBalance;
    double needValue;
    double denom;
    double spend;

    if (
        targetLocal <= 0.0
        || price <= 0.0
        || wallet->quoteBalance <= 0.0
        || totalValue <= 0.0
    ) {
        return 0.0;
    }
    needValue = (targetLocal * totalValue) - assetValue;
    if (needValue <= 0.0) {
        return 0.0;
    }
    denom = 1.0 - wallet->feeRate + (targetLocal * wallet->feeRate);
    if (denom <= 1e-12) {
        return 0.0;
    }
    spend = needValue / denom;
    return clipVal(spend, 0.0, wallet->quoteBalance);
}

/* Execute a scaled SELL trade against the phase state. */
static int applyScaledSell(
    WalletState* wallet,
    PhaseState* phase,
    int index,
    int64_t tsMs,
    double price,
    double scale,
    double maxSellValue
) {
    double targetValue;
    double reqPortions;
    double usePortions;
    double maxValue;
    double qty;
    int traded;

    if (
        phase->baseValue <= 0.0
        || price <= 0.0
        || scale <= 0.0
    ) {
        return 0;
    }
    targetValue = phase->baseValue * scale;
    if (targetValue <= 0.0) {
        return 0;
    }
    reqPortions = targetValue / phase->baseValue;
    usePortions = cappedPortions(
        reqPortions,
        phase->hasInfiniteRemaining,
        phase->portionsRemaining,
        phase->finalPortionPct
    );
    maxValue = phase->baseValue * usePortions;
    if (maxValue > maxSellValue) {
        maxValue = maxSellValue;
    }
    if (maxValue > wallet->baseBalance * price) {
        maxValue = wallet->baseBalance * price;
    }
    if (maxValue <= 0.0) {
        return 0;
    }
    qty = maxValue / price;
    traded = applySell(wallet, index, tsMs, price, &qty);
    if (!traded) {
        return 0;
    }
    phase->lastPrice = price;
    phase->hasLastPrice = 1;
    if (!phase->hasInfiniteRemaining) {
        phase->portionsRemaining -= usePortions;
        if (phase->portionsRemaining < 0.0) {
            phase->portionsRemaining = 0.0;
        }
    }
    return 1;
}

/* Return the max gross SELL value that preserves one asset floor. */
static double floorSellValueCap(
    const WalletState* wallet,
    double price,
    double floorPct
) {
    double floorLocal = clipVal(floorPct, 0.0, 1.0);
    double assetValue = wallet->baseBalance * price;
    double totalValue = assetValue + wallet->quoteBalance;
    double denom;
    double cap;

    if (floorLocal <= 0.0) {
        return assetValue;
    }
    denom = 1.0 - (floorLocal * wallet->feeRate);
    if (denom <= 1e-12) {
        return 0.0;
    }
    cap = (assetValue - (floorLocal * totalValue)) / denom;
    return clipVal(cap, 0.0, assetValue);
}

/* Buy only enough quote to reach one asset allocation target. */
static int buyToTargetPct(
    WalletState* wallet,
    int index,
    int64_t tsMs,
    double price,
    double targetPct
) {
    double spend;

    spend = buySpendToTargetPct(wallet, price, targetPct);
    if (spend <= 0.0) {
        return 0;
    }
    return applyBuy(wallet, index, tsMs, price, &spend);
}

/* Estimate bars/day from raw candle timestamps. */
static double barsPerDayFromTs(const int64_t* ts, int n) {
    int i;
    double diffMs = 0.0;
    const double dayMs = 24.0 * 60.0 * 60.0 * 1000.0;

    if (ts == NULL || n <= 1) {
        return 96.0;
    }

    for (i = 1; i < n; i++) {
        if (ts[i] > ts[i - 1]) {
            diffMs = (double)(ts[i] - ts[i - 1]);
            break;
        }
    }

    if (diffMs <= 0.0) {
        return 96.0;
    }

    return dayMs / diffMs;
}

/* Allocate reusable scratch arrays for one flag pass. */
static int initScratch(FlagScratch* scratch, int n) {
    scratch->meanA = NULL;
    scratch->stdA = NULL;
    scratch->meanB = NULL;
    scratch->stdB = NULL;

    scratch->meanA = (double*)malloc((size_t)n * sizeof(double));
    scratch->stdA = (double*)malloc((size_t)n * sizeof(double));
    scratch->meanB = (double*)malloc((size_t)n * sizeof(double));
    scratch->stdB = (double*)malloc((size_t)n * sizeof(double));

    if (
        scratch->meanA == NULL
        || scratch->stdA == NULL
        || scratch->meanB == NULL
        || scratch->stdB == NULL
    ) {
        free(scratch->meanA);
        free(scratch->stdA);
        free(scratch->meanB);
        free(scratch->stdB);
        return -1;
    }

    return 0;
}

/* Free reusable scratch arrays. */
static void freeScratch(FlagScratch* scratch) {
    free(scratch->meanA);
    free(scratch->stdA);
    free(scratch->meanB);
    free(scratch->stdB);
}

/* Allocate reusable per-row arrays once for one evaluation group. */
static int initRowScratch(RowScratch* scratch, int n) {
    int retN = n > 1 ? (n - 1) : 1;

    scratch->buyFlags = NULL;
    scratch->sellFlags = NULL;
    scratch->curveSim = NULL;
    scratch->retsSim = NULL;
    scratch->edgeRets = NULL;
    scratch->edgeVals = NULL;

    scratch->buyFlags = (unsigned char*)malloc(
        (size_t)n * sizeof(unsigned char)
    );
    scratch->sellFlags = (unsigned char*)malloc(
        (size_t)n * sizeof(unsigned char)
    );
    scratch->curveSim = (double*)malloc((size_t)n * sizeof(double));
    scratch->retsSim = (double*)malloc((size_t)retN * sizeof(double));
    scratch->edgeRets = (double*)malloc((size_t)retN * sizeof(double));
    scratch->edgeVals = (double*)malloc((size_t)n * sizeof(double));

    if (
        scratch->buyFlags == NULL
        || scratch->sellFlags == NULL
        || scratch->curveSim == NULL
        || scratch->retsSim == NULL
        || scratch->edgeRets == NULL
        || scratch->edgeVals == NULL
    ) {
        free(scratch->buyFlags);
        free(scratch->sellFlags);
        free(scratch->curveSim);
        free(scratch->retsSim);
        free(scratch->edgeRets);
        free(scratch->edgeVals);
        return -1;
    }

    return 0;
}

/* Free reusable per-row arrays. */
static void freeRowScratch(RowScratch* scratch) {
    free(scratch->buyFlags);
    free(scratch->sellFlags);
    free(scratch->curveSim);
    free(scratch->retsSim);
    free(scratch->edgeRets);
    free(scratch->edgeVals);
}

/* Allocate reusable rolling-metric arrays once for one evaluation group. */
static int initMetricScratch(MetricScratch* scratch, int n) {
    int retN = n > 1 ? (n - 1) : 1;

    scratch->cs = NULL;
    scratch->cs2 = NULL;
    scratch->csNeg = NULL;
    scratch->cs2Neg = NULL;
    scratch->csCntNeg = NULL;
    scratch->sharpeVals = NULL;
    scratch->sortinoVals = NULL;

    scratch->cs = (double*)malloc((size_t)(retN + 1) * sizeof(double));
    scratch->cs2 = (double*)malloc((size_t)(retN + 1) * sizeof(double));
    scratch->csNeg = (double*)malloc((size_t)(retN + 1) * sizeof(double));
    scratch->cs2Neg = (double*)malloc((size_t)(retN + 1) * sizeof(double));
    scratch->csCntNeg = (double*)malloc((size_t)(retN + 1) * sizeof(double));
    scratch->sharpeVals = (double*)malloc((size_t)retN * sizeof(double));
    scratch->sortinoVals = (double*)malloc((size_t)retN * sizeof(double));

    if (
        scratch->cs == NULL
        || scratch->cs2 == NULL
        || scratch->csNeg == NULL
        || scratch->cs2Neg == NULL
        || scratch->csCntNeg == NULL
        || scratch->sharpeVals == NULL
        || scratch->sortinoVals == NULL
    ) {
        free(scratch->cs);
        free(scratch->cs2);
        free(scratch->csNeg);
        free(scratch->cs2Neg);
        free(scratch->csCntNeg);
        free(scratch->sharpeVals);
        free(scratch->sortinoVals);
        return -1;
    }

    return 0;
}

/* Free reusable rolling-metric arrays. */
static void freeMetricScratch(MetricScratch* scratch) {
    free(scratch->cs);
    free(scratch->cs2);
    free(scratch->csNeg);
    free(scratch->cs2Neg);
    free(scratch->csCntNeg);
    free(scratch->sharpeVals);
    free(scratch->sortinoVals);
}

/* Zero one lazy z-series cache. */
static void initZSeriesCache(ZSeriesCache* cache) {
    cache->windows = NULL;
    cache->series = NULL;
    cache->count = 0;
    cache->cap = 0;
}

/* Free one lazy z-series cache and all cached arrays. */
static void freeZSeriesCache(ZSeriesCache* cache) {
    int i;

    for (i = 0; i < cache->count; i++) {
        free(cache->series[i]);
    }
    free(cache->windows);
    free(cache->series);
}

/* Zero all z-series caches for one evaluation group. */
static void initZCacheSet(ZCacheSet* cache) {
    initZSeriesCache(&cache->grad);
}

/* Free all z-series caches for one evaluation group. */
static void freeZCacheSet(ZCacheSet* cache) {
    freeZSeriesCache(&cache->grad);
}

/* Forward declare the z-series builder for lazy cache fills. */
static void buildZSeries(
    const double* series,
    int n,
    int window,
    double* meanArr,
    double* stdArr,
    double* out
);

/* Build one z-series once per family/window and reuse it across rows. */
static const double* cachedZSeries(
    ZSeriesCache* cache,
    const double* series,
    int n,
    int window,
    double* meanArr,
    double* stdArr
) {
    int i;
    int newCap;
    int* grownWindows;
    double** grownSeries;
    double* out;

    for (i = 0; i < cache->count; i++) {
        if (cache->windows[i] == window) {
            return cache->series[i];
        }
    }

    if (cache->count >= cache->cap) {
        newCap = cache->cap <= 0 ? 4 : cache->cap * 2;
        grownWindows = (int*)realloc(
            cache->windows,
            (size_t)newCap * sizeof(int)
        );
        if (grownWindows == NULL) {
            return NULL;
        }
        cache->windows = grownWindows;

        grownSeries = (double**)realloc(
            cache->series,
            (size_t)newCap * sizeof(double*)
        );
        if (grownSeries == NULL) {
            return NULL;
        }
        cache->series = grownSeries;
        cache->cap = newCap;
    }

    out = (double*)malloc((size_t)n * sizeof(double));
    if (out == NULL) {
        return NULL;
    }

    buildZSeries(series, n, window, meanArr, stdArr, out);
    cache->windows[cache->count] = window;
    cache->series[cache->count] = out;
    cache->count += 1;
    return out;
}

/* Allocate micro-derived work arrays. */
static int initMicroDerived(MicroDerived* derived, int n) {
    derived->barsPerDay = 96.0;
    derived->ema1 = NULL;
    derived->ema2 = NULL;
    derived->ema3 = NULL;
    derived->g1p1 = NULL;
    derived->trend = NULL;

    derived->ema1 = (double*)malloc((size_t)n * sizeof(double));
    derived->ema2 = (double*)malloc((size_t)n * sizeof(double));
    derived->ema3 = (double*)malloc((size_t)n * sizeof(double));
    derived->g1p1 = (double*)malloc((size_t)n * sizeof(double));
    derived->trend = (int*)malloc((size_t)n * sizeof(int));

    if (
        derived->ema1 == NULL
        || derived->ema2 == NULL
        || derived->ema3 == NULL
        || derived->g1p1 == NULL
        || derived->trend == NULL
    ) {
        free(derived->ema1);
        free(derived->ema2);
        free(derived->ema3);
        free(derived->g1p1);
        free(derived->trend);
        return -1;
    }

    return 0;
}

/* Free micro-derived work arrays. */
static void freeMicroDerived(MicroDerived* derived) {
    free(derived->ema1);
    free(derived->ema2);
    free(derived->ema3);
    free(derived->g1p1);
    free(derived->trend);
}

/* Allocate aligned macro arrays. */
static int initMacroDerived(MacroDerived* derived, int n) {
    derived->dyn = NULL;
    derived->dir = NULL;
    derived->mom = NULL;

    derived->dyn = (double*)malloc((size_t)n * sizeof(double));
    derived->dir = (int*)malloc((size_t)n * sizeof(int));
    derived->mom = (int*)malloc((size_t)n * sizeof(int));

    if (
        derived->dyn == NULL
        || derived->dir == NULL
        || derived->mom == NULL
    ) {
        free(derived->dyn);
        free(derived->dir);
        free(derived->mom);
        return -1;
    }

    return 0;
}

/* Free aligned macro arrays. */
static void freeMacroDerived(MacroDerived* derived) {
    free(derived->dyn);
    free(derived->dir);
    free(derived->mom);
}

/* Build a clipped rolling z-score series with NaN for invalid slots. */
static void buildZSeries(
    const double* series,
    int n,
    int window,
    double* meanArr,
    double* stdArr,
    double* out
) {
    int i;
    double zVal;

    rollingMeanAndStd(series, n, window, meanArr, stdArr);

    for (i = 0; i < n; i++) {
        out[i] = NAN;
        if (
            isnan(meanArr[i])
            || isnan(stdArr[i])
            || stdArr[i] <= 1e-6
        ) {
            continue;
        }
        zVal = (series[i] - meanArr[i]) / stdArr[i];
        out[i] = clipVal(zVal, -10.0, 10.0);
    }
}

/* Build micro math arrays from raw closes inside C. */
static void buildMicroDerived(
    const MicroSoa* micro,
    MicroDerived* derived
) {
    derived->barsPerDay = barsPerDayFromTs(micro->ts, micro->n);

    emaLpf(micro->closes, micro->n, micro->p1, derived->ema1);
    emaLpf(micro->closes, micro->n, micro->p2, derived->ema2);
    emaLpf(micro->closes, micro->n, micro->p3, derived->ema3);

    grad1Series(derived->ema1, micro->n, 100.0, derived->g1p1);
    trendCodes(
        derived->ema1,
        derived->ema2,
        derived->ema3,
        micro->n,
        derived->trend
    );
}

/* Align macro floats onto micro timestamps using last-known sample. */
static void alignDoubleSeries(
    const int64_t* tsMacro,
    const double* src,
    int nMacro,
    const int64_t* tsMicro,
    int nMicro,
    double* out
) {
    int i;
    int j = 0;
    int last = nMacro - 1;

    if (nMacro <= 0 || src == NULL || tsMacro == NULL) {
        for (i = 0; i < nMicro; i++) {
            out[i] = 0.0;
        }
        return;
    }

    for (i = 0; i < nMicro; i++) {
        while (j < last && tsMacro[j + 1] <= tsMicro[i]) {
            j += 1;
        }
        out[i] = src[j];
    }
}

/* Align one macro int series onto micro timestamps using last-known sample. */
static void alignIntSeries(
    const int64_t* tsMacro,
    const int* src,
    int nMacro,
    const int64_t* tsMicro,
    int nMicro,
    int* out
) {
    int i;
    int j = 0;
    int last = nMacro - 1;

    if (nMacro <= 0 || src == NULL || tsMacro == NULL) {
        for (i = 0; i < nMicro; i++) {
            out[i] = 0;
        }
        return;
    }

    for (i = 0; i < nMicro; i++) {
        while (j < last && tsMacro[j + 1] <= tsMicro[i]) {
            j += 1;
        }
        out[i] = src[j];
    }
}

/* Build aligned macro dyn/dir/mom arrays from raw macro closes inside C. */
static int buildMacroDerived(
    const MacroSoa* macro,
    const MicroSoa* micro,
    MacroDerived* derived
) {
    int i;
    double barsPerDay;
    double* ema1 = NULL;
    double* ema2 = NULL;
    double* ema3 = NULL;
    double* dynRaw = NULL;
    int* dirRaw = NULL;
    int* momRaw = NULL;

    if (
        macro == NULL
        || macro->n <= 0
        || macro->closes == NULL
        || macro->ts == NULL
    ) {
        for (i = 0; i < micro->n; i++) {
            derived->dyn[i] = 0.0;
            derived->dir[i] = 0;
            derived->mom[i] = 0;
        }
        return 0;
    }

    ema1 = (double*)malloc((size_t)macro->n * sizeof(double));
    ema2 = (double*)malloc((size_t)macro->n * sizeof(double));
    ema3 = (double*)malloc((size_t)macro->n * sizeof(double));
    dynRaw = (double*)malloc((size_t)macro->n * sizeof(double));
    dirRaw = (int*)malloc((size_t)macro->n * sizeof(int));
    momRaw = (int*)malloc((size_t)macro->n * sizeof(int));

    if (
        ema1 == NULL
        || ema2 == NULL
        || ema3 == NULL
        || dynRaw == NULL
        || dirRaw == NULL
        || momRaw == NULL
    ) {
        free(ema1);
        free(ema2);
        free(ema3);
        free(dynRaw);
        free(dirRaw);
        free(momRaw);
        return -1;
    }

    barsPerDay = barsPerDayFromTs(macro->ts, macro->n);
    emaLpf(macro->closes, macro->n, macro->p1, ema1);
    emaLpf(macro->closes, macro->n, macro->p2, ema2);
    emaLpf(macro->closes, macro->n, macro->p3, ema3);

    macroDynFromMas(
        ema1,
        ema2,
        ema3,
        macro->n,
        barsPerDay,
        macro->dynWinDays,
        macro->dynZMin,
        macro->dynZMax,
        macro->dynPctMax,
        macro->dynPctMin,
        macro->gradWinDays,
        macro->gradZMin,
        macro->gradZMax,
        macro->gradMultMin,
        macro->gradMultMax,
        dynRaw
    );

    for (i = 0; i < macro->n; i++) {
        dirRaw[i] = 0;
        momRaw[i] = 0;
        if (ema1[i] > ema3[i]) {
            dirRaw[i] = 1;
        }
        else if (ema1[i] < ema3[i]) {
            dirRaw[i] = -1;
        }
        if (ema1[i] > ema2[i]) {
            momRaw[i] = 1;
        }
        else if (ema1[i] < ema2[i]) {
            momRaw[i] = -1;
        }
    }

    alignDoubleSeries(
        macro->ts,
        dynRaw,
        macro->n,
        micro->ts,
        micro->n,
        derived->dyn
    );
    alignIntSeries(
        macro->ts,
        dirRaw,
        macro->n,
        micro->ts,
        micro->n,
        derived->dir
    );
    alignIntSeries(
        macro->ts,
        momRaw,
        macro->n,
        micro->ts,
        micro->n,
        derived->mom
    );

    free(ema1);
    free(ema2);
    free(ema3);
    free(dynRaw);
    free(dirRaw);
    free(momRaw);
    return 0;
}

/* Build per-side phase anchor indices from allowed regime runs. */
static void buildAnchors(const int* allowReg, int n, int* out) {
    int i;
    int lastAnchor = -1;
    int prevAllow = 0;

    for (i = 0; i < n; i++) {
        if (allowReg[i] && !prevAllow) {
            lastAnchor = i;
        }
        out[i] = lastAnchor;
        prevAllow = allowReg[i];
    }
}

/* Prepare all dataset-derived arrays once for one dataset pair. */
static int initPreparedDataset(
    const MicroSoa* micro,
    const MacroSoa* macro,
    PreparedDatasetState* state
) {
    int i;
    int n = micro->n;

    state->micro = micro;
    state->allowBuy = NULL;
    state->allowSell = NULL;
    state->buyAnchors = NULL;
    state->sellAnchors = NULL;
    initZCacheSet(&state->zCache);

    state->allowBuy = (int*)malloc((size_t)n * sizeof(int));
    state->allowSell = (int*)malloc((size_t)n * sizeof(int));
    state->buyAnchors = (int*)malloc((size_t)n * sizeof(int));
    state->sellAnchors = (int*)malloc((size_t)n * sizeof(int));

    if (
        state->allowBuy == NULL
        || state->allowSell == NULL
        || state->buyAnchors == NULL
        || state->sellAnchors == NULL
    ) {
        free(state->allowBuy);
        free(state->allowSell);
        free(state->buyAnchors);
        free(state->sellAnchors);
        return -1;
    }

    if (initScratch(&state->scratch, n) != 0) {
        free(state->allowBuy);
        free(state->allowSell);
        free(state->buyAnchors);
        free(state->sellAnchors);
        return -1;
    }

    if (initMicroDerived(&state->microDerived, n) != 0) {
        freeScratch(&state->scratch);
        freeZCacheSet(&state->zCache);
        free(state->allowBuy);
        free(state->allowSell);
        free(state->buyAnchors);
        free(state->sellAnchors);
        return -1;
    }

    if (initMacroDerived(&state->macroDerived, n) != 0) {
        freeMicroDerived(&state->microDerived);
        freeScratch(&state->scratch);
        freeZCacheSet(&state->zCache);
        free(state->allowBuy);
        free(state->allowSell);
        free(state->buyAnchors);
        free(state->sellAnchors);
        return -1;
    }

    buildMicroDerived(micro, &state->microDerived);
    if (buildMacroDerived(macro, micro, &state->macroDerived) != 0) {
        freeMacroDerived(&state->macroDerived);
        freeMicroDerived(&state->microDerived);
        freeScratch(&state->scratch);
        freeZCacheSet(&state->zCache);
        free(state->allowBuy);
        free(state->allowSell);
        free(state->buyAnchors);
        free(state->sellAnchors);
        return -1;
    }
    for (i = 0; i < n; i++) {
        state->allowBuy[i] = state->microDerived.trend[i] == -1;
        state->allowSell[i] = state->microDerived.trend[i] == 1;
    }

    buildAnchors(state->allowBuy, n, state->buyAnchors);
    buildAnchors(state->allowSell, n, state->sellAnchors);
    return 0;
}

/* Free all dataset-derived arrays for one dataset pair. */
static void freePreparedDataset(PreparedDatasetState* state) {
    freeMacroDerived(&state->macroDerived);
    freeMicroDerived(&state->microDerived);
    freeScratch(&state->scratch);
    freeZCacheSet(&state->zCache);
    free(state->allowBuy);
    free(state->allowSell);
    free(state->buyAnchors);
    free(state->sellAnchors);
}

/* Allocate per-lane scratch reused across scalar reference evaluations. */
static int initEvalLane(EvalLaneState* lane, int n) {
    if (initRowScratch(&lane->rowScratch, n) != 0) {
        return -1;
    }
    if (initMetricScratch(&lane->metricScratch, n) != 0) {
        freeRowScratch(&lane->rowScratch);
        return -1;
    }
    return 0;
}

/* Free one lane scratch bundle. */
static void freeEvalLane(EvalLaneState* lane) {
    freeMetricScratch(&lane->metricScratch);
    freeRowScratch(&lane->rowScratch);
}

/* Allocate reusable batch scratch for one fixed lane cap. */
static int initBatchLane(BatchLaneState* batch, int n, int cap) {
    batch->cap = cap;
    batch->n = n;
    batch->cooldown = NULL;
    batch->lastTrendCode = NULL;
    batch->lastSignalSellIndex = NULL;
    batch->buyLastCd = NULL;
    batch->sellLastCd = NULL;
    batch->buyLastAccepted = NULL;
    batch->sellLastAccepted = NULL;
    batch->buyLastPhase = NULL;
    batch->sellLastPhase = NULL;
    batch->buyCounts = NULL;
    batch->sellCounts = NULL;
    batch->benchQty = NULL;
    batch->curveSim = NULL;
    batch->dailyStrongDays = NULL;
    batch->dailyLockActive = NULL;
    batch->dailyLockStart = NULL;
    batch->dailyPrevStrong = NULL;
    batch->dailyPrevDown = NULL;
    batch->dailyEpisodeLocked = NULL;
    batch->dailyBridgeBars = NULL;
    batch->dailyLockReleaseOnStrong = NULL;
    batch->dailyCoastActive = NULL;
    batch->dailyCoastStart = NULL;
    batch->peakLong = NULL;
    batch->peakBearCount = NULL;
    batch->peakStrongGraceBars = NULL;
    batch->peakStrongReleases = NULL;
    batch->peakPrevStrong = NULL;
    batch->peakActive = NULL;
    batch->peakStart = NULL;
    batch->peakLocks = NULL;
    batch->peakCappedBuys = NULL;
    batch->peakLockHours = NULL;
    batch->peakUnlockSteps = NULL;
    batch->peakArmed = NULL;
    batch->dailyUltraEntryPrice = NULL;
    batch->dailyUltraPeakPrice = NULL;
    batch->dailyLockTargetPct = NULL;
    batch->dailyLockHoldDays = NULL;
    batch->dailyCoastTargetPct = NULL;
    batch->dailyCoastMinAssetPct = NULL;
    batch->dailyCoastMaxAssetPct = NULL;
    batch->peakMa = NULL;
    batch->peakIntegral = NULL;
    batch->peakPrevErr = NULL;
    batch->peakCap = NULL;
    batch->peakEdgeStart = NULL;
    batch->peakEdgeNow = NULL;
    batch->peakEdgePeak = NULL;
    batch->peakLockGain = NULL;
    batch->peakLockGainMax = NULL;
    batch->peakEdgeVals = NULL;
    batch->gradBuyZ = NULL;
    batch->gradSellZ = NULL;
    batch->wallets = NULL;
    batch->phases = NULL;
    batch->buyFlags = NULL;
    batch->sellFlags = NULL;
    batch->rows = NULL;

    batch->cooldown = (int*)malloc((size_t)cap * sizeof(int));
    batch->lastTrendCode = (int*)malloc((size_t)cap * sizeof(int));
    batch->lastSignalSellIndex = (int*)malloc((size_t)cap * sizeof(int));
    batch->buyLastCd = (int*)malloc((size_t)cap * sizeof(int));
    batch->sellLastCd = (int*)malloc((size_t)cap * sizeof(int));
    batch->buyLastAccepted = (int*)malloc((size_t)cap * sizeof(int));
    batch->sellLastAccepted = (int*)malloc((size_t)cap * sizeof(int));
    batch->buyLastPhase = (int*)malloc((size_t)cap * sizeof(int));
    batch->sellLastPhase = (int*)malloc((size_t)cap * sizeof(int));
    batch->buyCounts = (int*)malloc((size_t)cap * sizeof(int));
    batch->sellCounts = (int*)malloc((size_t)cap * sizeof(int));
    batch->benchQty = (double*)malloc((size_t)cap * sizeof(double));
    batch->dailyStrongDays = (int*)malloc((size_t)cap * sizeof(int));
    batch->dailyLockActive = (int*)malloc((size_t)cap * sizeof(int));
    batch->dailyLockStart = (int*)malloc((size_t)cap * sizeof(int));
    batch->dailyPrevStrong = (int*)malloc((size_t)cap * sizeof(int));
    batch->dailyPrevDown = (int*)malloc((size_t)cap * sizeof(int));
    batch->dailyEpisodeLocked = (int*)malloc((size_t)cap * sizeof(int));
    batch->dailyBridgeBars = (int*)malloc((size_t)cap * sizeof(int));
    batch->dailyLockReleaseOnStrong = (int*)malloc(
        (size_t)cap * sizeof(int)
    );
    batch->dailyCoastActive = (int*)malloc((size_t)cap * sizeof(int));
    batch->dailyCoastStart = (int*)malloc((size_t)cap * sizeof(int));
    batch->peakLong = (int*)malloc((size_t)cap * sizeof(int));
    batch->peakBearCount = (int*)malloc((size_t)cap * sizeof(int));
    batch->peakStrongGraceBars = (int*)malloc((size_t)cap * sizeof(int));
    batch->peakStrongReleases = (int*)malloc((size_t)cap * sizeof(int));
    batch->peakPrevStrong = (int*)malloc((size_t)cap * sizeof(int));
    batch->peakActive = (int*)malloc((size_t)cap * sizeof(int));
    batch->peakStart = (int*)malloc((size_t)cap * sizeof(int));
    batch->peakLocks = (int*)malloc((size_t)cap * sizeof(int));
    batch->peakCappedBuys = (int*)malloc((size_t)cap * sizeof(int));
    batch->peakLockHours = (int*)malloc((size_t)cap * sizeof(int));
    batch->peakUnlockSteps = (int*)malloc((size_t)cap * sizeof(int));
    batch->peakArmed = (int*)malloc((size_t)cap * sizeof(int));
    batch->dailyUltraEntryPrice = (double*)malloc(
        (size_t)cap * sizeof(double)
    );
    batch->dailyUltraPeakPrice = (double*)malloc(
        (size_t)cap * sizeof(double)
    );
    batch->dailyLockTargetPct = (double*)malloc(
        (size_t)cap * sizeof(double)
    );
    batch->dailyLockHoldDays = (double*)malloc(
        (size_t)cap * sizeof(double)
    );
    batch->dailyCoastTargetPct = (double*)malloc(
        (size_t)cap * sizeof(double)
    );
    batch->dailyCoastMinAssetPct = (double*)malloc(
        (size_t)cap * sizeof(double)
    );
    batch->dailyCoastMaxAssetPct = (double*)malloc(
        (size_t)cap * sizeof(double)
    );
    batch->peakMa = (double*)malloc((size_t)cap * sizeof(double));
    batch->peakIntegral = (double*)malloc((size_t)cap * sizeof(double));
    batch->peakPrevErr = (double*)malloc((size_t)cap * sizeof(double));
    batch->peakCap = (double*)malloc((size_t)cap * sizeof(double));
    batch->peakEdgeStart = (double*)malloc((size_t)cap * sizeof(double));
    batch->peakEdgeNow = (double*)malloc((size_t)cap * sizeof(double));
    batch->peakEdgePeak = (double*)malloc((size_t)cap * sizeof(double));
    batch->peakLockGain = (double*)malloc((size_t)cap * sizeof(double));
    batch->peakLockGainMax = (double*)malloc((size_t)cap * sizeof(double));
    batch->peakEdgeVals = (double*)malloc(
        (size_t)n * (size_t)cap * sizeof(double)
    );
    batch->curveSim = (double*)malloc(
        (size_t)n * (size_t)cap * sizeof(double)
    );
    batch->gradBuyZ = (const double**)malloc(
        (size_t)cap * sizeof(const double*)
    );
    batch->gradSellZ = (const double**)malloc(
        (size_t)cap * sizeof(const double*)
    );
    batch->wallets = (WalletState*)malloc(
        (size_t)cap * sizeof(WalletState)
    );
    batch->phases = (PhaseState*)malloc((size_t)cap * sizeof(PhaseState));
    batch->buyFlags = (unsigned char*)malloc(
        (size_t)n * (size_t)cap * sizeof(unsigned char)
    );
    batch->sellFlags = (unsigned char*)malloc(
        (size_t)n * (size_t)cap * sizeof(unsigned char)
    );
    batch->rows = (EvalRow*)malloc((size_t)cap * sizeof(EvalRow));

    if (
        batch->cooldown == NULL
        || batch->lastTrendCode == NULL
        || batch->lastSignalSellIndex == NULL
        || batch->buyLastCd == NULL
        || batch->sellLastCd == NULL
        || batch->buyLastAccepted == NULL
        || batch->sellLastAccepted == NULL
        || batch->buyLastPhase == NULL
        || batch->sellLastPhase == NULL
        || batch->buyCounts == NULL
        || batch->sellCounts == NULL
        || batch->benchQty == NULL
        || batch->dailyStrongDays == NULL
        || batch->dailyLockActive == NULL
        || batch->dailyLockStart == NULL
        || batch->dailyPrevStrong == NULL
        || batch->dailyPrevDown == NULL
        || batch->dailyEpisodeLocked == NULL
        || batch->dailyBridgeBars == NULL
        || batch->dailyLockReleaseOnStrong == NULL
        || batch->dailyCoastActive == NULL
        || batch->dailyCoastStart == NULL
        || batch->peakLong == NULL
        || batch->peakBearCount == NULL
        || batch->peakStrongGraceBars == NULL
        || batch->peakStrongReleases == NULL
        || batch->peakPrevStrong == NULL
        || batch->peakActive == NULL
        || batch->peakStart == NULL
        || batch->peakLocks == NULL
        || batch->peakCappedBuys == NULL
        || batch->peakLockHours == NULL
        || batch->peakUnlockSteps == NULL
        || batch->peakArmed == NULL
        || batch->dailyUltraEntryPrice == NULL
        || batch->dailyUltraPeakPrice == NULL
        || batch->dailyLockTargetPct == NULL
        || batch->dailyLockHoldDays == NULL
        || batch->dailyCoastTargetPct == NULL
        || batch->dailyCoastMinAssetPct == NULL
        || batch->dailyCoastMaxAssetPct == NULL
        || batch->peakMa == NULL
        || batch->peakIntegral == NULL
        || batch->peakPrevErr == NULL
        || batch->peakCap == NULL
        || batch->peakEdgeStart == NULL
        || batch->peakEdgeNow == NULL
        || batch->peakEdgePeak == NULL
        || batch->peakLockGain == NULL
        || batch->peakLockGainMax == NULL
        || batch->peakEdgeVals == NULL
        || batch->curveSim == NULL
        || batch->gradBuyZ == NULL
        || batch->gradSellZ == NULL
        || batch->wallets == NULL
        || batch->phases == NULL
        || batch->buyFlags == NULL
        || batch->sellFlags == NULL
        || batch->rows == NULL
    ) {
        free(batch->cooldown);
        free(batch->lastTrendCode);
        free(batch->lastSignalSellIndex);
        free(batch->buyLastCd);
        free(batch->sellLastCd);
        free(batch->buyLastAccepted);
        free(batch->sellLastAccepted);
        free(batch->buyLastPhase);
        free(batch->sellLastPhase);
        free(batch->buyCounts);
        free(batch->sellCounts);
        free(batch->benchQty);
        free(batch->dailyStrongDays);
        free(batch->dailyLockActive);
        free(batch->dailyLockStart);
        free(batch->dailyPrevStrong);
        free(batch->dailyPrevDown);
        free(batch->dailyEpisodeLocked);
        free(batch->dailyBridgeBars);
        free(batch->dailyLockReleaseOnStrong);
        free(batch->dailyCoastActive);
        free(batch->dailyCoastStart);
        free(batch->peakLong);
        free(batch->peakBearCount);
        free(batch->peakStrongGraceBars);
        free(batch->peakStrongReleases);
        free(batch->peakPrevStrong);
        free(batch->peakActive);
        free(batch->peakStart);
        free(batch->peakLocks);
        free(batch->peakCappedBuys);
        free(batch->peakLockHours);
        free(batch->peakUnlockSteps);
        free(batch->peakArmed);
        free(batch->dailyUltraEntryPrice);
        free(batch->dailyUltraPeakPrice);
        free(batch->dailyLockTargetPct);
        free(batch->dailyLockHoldDays);
        free(batch->dailyCoastTargetPct);
        free(batch->dailyCoastMinAssetPct);
        free(batch->dailyCoastMaxAssetPct);
        free(batch->peakMa);
        free(batch->peakIntegral);
        free(batch->peakPrevErr);
        free(batch->peakCap);
        free(batch->peakEdgeStart);
        free(batch->peakEdgeNow);
        free(batch->peakEdgePeak);
        free(batch->peakLockGain);
        free(batch->peakLockGainMax);
        free(batch->peakEdgeVals);
        free(batch->curveSim);
        free(batch->gradBuyZ);
        free(batch->gradSellZ);
        free(batch->wallets);
        free(batch->phases);
        free(batch->buyFlags);
        free(batch->sellFlags);
        free(batch->rows);
        return -1;
    }

    return 0;
}

/* Free one reusable batch scratch bundle. */
static void freeBatchLane(BatchLaneState* batch) {
    free(batch->cooldown);
    free(batch->lastTrendCode);
    free(batch->lastSignalSellIndex);
    free(batch->buyLastCd);
    free(batch->sellLastCd);
    free(batch->buyLastAccepted);
    free(batch->sellLastAccepted);
    free(batch->buyLastPhase);
    free(batch->sellLastPhase);
    free(batch->buyCounts);
    free(batch->sellCounts);
    free(batch->benchQty);
    free(batch->dailyStrongDays);
    free(batch->dailyLockActive);
    free(batch->dailyLockStart);
    free(batch->dailyPrevStrong);
    free(batch->dailyPrevDown);
    free(batch->dailyEpisodeLocked);
    free(batch->dailyBridgeBars);
    free(batch->dailyLockReleaseOnStrong);
    free(batch->dailyCoastActive);
    free(batch->dailyCoastStart);
    free(batch->peakLong);
    free(batch->peakBearCount);
    free(batch->peakStrongGraceBars);
    free(batch->peakStrongReleases);
    free(batch->peakPrevStrong);
    free(batch->peakActive);
    free(batch->peakStart);
    free(batch->peakLocks);
    free(batch->peakCappedBuys);
    free(batch->peakLockHours);
    free(batch->peakUnlockSteps);
    free(batch->peakArmed);
    free(batch->dailyUltraEntryPrice);
    free(batch->dailyUltraPeakPrice);
    free(batch->dailyLockTargetPct);
    free(batch->dailyLockHoldDays);
    free(batch->dailyCoastTargetPct);
    free(batch->dailyCoastMinAssetPct);
    free(batch->dailyCoastMaxAssetPct);
    free(batch->peakMa);
    free(batch->peakIntegral);
    free(batch->peakPrevErr);
    free(batch->peakCap);
    free(batch->peakEdgeStart);
    free(batch->peakEdgeNow);
    free(batch->peakEdgePeak);
    free(batch->peakLockGain);
    free(batch->peakLockGainMax);
    free(batch->peakEdgeVals);
    free(batch->curveSim);
    free(batch->gradBuyZ);
    free(batch->gradSellZ);
    free(batch->wallets);
    free(batch->phases);
    free(batch->buyFlags);
    free(batch->sellFlags);
    free(batch->rows);
}

/* Return one lane-major flag slice. */
static unsigned char* laneFlagSlice(
    unsigned char* flags,
    int lane,
    int n
) {
    return flags + ((size_t)lane * (size_t)n);
}

/* Return one lane-major curve slice. */
static double* laneCurveSlice(BatchLaneState* batch, int lane) {
    return batch->curveSim + ((size_t)lane * (size_t)batch->n);
}

/* Return one lane-major peak-lock edge-history slice. */
static double* lanePeakEdgeSlice(BatchLaneState* batch, int lane) {
    return batch->peakEdgeVals + ((size_t)lane * (size_t)batch->n);
}

/* Compare doubles for ascending qsort median passes. */
static int cmpDouble(const void* a, const void* b) {
    double x = *(const double*)a;
    double y = *(const double*)b;

    if (x < y) {
        return -1;
    }
    if (x > y) {
        return 1;
    }
    return 0;
}

/* Return the median of one finite double array. */
static double medianValue(double* values, int n) {
    int mid;

    if (n <= 0) {
        return nanVal();
    }

    qsort(values, (size_t)n, sizeof(double), cmpDouble);
    mid = n / 2;
    if ((n % 2) != 0) {
        return values[mid];
    }
    return (values[mid - 1] + values[mid]) * 0.5;
}

/* Build simple step returns from one equity curve. */
static int stepReturns(
    const double* curve,
    int n,
    double* out
) {
    int i;
    double prev;

    if (curve == NULL || out == NULL || n <= 1) {
        return 0;
    }

    for (i = 1; i < n; i++) {
        prev = curve[i - 1];
        if (prev == 0.0) {
            prev = 1e-12;
        }
        out[i - 1] = (curve[i] / prev) - 1.0;
    }
    return n - 1;
}

/* Compute max drawdown for one equity curve. */
static double maxDrawdownValue(const double* curve, int n) {
    int i;
    double peak;
    double mdd = 0.0;
    double dd;

    if (curve == NULL || n <= 0) {
        return nanVal();
    }

    peak = curve[0];
    for (i = 0; i < n; i++) {
        if (!isfinite(curve[i])) {
            continue;
        }
        if (curve[i] > peak) {
            peak = curve[i];
        }
        if (peak > 0.0) {
            dd = (peak - curve[i]) / peak;
            if (dd > mdd) {
                mdd = dd;
            }
        }
    }
    return mdd;
}

/* Compute CAGR for one equity curve and duration. */
static double cagrValue(
    const double* curve,
    int n,
    double years
) {
    double start;
    double end;

    if (curve == NULL || n <= 0 || years <= 0.0) {
        return nanVal();
    }
    start = curve[0];
    end = curve[n - 1];
    if (start <= 0.0) {
        return nanVal();
    }
    return pow(end / start, 1.0 / years) - 1.0;
}

/* Compute lifecycle edge metrics versus the HODL benchmark curve. */
static void lifecycleEdgeStats(
    const MicroSoa* micro,
    int begin,
    const double* curveSim,
    int curveLen,
    double benchQty,
    double* edgeVals,
    EvalRow* row
) {
    int i;
    int p25Idx;
    int belowCount = 0;
    int trackCount = 0;
    double sum = 0.0;
    double underSum = 0.0;
    double minEdge = INFINITY;
    double peak = -INFINITY;
    double edgeMdd = 0.0;
    double terminal;

    if (
        micro == NULL
        || curveSim == NULL
        || edgeVals == NULL
        || row == NULL
        || curveLen <= 0
    ) {
        row->lifecycleEdgeMean = nanVal();
        row->lifecycleEdgeMedian = nanVal();
        row->lifecycleEdgeP25 = nanVal();
        row->lifecycleEdgeMin = nanVal();
        row->lifecycleUnderwaterPct = nanVal();
        row->lifecycleUnderwaterMean = nanVal();
        row->lifecycleTrackingPct = nanVal();
        row->lifecycleEdgeMdd = nanVal();
        row->lifecycleEdgeScore = nanVal();
        return;
    }

    for (i = 0; i < curveLen; i++) {
        double bench = benchQty * micro->closes[begin + i];
        double edge;
        double draw;

        if (bench <= 0.0) {
            bench = 1e-12;
        }
        edge = ((curveSim[i] / bench) - 1.0) * 100.0;
        edgeVals[i] = edge;
        sum += edge;
        if (edge < minEdge) {
            minEdge = edge;
        }
        if (edge < 0.0) {
            belowCount += 1;
            underSum += -edge;
        }
        if (fabs(edge) < 5.0) {
            trackCount += 1;
        }
        if (edge > peak) {
            peak = edge;
        }
        draw = peak - edge;
        if (draw > edgeMdd) {
            edgeMdd = draw;
        }
    }

    qsort(edgeVals, (size_t)curveLen, sizeof(double), cmpDouble);
    p25Idx = (int)floor(0.25 * (double)(curveLen - 1));
    terminal = (
        (
            curveSim[curveLen - 1]
            / fmax(benchQty * micro->closes[begin + curveLen - 1], 1e-12)
        )
        - 1.0
    ) * 100.0;
    row->lifecycleEdgeMean = sum / (double)curveLen;
    row->lifecycleEdgeMedian = medianValue(edgeVals, curveLen);
    row->lifecycleEdgeP25 = edgeVals[p25Idx];
    row->lifecycleEdgeMin = minEdge;
    row->lifecycleUnderwaterPct = (
        ((double)belowCount / (double)curveLen) * 100.0
    );
    row->lifecycleUnderwaterMean = underSum / (double)curveLen;
    row->lifecycleTrackingPct = (
        ((double)trackCount / (double)curveLen) * 100.0
    );
    row->lifecycleEdgeMdd = edgeMdd;
    row->lifecycleEdgeScore = (
        terminal
        + (0.75 * row->lifecycleEdgeMedian)
        + row->lifecycleEdgeP25
        + (1.25 * row->lifecycleEdgeMin)
        - (0.75 * row->lifecycleUnderwaterMean)
        - (0.50 * row->lifecycleUnderwaterPct)
        - (0.35 * row->lifecycleTrackingPct)
        - (0.75 * row->lifecycleEdgeMdd)
    );
}

/* Compute rolling Sharpe/Sortino medians for one return series. */
static void rollingSharpeSortinoMedian(
    const double* returns,
    int n,
    double periodsPerYear,
    int window,
    MetricScratch* scratch,
    double* sharpeOut,
    double* sortinoOut
) {
    int i;
    int outN;
    int sharpeCount = 0;
    int sortinoCount = 0;
    double w;
    double sqrtPpy;
    double sumWin;
    double sum2Win;
    double meanWin;
    double varNum;
    double stdWin;
    double cntNeg;
    double sumNeg;
    double sum2Neg;
    double meanNeg;
    double varNumNeg;
    double dstd;
    double* cs;
    double* cs2;
    double* csNeg;
    double* cs2Neg;
    double* csCntNeg;
    double* sharpeVals;
    double* sortinoVals;

    *sharpeOut = nanVal();
    *sortinoOut = nanVal();

    if (
        returns == NULL
        || scratch == NULL
        || n <= 1
        || window <= 1
        || n < window
        || periodsPerYear <= 0.0
    ) {
        return;
    }

    outN = n - window + 1;
    cs = scratch->cs;
    cs2 = scratch->cs2;
    csNeg = scratch->csNeg;
    cs2Neg = scratch->cs2Neg;
    csCntNeg = scratch->csCntNeg;
    sharpeVals = scratch->sharpeVals;
    sortinoVals = scratch->sortinoVals;

    cs[0] = 0.0;
    cs2[0] = 0.0;
    csNeg[0] = 0.0;
    cs2Neg[0] = 0.0;
    csCntNeg[0] = 0.0;

    for (i = 0; i < n; i++) {
        cs[i + 1] = cs[i] + returns[i];
        cs2[i + 1] = cs2[i] + (returns[i] * returns[i]);
        if (returns[i] < 0.0) {
            csNeg[i + 1] = csNeg[i] + returns[i];
            cs2Neg[i + 1] = cs2Neg[i] + (returns[i] * returns[i]);
            csCntNeg[i + 1] = csCntNeg[i] + 1.0;
        }
        else {
            csNeg[i + 1] = csNeg[i];
            cs2Neg[i + 1] = cs2Neg[i];
            csCntNeg[i + 1] = csCntNeg[i];
        }
    }

    w = (double)window;
    sqrtPpy = sqrt(periodsPerYear);

    for (i = 0; i < outN; i++) {
        sumWin = cs[i + window] - cs[i];
        sum2Win = cs2[i + window] - cs2[i];
        meanWin = sumWin / w;
        varNum = sum2Win - ((sumWin * sumWin) / w);
        stdWin = sqrt(fmax(varNum / (double)(window - 1), 0.0));
        if (stdWin > 1e-12) {
            sharpeVals[sharpeCount] = (meanWin / stdWin) * sqrtPpy;
            sharpeCount += 1;
        }

        cntNeg = csCntNeg[i + window] - csCntNeg[i];
        if (cntNeg <= 1.5) {
            continue;
        }
        sumNeg = csNeg[i + window] - csNeg[i];
        sum2Neg = cs2Neg[i + window] - cs2Neg[i];
        meanNeg = sumNeg / cntNeg;
        varNumNeg = sum2Neg - (cntNeg * meanNeg * meanNeg);
        dstd = sqrt(fmax(varNumNeg / (cntNeg - 1.0), 0.0));
        if (dstd > 1e-12) {
            sortinoVals[sortinoCount] = (meanWin / dstd) * sqrtPpy;
            sortinoCount += 1;
        }
    }

    if (sharpeCount > 0) {
        *sharpeOut = medianValue(sharpeVals, sharpeCount);
    }
    if (sortinoCount > 0) {
        *sortinoOut = medianValue(sortinoVals, sortinoCount);
    }
}

/* Count one sweep axis product. */
uint64_t tuneAxesCount(const TuneAxes* axes) {
    uint64_t total = 1;

    if (axes == NULL) {
        return 1;
    }

    total *= (uint64_t)axes->grad1BuyZMin.n;
    total *= (uint64_t)axes->grad1SellZMin.n;
    total *= (uint64_t)axes->grad1BuyWinDays.n;
    total *= (uint64_t)axes->grad1SellWinDays.n;
    total *= (uint64_t)axes->phaseBuy.n;
    total *= (uint64_t)axes->phaseSell.n;
    total *= (uint64_t)axes->finalPortionPct.n;
    total *= (uint64_t)axes->cooldown.n;
    total *= (uint64_t)axes->taxMode.n;
    total *= (uint64_t)axes->seedAssetPct.n;
    total *= (uint64_t)axes->dailyStrongSellMult.n;
    total *= (uint64_t)axes->dailyStrongTargetPct.n;
    total *= (uint64_t)axes->dailyBridgeDays.n;
    total *= (uint64_t)axes->dailyDownBuyMult.n;
    total *= (uint64_t)axes->dailyCrabAssetCapPct.n;
    total *= (uint64_t)axes->dailyLockTargetPct.n;
    total *= (uint64_t)axes->dailyLockGainPct.n;
    total *= (uint64_t)axes->dailyLockNearHighPct.n;
    total *= (uint64_t)axes->dailyLockMaxDays.n;
    total *= (uint64_t)axes->postUltraCoastTargetPct.n;
    total *= (uint64_t)axes->postUltraGivebackPct.n;
    total *= (uint64_t)axes->postUltraReaccumPct.n;
    total *= (uint64_t)axes->postUltraDoubleTopPct.n;
    total *= (uint64_t)axes->postUltraMaxDays.n;
    total *= (uint64_t)axes->postUltraLockMinAssetPct.n;
    total *= (uint64_t)axes->postUltraLockMaxAssetPct.n;
    total *= (uint64_t)axes->postUltraLockGivebackPct.n;
    total *= (uint64_t)axes->postUltraLockReaccumPct.n;
    total *= (uint64_t)axes->postUltraLockDoubleTopPct.n;
    total *= (uint64_t)axes->postUltraLockMaxDays.n;
    total *= (uint64_t)axes->macroSellRelaxPct.n;
    total *= (uint64_t)axes->annualIncomeBase.n;
    total *= (uint64_t)axes->peakLockCapPct.n;
    total *= (uint64_t)axes->peakLockUnlockGainPct.n;
    total *= (uint64_t)axes->peakLockReentryStepPct.n;
    total *= (uint64_t)axes->peakLockArmGainPct.n;
    total *= (uint64_t)axes->peakLockGivebackPct.n;
    total *= (uint64_t)axes->peakLockMaxDays.n;
    total *= (uint64_t)axes->peakLockEdgeDrawPct.n;
    total *= (uint64_t)axes->peakLockEdgeSlopeDays.n;
    total *= (uint64_t)axes->peakLockRequireEdgeRisk.n;
    total *= (uint64_t)axes->peakLockMaDays.n;
    total *= (uint64_t)axes->peakLockKp.n;
    total *= (uint64_t)axes->peakLockKi.n;
    total *= (uint64_t)axes->peakLockKd.n;
    total *= (uint64_t)axes->peakLockIntegralDecay.n;
    total *= (uint64_t)axes->peakLockEntryThreshold.n;
    total *= (uint64_t)axes->peakLockExitThreshold.n;
    total *= (uint64_t)axes->peakLockConfirmBars.n;
    total *= (uint64_t)axes->peakLockReleaseTargetPct.n;
    total *= (uint64_t)axes->peakLockUltraGraceDays.n;
    return total;
}

/* Decode one cartesian-product index into one TuneParams row. */
static void decodeSweepParam(
    const TuneParams* baseParam,
    const TuneAxes* axes,
    uint64_t comboIdx,
    TuneParams* out
) {
    uint64_t axisIdx = comboIdx;

    *out = *baseParam;
    if (axes == NULL) {
        return;
    }

    out->grad1BuyZMin =
        axes->grad1BuyZMin.values[axisIdx % (uint64_t)axes->grad1BuyZMin.n];
    axisIdx /= (uint64_t)axes->grad1BuyZMin.n;
    out->grad1SellZMin =
        axes->grad1SellZMin.values[axisIdx % (uint64_t)axes->grad1SellZMin.n];
    axisIdx /= (uint64_t)axes->grad1SellZMin.n;
    out->grad1BuyWinDays = axes->grad1BuyWinDays.values[
        axisIdx % (uint64_t)axes->grad1BuyWinDays.n
    ];
    axisIdx /= (uint64_t)axes->grad1BuyWinDays.n;
    out->grad1SellWinDays = axes->grad1SellWinDays.values[
        axisIdx % (uint64_t)axes->grad1SellWinDays.n
    ];
    axisIdx /= (uint64_t)axes->grad1SellWinDays.n;
    out->phaseBuy =
        axes->phaseBuy.values[axisIdx % (uint64_t)axes->phaseBuy.n];
    axisIdx /= (uint64_t)axes->phaseBuy.n;
    out->phaseSell =
        axes->phaseSell.values[axisIdx % (uint64_t)axes->phaseSell.n];
    axisIdx /= (uint64_t)axes->phaseSell.n;
    out->finalPortionPct = axes->finalPortionPct.values[
        axisIdx % (uint64_t)axes->finalPortionPct.n
    ];
    axisIdx /= (uint64_t)axes->finalPortionPct.n;
    out->cooldown =
        axes->cooldown.values[axisIdx % (uint64_t)axes->cooldown.n];
    axisIdx /= (uint64_t)axes->cooldown.n;
    out->taxMode = axes->taxMode.values[
        axisIdx % (uint64_t)axes->taxMode.n
    ];
    axisIdx /= (uint64_t)axes->taxMode.n;
    out->seedAssetPct = axes->seedAssetPct.values[
        axisIdx % (uint64_t)axes->seedAssetPct.n
    ];
    axisIdx /= (uint64_t)axes->seedAssetPct.n;
    out->dailyStrongSellMult = axes->dailyStrongSellMult.values[
        axisIdx % (uint64_t)axes->dailyStrongSellMult.n
    ];
    axisIdx /= (uint64_t)axes->dailyStrongSellMult.n;
    out->dailyStrongTargetPct = axes->dailyStrongTargetPct.values[
        axisIdx % (uint64_t)axes->dailyStrongTargetPct.n
    ];
    axisIdx /= (uint64_t)axes->dailyStrongTargetPct.n;
    out->dailyBridgeDays = axes->dailyBridgeDays.values[
        axisIdx % (uint64_t)axes->dailyBridgeDays.n
    ];
    axisIdx /= (uint64_t)axes->dailyBridgeDays.n;
    out->dailyDownBuyMult = axes->dailyDownBuyMult.values[
        axisIdx % (uint64_t)axes->dailyDownBuyMult.n
    ];
    axisIdx /= (uint64_t)axes->dailyDownBuyMult.n;
    out->dailyCrabAssetCapPct = axes->dailyCrabAssetCapPct.values[
        axisIdx % (uint64_t)axes->dailyCrabAssetCapPct.n
    ];
    axisIdx /= (uint64_t)axes->dailyCrabAssetCapPct.n;
    out->dailyLockTargetPct = axes->dailyLockTargetPct.values[
        axisIdx % (uint64_t)axes->dailyLockTargetPct.n
    ];
    axisIdx /= (uint64_t)axes->dailyLockTargetPct.n;
    out->dailyLockGainPct = axes->dailyLockGainPct.values[
        axisIdx % (uint64_t)axes->dailyLockGainPct.n
    ];
    axisIdx /= (uint64_t)axes->dailyLockGainPct.n;
    out->dailyLockNearHighPct = axes->dailyLockNearHighPct.values[
        axisIdx % (uint64_t)axes->dailyLockNearHighPct.n
    ];
    axisIdx /= (uint64_t)axes->dailyLockNearHighPct.n;
    out->dailyLockMaxDays = axes->dailyLockMaxDays.values[
        axisIdx % (uint64_t)axes->dailyLockMaxDays.n
    ];
    axisIdx /= (uint64_t)axes->dailyLockMaxDays.n;
    out->postUltraCoastTargetPct =
        axes->postUltraCoastTargetPct.values[
            axisIdx % (uint64_t)axes->postUltraCoastTargetPct.n
        ];
    axisIdx /= (uint64_t)axes->postUltraCoastTargetPct.n;
    out->postUltraGivebackPct = axes->postUltraGivebackPct.values[
        axisIdx % (uint64_t)axes->postUltraGivebackPct.n
    ];
    axisIdx /= (uint64_t)axes->postUltraGivebackPct.n;
    out->postUltraReaccumPct = axes->postUltraReaccumPct.values[
        axisIdx % (uint64_t)axes->postUltraReaccumPct.n
    ];
    axisIdx /= (uint64_t)axes->postUltraReaccumPct.n;
    out->postUltraDoubleTopPct =
        axes->postUltraDoubleTopPct.values[
            axisIdx % (uint64_t)axes->postUltraDoubleTopPct.n
        ];
    axisIdx /= (uint64_t)axes->postUltraDoubleTopPct.n;
    out->postUltraMaxDays = axes->postUltraMaxDays.values[
        axisIdx % (uint64_t)axes->postUltraMaxDays.n
    ];
    axisIdx /= (uint64_t)axes->postUltraMaxDays.n;
    out->postUltraLockMinAssetPct =
        axes->postUltraLockMinAssetPct.values[
            axisIdx % (uint64_t)axes->postUltraLockMinAssetPct.n
        ];
    axisIdx /= (uint64_t)axes->postUltraLockMinAssetPct.n;
    out->postUltraLockMaxAssetPct =
        axes->postUltraLockMaxAssetPct.values[
            axisIdx % (uint64_t)axes->postUltraLockMaxAssetPct.n
        ];
    axisIdx /= (uint64_t)axes->postUltraLockMaxAssetPct.n;
    out->postUltraLockGivebackPct =
        axes->postUltraLockGivebackPct.values[
            axisIdx % (uint64_t)axes->postUltraLockGivebackPct.n
        ];
    axisIdx /= (uint64_t)axes->postUltraLockGivebackPct.n;
    out->postUltraLockReaccumPct =
        axes->postUltraLockReaccumPct.values[
            axisIdx % (uint64_t)axes->postUltraLockReaccumPct.n
        ];
    axisIdx /= (uint64_t)axes->postUltraLockReaccumPct.n;
    out->postUltraLockDoubleTopPct =
        axes->postUltraLockDoubleTopPct.values[
            axisIdx % (uint64_t)axes->postUltraLockDoubleTopPct.n
        ];
    axisIdx /= (uint64_t)axes->postUltraLockDoubleTopPct.n;
    out->postUltraLockMaxDays =
        axes->postUltraLockMaxDays.values[
            axisIdx % (uint64_t)axes->postUltraLockMaxDays.n
        ];
    axisIdx /= (uint64_t)axes->postUltraLockMaxDays.n;
    out->macroSellRelaxPct = axes->macroSellRelaxPct.values[
        axisIdx % (uint64_t)axes->macroSellRelaxPct.n
    ];
    axisIdx /= (uint64_t)axes->macroSellRelaxPct.n;
    out->annualIncomeBase = axes->annualIncomeBase.values[
        axisIdx % (uint64_t)axes->annualIncomeBase.n
    ];
    axisIdx /= (uint64_t)axes->annualIncomeBase.n;
    out->peakLockCapPct = axes->peakLockCapPct.values[
        axisIdx % (uint64_t)axes->peakLockCapPct.n
    ];
    axisIdx /= (uint64_t)axes->peakLockCapPct.n;
    out->peakLockUnlockGainPct = axes->peakLockUnlockGainPct.values[
        axisIdx % (uint64_t)axes->peakLockUnlockGainPct.n
    ];
    axisIdx /= (uint64_t)axes->peakLockUnlockGainPct.n;
    out->peakLockReentryStepPct = axes->peakLockReentryStepPct.values[
        axisIdx % (uint64_t)axes->peakLockReentryStepPct.n
    ];
    axisIdx /= (uint64_t)axes->peakLockReentryStepPct.n;
    out->peakLockArmGainPct = axes->peakLockArmGainPct.values[
        axisIdx % (uint64_t)axes->peakLockArmGainPct.n
    ];
    axisIdx /= (uint64_t)axes->peakLockArmGainPct.n;
    out->peakLockGivebackPct = axes->peakLockGivebackPct.values[
        axisIdx % (uint64_t)axes->peakLockGivebackPct.n
    ];
    axisIdx /= (uint64_t)axes->peakLockGivebackPct.n;
    out->peakLockMaxDays = axes->peakLockMaxDays.values[
        axisIdx % (uint64_t)axes->peakLockMaxDays.n
    ];
    axisIdx /= (uint64_t)axes->peakLockMaxDays.n;
    out->peakLockEdgeDrawPct = axes->peakLockEdgeDrawPct.values[
        axisIdx % (uint64_t)axes->peakLockEdgeDrawPct.n
    ];
    axisIdx /= (uint64_t)axes->peakLockEdgeDrawPct.n;
    out->peakLockEdgeSlopeDays = axes->peakLockEdgeSlopeDays.values[
        axisIdx % (uint64_t)axes->peakLockEdgeSlopeDays.n
    ];
    axisIdx /= (uint64_t)axes->peakLockEdgeSlopeDays.n;
    out->peakLockRequireEdgeRisk = axes->peakLockRequireEdgeRisk.values[
        axisIdx % (uint64_t)axes->peakLockRequireEdgeRisk.n
    ];
    axisIdx /= (uint64_t)axes->peakLockRequireEdgeRisk.n;
    out->peakLockMaDays = axes->peakLockMaDays.values[
        axisIdx % (uint64_t)axes->peakLockMaDays.n
    ];
    axisIdx /= (uint64_t)axes->peakLockMaDays.n;
    out->peakLockKp = axes->peakLockKp.values[
        axisIdx % (uint64_t)axes->peakLockKp.n
    ];
    axisIdx /= (uint64_t)axes->peakLockKp.n;
    out->peakLockKi = axes->peakLockKi.values[
        axisIdx % (uint64_t)axes->peakLockKi.n
    ];
    axisIdx /= (uint64_t)axes->peakLockKi.n;
    out->peakLockKd = axes->peakLockKd.values[
        axisIdx % (uint64_t)axes->peakLockKd.n
    ];
    axisIdx /= (uint64_t)axes->peakLockKd.n;
    out->peakLockIntegralDecay = axes->peakLockIntegralDecay.values[
        axisIdx % (uint64_t)axes->peakLockIntegralDecay.n
    ];
    axisIdx /= (uint64_t)axes->peakLockIntegralDecay.n;
    out->peakLockEntryThreshold = axes->peakLockEntryThreshold.values[
        axisIdx % (uint64_t)axes->peakLockEntryThreshold.n
    ];
    axisIdx /= (uint64_t)axes->peakLockEntryThreshold.n;
    out->peakLockExitThreshold = axes->peakLockExitThreshold.values[
        axisIdx % (uint64_t)axes->peakLockExitThreshold.n
    ];
    axisIdx /= (uint64_t)axes->peakLockExitThreshold.n;
    out->peakLockConfirmBars = axes->peakLockConfirmBars.values[
        axisIdx % (uint64_t)axes->peakLockConfirmBars.n
    ];
    axisIdx /= (uint64_t)axes->peakLockConfirmBars.n;
    out->peakLockReleaseTargetPct = axes->peakLockReleaseTargetPct.values[
        axisIdx % (uint64_t)axes->peakLockReleaseTargetPct.n
    ];
    axisIdx /= (uint64_t)axes->peakLockReleaseTargetPct.n;
    out->peakLockUltraGraceDays = axes->peakLockUltraGraceDays.values[
        axisIdx % (uint64_t)axes->peakLockUltraGraceDays.n
    ];
}

/* Count accepted BUY or SELL flags for one parameter row. */
static int markSideFlags(
    const MicroSoa* microRaw,
    const MicroDerived* micro,
    const MacroDerived* macro,
    const TuneParams* param,
    int startIdx,
    int isBuy,
    const int* allowReg,
    const int* anchorIdx,
    FlagScratch* scratch,
    ZCacheSet* zCache,
    unsigned char* outFlags
) {
    int i;
    int n = microRaw->n;
    int cooldown;
    int gradBars;
    int begin;
    int lastCd;
    int lastAccepted = -1;
    int lastPhase = -1;
    int refIdx;
    double gradSign;
    double gradThresh;
    double gradScore;
    double req;
    double mult;
    double priceNow;
    double priceRef;
    double deltaPct;
    const double* gradZ;
    int count = 0;

    if (outFlags != NULL) {
        for (i = 0; i < n; i++) {
            outFlags[i] = 0;
        }
    }

    gradBars = (int)round(
        (double)(
            isBuy ? param->grad1BuyWinDays : param->grad1SellWinDays
        ) * micro->barsPerDay
    );
    if (gradBars < 1) {
        gradBars = 1;
    }

    gradSign = isBuy ? -1.0 : 1.0;
    gradThresh = isBuy
        ? param->grad1BuyZMin
        : param->grad1SellZMin;

    gradZ = cachedZSeries(
        &zCache->grad,
        micro->g1p1,
        n,
        gradBars,
        scratch->meanA,
        scratch->stdA
    );

    if (gradZ == NULL) {
        return -1;
    }

    cooldown = param->cooldown;
    if (cooldown < 0) {
        cooldown = 0;
    }

    begin = startIdx;
    if (begin < 0) {
        begin = 0;
    }
    if (begin >= n) {
        return 0;
    }

    lastCd = -cooldown;

    for (i = begin; i < n; i++) {
        if (!allowReg[i]) {
            continue;
        }

        gradScore = gradZ[i] * gradSign;
        if (!isfinite(gradScore) || gradScore < gradThresh) {
            continue;
        }

        if ((i - lastCd) < cooldown) {
            continue;
        }
        lastCd = i;

        if (anchorIdx[i] != lastPhase) {
            lastAccepted = -1;
            lastPhase = anchorIdx[i];
        }

        refIdx = lastAccepted >= 0 ? lastAccepted : anchorIdx[i];
        if (refIdx >= 0) {
            priceNow = microRaw->closes[i];
            priceRef = microRaw->closes[refIdx];
            if (priceNow > 0.0 && priceRef > 0.0) {
                if (isBuy) {
                    deltaPct = ((priceRef / priceNow) - 1.0) * 100.0;
                }
                else {
                    deltaPct = ((priceNow / priceRef) - 1.0) * 100.0;
                }
            }
            else {
                deltaPct = 0.0;
            }

            mult = 1.0;
            req = fabs(macro->dyn[i]) * mult;
            if (!isBuy) {
                req *= 1.0 - (clipVal(
                    param->macroSellRelaxPct,
                    0.0,
                    100.0
                ) / 100.0);
            }
            if (req < 0.0) {
                req = 0.0;
            }
            if (deltaPct < req) {
                continue;
            }
        }

        lastAccepted = i;
        if (outFlags != NULL) {
            outFlags[i] = 1;
        }
        count += 1;
    }

    return count;
}

/* Finalize one row once the curve and wallet state are complete. */
static void finishEvalRow(
    const MicroSoa* micro,
    PreparedDatasetState* prepared,
    EvalLaneState* lane,
    int begin,
    const WalletState* walletTrade,
    const double* curveSim,
    int curveLen,
    double benchQty,
    EvalRow* row
) {
    int i;
    int retLen;
    double periodsPerYear;
    double durationDays;
    double durationYears;
    double simValueRaw;
    double benchValueRaw;
    double sharpe4w;
    double sortino4w;
    double sharpe13w;
    double sortino13w;
    double* retsSim = lane->rowScratch.retsSim;
    double* edgeRets = lane->rowScratch.edgeRets;

    simValueRaw = portfolioValue(walletTrade, micro->closes[micro->n - 1]);
    benchValueRaw = benchQty * micro->closes[micro->n - 1];

    row->simValue = simValueRaw;
    row->simPostTax = simValueRaw;
    row->benchValue = benchValueRaw;
    row->benchPostTax = benchValueRaw;
    row->preTaxEdge = simValueRaw - benchValueRaw;
    row->postTaxEdge = simValueRaw - benchValueRaw;
    row->netPctVsHodl = (
        benchValueRaw > 0.0
        ? ((simValueRaw / benchValueRaw) - 1.0) * 100.0
        : nanVal()
    );
    row->fees = walletTrade->feesPaidQuote;
    row->tax = 0.0;
    row->potentialProfit = nanVal();
    row->potentialProfitBench = nanVal();
    row->netAfterTaxProfit = nanVal();
    row->netAfterTaxProfitBench = nanVal();
    row->trades = walletTrade->tradeCount;
    row->buyTrades = walletTrade->buyTrades;
    row->sellTrades = walletTrade->sellTrades;

    periodsPerYear = prepared->microDerived.barsPerDay * 365.0;
    if (periodsPerYear <= 0.0) {
        periodsPerYear = 365.0;
    }

    retLen = stepReturns(curveSim, curveLen, retsSim);
    for (i = 0; i < retLen; i++) {
        double prevBench = micro->closes[begin + i];
        double benchRet;

        if (prevBench == 0.0) {
            prevBench = 1e-12;
        }
        benchRet = (micro->closes[begin + i + 1] / prevBench) - 1.0;
        edgeRets[i] = retsSim[i] - benchRet;
    }

    durationDays = (
        (double)(micro->ts[micro->n - 1] - micro->ts[begin])
        / (24.0 * 60.0 * 60.0 * 1000.0)
    );
    if (durationDays < 1.0) {
        durationDays = 1.0;
    }
    durationYears = durationDays / 365.0;

    row->sharpe = nanVal();
    row->sortino = nanVal();
    row->mdd = maxDrawdownValue(curveSim, curveLen);
    row->cagr = cagrValue(curveSim, curveLen, durationYears);
    lifecycleEdgeStats(
        micro,
        begin,
        curveSim,
        curveLen,
        benchQty,
        lane->rowScratch.edgeVals,
        row
    );

    rollingSharpeSortinoMedian(
        edgeRets,
        retLen,
        periodsPerYear,
        fmax((int)round(28.0 * prepared->microDerived.barsPerDay), 2),
        &lane->metricScratch,
        &sharpe4w,
        &sortino4w
    );
    rollingSharpeSortinoMedian(
        edgeRets,
        retLen,
        periodsPerYear,
        fmax((int)round(91.0 * prepared->microDerived.barsPerDay), 2),
        &lane->metricScratch,
        &sharpe13w,
        &sortino13w
    );

    row->sharpe1w = nanVal();
    row->sortino1w = nanVal();
    row->sharpe4w = sharpe4w;
    row->sortino4w = sortino4w;
    row->sharpe13w = sharpe13w;
    row->sortino13w = sortino13w;
    row->sharpe1wAbs = nanVal();
    row->sortino1wAbs = nanVal();
    row->sharpe4wAbs = nanVal();
    row->sortino4wAbs = nanVal();
    row->sharpe13wAbs = nanVal();
    row->sortino13wAbs = nanVal();
}

/* Evaluate one parameter row once flags are already available. */
static int evaluateRowFlags(
    const MicroSoa* micro,
    PreparedDatasetState* prepared,
    EvalLaneState* lane,
    const TuneParams* param,
    int startIdx,
    const unsigned char* buyFlags,
    const unsigned char* sellFlags,
    int buyCount,
    int sellCount,
    EvalRow* row
) {
    int i;
    int begin;
    int curveLen;
    int currentTrend;
    int lastTrendCode = 0;
    int newBearRegime;
    int newBullRegime;
    int traded;
    double price;
    double price0;
    double benchQty;
    double seedSpend;
    double* curveSim;
    WalletState walletTrade;
    PhaseState phase;

    begin = startIdx;
    if (begin < 0) {
        begin = 0;
    }
    if (begin >= micro->n) {
        return 0;
    }

    curveLen = micro->n - begin;
    curveSim = lane->rowScratch.curveSim;

    row->buyFlags = buyCount;
    row->sellFlags = sellCount;
    row->flagCount = row->buyFlags + row->sellFlags;

    if (
        initWallet(
            &walletTrade,
            param->feeRate,
            0.0,
            TAX_CGT,
            0.0
        ) != 0
    ) {
        return -1;
    }

    phase.side = 0;
    phase.baseValue = 0.0;
    phase.lastPrice = 0.0;
    phase.hasLastPrice = 0;
    phase.portionsRemaining = 0.0;
    phase.hasInfiniteRemaining = 0;
    phase.finalPortionPct = param->finalPortionPct;

    price0 = micro->closes[begin];
    walletTrade.quoteBalance += param->seedQuote;
    seedSpend = param->seedQuote * clipVal(param->seedAssetPct, 0.0, 1.0);
    if (seedSpend > 0.0) {
        applyBuy(&walletTrade, begin, micro->ts[begin], price0, &seedSpend);
    }
    benchQty = (
        price0 > 0.0
        ? (param->seedQuote * (1.0 - param->feeRate)) / price0
        : 0.0
    );

    for (i = begin; i < micro->n; i++) {
        price = micro->closes[i];
        if (sellFlags[i] || buyFlags[i]) {
            currentTrend = prepared->microDerived.trend[i];
            newBearRegime = (
                currentTrend == -1
                && lastTrendCode != -1
            );
            newBullRegime = (
                currentTrend == 1
                && lastTrendCode != 1
            );
            lastTrendCode = currentTrend;

            if (sellFlags[i]) {
                if (currentTrend == 1 && walletTrade.baseBalance > 0.0) {
                    if (phase.side != -1 || newBullRegime) {
                        phase.side = -1;
                        phase.hasLastPrice = 0;
                        phase.baseValue = enterSellPhase(
                            &walletTrade,
                            price,
                            param->phaseSell
                        );
                        phase.finalPortionPct = param->finalPortionPct;
                        phase.hasInfiniteRemaining = (
                            param->finalPortionPct >= (1.0 - 1e-9)
                        );
                        phase.portionsRemaining = (
                            phase.baseValue > 0.0
                            ? (double)param->phaseSell
                            : 0.0
                        );
                    }
                    traded = applyScaledSell(
                        &walletTrade,
                        &phase,
                        i,
                        micro->ts[i],
                        price,
                        calcSellScale(
                            phase.hasLastPrice,
                            phase.lastPrice,
                            price
                        ),
                        walletTrade.baseBalance * price
                    );
                    (void)traded;
                }
            }

            if (buyFlags[i]) {
                if (currentTrend == -1 && walletTrade.quoteBalance > 0.0) {
                    if (phase.side != 1 || newBearRegime) {
                        phase.side = 1;
                        phase.hasLastPrice = 0;
                        phase.baseValue = enterBuyPhase(
                            &walletTrade,
                            param->phaseBuy
                        );
                        phase.finalPortionPct = param->finalPortionPct;
                        phase.hasInfiniteRemaining = (
                            param->finalPortionPct >= (1.0 - 1e-9)
                        );
                        phase.portionsRemaining = (
                            phase.baseValue > 0.0
                            ? (double)param->phaseBuy
                            : 0.0
                        );
                    }
                    traded = applyScaledBuy(
                        &walletTrade,
                        &phase,
                        i,
                        micro->ts[i],
                        price,
                        calcBuyScale(
                            phase.hasLastPrice,
                            phase.lastPrice,
                            price
                        ),
                        -1.0
                    );
                    (void)traded;
                }
            }
        }

        curveSim[i - begin] = portfolioValue(&walletTrade, price);
    }

    finishEvalRow(
        micro,
        prepared,
        lane,
        begin,
        &walletTrade,
        curveSim,
        curveLen,
        benchQty,
        row
    );

    freeWallet(&walletTrade);
    return 0;
}

/* Evaluate one parameter row end-to-end inside C. */
static int evaluateRow(
    const MicroSoa* micro,
    PreparedDatasetState* prepared,
    EvalLaneState* lane,
    const TuneParams* param,
    int startIdx,
    EvalRow* row
) {
    unsigned char* buyFlags = lane->rowScratch.buyFlags;
    unsigned char* sellFlags = lane->rowScratch.sellFlags;
    int buyCount;
    int sellCount;

    buyCount = markSideFlags(
        micro,
        &prepared->microDerived,
        &prepared->macroDerived,
        param,
        startIdx,
        1,
        prepared->allowBuy,
        prepared->buyAnchors,
        &prepared->scratch,
        &prepared->zCache,
        buyFlags
    );
    sellCount = markSideFlags(
        micro,
        &prepared->microDerived,
        &prepared->macroDerived,
        param,
        startIdx,
        0,
        prepared->allowSell,
        prepared->sellAnchors,
        &prepared->scratch,
        &prepared->zCache,
        sellFlags
    );
    if (buyCount < 0 || sellCount < 0) {
        return -1;
    }

    return evaluateRowFlags(
        micro,
        prepared,
        lane,
        param,
        startIdx,
        buyFlags,
        sellFlags,
        buyCount,
        sellCount,
        row
    );
}

/* Zero one result row. */
static void zeroRow(EvalRow* row) {
    row->simValue = 0.0;
    row->simPostTax = 0.0;
    row->benchValue = 0.0;
    row->benchPostTax = 0.0;
    row->preTaxEdge = 0.0;
    row->postTaxEdge = 0.0;
    row->netPctVsHodl = nanVal();
    row->fees = 0.0;
    row->tax = 0.0;
    row->potentialProfit = 0.0;
    row->potentialProfitBench = 0.0;
    row->netAfterTaxProfit = 0.0;
    row->netAfterTaxProfitBench = 0.0;
    row->sharpe = nanVal();
    row->sortino = nanVal();
    row->mdd = nanVal();
    row->cagr = nanVal();
    row->sharpe1w = nanVal();
    row->sortino1w = nanVal();
    row->sharpe4w = nanVal();
    row->sortino4w = nanVal();
    row->sharpe13w = nanVal();
    row->sortino13w = nanVal();
    row->sharpe1wAbs = nanVal();
    row->sortino1wAbs = nanVal();
    row->sharpe4wAbs = nanVal();
    row->sortino4wAbs = nanVal();
    row->sharpe13wAbs = nanVal();
    row->sortino13wAbs = nanVal();
    row->lifecycleEdgeMean = nanVal();
    row->lifecycleEdgeMedian = nanVal();
    row->lifecycleEdgeP25 = nanVal();
    row->lifecycleEdgeMin = nanVal();
    row->lifecycleUnderwaterPct = nanVal();
    row->lifecycleUnderwaterMean = nanVal();
    row->lifecycleTrackingPct = nanVal();
    row->lifecycleEdgeMdd = nanVal();
    row->lifecycleEdgeScore = nanVal();
    row->trades = 0;
    row->buyTrades = 0;
    row->sellTrades = 0;
    row->flagCount = 0;
    row->buyFlags = 0;
    row->sellFlags = 0;
}

/* Mirror Python's tax-aware edge selection. */
static double edgeVsBenchValue(
    const TuneParams* param,
    const EvalRow* row
) {
    double grossEdge = row->simValue - row->benchValue;
    double netEdge = row->simPostTax - row->benchPostTax;

    if (param->taxMode == TAX_INCOME) {
        return grossEdge;
    }
    return netEdge;
}

/* Return the finite minimum of two values. */
static double nanMinPair(double a, double b) {
    if (isfinite(a) && isfinite(b)) {
        return fmin(a, b);
    }
    if (isfinite(a)) {
        return a;
    }
    if (isfinite(b)) {
        return b;
    }
    return nanVal();
}

static double positivePart(double value) {
    return value > 0.0 ? value : 0.0;
}

static double tradeCountPenalty(const EvalRow* row) {
    return 0.03 * positivePart((double)row->trades - 500.0);
}

static double lifecycleScoreValue(const EvalRow* row) {
    if (!isfinite(row->lifecycleEdgeScore)) {
        return -INFINITY;
    }
    return row->lifecycleEdgeScore;
}

/* Match the Python risk-selection score. */
static double riskScoreValue(const EvalRow* row) {
    double sharpeWorst;
    double sortinoWorst;
    double mar;
    double drawdownPenalty;
    double score;

    if (
        !isfinite(row->lifecycleEdgeScore)
        || !isfinite(row->cagr)
        || !isfinite(row->mdd)
        || row->mdd <= 1e-12
    ) {
        return -INFINITY;
    }

    sharpeWorst = nanMinPair(row->sharpe4w, row->sharpe13w);
    sortinoWorst = nanMinPair(row->sortino4w, row->sortino13w);
    mar = row->cagr / row->mdd;
    drawdownPenalty = (
        (0.35 * row->mdd)
        + (1.25 * positivePart(row->mdd - 0.55))
    );
    score = row->lifecycleEdgeScore
        + (2.0 * mar)
        + (4.0 * sharpeWorst)
        + (4.0 * sortinoWorst)
        - (100.0 * drawdownPenalty)
        - tradeCountPenalty(row);
    if (!isfinite(score)) {
        return -INFINITY;
    }
    return score;
}

/* Reset one tune-group result container. */
static void zeroRunResult(TuneRunResult* out) {
    zeroRow(&out->bestRow);
    zeroRow(&out->statsRow);
    out->bestComboIdx = 0;
    out->statsComboIdx = 0;
    out->bestGrossPct = -INFINITY;
    out->statsScore = -INFINITY;
    out->evalCount = 0;
    out->elapsedSecs = 0.0;
}

/* Write the stable tuner CSV header once. */
static int writeCsvHeader(FILE* fp) {
    if (
        fprintf(
            fp,
            "ticker,interval,days,p1,p2,p3,"
            "GRAD1_BUY_Z_MIN,GRAD1_SELL_Z_MIN,"
            "GRAD1_BUY_WIN_DAYS,GRAD1_SELL_WIN_DAYS,"
            "PHASE_BUY_PORTIONS,PHASE_SELL_PORTIONS,FINAL_PORTION_PCT,"
            "COOLDOWN,MACRO_INTERVAL,MACRO_P1,MACRO_GRAD_PERIOD,MACRO_P3,"
            "MACRO_NRG_WIN_DAYS,MACRO_NRG_Z_MIN,MACRO_NRG_Z_MAX,"
            "MACRO_DYN_PCT_MIN,MACRO_DYN_PCT_MAX,"
            "MACRO_GRAD_WIN_DAYS,MACRO_GRAD_Z_MIN,MACRO_GRAD_Z_MAX,"
            "MACRO_MULT_GRAD_MIN,MACRO_MULT_GRAD_MAX,"
            "MACRO_SELL_RELAX_PCT,"
            "ULTRA_SELL_MULT,"
            "ULTRA_EXPOSURE_TARGET,"
            "ULTRA_BRIDGE_DAYS,"
            "DAILY_DOWN_BUY_MULT,"
            "CRAB_ASSET_CAP_PCT,"
            "ULTRA_EXIT_DEPTH,"
            "ULTRA_GAIN_MIN_PCT,ULTRA_GAIN_MAX_PCT,"
            "ULTRA_EXIT_HOLD_DAYS,"
            "POST_ULTRA_COAST_TARGET_PCT,"
            "POST_ULTRA_GIVEBACK_PCT,"
            "POST_ULTRA_REACCUM_PCT,"
            "POST_ULTRA_DOUBLE_TOP_PCT,"
            "POST_ULTRA_MAX_DAYS,"
            "POST_ULTRA_LOCK_MIN_ASSET_PCT,"
            "POST_ULTRA_LOCK_MAX_ASSET_PCT,"
            "POST_ULTRA_LOCK_GIVEBACK_PCT,"
            "POST_ULTRA_LOCK_REACCUM_PCT,"
            "POST_ULTRA_LOCK_DOUBLE_TOP_PCT,"
            "POST_ULTRA_LOCK_MAX_DAYS,"
            "PEAK_LOCK_CAP_PCT,"
            "PEAK_LOCK_UNLOCK_GAIN_PCT,"
            "PEAK_LOCK_REENTRY_STEP_PCT,"
            "PEAK_LOCK_ARM_GAIN_PCT,"
            "PEAK_LOCK_GIVEBACK_PCT,"
            "PEAK_LOCK_MAX_DAYS,"
            "PEAK_LOCK_EDGE_DRAW_PCT,"
            "PEAK_LOCK_EDGE_SLOPE_DAYS,"
            "PEAK_LOCK_REQUIRE_EDGE_RISK,"
            "PEAK_LOCK_MA_DAYS,"
            "PEAK_LOCK_KP,"
            "PEAK_LOCK_KI,"
            "PEAK_LOCK_KD,"
            "PEAK_LOCK_INTEGRAL_DECAY,"
            "PEAK_LOCK_ENTRY_THRESHOLD,"
            "PEAK_LOCK_EXIT_THRESHOLD,"
            "PEAK_LOCK_CONFIRM_BARS,"
            "PEAK_LOCK_RELEASE_TARGET_PCT,"
            "PEAK_LOCK_ULTRA_GRACE_DAYS,"
            "WALLET_SEED_ASSET_PCT,"
            "TAX_MODE,"
            "preTaxEdge,postTaxEdge,netPctVsHodl,"
            "simValue,simPostTax,benchValue,benchPostTax,"
            "trades,fees,tax,"
            "potentialProfit,potentialProfitBench,"
            "netAfterTaxProfit,netAfterTaxProfitBench,"
            "grossEdgeVsBench,netEdgeVsBench,edgeVsBench,"
            "sharpe,sortino,mdd,cagr,"
            "sharpe4w,sortino4w,sharpe13w,sortino13w,"
            "sharpe4wAbs,sortino4wAbs,"
            "sharpe13wAbs,sortino13wAbs,"
            "lifecycleEdgeMean,lifecycleEdgeMedian,"
            "lifecycleEdgeP25,lifecycleEdgeMin,"
            "lifecycleUnderwaterPct,lifecycleUnderwaterMean,"
            "lifecycleTrackingPct,lifecycleEdgeMdd,"
            "lifecycleEdgeScore,"
            "scoreMetric\n"
        ) < 0
    ) {
        return -1;
    }
    return 0;
}

/* Write one evaluated row in the stable tuner CSV order. */
static int writeCsvRow(
    FILE* fp,
    const TuneGroupMeta* meta,
    const TuneParams* param,
    const EvalRow* row
) {
    double grossEdgeVsBench = row->simValue - row->benchValue;
    double netEdgeVsBench = row->simPostTax - row->benchPostTax;
    double edgeVsBench = edgeVsBenchValue(param, row);
    double scoreMetric = roundMetric(row->lifecycleEdgeScore);

    if (
        fprintf(
            fp,
            "%s,%s,%d,%d,%d,%d,"
            "%.15g,%.15g,%d,%d,"
            "%d,%d,%.15g,%d,"
            "%s,%d,%d,%d,"
            "%d,%.15g,%.15g,%.15g,%.15g,"
            "%d,%.15g,%.15g,%.15g,%.15g,"
            "%.15g,"
            "%.15g,%.15g,%.15g,"
            "%.15g,%.15g,"
            "%.15g,%.15g,%.15g,%d,"
            "%.15g,%.15g,%.15g,%.15g,%.15g,"
            "%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,"
            "%.15g,%.15g,%.15g,%.15g,%.15g,"
            "%.15g,%.15g,%.15g,%d,"
            "%.15g,%.15g,%.15g,%.15g,%.15g,"
            "%.15g,%.15g,%d,%.15g,%.15g,"
            "%.15g,%s,",
            meta->ticker,
            meta->interval,
            meta->days,
            meta->p1,
            meta->p2,
            meta->p3,
            param->grad1BuyZMin,
            param->grad1SellZMin,
            param->grad1BuyWinDays,
            param->grad1SellWinDays,
            param->phaseBuy,
            param->phaseSell,
            param->finalPortionPct,
            param->cooldown,
            meta->macroInterval,
            meta->macroP1,
            meta->macroGradPeriod,
            meta->macroP3,
            meta->macroDynWinDays,
            meta->macroDynZMin,
            meta->macroDynZMax,
            meta->macroDynPctMin,
            meta->macroDynPctMax,
            meta->macroGradWinDays,
            meta->macroGradZMin,
            meta->macroGradZMax,
            meta->macroGradMultMin,
            meta->macroGradMultMax,
            param->macroSellRelaxPct,
            param->dailyStrongSellMult,
            param->dailyStrongTargetPct,
            param->dailyBridgeDays,
            param->dailyDownBuyMult,
            param->dailyCrabAssetCapPct,
            param->dailyLockTargetPct,
            param->dailyLockGainPct,
            param->dailyLockNearHighPct,
            param->dailyLockMaxDays,
            param->postUltraCoastTargetPct,
            param->postUltraGivebackPct,
            param->postUltraReaccumPct,
            param->postUltraDoubleTopPct,
            param->postUltraMaxDays,
            param->postUltraLockMinAssetPct,
            param->postUltraLockMaxAssetPct,
            param->postUltraLockGivebackPct,
            param->postUltraLockReaccumPct,
            param->postUltraLockDoubleTopPct,
            param->postUltraLockMaxDays,
            param->peakLockCapPct,
            param->peakLockUnlockGainPct,
            param->peakLockReentryStepPct,
            param->peakLockArmGainPct,
            param->peakLockGivebackPct,
            param->peakLockMaxDays,
            param->peakLockEdgeDrawPct,
            param->peakLockEdgeSlopeDays,
            param->peakLockRequireEdgeRisk,
            param->peakLockMaDays,
            param->peakLockKp,
            param->peakLockKi,
            param->peakLockKd,
            param->peakLockIntegralDecay,
            param->peakLockEntryThreshold,
            param->peakLockExitThreshold,
            param->peakLockConfirmBars,
            param->peakLockReleaseTargetPct,
            param->peakLockUltraGraceDays,
            param->seedAssetPct,
            taxName(param->taxMode)
        ) < 0
    ) {
        return -1;
    }

    if (
        fprintf(
            fp,
            "%.6f,%.6f,%.6f,"
            "%.6f,%.6f,%.6f,%.6f,"
            "%d,%.6f,%.6f,"
            "%.6f,%.6f,%.6f,%.6f,"
            "%.6f,%.6f,%.6f,"
            "%.6f,%.6f,%.6f,%.6f,"
            "%.6f,%.6f,%.6f,%.6f,"
            "%.6f,%.6f,%.6f,%.6f,"
            "%.6f,%.6f,%.6f,%.6f,"
            "%.6f,%.6f,%.6f,%.6f,"
            "%.6f,"
            "%.6f\n",
            roundMetric(row->preTaxEdge),
            roundMetric(row->postTaxEdge),
            roundMetric(row->netPctVsHodl),
            roundMetric(row->simValue),
            roundMetric(row->simPostTax),
            roundMetric(row->benchValue),
            roundMetric(row->benchPostTax),
            row->trades,
            roundMetric(row->fees),
            roundMetric(row->tax),
            roundMetric(row->potentialProfit),
            roundMetric(row->potentialProfitBench),
            roundMetric(row->netAfterTaxProfit),
            roundMetric(row->netAfterTaxProfitBench),
            roundMetric(grossEdgeVsBench),
            roundMetric(netEdgeVsBench),
            roundMetric(edgeVsBench),
            roundMetric(row->sharpe),
            roundMetric(row->sortino),
            roundMetric(row->mdd),
            roundMetric(row->cagr),
            roundMetric(row->sharpe4w),
            roundMetric(row->sortino4w),
            roundMetric(row->sharpe13w),
            roundMetric(row->sortino13w),
            roundMetric(row->sharpe4wAbs),
            roundMetric(row->sortino4wAbs),
            roundMetric(row->sharpe13wAbs),
            roundMetric(row->sortino13wAbs),
            roundMetric(row->lifecycleEdgeMean),
            roundMetric(row->lifecycleEdgeMedian),
            roundMetric(row->lifecycleEdgeP25),
            roundMetric(row->lifecycleEdgeMin),
            roundMetric(row->lifecycleUnderwaterPct),
            roundMetric(row->lifecycleUnderwaterMean),
            roundMetric(row->lifecycleTrackingPct),
            roundMetric(row->lifecycleEdgeMdd),
            roundMetric(row->lifecycleEdgeScore),
            scoreMetric
        ) < 0
    ) {
        return -1;
    }

    return 0;
}

/* Return whether one SoA batch bundle exposes all required lane arrays. */
static int validBatchParamsSoa(const BatchParamsSoa* params) {
    if (params == NULL) {
        return 0;
    }
    if (params->count < 0) {
        return 0;
    }
    if (params->count == 0) {
        return 1;
    }
    return (
        params->grad1BuyZMin != NULL
        && params->grad1SellZMin != NULL
        && params->grad1BuyWinDays != NULL
        && params->grad1SellWinDays != NULL
        && params->phaseBuy != NULL
        && params->phaseSell != NULL
        && params->finalPortionPct != NULL
        && params->cooldown != NULL
        && params->feeRate != NULL
        && params->seedQuote != NULL
        && params->seedAssetPct != NULL
        && params->taxMode != NULL
        && params->annualIncomeBase != NULL
        && params->dailyStrongSellMult != NULL
        && params->dailyStrongTargetPct != NULL
        && params->dailyBridgeDays != NULL
        && params->dailyDownBuyMult != NULL
        && params->dailyCrabAssetCapPct != NULL
        && params->dailyLockTargetPct != NULL
        && params->dailyLockGainPct != NULL
        && params->dailyLockNearHighPct != NULL
        && params->dailyLockMaxDays != NULL
        && params->postUltraCoastTargetPct != NULL
        && params->postUltraGivebackPct != NULL
        && params->postUltraReaccumPct != NULL
        && params->postUltraDoubleTopPct != NULL
        && params->postUltraMaxDays != NULL
        && params->postUltraLockMinAssetPct != NULL
        && params->postUltraLockMaxAssetPct != NULL
        && params->postUltraLockGivebackPct != NULL
        && params->postUltraLockReaccumPct != NULL
        && params->postUltraLockDoubleTopPct != NULL
        && params->postUltraLockMaxDays != NULL
        && params->macroSellRelaxPct != NULL
        && params->peakLockCapPct != NULL
        && params->peakLockUnlockGainPct != NULL
        && params->peakLockReentryStepPct != NULL
        && params->peakLockArmGainPct != NULL
        && params->peakLockGivebackPct != NULL
        && params->peakLockMaxDays != NULL
        && params->peakLockEdgeDrawPct != NULL
        && params->peakLockEdgeSlopeDays != NULL
        && params->peakLockRequireEdgeRisk != NULL
        && params->peakLockMaDays != NULL
        && params->peakLockKp != NULL
        && params->peakLockKi != NULL
        && params->peakLockKd != NULL
        && params->peakLockIntegralDecay != NULL
        && params->peakLockEntryThreshold != NULL
        && params->peakLockExitThreshold != NULL
        && params->peakLockConfirmBars != NULL
        && params->peakLockReleaseTargetPct != NULL
        && params->peakLockUltraGraceDays != NULL
    );
}

/* Decode one SoA lane into the scalar reference row. */
static void decodeBatchLaneParam(
    const BatchParamsSoa* params,
    int lane,
    BatchLaneParam* laneParam
) {
    TuneParams* out = &laneParam->param;

    out->grad1BuyZMin = params->grad1BuyZMin[lane];
    out->grad1SellZMin = params->grad1SellZMin[lane];
    out->grad1BuyWinDays = params->grad1BuyWinDays[lane];
    out->grad1SellWinDays = params->grad1SellWinDays[lane];
    out->phaseBuy = params->phaseBuy[lane];
    out->phaseSell = params->phaseSell[lane];
    out->finalPortionPct = params->finalPortionPct[lane];
    out->cooldown = params->cooldown[lane];
    out->feeRate = params->feeRate[lane];
    out->seedQuote = params->seedQuote[lane];
    out->seedAssetPct = params->seedAssetPct[lane];
    out->taxMode = params->taxMode[lane];
    out->annualIncomeBase = params->annualIncomeBase[lane];
    out->dailyStrongSellMult = params->dailyStrongSellMult[lane];
    out->dailyStrongTargetPct = params->dailyStrongTargetPct[lane];
    out->dailyBridgeDays = params->dailyBridgeDays[lane];
    out->dailyDownBuyMult = params->dailyDownBuyMult[lane];
    out->dailyCrabAssetCapPct = params->dailyCrabAssetCapPct[lane];
    out->dailyLockTargetPct = params->dailyLockTargetPct[lane];
    out->dailyLockGainPct = params->dailyLockGainPct[lane];
    out->dailyLockNearHighPct = params->dailyLockNearHighPct[lane];
    out->dailyLockMaxDays = params->dailyLockMaxDays[lane];
    out->postUltraCoastTargetPct =
        params->postUltraCoastTargetPct[lane];
    out->postUltraGivebackPct = params->postUltraGivebackPct[lane];
    out->postUltraReaccumPct = params->postUltraReaccumPct[lane];
    out->postUltraDoubleTopPct = params->postUltraDoubleTopPct[lane];
    out->postUltraMaxDays = params->postUltraMaxDays[lane];
    out->postUltraLockMinAssetPct =
        params->postUltraLockMinAssetPct[lane];
    out->postUltraLockMaxAssetPct =
        params->postUltraLockMaxAssetPct[lane];
    out->postUltraLockGivebackPct = params->postUltraLockGivebackPct[lane];
    out->postUltraLockReaccumPct = params->postUltraLockReaccumPct[lane];
    out->postUltraLockDoubleTopPct =
        params->postUltraLockDoubleTopPct[lane];
    out->postUltraLockMaxDays = params->postUltraLockMaxDays[lane];
    out->macroSellRelaxPct = params->macroSellRelaxPct[lane];
    out->peakLockCapPct = params->peakLockCapPct[lane];
    out->peakLockUnlockGainPct = params->peakLockUnlockGainPct[lane];
    out->peakLockReentryStepPct = params->peakLockReentryStepPct[lane];
    out->peakLockArmGainPct = params->peakLockArmGainPct[lane];
    out->peakLockGivebackPct = params->peakLockGivebackPct[lane];
    out->peakLockMaxDays = params->peakLockMaxDays[lane];
    out->peakLockEdgeDrawPct = params->peakLockEdgeDrawPct[lane];
    out->peakLockEdgeSlopeDays = params->peakLockEdgeSlopeDays[lane];
    out->peakLockRequireEdgeRisk = params->peakLockRequireEdgeRisk[lane];
    out->peakLockMaDays = params->peakLockMaDays[lane];
    out->peakLockKp = params->peakLockKp[lane];
    out->peakLockKi = params->peakLockKi[lane];
    out->peakLockKd = params->peakLockKd[lane];
    out->peakLockIntegralDecay = params->peakLockIntegralDecay[lane];
    out->peakLockEntryThreshold = params->peakLockEntryThreshold[lane];
    out->peakLockExitThreshold = params->peakLockExitThreshold[lane];
    out->peakLockConfirmBars = params->peakLockConfirmBars[lane];
    out->peakLockReleaseTargetPct = params->peakLockReleaseTargetPct[lane];
    out->peakLockUltraGraceDays = params->peakLockUltraGraceDays[lane];
}

/* Point one fixed-cap chunk view at its embedded lane arrays. */
static void initBatchParamChunk(BatchParamChunk* chunk) {
    chunk->params.count = 0;
    chunk->params.grad1BuyZMin = chunk->grad1BuyZMin;
    chunk->params.grad1SellZMin = chunk->grad1SellZMin;
    chunk->params.grad1BuyWinDays = chunk->grad1BuyWinDays;
    chunk->params.grad1SellWinDays = chunk->grad1SellWinDays;
    chunk->params.phaseBuy = chunk->phaseBuy;
    chunk->params.phaseSell = chunk->phaseSell;
    chunk->params.finalPortionPct = chunk->finalPortionPct;
    chunk->params.cooldown = chunk->cooldown;
    chunk->params.feeRate = chunk->feeRate;
    chunk->params.seedQuote = chunk->seedQuote;
    chunk->params.seedAssetPct = chunk->seedAssetPct;
    chunk->params.taxMode = chunk->taxMode;
    chunk->params.annualIncomeBase = chunk->annualIncomeBase;
    chunk->params.dailyStrongSellMult = chunk->dailyStrongSellMult;
    chunk->params.dailyStrongTargetPct = chunk->dailyStrongTargetPct;
    chunk->params.dailyBridgeDays = chunk->dailyBridgeDays;
    chunk->params.dailyDownBuyMult = chunk->dailyDownBuyMult;
    chunk->params.dailyCrabAssetCapPct = chunk->dailyCrabAssetCapPct;
    chunk->params.dailyLockTargetPct = chunk->dailyLockTargetPct;
    chunk->params.dailyLockGainPct = chunk->dailyLockGainPct;
    chunk->params.dailyLockNearHighPct = chunk->dailyLockNearHighPct;
    chunk->params.dailyLockMaxDays = chunk->dailyLockMaxDays;
    chunk->params.postUltraCoastTargetPct =
        chunk->postUltraCoastTargetPct;
    chunk->params.postUltraGivebackPct = chunk->postUltraGivebackPct;
    chunk->params.postUltraReaccumPct = chunk->postUltraReaccumPct;
    chunk->params.postUltraDoubleTopPct = chunk->postUltraDoubleTopPct;
    chunk->params.postUltraMaxDays = chunk->postUltraMaxDays;
    chunk->params.postUltraLockMinAssetPct =
        chunk->postUltraLockMinAssetPct;
    chunk->params.postUltraLockMaxAssetPct =
        chunk->postUltraLockMaxAssetPct;
    chunk->params.postUltraLockGivebackPct =
        chunk->postUltraLockGivebackPct;
    chunk->params.postUltraLockReaccumPct =
        chunk->postUltraLockReaccumPct;
    chunk->params.postUltraLockDoubleTopPct =
        chunk->postUltraLockDoubleTopPct;
    chunk->params.postUltraLockMaxDays = chunk->postUltraLockMaxDays;
    chunk->params.macroSellRelaxPct = chunk->macroSellRelaxPct;
    chunk->params.peakLockCapPct = chunk->peakLockCapPct;
    chunk->params.peakLockUnlockGainPct = chunk->peakLockUnlockGainPct;
    chunk->params.peakLockReentryStepPct = chunk->peakLockReentryStepPct;
    chunk->params.peakLockArmGainPct = chunk->peakLockArmGainPct;
    chunk->params.peakLockGivebackPct = chunk->peakLockGivebackPct;
    chunk->params.peakLockMaxDays = chunk->peakLockMaxDays;
    chunk->params.peakLockEdgeDrawPct = chunk->peakLockEdgeDrawPct;
    chunk->params.peakLockEdgeSlopeDays = chunk->peakLockEdgeSlopeDays;
    chunk->params.peakLockRequireEdgeRisk = chunk->peakLockRequireEdgeRisk;
    chunk->params.peakLockMaDays = chunk->peakLockMaDays;
    chunk->params.peakLockKp = chunk->peakLockKp;
    chunk->params.peakLockKi = chunk->peakLockKi;
    chunk->params.peakLockKd = chunk->peakLockKd;
    chunk->params.peakLockIntegralDecay = chunk->peakLockIntegralDecay;
    chunk->params.peakLockEntryThreshold = chunk->peakLockEntryThreshold;
    chunk->params.peakLockExitThreshold = chunk->peakLockExitThreshold;
    chunk->params.peakLockConfirmBars = chunk->peakLockConfirmBars;
    chunk->params.peakLockReleaseTargetPct = chunk->peakLockReleaseTargetPct;
    chunk->params.peakLockUltraGraceDays = chunk->peakLockUltraGraceDays;
}

/* Slice one SoA batch bundle without copying lane arrays. */
static void sliceBatchParamsSoa(
    const BatchParamsSoa* params,
    int start,
    int count,
    BatchParamsSoa* out
) {
    out->count = count;
    out->grad1BuyZMin = params->grad1BuyZMin + start;
    out->grad1SellZMin = params->grad1SellZMin + start;
    out->grad1BuyWinDays = params->grad1BuyWinDays + start;
    out->grad1SellWinDays = params->grad1SellWinDays + start;
    out->phaseBuy = params->phaseBuy + start;
    out->phaseSell = params->phaseSell + start;
    out->finalPortionPct = params->finalPortionPct + start;
    out->cooldown = params->cooldown + start;
    out->feeRate = params->feeRate + start;
    out->seedQuote = params->seedQuote + start;
    out->seedAssetPct = params->seedAssetPct + start;
    out->taxMode = params->taxMode + start;
    out->annualIncomeBase = params->annualIncomeBase + start;
    out->dailyStrongSellMult = params->dailyStrongSellMult + start;
    out->dailyStrongTargetPct = params->dailyStrongTargetPct + start;
    out->dailyBridgeDays = params->dailyBridgeDays + start;
    out->dailyDownBuyMult = params->dailyDownBuyMult + start;
    out->dailyCrabAssetCapPct = params->dailyCrabAssetCapPct + start;
    out->dailyLockTargetPct = params->dailyLockTargetPct + start;
    out->dailyLockGainPct = params->dailyLockGainPct + start;
    out->dailyLockNearHighPct = params->dailyLockNearHighPct + start;
    out->dailyLockMaxDays = params->dailyLockMaxDays + start;
    out->postUltraCoastTargetPct =
        params->postUltraCoastTargetPct + start;
    out->postUltraGivebackPct = params->postUltraGivebackPct + start;
    out->postUltraReaccumPct = params->postUltraReaccumPct + start;
    out->postUltraDoubleTopPct = params->postUltraDoubleTopPct + start;
    out->postUltraMaxDays = params->postUltraMaxDays + start;
    out->postUltraLockMinAssetPct =
        params->postUltraLockMinAssetPct + start;
    out->postUltraLockMaxAssetPct =
        params->postUltraLockMaxAssetPct + start;
    out->postUltraLockGivebackPct =
        params->postUltraLockGivebackPct + start;
    out->postUltraLockReaccumPct =
        params->postUltraLockReaccumPct + start;
    out->postUltraLockDoubleTopPct =
        params->postUltraLockDoubleTopPct + start;
    out->postUltraLockMaxDays = params->postUltraLockMaxDays + start;
    out->macroSellRelaxPct = params->macroSellRelaxPct + start;
    out->peakLockCapPct = params->peakLockCapPct + start;
    out->peakLockUnlockGainPct = params->peakLockUnlockGainPct + start;
    out->peakLockReentryStepPct = params->peakLockReentryStepPct + start;
    out->peakLockArmGainPct = params->peakLockArmGainPct + start;
    out->peakLockGivebackPct = params->peakLockGivebackPct + start;
    out->peakLockMaxDays = params->peakLockMaxDays + start;
    out->peakLockEdgeDrawPct = params->peakLockEdgeDrawPct + start;
    out->peakLockEdgeSlopeDays = params->peakLockEdgeSlopeDays + start;
    out->peakLockRequireEdgeRisk = params->peakLockRequireEdgeRisk + start;
    out->peakLockMaDays = params->peakLockMaDays + start;
    out->peakLockKp = params->peakLockKp + start;
    out->peakLockKi = params->peakLockKi + start;
    out->peakLockKd = params->peakLockKd + start;
    out->peakLockIntegralDecay = params->peakLockIntegralDecay + start;
    out->peakLockEntryThreshold = params->peakLockEntryThreshold + start;
    out->peakLockExitThreshold = params->peakLockExitThreshold + start;
    out->peakLockConfirmBars = params->peakLockConfirmBars + start;
    out->peakLockReleaseTargetPct = params->peakLockReleaseTargetPct + start;
    out->peakLockUltraGraceDays = params->peakLockUltraGraceDays + start;
}

/* Materialize one cartesian sweep chunk into the fixed batch SoA shape. */
static void fillBatchParamChunk(
    BatchParamChunk* chunk,
    const TuneParams* baseParam,
    const TuneAxes* axes,
    uint64_t comboStart,
    int comboCount
) {
    int lane;
    TuneParams param;

    chunk->params.count = comboCount;
    for (lane = 0; lane < comboCount; lane++) {
        decodeSweepParam(
            baseParam,
            axes,
            comboStart + (uint64_t)lane,
            &param
        );
        chunk->grad1BuyZMin[lane] = param.grad1BuyZMin;
        chunk->grad1SellZMin[lane] = param.grad1SellZMin;
        chunk->grad1BuyWinDays[lane] = param.grad1BuyWinDays;
        chunk->grad1SellWinDays[lane] = param.grad1SellWinDays;
        chunk->phaseBuy[lane] = param.phaseBuy;
        chunk->phaseSell[lane] = param.phaseSell;
        chunk->finalPortionPct[lane] = param.finalPortionPct;
        chunk->cooldown[lane] = param.cooldown;
        chunk->feeRate[lane] = param.feeRate;
        chunk->seedQuote[lane] = param.seedQuote;
        chunk->seedAssetPct[lane] = param.seedAssetPct;
        chunk->taxMode[lane] = param.taxMode;
        chunk->annualIncomeBase[lane] = param.annualIncomeBase;
        chunk->dailyStrongSellMult[lane] = param.dailyStrongSellMult;
        chunk->dailyStrongTargetPct[lane] = param.dailyStrongTargetPct;
        chunk->dailyBridgeDays[lane] = param.dailyBridgeDays;
        chunk->dailyDownBuyMult[lane] = param.dailyDownBuyMult;
        chunk->dailyCrabAssetCapPct[lane] = param.dailyCrabAssetCapPct;
        chunk->dailyLockTargetPct[lane] = param.dailyLockTargetPct;
        chunk->dailyLockGainPct[lane] = param.dailyLockGainPct;
        chunk->dailyLockNearHighPct[lane] = param.dailyLockNearHighPct;
        chunk->dailyLockMaxDays[lane] = param.dailyLockMaxDays;
        chunk->postUltraCoastTargetPct[lane] =
            param.postUltraCoastTargetPct;
        chunk->postUltraGivebackPct[lane] = param.postUltraGivebackPct;
        chunk->postUltraReaccumPct[lane] = param.postUltraReaccumPct;
        chunk->postUltraDoubleTopPct[lane] = param.postUltraDoubleTopPct;
        chunk->postUltraMaxDays[lane] = param.postUltraMaxDays;
        chunk->postUltraLockMinAssetPct[lane] =
            param.postUltraLockMinAssetPct;
        chunk->postUltraLockMaxAssetPct[lane] =
            param.postUltraLockMaxAssetPct;
        chunk->postUltraLockGivebackPct[lane] =
            param.postUltraLockGivebackPct;
        chunk->postUltraLockReaccumPct[lane] =
            param.postUltraLockReaccumPct;
        chunk->postUltraLockDoubleTopPct[lane] =
            param.postUltraLockDoubleTopPct;
        chunk->postUltraLockMaxDays[lane] = param.postUltraLockMaxDays;
        chunk->macroSellRelaxPct[lane] = param.macroSellRelaxPct;
        chunk->peakLockCapPct[lane] = param.peakLockCapPct;
        chunk->peakLockUnlockGainPct[lane] = param.peakLockUnlockGainPct;
        chunk->peakLockReentryStepPct[lane] = param.peakLockReentryStepPct;
        chunk->peakLockArmGainPct[lane] = param.peakLockArmGainPct;
        chunk->peakLockGivebackPct[lane] = param.peakLockGivebackPct;
        chunk->peakLockMaxDays[lane] = param.peakLockMaxDays;
        chunk->peakLockEdgeDrawPct[lane] = param.peakLockEdgeDrawPct;
        chunk->peakLockEdgeSlopeDays[lane] = param.peakLockEdgeSlopeDays;
        chunk->peakLockRequireEdgeRisk[lane] =
            param.peakLockRequireEdgeRisk;
        chunk->peakLockMaDays[lane] = param.peakLockMaDays;
        chunk->peakLockKp[lane] = param.peakLockKp;
        chunk->peakLockKi[lane] = param.peakLockKi;
        chunk->peakLockKd[lane] = param.peakLockKd;
        chunk->peakLockIntegralDecay[lane] = param.peakLockIntegralDecay;
        chunk->peakLockEntryThreshold[lane] = param.peakLockEntryThreshold;
        chunk->peakLockExitThreshold[lane] = param.peakLockExitThreshold;
        chunk->peakLockConfirmBars[lane] = param.peakLockConfirmBars;
        chunk->peakLockReleaseTargetPct[lane] =
            param.peakLockReleaseTargetPct;
        chunk->peakLockUltraGraceDays[lane] = param.peakLockUltraGraceDays;
    }
}

/* Prime z-series refs and reset per-lane flag state for one chunk. */
static int primeBatchFlags(
    PreparedDatasetState* prepared,
    const BatchParamsSoa* params,
    BatchLaneState* batch
) {
    int lane;
    int n = prepared->micro->n;

    for (lane = 0; lane < params->count; lane++) {
        int cooldown;
        int gradBuyBars;
        int gradSellBars;

        memset(laneFlagSlice(batch->buyFlags, lane, n), 0, (size_t)n);
        memset(laneFlagSlice(batch->sellFlags, lane, n), 0, (size_t)n);

        cooldown = params->cooldown[lane];
        if (cooldown < 0) {
            cooldown = 0;
        }
        batch->cooldown[lane] = cooldown;
        batch->buyLastCd[lane] = -cooldown;
        batch->sellLastCd[lane] = -cooldown;
        batch->buyLastAccepted[lane] = -1;
        batch->sellLastAccepted[lane] = -1;
        batch->buyLastPhase[lane] = -1;
        batch->sellLastPhase[lane] = -1;
        batch->buyCounts[lane] = 0;
        batch->sellCounts[lane] = 0;

        gradBuyBars = (int)round(
            (double)params->grad1BuyWinDays[lane]
            * prepared->microDerived.barsPerDay
        );
        if (gradBuyBars < 1) {
            gradBuyBars = 1;
        }
        gradSellBars = (int)round(
            (double)params->grad1SellWinDays[lane]
            * prepared->microDerived.barsPerDay
        );
        if (gradSellBars < 1) {
            gradSellBars = 1;
        }

        batch->gradBuyZ[lane] = cachedZSeries(
            &prepared->zCache.grad,
            prepared->microDerived.g1p1,
            n,
            gradBuyBars,
            prepared->scratch.meanA,
            prepared->scratch.stdA
        );
        batch->gradSellZ[lane] = cachedZSeries(
            &prepared->zCache.grad,
            prepared->microDerived.g1p1,
            n,
            gradSellBars,
            prepared->scratch.meanA,
            prepared->scratch.stdA
        );

        if (
            batch->gradBuyZ[lane] == NULL
            || batch->gradSellZ[lane] == NULL
        ) {
            return -1;
        }
    }

    return 0;
}

/* Walk the candles once and emit BUY/SELL flags for the full lane chunk. */
static int markFlagsBatch(
    const MicroSoa* micro,
    PreparedDatasetState* prepared,
    int startIdx,
    const BatchParamsSoa* params,
    BatchLaneState* batch
) {
    int i;
    int lane;
    int begin;
    int n = micro->n;

    begin = startIdx;
    if (begin < 0) {
        begin = 0;
    }
    if (begin >= n) {
        return 0;
    }
    if (primeBatchFlags(prepared, params, batch) != 0) {
        return -1;
    }
    for (i = begin; i < n; i++) {
        double priceNow = micro->closes[i];
        double dynAbs = fabs(prepared->macroDerived.dyn[i]);
        int buyAnchor = prepared->buyAnchors[i];
        int sellAnchor = prepared->sellAnchors[i];

        if (prepared->allowBuy[i]) {
            for (lane = 0; lane < params->count; lane++) {
                double gradScore = -batch->gradBuyZ[lane][i];
                double deltaPct;
                double mult;
                double priceRef;
                double req;
                int refIdx;

                if (
                    !isfinite(gradScore)
                    || gradScore < params->grad1BuyZMin[lane]
                ) {
                    continue;
                }
                if ((i - batch->buyLastCd[lane]) < batch->cooldown[lane]) {
                    continue;
                }
                batch->buyLastCd[lane] = i;
                if (buyAnchor != batch->buyLastPhase[lane]) {
                    batch->buyLastAccepted[lane] = -1;
                    batch->buyLastPhase[lane] = buyAnchor;
                }

                refIdx = batch->buyLastAccepted[lane] >= 0
                    ? batch->buyLastAccepted[lane]
                    : buyAnchor;
                if (refIdx >= 0) {
                    priceRef = micro->closes[refIdx];
                    if (priceNow > 0.0 && priceRef > 0.0) {
                        deltaPct = ((priceRef / priceNow) - 1.0) * 100.0;
                    }
                    else {
                        deltaPct = 0.0;
                    }
                    mult = 1.0;
                    req = dynAbs * mult;
                    if (req < 0.0) {
                        req = 0.0;
                    }
                    if (deltaPct < req) {
                        continue;
                    }
                }

                batch->buyLastAccepted[lane] = i;
                laneFlagSlice(batch->buyFlags, lane, n)[i] = 1;
                batch->buyCounts[lane] += 1;
            }
        }

        if (prepared->allowSell[i]) {
            for (lane = 0; lane < params->count; lane++) {
                double gradScore = batch->gradSellZ[lane][i];
                double deltaPct;
                double mult;
                double priceRef;
                double req;
                int refIdx;

                if (
                    !isfinite(gradScore)
                    || gradScore < params->grad1SellZMin[lane]
                ) {
                    continue;
                }
                if (
                    (i - batch->sellLastCd[lane]) < batch->cooldown[lane]
                ) {
                    continue;
                }
                batch->sellLastCd[lane] = i;
                if (sellAnchor != batch->sellLastPhase[lane]) {
                    batch->sellLastAccepted[lane] = -1;
                    batch->sellLastPhase[lane] = sellAnchor;
                }

                refIdx = batch->sellLastAccepted[lane] >= 0
                    ? batch->sellLastAccepted[lane]
                    : sellAnchor;
                if (refIdx >= 0) {
                    priceRef = micro->closes[refIdx];
                    if (priceNow > 0.0 && priceRef > 0.0) {
                        deltaPct = ((priceNow / priceRef) - 1.0) * 100.0;
                    }
                    else {
                        deltaPct = 0.0;
                    }
                    mult = 1.0;
                    req = dynAbs * mult;
                    req *= 1.0 - (clipVal(
                        params->macroSellRelaxPct[lane],
                        0.0,
                        100.0
                    ) / 100.0);
                    if (req < 0.0) {
                        req = 0.0;
                    }
                    if (deltaPct < req) {
                        continue;
                    }
                }

                batch->sellLastAccepted[lane] = i;
                laneFlagSlice(batch->sellFlags, lane, n)[i] = 1;
                batch->sellCounts[lane] += 1;
            }
        }
    }

    return 0;
}

/* Format an ETA countdown as HH:MM:SS. */
static void formatEta(double seconds, char* out, int outLen) {
    int total;
    int hours;
    int minutes;
    int secs;

    if (seconds < 0.0) {
        seconds = 0.0;
    }
    total = (int)seconds;
    hours = total / 3600;
    minutes = (total % 3600) / 60;
    secs = total % 60;
    snprintf(out, (size_t)outLen, "%02d:%02d:%02d", hours, minutes, secs);
}

/* Print one throttled runtime progress update to stderr. */
static void printProgress(
    uint64_t doneCount,
    uint64_t totalCount,
    double startSecs,
    const TuneGroupMeta* meta
) {
    char etaBuf[16];
    double elapsed;
    double fraction;
    double rate;
    double remaining;

    if (totalCount == 0) {
        return;
    }

    elapsed = nowSecs() - startSecs;
    if (elapsed < 1e-6) {
        elapsed = 1e-6;
    }
    fraction = (double)doneCount / (double)totalCount;
    if (fraction < 0.0) {
        fraction = 0.0;
    }
    if (fraction > 1.0) {
        fraction = 1.0;
    }
    rate = (double)doneCount / elapsed;
    remaining = rate > 0.0
        ? ((double)(totalCount - doneCount) / rate)
        : 0.0;
    formatEta(remaining, etaBuf, (int)sizeof(etaBuf));
    if (meta != NULL && meta->interval != NULL && meta->macroInterval != NULL) {
        fprintf(
            stderr,
            "\r[host] %6.2f%% %llu/%llu %s(%d,%d,%d) %s(%d,%d,%d) "
            "%.0f/s ETA %s",
            fraction * 100.0,
            (unsigned long long)doneCount,
            (unsigned long long)totalCount,
            meta->interval,
            meta->p1,
            meta->p2,
            meta->p3,
            meta->macroInterval,
            meta->macroP1,
            meta->macroGradPeriod,
            meta->macroP3,
            rate,
            etaBuf
        );
    }
    else {
        fprintf(
            stderr,
            "\r[host] %6.2f%% %llu/%llu %.0f/s ETA %s",
            fraction * 100.0,
            (unsigned long long)doneCount,
            (unsigned long long)totalCount,
            rate,
            etaBuf
        );
    }
    if (doneCount >= totalCount) {
        fputc('\n', stderr);
    }
    fflush(stderr);
}

static int dailyClusterIsDown(int cluster, int mask) {
    if (cluster < 0 || cluster >= 30) {
        return 0;
    }
    return (mask & (1 << cluster)) != 0;
}

static int dailyPostureStepRaw(
    const MicroSoa* micro,
    const BatchParamsSoa* params,
    BatchLaneState* batch,
    int lane,
    int index,
    double price,
    double barsDay,
    int* strongOut,
    int* downOut,
    int* downEntryOut,
    int* forceLockOut,
    double* lockTargetOut
) {
    int cluster;
    int rawStrong;
    int strongNow;
    int downNow;
    int downEntry;
    int exitTrigger;
    int wasActive;
    int bridgeMaxBars;
    double entryPrice;
    double gainPct;
    double peakGainPct;
    double gainMinPct;
    double gainMaxPct;
    double givebackPct;
    double gainSpan;
    double score;
    double exitDepth;
    double targetPct;
    double lockDays;
    double coastTargetPct;
    double coastGivebackPct;
    double coastReaccumPct;
    double coastDoubleTopPct;
    double coastMaxDays;
    double cloudMinAssetPct;
    double cloudMaxAssetPct;
    double cloudGivebackPct;
    double cloudReaccumPct;
    double cloudDoubleTopPct;
    double cloudMaxDays;
    double oldPeakPrice;
    double reaccumPrice;
    double coastDays;
    int cloudEnabled;
    int cloudTriggered;
    int coastRelease;
    int doubleTopRelease;
    int releaseTimed;

    *strongOut = 0;
    *downOut = 0;
    *downEntryOut = 0;
    *forceLockOut = 0;
    *lockTargetOut = 1.0;
    if (
        micro->dailyCluster == NULL
        || micro->dailyRet30 == NULL
        || micro->dailyNearHigh == NULL
    ) {
        return 0;
    }

    cluster = micro->dailyCluster[index];
    rawStrong = cluster == DAILY_STRONG_CLUSTER;
    downNow = dailyClusterIsDown(cluster, DAILY_DOWN_MASK);
    *downOut = downNow;
    downEntry = downNow && !batch->dailyPrevDown[lane];
    *downEntryOut = downEntry;
    batch->dailyPrevDown[lane] = downNow;
    wasActive = batch->dailyPrevStrong[lane];
    bridgeMaxBars = (int)round(
        clipVal(params->dailyBridgeDays[lane], 0.0, 3650.0)
        * (barsDay > 0.0 ? barsDay : 1.0)
    );
    coastTargetPct = clipVal(
        params->postUltraCoastTargetPct[lane],
        0.0,
        1.0
    );
    coastGivebackPct = fmax(params->postUltraGivebackPct[lane], 0.0);
    coastReaccumPct = clipVal(
        params->postUltraReaccumPct[lane],
        -95.0,
        1000.0
    );
    coastDoubleTopPct = fmax(params->postUltraDoubleTopPct[lane], 0.0);
    coastMaxDays = fmax(params->postUltraMaxDays[lane], 0.0);
    cloudMinAssetPct = clipVal(
        params->postUltraLockMinAssetPct[lane],
        0.0,
        1.0
    );
    cloudMaxAssetPct = clipVal(
        params->postUltraLockMaxAssetPct[lane],
        0.0,
        1.0
    );
    if (
        params->postUltraLockGivebackPct[lane] <= 0.0
        && coastGivebackPct > 0.0
        && cloudMaxAssetPct >= 1.0 - 1e-9
    ) {
        cloudMinAssetPct = coastTargetPct;
        cloudMaxAssetPct = coastTargetPct;
    }
    if (cloudMaxAssetPct < cloudMinAssetPct) {
        cloudMaxAssetPct = cloudMinAssetPct;
    }
    cloudGivebackPct = (
        params->postUltraLockGivebackPct[lane] > 0.0
        ? fmax(params->postUltraLockGivebackPct[lane], 0.0)
        : coastGivebackPct
    );
    cloudReaccumPct = clipVal(
        (
            params->postUltraLockGivebackPct[lane] > 0.0
            ? params->postUltraLockReaccumPct[lane]
            : coastReaccumPct
        ),
        -95.0,
        1000.0
    );
    cloudDoubleTopPct = (
        params->postUltraLockGivebackPct[lane] > 0.0
        ? fmax(params->postUltraLockDoubleTopPct[lane], 0.0)
        : coastDoubleTopPct
    );
    cloudMaxDays = fmax(
        (
            params->postUltraLockGivebackPct[lane] > 0.0
            ? params->postUltraLockMaxDays[lane]
            : coastMaxDays
        ),
        0.0
    );
    cloudEnabled = cloudMaxAssetPct < 1.0 - 1e-9
        && cloudGivebackPct > 0.0;
    oldPeakPrice = batch->dailyUltraPeakPrice[lane];
    cloudTriggered = 0;
    coastRelease = 0;
    doubleTopRelease = (
        batch->dailyCoastActive[lane]
        && rawStrong
        && cloudDoubleTopPct > 0.0
        && oldPeakPrice > 0.0
        && price >= oldPeakPrice * (1.0 - (cloudDoubleTopPct / 100.0))
    );
    if (doubleTopRelease) {
        batch->dailyCoastActive[lane] = 0;
        batch->dailyCoastStart[lane] = -1;
        batch->dailyCoastTargetPct[lane] = 1.0;
        batch->dailyCoastMinAssetPct[lane] = 0.0;
        batch->dailyCoastMaxAssetPct[lane] = 1.0;
        batch->dailyLockActive[lane] = 0;
        batch->dailyLockStart[lane] = -1;
        batch->dailyLockTargetPct[lane] = 1.0;
        batch->dailyLockHoldDays[lane] = 0.0;
        batch->dailyLockReleaseOnStrong[lane] = 0;
        coastRelease = 1;
    }

    if (rawStrong) {
        if (!wasActive && !doubleTopRelease) {
            batch->dailyEpisodeLocked[lane] = 0;
            batch->dailyUltraEntryPrice[lane] = price;
            batch->dailyUltraPeakPrice[lane] = price;
        }
        else if (doubleTopRelease) {
            if (batch->dailyUltraEntryPrice[lane] <= 0.0) {
                batch->dailyUltraEntryPrice[lane] = price;
            }
            if (price > batch->dailyUltraPeakPrice[lane]) {
                batch->dailyUltraPeakPrice[lane] = price;
            }
        }
        batch->dailyBridgeBars[lane] = 0;
        strongNow = 1;
        batch->dailyStrongDays[lane] = (
            wasActive ? batch->dailyStrongDays[lane] + 1 : 1
        );
    } else if (wasActive) {
        batch->dailyBridgeBars[lane] += 1;
        strongNow = batch->dailyBridgeBars[lane] <= bridgeMaxBars;
        if (strongNow) {
            batch->dailyStrongDays[lane] += 1;
        } else {
            batch->dailyStrongDays[lane] = 0;
        }
    } else {
        strongNow = 0;
        batch->dailyBridgeBars[lane] = 0;
        batch->dailyStrongDays[lane] = 0;
    }

    if (strongNow && price > batch->dailyUltraPeakPrice[lane]) {
        batch->dailyUltraPeakPrice[lane] = price;
    }
    *strongOut = strongNow;

    entryPrice = batch->dailyUltraEntryPrice[lane];
    peakGainPct = (
        entryPrice > 0.0
        ? ((batch->dailyUltraPeakPrice[lane] / entryPrice) - 1.0) * 100.0
        : 0.0
    );
    givebackPct = (
        batch->dailyUltraPeakPrice[lane] > 0.0 && price > 0.0
        ? ((batch->dailyUltraPeakPrice[lane] / price) - 1.0) * 100.0
        : 0.0
    );
    gainMinPct = params->dailyLockGainPct[lane];

    if (strongNow && !batch->dailyEpisodeLocked[lane]) {
        gainMaxPct = params->dailyLockNearHighPct[lane];
        exitDepth = clipVal(params->dailyLockTargetPct[lane], 0.0, 1.0);
        if (exitDepth > 0.0 && peakGainPct >= gainMaxPct) {
            targetPct = clipVal(1.0 - exitDepth, 0.0, 1.0);
            batch->dailyLockTargetPct[lane] = targetPct;
            batch->dailyLockHoldDays[lane] =
                (double)params->dailyLockMaxDays[lane];
            batch->dailyLockReleaseOnStrong[lane] = 0;
            *forceLockOut = 1;
            *lockTargetOut = targetPct;
        }
    }

    exitTrigger = wasActive
        && !strongNow
        && !batch->dailyEpisodeLocked[lane];
    if (exitTrigger) {
        gainPct = peakGainPct;
        gainMaxPct = params->dailyLockNearHighPct[lane];
        gainSpan = gainMaxPct - gainMinPct;
        if (gainSpan < 1e-12) {
            gainSpan = 1e-12;
        }
        score = clipVal((gainPct - gainMinPct) / gainSpan, 0.0, 1.0);
        exitDepth = clipVal(params->dailyLockTargetPct[lane], 0.0, 1.0);
        targetPct = clipVal(1.0 - (exitDepth * score), 0.0, 1.0);
        batch->dailyLockTargetPct[lane] = targetPct;
        batch->dailyLockHoldDays[lane] = (
            (double)params->dailyLockMaxDays[lane] * score
        );
        batch->dailyLockReleaseOnStrong[lane] = 1;
        *forceLockOut = score > 0.0 && exitDepth > 0.0;
        *lockTargetOut = targetPct;
    }

    cloudTriggered = (
        cloudEnabled
        && !batch->dailyCoastActive[lane]
        && peakGainPct >= gainMinPct
        && givebackPct >= cloudGivebackPct
    );
    if (cloudTriggered) {
        batch->dailyCoastActive[lane] = 1;
        batch->dailyCoastStart[lane] = index;
        batch->dailyCoastTargetPct[lane] = cloudMaxAssetPct;
        batch->dailyCoastMinAssetPct[lane] = cloudMinAssetPct;
        batch->dailyCoastMaxAssetPct[lane] = cloudMaxAssetPct;
        batch->dailyLockTargetPct[lane] = cloudMinAssetPct;
        batch->dailyLockHoldDays[lane] = cloudMaxDays;
        batch->dailyLockReleaseOnStrong[lane] = 0;
        *forceLockOut = 1;
        *lockTargetOut = cloudMinAssetPct;
    }

    if (!strongNow) {
        batch->dailyBridgeBars[lane] = 0;
    }

    lockDays = (
        batch->dailyLockStart[lane] >= 0
        ? (
            (double)(index - batch->dailyLockStart[lane])
            / (barsDay > 0.0 ? barsDay : 1.0)
        )
        : 0.0
    );
    if (
        batch->dailyLockActive[lane]
        && (
            (
                strongNow
                && batch->dailyLockReleaseOnStrong[lane]
            )
            || lockDays >= batch->dailyLockHoldDays[lane]
        )
    ) {
        releaseTimed = lockDays >= batch->dailyLockHoldDays[lane];
        if (
            batch->dailyCoastActive[lane]
            && batch->dailyLockHoldDays[lane] <= 0.0
        ) {
            releaseTimed = 0;
        }
        if (
            (
                strongNow
                && batch->dailyLockReleaseOnStrong[lane]
            )
            || releaseTimed
        ) {
            batch->dailyLockActive[lane] = 0;
            batch->dailyLockStart[lane] = -1;
            batch->dailyLockTargetPct[lane] = 1.0;
            batch->dailyLockHoldDays[lane] = 0.0;
            batch->dailyLockReleaseOnStrong[lane] = 0;
            if (batch->dailyCoastActive[lane]) {
                batch->dailyCoastActive[lane] = 0;
                batch->dailyCoastStart[lane] = -1;
                batch->dailyCoastTargetPct[lane] = 1.0;
                batch->dailyCoastMinAssetPct[lane] = 0.0;
                batch->dailyCoastMaxAssetPct[lane] = 1.0;
            }
        }
    }
    if (batch->dailyCoastActive[lane] && !cloudTriggered) {
        reaccumPrice = entryPrice * (1.0 + (cloudReaccumPct / 100.0));
        coastDays = (
            batch->dailyCoastStart[lane] >= 0
            ? (
                (double)(index - batch->dailyCoastStart[lane])
                / (barsDay > 0.0 ? barsDay : 1.0)
            )
            : 0.0
        );
        if (entryPrice > 0.0 && price <= reaccumPrice) {
            coastRelease = 1;
        }
        if (cloudMaxDays > 0.0 && coastDays >= cloudMaxDays) {
            coastRelease = 1;
        }
        if (coastRelease) {
            batch->dailyCoastActive[lane] = 0;
            batch->dailyCoastStart[lane] = -1;
            batch->dailyCoastTargetPct[lane] = 1.0;
            batch->dailyCoastMinAssetPct[lane] = 0.0;
            batch->dailyCoastMaxAssetPct[lane] = 1.0;
            batch->dailyLockActive[lane] = 0;
            batch->dailyLockStart[lane] = -1;
            batch->dailyLockTargetPct[lane] = 1.0;
            batch->dailyLockHoldDays[lane] = 0.0;
            batch->dailyLockReleaseOnStrong[lane] = 0;
        }
    }
    batch->dailyPrevStrong[lane] = strongNow;
    return 0;
}

/* Evaluate one decoded sweep chunk against prepared dataset + lane scratch. */
static int evalSweepRows(
    const MicroSoa* micro,
    PreparedDatasetState* prepared,
    EvalLaneState* lane,
    int startIdx,
    const TuneParams* baseParam,
    const TuneAxes* axes,
    uint64_t comboStart,
    int comboCount,
    EvalRow* rows
) {
    int i;
    TuneParams param;

    if (
        micro == NULL
        || prepared == NULL
        || lane == NULL
        || baseParam == NULL
        || rows == NULL
    ) {
        return -1;
    }

    for (i = 0; i < comboCount; i++) {
        zeroRow(&rows[i]);
    }

    if (micro->n == 0 || comboCount <= 0) {
        return 0;
    }

    for (i = 0; i < comboCount; i++) {
        decodeSweepParam(
            baseParam,
            axes,
            comboStart + (uint64_t)i,
            &param
        );
        if (
            evaluateRow(
                micro,
                prepared,
                lane,
                &param,
                startIdx,
                &rows[i]
            ) != 0
        ) {
            return -1;
        }
    }

    return 0;
}

/* Evaluate a batch of tuning rows. */
int evalBatch(
    const MicroSoa* micro,
    const MacroSoa* macro,
    int startIdx,
    const TuneParams* params,
    int paramCount,
    EvalRow* rows
) {
    int i;
    int n;
    PreparedDatasetState prepared;
    EvalLaneState lane;

    if (micro == NULL || params == NULL || rows == NULL) {
        return -1;
    }

    n = micro->n;
    if (n < 0 || micro->closes == NULL || micro->ts == NULL) {
        return -1;
    }

    for (i = 0; i < paramCount; i++) {
        zeroRow(&rows[i]);
    }

    if (n == 0) {
        return 0;
    }

    if (initPreparedDataset(micro, macro, &prepared) != 0) {
        return -1;
    }
    if (initEvalLane(&lane, n) != 0) {
        freePreparedDataset(&prepared);
        return -1;
    }

    for (i = 0; i < paramCount; i++) {
        if (
            evaluateRow(
                micro,
                &prepared,
                &lane,
                &params[i],
                startIdx,
                &rows[i]
            ) != 0
        ) {
            freeEvalLane(&lane);
            freePreparedDataset(&prepared);
            return -1;
        }
    }

    freeEvalLane(&lane);
    freePreparedDataset(&prepared);
    return 0;
}

static int peakLockEnabled(const BatchParamsSoa* params, int lane) {
    return params->peakLockCapPct[lane] < (1.0 - 1e-9);
}

static int peakMaBars(
    const BatchParamsSoa* params,
    int lane,
    double barsDay
) {
    int bars = (int)round(params->peakLockMaDays[lane] * barsDay);

    return bars < 2 ? 2 : bars;
}

static void warmPeakPidRaw(
    const MicroSoa* micro,
    const BatchParamsSoa* params,
    BatchLaneState* batch,
    int lane,
    int startIdx,
    double barsDay
) {
    int i;
    double alpha;

    batch->peakMa[lane] = micro->n > 0 ? micro->closes[0] : 0.0;
    batch->peakIntegral[lane] = 0.0;
    batch->peakPrevErr[lane] = 0.0;
    batch->peakLong[lane] = 0;
    batch->peakBearCount[lane] = 0;
    if (!peakLockEnabled(params, lane)) {
        return;
    }
    alpha = 2.0 / ((double)peakMaBars(params, lane, barsDay) + 1.0);
    for (i = 0; i < startIdx && i < micro->n; i++) {
        double price = micro->closes[i];
        double err;

        batch->peakMa[lane] = (
            (alpha * price) + ((1.0 - alpha) * batch->peakMa[lane])
        );
        err = (
            batch->peakMa[lane] > 1e-12
            ? (price - batch->peakMa[lane]) / batch->peakMa[lane]
            : 0.0
        );
        batch->peakIntegral[lane] = (
            params->peakLockIntegralDecay[lane] * batch->peakIntegral[lane]
            + err
        );
        batch->peakPrevErr[lane] = err;
    }
    batch->peakBearCount[lane] = 0;
}

static void stepPeakPidRaw(
    const BatchParamsSoa* params,
    BatchLaneState* batch,
    int lane,
    double price,
    double barsDay
) {
    double alpha;
    double err;
    double deriv;
    double raw;

    if (!peakLockEnabled(params, lane)) {
        return;
    }
    alpha = 2.0 / ((double)peakMaBars(params, lane, barsDay) + 1.0);
    batch->peakMa[lane] = (
        (alpha * price) + ((1.0 - alpha) * batch->peakMa[lane])
    );
    err = (
        batch->peakMa[lane] > 1e-12
        ? (price - batch->peakMa[lane]) / batch->peakMa[lane]
        : 0.0
    );
    deriv = err - batch->peakPrevErr[lane];
    batch->peakIntegral[lane] = (
        params->peakLockIntegralDecay[lane] * batch->peakIntegral[lane]
        + err
    );
    raw = (
        params->peakLockKp[lane] * err
        + params->peakLockKi[lane] * batch->peakIntegral[lane]
        + params->peakLockKd[lane] * deriv
    );
    if (raw > params->peakLockEntryThreshold[lane]) {
        batch->peakLong[lane] = 1;
    } else if (raw < params->peakLockExitThreshold[lane]) {
        batch->peakLong[lane] = 0;
    }
    if (batch->peakLong[lane]) {
        batch->peakBearCount[lane] = 0;
    } else {
        batch->peakBearCount[lane] += 1;
    }
    batch->peakPrevErr[lane] = err;
}

static void stepPeakStrongRaw(
    const BatchParamsSoa* params,
    BatchLaneState* batch,
    int lane,
    int strongNow,
    double barsDay,
    int* strongEntryOut,
    int* graceOut
) {
    int graceBars;

    *strongEntryOut = strongNow && !batch->peakPrevStrong[lane];
    *graceOut = 0;
    if (!peakLockEnabled(params, lane)) {
        batch->peakPrevStrong[lane] = strongNow;
        return;
    }
    graceBars = (int)round(
        clipVal(params->peakLockUltraGraceDays[lane], 0.0, 3650.0)
        * barsDay
    );
    if (strongNow) {
        batch->peakStrongGraceBars[lane] = graceBars;
    } else if (batch->peakStrongGraceBars[lane] > 0) {
        batch->peakStrongGraceBars[lane] -= 1;
    }
    *graceOut = !strongNow && batch->peakStrongGraceBars[lane] > 0;
    batch->peakPrevStrong[lane] = strongNow;
}

static void armPeakLockRaw(
    const BatchParamsSoa* params,
    BatchLaneState* batch,
    int lane,
    int strongNow,
    double ultraGainPct
) {
    if (
        peakLockEnabled(params, lane)
        && strongNow
        && ultraGainPct >= params->peakLockArmGainPct[lane]
    ) {
        batch->peakArmed[lane] = 1;
    }
}

static int evaluatePeakLockRaw(
    const BatchParamsSoa* params,
    BatchLaneState* batch,
    WalletState* wallet,
    int lane,
    int offset,
    double price,
    double givebackPct,
    int strongEntry,
    int graceActive,
    double barsDay
) {
    double benchValue;
    double walletValue;
    double edgeDraw;
    double slope = 0.0;
    double* edgeVals;
    int edgeSlopeBars;
    int peakRisk;
    int edgeRisk;

    if (!peakLockEnabled(params, lane)) {
        return 0;
    }
    benchValue = batch->benchQty[lane] * price;
    walletValue = portfolioValue(wallet, price);
    batch->peakEdgeNow[lane] = (
        benchValue > 0.0
        ? ((walletValue / benchValue) - 1.0) * 100.0
        : 0.0
    );
    if (batch->peakEdgeNow[lane] > batch->peakEdgePeak[lane]) {
        batch->peakEdgePeak[lane] = batch->peakEdgeNow[lane];
    }
    edgeDraw = batch->peakEdgePeak[lane] - batch->peakEdgeNow[lane];
    edgeVals = lanePeakEdgeSlice(batch, lane);
    edgeVals[offset] = batch->peakEdgeNow[lane];
    edgeSlopeBars = (int)round(params->peakLockEdgeSlopeDays[lane] * barsDay);
    if (edgeSlopeBars < 0) {
        edgeSlopeBars = 0;
    }
    if (offset >= edgeSlopeBars && edgeSlopeBars > 0) {
        slope = edgeVals[offset] - edgeVals[offset - edgeSlopeBars];
    }
    if (
        strongEntry
        && batch->peakActive[lane]
        && params->peakLockReleaseTargetPct[lane] > 0.0
    ) {
        double oldCap = batch->peakCap[lane];

        batch->peakCap[lane] = fmax(
            batch->peakCap[lane],
            clipVal(params->peakLockReleaseTargetPct[lane], 0.0, 1.0)
        );
        if (batch->peakCap[lane] > oldCap + 1e-9) {
            batch->peakStrongReleases[lane] += 1;
            batch->peakEdgeStart[lane] = batch->peakEdgeNow[lane];
        }
        if (batch->peakCap[lane] >= 1.0 - 1e-9) {
            batch->peakActive[lane] = 0;
            batch->peakStart[lane] = -1;
            batch->peakCap[lane] = 1.0;
        }
        batch->peakArmed[lane] = 0;
        batch->peakBearCount[lane] = 0;
        batch->peakLong[lane] = 1;
    }

    peakRisk = givebackPct >= params->peakLockGivebackPct[lane];
    edgeRisk = (
        edgeDraw >= params->peakLockEdgeDrawPct[lane]
        && slope < 0.0
    );
    return (
        batch->peakArmed[lane]
        && !batch->peakActive[lane]
        && batch->peakBearCount[lane] >= params->peakLockConfirmBars[lane]
        && !graceActive
        && peakRisk
        && (edgeRisk || !params->peakLockRequireEdgeRisk[lane])
    );
}

static void recordPeakLockRaw(
    const BatchParamsSoa* params,
    BatchLaneState* batch,
    int lane,
    int index
) {
    batch->peakLocks[lane] += 1;
    batch->peakActive[lane] = 1;
    batch->peakStart[lane] = index;
    batch->peakCap[lane] = clipVal(params->peakLockCapPct[lane], 0.0, 1.0);
    batch->peakEdgeStart[lane] = batch->peakEdgeNow[lane];
    batch->peakLockGain[lane] = 0.0;
    batch->peakArmed[lane] = 0;
}

static void stepActivePeakLockRaw(
    const BatchParamsSoa* params,
    BatchLaneState* batch,
    int lane,
    int index,
    double barsDay
) {
    int maxBars;
    int age;
    int canStep;
    double baseCap;

    if (!peakLockEnabled(params, lane) || !batch->peakActive[lane]) {
        return;
    }
    batch->peakLockHours[lane] += 1;
    batch->peakLockGain[lane] = (
        batch->peakEdgeNow[lane] - batch->peakEdgeStart[lane]
    );
    if (batch->peakLockGain[lane] > batch->peakLockGainMax[lane]) {
        batch->peakLockGainMax[lane] = batch->peakLockGain[lane];
    }
    age = index - batch->peakStart[lane];
    baseCap = clipVal(params->peakLockCapPct[lane], 0.0, 1.0);
    canStep = (
        params->peakLockReentryStepPct[lane] > 1e-12
        && batch->peakCap[lane] <= baseCap + 1e-9
    );
    if (
        batch->peakLockGain[lane] >= params->peakLockUnlockGainPct[lane]
        && batch->peakCap[lane] < 1.0
        && canStep
    ) {
        batch->peakCap[lane] = clipVal(
            batch->peakCap[lane] + params->peakLockReentryStepPct[lane],
            0.0,
            1.0
        );
        batch->peakEdgeStart[lane] = batch->peakEdgeNow[lane];
        batch->peakUnlockSteps[lane] += 1;
    }
    maxBars = (int)round(params->peakLockMaxDays[lane] * barsDay);
    if (batch->peakCap[lane] >= 1.0 - 1e-9 || age >= maxBars) {
        batch->peakActive[lane] = 0;
        batch->peakStart[lane] = -1;
        batch->peakCap[lane] = 1.0;
    }
}

/* Run the wallet/phase loop once over all lanes in the active chunk. */
static int evalBatchWalletRows(
    EvalSession* session,
    int startIdx,
    const BatchParamsSoa* params,
    BatchResults* out
) {
    BatchLaneState* batch = &session->batch;
    int begin;
    int curveLen;
    int lane;
    int i;
    int initCount = 0;
    double price0;
    double seedSpend;

    begin = startIdx;
    if (begin < 0) {
        begin = 0;
    }
    if (begin >= session->micro->n) {
        return 0;
    }
    curveLen = session->micro->n - begin;
    price0 = session->micro->closes[begin];

    for (lane = 0; lane < params->count; lane++) {
        WalletState* wallet = &batch->wallets[lane];
        PhaseState* phase = &batch->phases[lane];

        out->rows[lane].buyFlags = batch->buyCounts[lane];
        out->rows[lane].sellFlags = batch->sellCounts[lane];
        out->rows[lane].flagCount = (
            batch->buyCounts[lane] + batch->sellCounts[lane]
        );
        if (
            initWallet(
                wallet,
                params->feeRate[lane],
                0.0,
                TAX_CGT,
                0.0
            ) != 0
        ) {
            goto fail;
        }
        initCount += 1;

        phase->side = 0;
        phase->baseValue = 0.0;
        phase->lastPrice = 0.0;
        phase->hasLastPrice = 0;
        phase->portionsRemaining = 0.0;
        phase->hasInfiniteRemaining = 0;
        phase->finalPortionPct = params->finalPortionPct[lane];

        batch->lastTrendCode[lane] = 0;
        batch->lastSignalSellIndex[lane] = -1000000000;
        batch->dailyStrongDays[lane] = 0;
        batch->dailyLockActive[lane] = 0;
        batch->dailyLockStart[lane] = -1;
        batch->dailyPrevStrong[lane] = 0;
        batch->dailyPrevDown[lane] = 0;
        batch->dailyEpisodeLocked[lane] = 0;
        batch->dailyBridgeBars[lane] = 0;
        batch->dailyLockReleaseOnStrong[lane] = 0;
        batch->dailyCoastActive[lane] = 0;
        batch->dailyCoastStart[lane] = -1;
        batch->dailyUltraEntryPrice[lane] = 0.0;
        batch->dailyUltraPeakPrice[lane] = 0.0;
        batch->dailyLockTargetPct[lane] = 1.0;
        batch->dailyLockHoldDays[lane] = 0.0;
        batch->dailyCoastTargetPct[lane] = 1.0;
        batch->dailyCoastMinAssetPct[lane] = 0.0;
        batch->dailyCoastMaxAssetPct[lane] = 1.0;
        batch->peakStrongGraceBars[lane] = 0;
        batch->peakStrongReleases[lane] = 0;
        batch->peakPrevStrong[lane] = 0;
        batch->peakActive[lane] = 0;
        batch->peakStart[lane] = -1;
        batch->peakLocks[lane] = 0;
        batch->peakCappedBuys[lane] = 0;
        batch->peakLockHours[lane] = 0;
        batch->peakUnlockSteps[lane] = 0;
        batch->peakArmed[lane] = 0;
        batch->peakCap[lane] = 1.0;
        batch->peakEdgeStart[lane] = 0.0;
        batch->peakEdgeNow[lane] = 0.0;
        batch->peakEdgePeak[lane] = 0.0;
        batch->peakLockGain[lane] = 0.0;
        batch->peakLockGainMax[lane] = 0.0;
        memset(
            lanePeakEdgeSlice(batch, lane),
            0,
            (size_t)batch->n * sizeof(double)
        );
        warmPeakPidRaw(
            session->micro,
            params,
            batch,
            lane,
            begin,
            session->prepared->microDerived.barsPerDay
        );
        wallet->quoteBalance += params->seedQuote[lane];
        seedSpend = (
            params->seedQuote[lane]
            * clipVal(params->seedAssetPct[lane], 0.0, 1.0)
        );
        if (seedSpend > 0.0) {
            applyBuy(
                wallet,
                begin,
                session->micro->ts[begin],
                price0,
                &seedSpend
            );
        }
        batch->benchQty[lane] = (
            price0 > 0.0
            ? (
                params->seedQuote[lane]
                * (1.0 - params->feeRate[lane])
            ) / price0
            : 0.0
        );
    }

    for (i = begin; i < session->micro->n; i++) {
        double price = session->micro->closes[i];
        int currentTrend = session->prepared->microDerived.trend[i];
        int64_t tsMs = session->micro->ts[i];
        int offset = i - begin;

        for (lane = 0; lane < params->count; lane++) {
            WalletState* wallet = &batch->wallets[lane];
            PhaseState* phase = &batch->phases[lane];
            unsigned char* buyFlags = laneFlagSlice(
                batch->buyFlags,
                lane,
                batch->n
            );
            unsigned char* sellFlags = laneFlagSlice(
                batch->sellFlags,
                lane,
                batch->n
            );
            double* curveSim = laneCurveSlice(batch, lane);
            int dailyStrong = 0;
            int dailyDown = 0;
            int dailyDownEntry = 0;
            int dailyForceLock = 0;
            int dailyCloudActive = 0;
            int peakStrongEntry = 0;
            int peakGraceActive = 0;
            int rawStrongTarget = 0;
            int targetBlockBars = 0;
            int targetBuyBlocked = 0;
            int targetBuyPeakBlocked = 0;
            double targetBuyPct = 0.0;
            double dailyLockTarget = 1.0;
            double ultraEntry = 0.0;
            double ultraPeak = 0.0;
            double ultraGainPct = 0.0;
            double givebackPct = 0.0;
            double crabCap = clipVal(
                params->dailyCrabAssetCapPct[lane],
                0.0,
                1.0
            );
            rawStrongTarget = (
                session->micro->dailyCluster != NULL
                && session->micro->dailyCluster[i] == DAILY_STRONG_CLUSTER
            );
            targetBlockBars = (int)round(
                session->prepared->microDerived.barsPerDay
            );
            if (targetBlockBars < params->cooldown[lane]) {
                targetBlockBars = params->cooldown[lane];
            }
            targetBuyBlocked = (
                sellFlags[i]
                || (
                    i - batch->lastSignalSellIndex[lane]
                    <= targetBlockBars
                )
            );
            dailyPostureStepRaw(
                session->micro,
                params,
                batch,
                lane,
                i,
                price,
                session->prepared->microDerived.barsPerDay,
                &dailyStrong,
                &dailyDown,
                &dailyDownEntry,
                &dailyForceLock,
                &dailyLockTarget
            );
            dailyCloudActive = batch->dailyCoastActive[lane];
            if (!dailyCloudActive) {
                stepPeakPidRaw(
                    params,
                    batch,
                    lane,
                    price,
                    session->prepared->microDerived.barsPerDay
                );
                stepPeakStrongRaw(
                    params,
                    batch,
                    lane,
                    dailyStrong,
                    session->prepared->microDerived.barsPerDay,
                    &peakStrongEntry,
                    &peakGraceActive
                );
            }
            ultraEntry = batch->dailyUltraEntryPrice[lane];
            ultraPeak = batch->dailyUltraPeakPrice[lane];
            ultraGainPct = (
                ultraEntry > 0.0
                ? ((ultraPeak / ultraEntry) - 1.0) * 100.0
                : 0.0
            );
            givebackPct = (
                ultraPeak > 0.0
                ? ((ultraPeak / price) - 1.0) * 100.0
                : 0.0
            );
            if (!dailyCloudActive) {
                armPeakLockRaw(
                    params,
                    batch,
                    lane,
                    dailyStrong,
                    ultraGainPct
                );
            }
            if (dailyForceLock) {
                double maxSellValue = floorSellValueCap(
                    wallet,
                    price,
                    dailyLockTarget
                );
                if (maxSellValue > 0.0 && price > 0.0) {
                    double sellQty = maxSellValue / price;
                    if (applySell(wallet, i, tsMs, price, &sellQty)) {
                        phase->side = 0;
                        phase->hasLastPrice = 0;
                        phase->portionsRemaining = 0.0;
                    }
                }
                batch->dailyEpisodeLocked[lane] = 1;
                batch->dailyLockActive[lane] = 1;
                batch->dailyLockStart[lane] = i;
            }
            if (
                !dailyCloudActive
                &&
                evaluatePeakLockRaw(
                    params,
                    batch,
                    wallet,
                    lane,
                    offset,
                    price,
                    givebackPct,
                    peakStrongEntry,
                    peakGraceActive,
                    session->prepared->microDerived.barsPerDay
                )
            ) {
                double maxSellValue = floorSellValueCap(
                    wallet,
                    price,
                    clipVal(params->peakLockCapPct[lane], 0.0, 1.0)
                );
                if (maxSellValue > 0.0 && price > 0.0) {
                    double sellQty = maxSellValue / price;
                    if (applySell(wallet, i, tsMs, price, &sellQty)) {
                        recordPeakLockRaw(params, batch, lane, i);
                        phase->side = 0;
                        phase->hasLastPrice = 0;
                        phase->portionsRemaining = 0.0;
                    }
                }
            }
            if (!dailyCloudActive) {
                stepActivePeakLockRaw(
                    params,
                    batch,
                    lane,
                    i,
                    session->prepared->microDerived.barsPerDay
                );
            }
            if (
                dailyDownEntry
                && !dailyCloudActive
                && crabCap < 1.0 - 1e-9
            ) {
                double maxSellValue = floorSellValueCap(
                    wallet,
                    price,
                    crabCap
                );
                if (maxSellValue > 0.0 && price > 0.0) {
                    double sellQty = maxSellValue / price;
                    if (applySell(wallet, i, tsMs, price, &sellQty)) {
                        phase->side = 0;
                        phase->hasLastPrice = 0;
                        phase->portionsRemaining = 0.0;
                    }
                }
            }
            targetBuyPeakBlocked = (
                !dailyCloudActive
                &&
                peakLockEnabled(params, lane)
                && batch->peakActive[lane]
                && batch->peakCap[lane]
                    <= params->peakLockCapPct[lane] + 1e-9
            );
            targetBuyPct = params->dailyStrongTargetPct[lane];
            if (
                peakLockEnabled(params, lane)
                && !dailyCloudActive
                && batch->peakActive[lane]
                && batch->peakCap[lane] < 1.0 - 1e-9
                && !targetBuyPeakBlocked
            ) {
                targetBuyPct = fmin(targetBuyPct, batch->peakCap[lane]);
            }
            if (
                rawStrongTarget
                && !batch->dailyLockActive[lane]
                && !dailyCloudActive
                && targetBuyPct > 0.0
                && !targetBuyPeakBlocked
                && !targetBuyBlocked
            ) {
                if (buyToTargetPct(
                    wallet,
                    i,
                    tsMs,
                    price,
                    targetBuyPct
                )) {
                    phase->side = 0;
                    phase->hasLastPrice = 0;
                    phase->portionsRemaining = 0.0;
                }
            }

            if (sellFlags[i] || buyFlags[i]) {
                int newBearRegime;
                int newBullRegime;
                int traded;

                newBearRegime = (
                    currentTrend == -1
                    && batch->lastTrendCode[lane] != -1
                );
                newBullRegime = (
                    currentTrend == 1
                    && batch->lastTrendCode[lane] != 1
                );
                batch->lastTrendCode[lane] = currentTrend;

                if (sellFlags[i]) {
                    if (currentTrend == 1 && wallet->baseBalance > 0.0) {
                        if (phase->side != -1 || newBullRegime) {
                            phase->side = -1;
                            phase->hasLastPrice = 0;
                            phase->baseValue = enterSellPhase(
                                wallet,
                                price,
                                params->phaseSell[lane]
                            );
                            phase->finalPortionPct =
                                params->finalPortionPct[lane];
                            phase->hasInfiniteRemaining = (
                                params->finalPortionPct[lane]
                                >= (1.0 - 1e-9)
                            );
                            phase->portionsRemaining = (
                                phase->baseValue > 0.0
                                ? (double)params->phaseSell[lane]
                                : 0.0
                            );
                        }
                        double scale = calcSellScale(
                            phase->hasLastPrice,
                            phase->lastPrice,
                            price
                        );
                        double maxSellValue = wallet->baseBalance * price;
                        double sellFloorPct = 0.0;
                        if (dailyStrong && !dailyCloudActive) {
                            double strongFloorPct;

                            strongFloorPct =
                                params->dailyStrongTargetPct[lane];
                            if (
                                peakLockEnabled(params, lane)
                                && batch->peakActive[lane]
                                && batch->peakCap[lane] < (1.0 - 1e-9)
                            ) {
                                strongFloorPct = fmin(
                                    strongFloorPct,
                                    batch->peakCap[lane]
                                );
                            }
                            sellFloorPct = fmax(
                                sellFloorPct,
                                strongFloorPct
                            );
                            scale *= params->dailyStrongSellMult[lane];
                        }
                        if (batch->dailyLockActive[lane]) {
                            sellFloorPct = fmax(
                                sellFloorPct,
                                batch->dailyLockTargetPct[lane]
                            );
                        }
                        if (sellFloorPct > 0.0) {
                            maxSellValue = floorSellValueCap(
                                wallet,
                                price,
                                sellFloorPct
                            );
                        }
                        traded = applyScaledSell(
                            wallet,
                            phase,
                            i,
                            tsMs,
                            price,
                            scale,
                            maxSellValue
                        );
                        if (traded) {
                            batch->lastSignalSellIndex[lane] = i;
                        }
                    }
                }

                if (buyFlags[i]) {
                    if (currentTrend == -1 && wallet->quoteBalance > 0.0) {
                        int peakCapBuyApplied = 0;
                        if (phase->side != 1 || newBearRegime) {
                            phase->side = 1;
                            phase->hasLastPrice = 0;
                            phase->baseValue = enterBuyPhase(
                                wallet,
                                params->phaseBuy[lane]
                            );
                            phase->finalPortionPct =
                                params->finalPortionPct[lane];
                            phase->hasInfiniteRemaining = (
                                params->finalPortionPct[lane]
                                >= (1.0 - 1e-9)
                            );
                            phase->portionsRemaining = (
                                phase->baseValue > 0.0
                                ? (double)params->phaseBuy[lane]
                                : 0.0
                            );
                        }
                        double scale = calcBuyScale(
                                phase->hasLastPrice,
                                phase->lastPrice,
                                price
                        );
                        double maxSpendValue = -1.0;
                        if (dailyDown) {
                            scale *= params->dailyDownBuyMult[lane];
                            if (crabCap < 1.0 - 1e-9) {
                                maxSpendValue = buySpendToTargetPct(
                                    wallet,
                                    price,
                                    crabCap
                                );
                            }
                        }
                        if (
                            batch->dailyCoastActive[lane]
                            && batch->dailyCoastMaxAssetPct[lane]
                                < 1.0 - 1e-9
                        ) {
                            double coastSpendValue = buySpendToTargetPct(
                                wallet,
                                price,
                                batch->dailyCoastMaxAssetPct[lane]
                            );

                            if (maxSpendValue >= 0.0) {
                                maxSpendValue = fmin(
                                    maxSpendValue,
                                    coastSpendValue
                                );
                            }
                            else {
                                maxSpendValue = coastSpendValue;
                            }
                            if (maxSpendValue <= 0.0) {
                                scale = 0.0;
                            }
                        }
                        if (
                            peakLockEnabled(params, lane)
                            && !dailyCloudActive
                            && batch->peakActive[lane]
                            && batch->peakCap[lane] < 1.0 - 1e-9
                        ) {
                            double capTarget = batch->peakCap[lane];
                            double capSpendValue = 0.0;
                            double baseCap = clipVal(
                                params->peakLockCapPct[lane],
                                0.0,
                                1.0
                            );

                            if (crabCap < 1.0 - 1e-9) {
                                capTarget = fmin(capTarget, crabCap);
                            }
                            capSpendValue = buySpendToTargetPct(
                                wallet,
                                price,
                                capTarget
                            );
                            if (batch->peakCap[lane] <= baseCap + 1e-9) {
                                scale = 0.0;
                            }
                            else if (maxSpendValue >= 0.0) {
                                maxSpendValue = fmin(
                                    maxSpendValue,
                                    capSpendValue
                                );
                                peakCapBuyApplied = 1;
                            }
                            else {
                                maxSpendValue = capSpendValue;
                                peakCapBuyApplied = 1;
                            }
                            if (maxSpendValue <= 0.0) {
                                scale = 0.0;
                            }
                        }
                        traded = applyScaledBuy(
                            wallet,
                            phase,
                            i,
                            tsMs,
                            price,
                            scale,
                            maxSpendValue
                        );
                        if (traded && peakCapBuyApplied) {
                            batch->peakCappedBuys[lane] += 1;
                        }
                    }
                }
            }

            curveSim[offset] = portfolioValue(wallet, price);
        }
    }

    for (lane = 0; lane < params->count; lane++) {
        finishEvalRow(
            session->micro,
            session->prepared,
            &session->lane,
            begin,
            &batch->wallets[lane],
            laneCurveSlice(batch, lane),
            curveLen,
            batch->benchQty[lane],
            &out->rows[lane]
        );
        freeWallet(&batch->wallets[lane]);
    }

    return 0;

fail:
    for (lane = 0; lane < initCount; lane++) {
        freeWallet(&batch->wallets[lane]);
    }
    return -1;
}

/* Evaluate one SoA chunk that fits inside the reusable batch scratch. */
static int evalBatchChunkSession(
    EvalSession* session,
    int startIdx,
    const BatchParamsSoa* params,
    BatchResults* out
) {
    int lane;

    if (
        session == NULL
        || params == NULL
        || out == NULL
        || params->count > session->batch.cap
    ) {
        return -1;
    }
    if (out->count != params->count) {
        return -1;
    }
    if (out->count > 0 && out->rows == NULL) {
        return -1;
    }

    for (lane = 0; lane < out->count; lane++) {
        zeroRow(&out->rows[lane]);
    }
    if (params->count == 0 || session->micro->n == 0) {
        return 0;
    }

    if (
        markFlagsBatch(
            session->micro,
            session->prepared,
            startIdx,
            params,
            &session->batch
        ) != 0
    ) {
        return -1;
    }

    return evalBatchWalletRows(session, startIdx, params, out);
}

/* Evaluate one SoA lane batch against a reusable dataset session. */
int evalBatchSoaSession(
    void* sessionHandle,
    int startIdx,
    const BatchParamsSoa* params,
    BatchResults* out
) {
    EvalSession* session = (EvalSession*)sessionHandle;
    BatchParamsSoa chunkParams;
    BatchResults chunkOut;
    int start;

    if (session == NULL || out == NULL || !validBatchParamsSoa(params)) {
        return -1;
    }
    if (out->count != params->count) {
        return -1;
    }
    if (out->count > 0 && out->rows == NULL) {
        return -1;
    }

    if (params->count == 0) {
        return 0;
    }

    for (start = 0; start < params->count; start += session->batch.cap) {
        int chunkCount = params->count - start;

        if (chunkCount > session->batch.cap) {
            chunkCount = session->batch.cap;
        }
        sliceBatchParamsSoa(params, start, chunkCount, &chunkParams);
        chunkOut.count = chunkCount;
        chunkOut.rows = out->rows + start;
        if (
            evalBatchChunkSession(
                session,
                startIdx,
                &chunkParams,
                &chunkOut
            ) != 0
        ) {
            return -1;
        }
    }

    return 0;
}

/* Evaluate one SoA lane batch without caller-managed session lifetime. */
int evalBatchSoa(
    const MicroSoa* micro,
    const MacroSoa* macro,
    int startIdx,
    const BatchParamsSoa* params,
    BatchResults* out
) {
    int rc;
    void* sessionHandle;

    if (micro == NULL || out == NULL) {
        return -1;
    }

    sessionHandle = createEvalSession(micro, macro);
    if (sessionHandle == NULL) {
        return -1;
    }
    rc = evalBatchSoaSession(
        sessionHandle,
        startIdx,
        params,
        out
    );
    destroyEvalSession(sessionHandle);
    return rc;
}

/* Prepare one reusable dataset block for a fixed micro/macro pair. */
void* createPreparedDataset(
    const MicroSoa* micro,
    const MacroSoa* macro
) {
    PreparedDatasetState* prepared;

    if (micro == NULL || macro == NULL) {
        return NULL;
    }
    if (
        micro->n < 0
        || micro->closes == NULL
        || micro->ts == NULL
    ) {
        return NULL;
    }

    prepared = (PreparedDatasetState*)malloc(sizeof(PreparedDatasetState));
    if (prepared == NULL) {
        return NULL;
    }
    if (initPreparedDataset(micro, macro, prepared) != 0) {
        free(prepared);
        return NULL;
    }

    return (void*)prepared;
}

/* Destroy one prepared dataset block. */
void destroyPreparedDataset(void* preparedHandle) {
    PreparedDatasetState* prepared = (PreparedDatasetState*)preparedHandle;

    if (prepared == NULL) {
        return;
    }
    freePreparedDataset(prepared);
    free(prepared);
}

/* Create one reusable evaluation session from prepared dataset state. */
void* createEvalSessionPrepared(void* preparedHandle) {
    PreparedDatasetState* prepared = (PreparedDatasetState*)preparedHandle;
    EvalSession* session;

    if (prepared == NULL || prepared->micro == NULL) {
        return NULL;
    }

    session = (EvalSession*)malloc(sizeof(EvalSession));
    if (session == NULL) {
        return NULL;
    }
    session->micro = prepared->micro;
    session->prepared = prepared;
    session->ownsPrepared = 0;
    if (initEvalLane(&session->lane, prepared->micro->n) != 0) {
        free(session);
        return NULL;
    }
    if (
        initBatchLane(
            &session->batch,
            prepared->micro->n,
            TUNE_BATCH_LANES
        ) != 0
    ) {
        freeEvalLane(&session->lane);
        free(session);
        return NULL;
    }

    return (void*)session;
}

/* Create one reusable evaluation session for a fixed dataset. */
void* createEvalSession(
    const MicroSoa* micro,
    const MacroSoa* macro
) {
    EvalSession* session;
    void* preparedHandle;

    preparedHandle = createPreparedDataset(micro, macro);
    if (preparedHandle == NULL) {
        return NULL;
    }
    session = (EvalSession*)createEvalSessionPrepared(preparedHandle);
    if (session == NULL) {
        destroyPreparedDataset(preparedHandle);
        return NULL;
    }
    session->ownsPrepared = 1;
    return (void*)session;
}

/* Destroy one reusable evaluation session. */
void destroyEvalSession(void* sessionHandle) {
    EvalSession* session = (EvalSession*)sessionHandle;

    if (session == NULL) {
        return;
    }
    freeBatchLane(&session->batch);
    freeEvalLane(&session->lane);
    if (session->ownsPrepared) {
        destroyPreparedDataset((void*)session->prepared);
    }
    free(session);
}

/* Evaluate one sweep chunk against a reusable evaluation session. */
int evalSweepSession(
    void* sessionHandle,
    int startIdx,
    const TuneParams* baseParam,
    const TuneAxes* axes,
    uint64_t comboStart,
    int comboCount,
    EvalRow* rows
) {
    EvalSession* session = (EvalSession*)sessionHandle;

    if (session == NULL) {
        return -1;
    }

    return evalSweepRows(
        session->micro,
        session->prepared,
        &session->lane,
        startIdx,
        baseParam,
        axes,
        comboStart,
        comboCount,
        rows
    );
}

/* Evaluate one chunk of a cartesian sweep inside C. */
int evalSweep(
    const MicroSoa* micro,
    const MacroSoa* macro,
    int startIdx,
    const TuneParams* baseParam,
    const TuneAxes* axes,
    uint64_t comboStart,
    int comboCount,
    EvalRow* rows
) {
    int rc;
    void* sessionHandle;

    if (micro == NULL || baseParam == NULL || rows == NULL) {
        return -1;
    }

    sessionHandle = createEvalSession(micro, macro);
    if (sessionHandle == NULL) {
        return -1;
    }
    rc = evalSweepSession(
        sessionHandle,
        startIdx,
        baseParam,
        axes,
        comboStart,
        comboCount,
        rows
    );
    destroyEvalSession(sessionHandle);
    return rc;
}

/* Run one full inner sweep against an initialized reusable session. */
int runTuneGroupSession(
    void* sessionHandle,
    int startIdx,
    const TuneParams* baseParam,
    const TuneAxes* axes,
    const TuneGroupMeta* meta,
    const TuneRunOptions* options,
    TuneRunResult* out
) {
    EvalSession* session = (EvalSession*)sessionHandle;
    FILE* fp = NULL;
    TuneParams param;
    EvalRow row;
    uint64_t comboTotal;
    uint64_t progressRows;
    uint64_t doneBase;
    uint64_t totalCount;
    double progressSecs;
    double startSecs;
    double lastPrint;
    double beginSecs;
    int wroteAny = 0;
    uint64_t comboIdx;

    if (
        session == NULL
        || baseParam == NULL
        || meta == NULL
        || out == NULL
    ) {
        return -1;
    }

    zeroRunResult(out);
    comboTotal = tuneAxesCount(axes);
    out->evalCount = comboTotal;
    if (comboTotal == 0) {
        return 0;
    }

    doneBase = options != NULL ? options->doneCount : 0;
    totalCount = options != NULL ? options->totalCount : comboTotal;
    startSecs = options != NULL ? options->startSecs : nowSecs();
    progressRows = (
        options != NULL && options->progressRows > 0
        ? options->progressRows
        : 4096
    );
    progressSecs = (
        options != NULL && options->progressSecs > 0.0
        ? options->progressSecs
        : 0.25
    );
    lastPrint = startSecs;
    beginSecs = nowSecs();

    if (options != NULL && options->csvPath != NULL && options->csvPath[0]) {
        fp = fopen(options->csvPath, options->appendCsv ? "a" : "w");
        if (fp == NULL) {
            return -1;
        }
        setvbuf(fp, NULL, _IOFBF, 1 << 20);
        if (!options->appendCsv && writeCsvHeader(fp) != 0) {
            fclose(fp);
            return -1;
        }
    }

    for (comboIdx = 0; comboIdx < comboTotal; comboIdx++) {
        double grossPct;
        double riskScore;
        uint64_t doneCount;
        double now;
        int shouldPrint = 0;

        zeroRow(&row);
        decodeSweepParam(baseParam, axes, comboIdx, &param);
        if (
            evaluateRow(
                session->micro,
                session->prepared,
                &session->lane,
                &param,
                startIdx,
                &row
            ) != 0
        ) {
            if (fp != NULL) {
                fclose(fp);
            }
            return -1;
        }

        if (fp != NULL && writeCsvRow(fp, meta, &param, &row) != 0) {
            fclose(fp);
            return -1;
        }

        grossPct = lifecycleScoreValue(&row);
        if (!wroteAny || grossPct > out->bestGrossPct) {
            out->bestRow = row;
            out->bestComboIdx = comboIdx;
            out->bestGrossPct = grossPct;
        }

        riskScore = riskScoreValue(&row);
        if (!wroteAny || riskScore > out->statsScore) {
            out->statsRow = row;
            out->statsComboIdx = comboIdx;
            out->statsScore = riskScore;
        }

        wroteAny = 1;
        doneCount = doneBase + comboIdx + 1;
        now = nowSecs();
        if (((comboIdx + 1) % progressRows) == 0) {
            shouldPrint = 1;
        }
        if ((now - lastPrint) >= progressSecs) {
            shouldPrint = 1;
        }
        if (doneCount >= totalCount) {
            shouldPrint = 1;
        }
        if (shouldPrint) {
            printProgress(doneCount, totalCount, startSecs, meta);
            lastPrint = now;
        }
    }

    if (fp != NULL) {
        fclose(fp);
    }
    out->elapsedSecs = nowSecs() - beginSecs;
    return 0;
}

/* Run one full inner sweep against an initialized batch session. */
int runTuneGroupBatchSession(
    void* sessionHandle,
    int startIdx,
    const TuneParams* baseParam,
    const TuneAxes* axes,
    const TuneGroupMeta* meta,
    const TuneRunOptions* options,
    TuneRunResult* out
) {
    EvalSession* session = (EvalSession*)sessionHandle;
    FILE* fp = NULL;
    BatchParamChunk chunk;
    BatchResults chunkOut;
    BatchLaneParam laneParam;
    uint64_t comboTotal;
    uint64_t progressRows;
    uint64_t doneBase;
    uint64_t totalCount;
    uint64_t lastPrintCount;
    double progressSecs;
    double startSecs;
    double lastPrint;
    double beginSecs;
    int wroteAny = 0;
    uint64_t comboStart;

    if (
        session == NULL
        || baseParam == NULL
        || meta == NULL
        || out == NULL
    ) {
        return -1;
    }

    initBatchParamChunk(&chunk);
    zeroRunResult(out);
    comboTotal = tuneAxesCount(axes);
    out->evalCount = comboTotal;
    if (comboTotal == 0) {
        return 0;
    }

    doneBase = options != NULL ? options->doneCount : 0;
    totalCount = options != NULL ? options->totalCount : comboTotal;
    startSecs = options != NULL ? options->startSecs : nowSecs();
    progressRows = (
        options != NULL && options->progressRows > 0
        ? options->progressRows
        : 4096
    );
    progressSecs = (
        options != NULL && options->progressSecs > 0.0
        ? options->progressSecs
        : 0.25
    );
    lastPrint = startSecs;
    lastPrintCount = doneBase;
    beginSecs = nowSecs();

    if (options != NULL && options->csvPath != NULL && options->csvPath[0]) {
        fp = fopen(options->csvPath, options->appendCsv ? "a" : "w");
        if (fp == NULL) {
            return -1;
        }
        setvbuf(fp, NULL, _IOFBF, 1 << 20);
        if (!options->appendCsv && writeCsvHeader(fp) != 0) {
            fclose(fp);
            return -1;
        }
    }

    for (comboStart = 0; comboStart < comboTotal; ) {
        int lane;
        int chunkCount = (int)(comboTotal - comboStart);
        uint64_t doneCount;
        double now;
        int shouldPrint = 0;

        if (chunkCount > session->batch.cap) {
            chunkCount = session->batch.cap;
        }
        fillBatchParamChunk(
            &chunk,
            baseParam,
            axes,
            comboStart,
            chunkCount
        );
        chunkOut.count = chunkCount;
        chunkOut.rows = session->batch.rows;
        if (
            evalBatchChunkSession(
                session,
                startIdx,
                &chunk.params,
                &chunkOut
            ) != 0
        ) {
            if (fp != NULL) {
                fclose(fp);
            }
            return -1;
        }

        for (lane = 0; lane < chunkCount; lane++) {
            double grossPct;
            double riskScore;

            decodeBatchLaneParam(&chunk.params, lane, &laneParam);
            if (
                fp != NULL
                && writeCsvRow(
                    fp,
                    meta,
                    &laneParam.param,
                    &session->batch.rows[lane]
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }

            grossPct = lifecycleScoreValue(&session->batch.rows[lane]);
            if (!wroteAny || grossPct > out->bestGrossPct) {
                out->bestRow = session->batch.rows[lane];
                out->bestComboIdx = comboStart + (uint64_t)lane;
                out->bestGrossPct = grossPct;
            }

            riskScore = riskScoreValue(&session->batch.rows[lane]);
            if (!wroteAny || riskScore > out->statsScore) {
                out->statsRow = session->batch.rows[lane];
                out->statsComboIdx = comboStart + (uint64_t)lane;
                out->statsScore = riskScore;
            }

            wroteAny = 1;
        }

        comboStart += (uint64_t)chunkCount;
        doneCount = doneBase + comboStart;
        now = nowSecs();
        if ((doneCount - lastPrintCount) >= progressRows) {
            shouldPrint = 1;
        }
        if ((now - lastPrint) >= progressSecs) {
            shouldPrint = 1;
        }
        if (doneCount >= totalCount) {
            shouldPrint = 1;
        }
        if (shouldPrint) {
            printProgress(doneCount, totalCount, startSecs, meta);
            lastPrint = now;
            lastPrintCount = doneCount;
        }
    }

    if (fp != NULL) {
        fclose(fp);
    }
    out->elapsedSecs = nowSecs() - beginSecs;
    return 0;
}

/* Run one full inner sweep by creating and owning one reusable session. */
int runTuneGroup(
    const MicroSoa* micro,
    const MacroSoa* macro,
    int startIdx,
    const TuneParams* baseParam,
    const TuneAxes* axes,
    const TuneGroupMeta* meta,
    const TuneRunOptions* options,
    TuneRunResult* out
) {
    int rc;
    void* sessionHandle;

    if (micro == NULL || baseParam == NULL || out == NULL) {
        return -1;
    }

    sessionHandle = createEvalSession(micro, macro);
    if (sessionHandle == NULL) {
        return -1;
    }
    rc = runTuneGroupSession(
        sessionHandle,
        startIdx,
        baseParam,
        axes,
        meta,
        options,
        out
    );
    destroyEvalSession(sessionHandle);
    return rc;
}

/* Run one full inner sweep via the explicit batch-facing API. */
int runTuneGroupBatch(
    const MicroSoa* micro,
    const MacroSoa* macro,
    int startIdx,
    const TuneParams* baseParam,
    const TuneAxes* axes,
    const TuneGroupMeta* meta,
    const TuneRunOptions* options,
    TuneRunResult* out
) {
    int rc;
    void* sessionHandle;

    if (micro == NULL || baseParam == NULL || out == NULL) {
        return -1;
    }

    sessionHandle = createEvalSession(micro, macro);
    if (sessionHandle == NULL) {
        return -1;
    }
    rc = runTuneGroupBatchSession(
        sessionHandle,
        startIdx,
        baseParam,
        axes,
        meta,
        options,
        out
    );
    destroyEvalSession(sessionHandle);
    return rc;
}
