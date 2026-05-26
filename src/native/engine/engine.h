#ifndef ENGINE_H
#define ENGINE_H

#include <stdint.h>

#define TAX_CGT 0
#define TAX_INCOME 1

typedef struct{
    int n;
    int p1;
    int p2;
    int p3;
    const int64_t* ts;
    const double* opens;
    const double* highs;
    const double* lows;
    const double* closes;
    const double* volumes;
    const int* dailyCluster;
    const double* dailyRet30;
    const double* dailyNearHigh;
} MicroSoa;

typedef struct{
    int n;
    int p1;
    int p2;
    int p3;
    double dynWinDays;
    double dynZMin;
    double dynZMax;
    double dynPctMax;
    double dynPctMin;
    double gradWinDays;
    double gradZMin;
    double gradZMax;
    double gradMultMin;
    double gradMultMax;
    const int64_t* ts;
    const double* closes;
} MacroSoa;

typedef struct{
    double grad1BuyZMin;
    double grad1SellZMin;
    int grad1BuyWinDays;
    int grad1SellWinDays;
    int phaseBuy;
    int phaseSell;
    double finalPortionPct;
    int cooldown;
    double feeRate;
    double seedQuote;
    double seedAssetPct;
    int taxMode;
    double annualIncomeBase;
    double dailyStrongSellMult;
    double dailyStrongTargetPct;
    double dailyBridgeDays;
    double dailyDownBuyMult;
    double dailyCrabAssetCapPct;
    double dailyLockTargetPct;
    double dailyLockGainPct;
    double dailyLockNearHighPct;
    int dailyLockMaxDays;
    double postUltraCoastTargetPct;
    double postUltraGivebackPct;
    double postUltraReaccumPct;
    double postUltraDoubleTopPct;
    double postUltraMaxDays;
    double postUltraLockMinAssetPct;
    double postUltraLockMaxAssetPct;
    double postUltraLockGivebackPct;
    double postUltraLockReaccumPct;
    double postUltraLockDoubleTopPct;
    double postUltraLockMaxDays;
    double macroSellRelaxPct;
    double peakLockCapPct;
    double peakLockUnlockGainPct;
    double peakLockReentryStepPct;
    double peakLockArmGainPct;
    double peakLockGivebackPct;
    double peakLockMaxDays;
    double peakLockEdgeDrawPct;
    double peakLockEdgeSlopeDays;
    int peakLockRequireEdgeRisk;
    double peakLockMaDays;
    double peakLockKp;
    double peakLockKi;
    double peakLockKd;
    double peakLockIntegralDecay;
    double peakLockEntryThreshold;
    double peakLockExitThreshold;
    int peakLockConfirmBars;
    double peakLockReleaseTargetPct;
    double peakLockUltraGraceDays;
} TuneParams;

typedef struct{
    double simValue;
    double simPostTax;
    double benchValue;
    double benchPostTax;
    double preTaxEdge;
    double postTaxEdge;
    double netPctVsHodl;
    double fees;
    double tax;
    double potentialProfit;
    double potentialProfitBench;
    double netAfterTaxProfit;
    double netAfterTaxProfitBench;
    double sharpe;
    double sortino;
    double mdd;
    double cagr;
    double sharpe1w;
    double sortino1w;
    double sharpe4w;
    double sortino4w;
    double sharpe13w;
    double sortino13w;
    double sharpe1wAbs;
    double sortino1wAbs;
    double sharpe4wAbs;
    double sortino4wAbs;
    double sharpe13wAbs;
    double sortino13wAbs;
    double lifecycleEdgeMean;
    double lifecycleEdgeMedian;
    double lifecycleEdgeP25;
    double lifecycleEdgeMin;
    double lifecycleUnderwaterPct;
    double lifecycleUnderwaterMean;
    double lifecycleTrackingPct;
    double lifecycleEdgeMdd;
    double lifecycleEdgeScore;
    int trades;
    int buyTrades;
    int sellTrades;
    int flagCount;
    int buyFlags;
    int sellFlags;
} EvalRow;

typedef struct{
    int n;
    const int* values;
} IntAxis;

typedef struct{
    int n;
    const double* values;
} DoubleAxis;

typedef struct{
    DoubleAxis grad1BuyZMin;
    DoubleAxis grad1SellZMin;
    IntAxis grad1BuyWinDays;
    IntAxis grad1SellWinDays;
    IntAxis phaseBuy;
    IntAxis phaseSell;
    DoubleAxis finalPortionPct;
    IntAxis cooldown;
    IntAxis taxMode;
    DoubleAxis seedAssetPct;
    DoubleAxis dailyStrongSellMult;
    DoubleAxis dailyStrongTargetPct;
    DoubleAxis dailyBridgeDays;
    DoubleAxis dailyDownBuyMult;
    DoubleAxis dailyCrabAssetCapPct;
    DoubleAxis dailyLockTargetPct;
    DoubleAxis dailyLockGainPct;
    DoubleAxis dailyLockNearHighPct;
    IntAxis dailyLockMaxDays;
    DoubleAxis postUltraCoastTargetPct;
    DoubleAxis postUltraGivebackPct;
    DoubleAxis postUltraReaccumPct;
    DoubleAxis postUltraDoubleTopPct;
    DoubleAxis postUltraMaxDays;
    DoubleAxis postUltraLockMinAssetPct;
    DoubleAxis postUltraLockMaxAssetPct;
    DoubleAxis postUltraLockGivebackPct;
    DoubleAxis postUltraLockReaccumPct;
    DoubleAxis postUltraLockDoubleTopPct;
    DoubleAxis postUltraLockMaxDays;
    DoubleAxis macroSellRelaxPct;
    DoubleAxis annualIncomeBase;
    DoubleAxis peakLockCapPct;
    DoubleAxis peakLockUnlockGainPct;
    DoubleAxis peakLockReentryStepPct;
    DoubleAxis peakLockArmGainPct;
    DoubleAxis peakLockGivebackPct;
    DoubleAxis peakLockMaxDays;
    DoubleAxis peakLockEdgeDrawPct;
    DoubleAxis peakLockEdgeSlopeDays;
    IntAxis peakLockRequireEdgeRisk;
    DoubleAxis peakLockMaDays;
    DoubleAxis peakLockKp;
    DoubleAxis peakLockKi;
    DoubleAxis peakLockKd;
    DoubleAxis peakLockIntegralDecay;
    DoubleAxis peakLockEntryThreshold;
    DoubleAxis peakLockExitThreshold;
    IntAxis peakLockConfirmBars;
    DoubleAxis peakLockReleaseTargetPct;
    DoubleAxis peakLockUltraGraceDays;
} TuneAxes;

typedef struct{
    int count;
    const double* grad1BuyZMin;
    const double* grad1SellZMin;
    const int* grad1BuyWinDays;
    const int* grad1SellWinDays;
    const int* phaseBuy;
    const int* phaseSell;
    const double* finalPortionPct;
    const int* cooldown;
    const double* feeRate;
    const double* seedQuote;
    const double* seedAssetPct;
    const int* taxMode;
    const double* annualIncomeBase;
    const double* dailyStrongSellMult;
    const double* dailyStrongTargetPct;
    const double* dailyBridgeDays;
    const double* dailyDownBuyMult;
    const double* dailyCrabAssetCapPct;
    const double* dailyLockTargetPct;
    const double* dailyLockGainPct;
    const double* dailyLockNearHighPct;
    const int* dailyLockMaxDays;
    const double* postUltraCoastTargetPct;
    const double* postUltraGivebackPct;
    const double* postUltraReaccumPct;
    const double* postUltraDoubleTopPct;
    const double* postUltraMaxDays;
    const double* postUltraLockMinAssetPct;
    const double* postUltraLockMaxAssetPct;
    const double* postUltraLockGivebackPct;
    const double* postUltraLockReaccumPct;
    const double* postUltraLockDoubleTopPct;
    const double* postUltraLockMaxDays;
    const double* macroSellRelaxPct;
    const double* peakLockCapPct;
    const double* peakLockUnlockGainPct;
    const double* peakLockReentryStepPct;
    const double* peakLockArmGainPct;
    const double* peakLockGivebackPct;
    const double* peakLockMaxDays;
    const double* peakLockEdgeDrawPct;
    const double* peakLockEdgeSlopeDays;
    const int* peakLockRequireEdgeRisk;
    const double* peakLockMaDays;
    const double* peakLockKp;
    const double* peakLockKi;
    const double* peakLockKd;
    const double* peakLockIntegralDecay;
    const double* peakLockEntryThreshold;
    const double* peakLockExitThreshold;
    const int* peakLockConfirmBars;
    const double* peakLockReleaseTargetPct;
    const double* peakLockUltraGraceDays;
} BatchParamsSoa;

typedef struct{
    const char* ticker;
    const char* interval;
    int days;
    int p1;
    int p2;
    int p3;
    const char* macroInterval;
    int macroP1;
    int macroP3;
    int macroGradPeriod;
    int macroDynWinDays;
    double macroDynZMin;
    double macroDynZMax;
    double macroDynPctMin;
    double macroDynPctMax;
    int macroGradWinDays;
    double macroGradZMin;
    double macroGradZMax;
    double macroGradMultMin;
    double macroGradMultMax;
} TuneGroupMeta;

typedef struct{
    const char* csvPath;
    int appendCsv;
    uint64_t doneCount;
    uint64_t totalCount;
    double startSecs;
    uint64_t progressRows;
    double progressSecs;
} TuneRunOptions;

typedef struct{
    EvalRow bestRow;
    EvalRow statsRow;
    uint64_t bestComboIdx;
    uint64_t statsComboIdx;
    double bestGrossPct;
    double statsScore;
    uint64_t evalCount;
    double elapsedSecs;
} TuneRunResult;

typedef struct{
    int count;
    EvalRow* rows;
} BatchResults;

/* Build EMA series. */
void emaLpf(
    const double* values,
    int n,
    int period,
    double* out
);

/* Build first derivative series. */
void grad1Series(
    const double* values,
    int n,
    double target,
    double* out
);

/* Build trend code array. */
void trendCodes(
    const double* m1,
    const double* m2,
    const double* m3,
    int n,
    int* out
);

/* Build rolling mean and std arrays. */
void rollingMeanAndStd(
    const double* series,
    int n,
    int window,
    double* meanOut,
    double* stdOut
);

/* Build regime energy array. */
void energyCsum(
    const double* m1,
    const double* m2,
    const double* m3,
    const int* trendCode,
    int n,
    int leg,
    double* out
);

/* Build regime spread peak ratio array. */
void spreadPeakRatioFromMas(
    const double* mA,
    const double* mB,
    const int* trendCode,
    int n,
    double* out
);

/* Build macro dynamic threshold series. */
void macroDynFromMas(
    const double* m1,
    const double* m2,
    const double* m3,
    int n,
    double barsPerDay,
    double winDays,
    double zMin,
    double zMax,
    double pctMax,
    double pctMin,
    double gradWinDays,
    double gradZMin,
    double gradZMax,
    double gradMultMin,
    double gradMultMax,
    double* out
);

/* Evaluate a batch of tuning rows. */
int evalBatch(
    const MicroSoa* micro,
    const MacroSoa* macro,
    int startIdx,
    const TuneParams* params,
    int paramCount,
    EvalRow* rows
);

/* Evaluate one SoA lane batch against an owned reusable session. */
int evalBatchSoaSession(
    void* sessionHandle,
    int startIdx,
    const BatchParamsSoa* params,
    BatchResults* out
);

/* Evaluate one SoA lane batch without caller-managed session lifetime. */
int evalBatchSoa(
    const MicroSoa* micro,
    const MacroSoa* macro,
    int startIdx,
    const BatchParamsSoa* params,
    BatchResults* out
);

/* Count the cartesian product size for sweep axes. */
uint64_t tuneAxesCount(const TuneAxes* axes);

/* Prepare dataset-derived arrays once for a fixed micro/macro pair. */
void* createPreparedDataset(
    const MicroSoa* micro,
    const MacroSoa* macro
);

/* Destroy one prepared dataset block. */
void destroyPreparedDataset(void* preparedHandle);

/* Create one reusable evaluation session from prepared dataset state. */
void* createEvalSessionPrepared(void* preparedHandle);

/* Create one reusable evaluation session for a fixed micro/macro dataset. */
void* createEvalSession(
    const MicroSoa* micro,
    const MacroSoa* macro
);

/* Destroy one reusable evaluation session. */
void destroyEvalSession(void* sessionHandle);

/* Evaluate one sweep chunk against a reusable evaluation session. */
int evalSweepSession(
    void* sessionHandle,
    int startIdx,
    const TuneParams* baseParam,
    const TuneAxes* axes,
    uint64_t comboStart,
    int comboCount,
    EvalRow* rows
);

/* Evaluate a sweep chunk without Python materializing TuneParams rows. */
int evalSweep(
    const MicroSoa* micro,
    const MacroSoa* macro,
    int startIdx,
    const TuneParams* baseParam,
    const TuneAxes* axes,
    uint64_t comboStart,
    int comboCount,
    EvalRow* rows
);

/* Run one full inner sweep group and return only winners plus timing. */
int runTuneGroup(
    const MicroSoa* micro,
    const MacroSoa* macro,
    int startIdx,
    const TuneParams* baseParam,
    const TuneAxes* axes,
    const TuneGroupMeta* meta,
    const TuneRunOptions* options,
    TuneRunResult* out
);

/* Run one full inner sweep against an existing reusable session. */
int runTuneGroupSession(
    void* sessionHandle,
    int startIdx,
    const TuneParams* baseParam,
    const TuneAxes* axes,
    const TuneGroupMeta* meta,
    const TuneRunOptions* options,
    TuneRunResult* out
);

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
);

/* Run one full inner sweep against an existing batch session. */
int runTuneGroupBatchSession(
    void* sessionHandle,
    int startIdx,
    const TuneParams* baseParam,
    const TuneAxes* axes,
    const TuneGroupMeta* meta,
    const TuneRunOptions* options,
    TuneRunResult* out
);

#endif
