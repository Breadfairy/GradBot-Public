#ifndef TUNE_SPEC_H
#define TUNE_SPEC_H

#include "../engine/engine.h"

typedef struct{
    char ticker[64];
    char cacheRoot[1024];
    char outDir[1024];
    char resultsCsvPath[1024];
    char bestRowCsvPath[1024];
    char statsRowCsvPath[1024];
    int64_t anchorMs;
    int totalDays;
    int primerDays;
    int trainingDays;
    int tunerDays;
    int holdoutDays;
    double feeRate;
    double seedQuote;
    char dailyClusterPath[1024];
    char dailyClusterModelPath[1024];
} TuneHostMeta;

typedef struct{
    char interval[32];
    int p1;
    int p2;
    int p3;
} IntervalGroup;

typedef struct{
    char macroInterval[32];
    int macroDynWin;
    double macroDynZMin;
    double macroDynZMax;
    double macroDynPctMin;
    double macroDynPctMax;
    int macroP1;
    int macroP3;
    int macroGradPeriod;
    int macroGradWinDays;
    double macroGradZMin;
    double macroGradZMax;
    double macroGradMultMin;
    double macroGradMultMax;
} MacroGroup;

typedef struct{
    int count;
    IntervalGroup* items;
} IntervalGroupList;

typedef struct{
    int count;
    MacroGroup* items;
} MacroGroupList;

typedef struct{
    double* grad1BuyZMinVals;
    double* grad1SellZMinVals;
    int* grad1BuyWinDaysVals;
    int* grad1SellWinDaysVals;
    int* phaseBuyVals;
    int* phaseSellVals;
    double* finalPortionPctVals;
    int* cooldownVals;
    int* taxModeVals;
    double* seedAssetPctVals;
    double* dailyStrongSellMultVals;
    double* dailyStrongTargetPctVals;
    double* dailyBridgeDaysVals;
    double* dailyDownBuyMultVals;
    double* dailyCrabAssetCapPctVals;
    double* dailyLockTargetPctVals;
    double* dailyLockGainPctVals;
    double* dailyLockNearHighPctVals;
    int* dailyLockMaxDaysVals;
    double* postUltraCoastTargetPctVals;
    double* postUltraGivebackPctVals;
    double* postUltraReaccumPctVals;
    double* postUltraDoubleTopPctVals;
    double* postUltraMaxDaysVals;
    double* postUltraLockMinAssetPctVals;
    double* postUltraLockMaxAssetPctVals;
    double* postUltraLockGivebackPctVals;
    double* postUltraLockReaccumPctVals;
    double* postUltraLockDoubleTopPctVals;
    double* postUltraLockMaxDaysVals;
    double* macroSellRelaxPctVals;
    double* annualIncomeBaseVals;
    double* peakLockCapPctVals;
    double* peakLockUnlockGainPctVals;
    double* peakLockReentryStepPctVals;
    double* peakLockArmGainPctVals;
    double* peakLockGivebackPctVals;
    double* peakLockMaxDaysVals;
    double* peakLockEdgeDrawPctVals;
    double* peakLockEdgeSlopeDaysVals;
    int* peakLockRequireEdgeRiskVals;
    double* peakLockMaDaysVals;
    double* peakLockKpVals;
    double* peakLockKiVals;
    double* peakLockKdVals;
    double* peakLockIntegralDecayVals;
    double* peakLockEntryThresholdVals;
    double* peakLockExitThresholdVals;
    int* peakLockConfirmBarsVals;
    double* peakLockReleaseTargetPctVals;
    double* peakLockUltraGraceDaysVals;
    TuneAxes axes;
} ParsedAxes;

int loadMeta(const char* path, TuneHostMeta* out);
int loadIntervalGroups(const char* path, IntervalGroupList* out);
int loadMacroGroups(const char* path, MacroGroupList* out);
int loadAxes(const char* path, ParsedAxes* out);
void freeIntervalGroups(IntervalGroupList* groups);
void freeMacroGroups(MacroGroupList* groups);
void freeAxes(ParsedAxes* axes);

#endif
