#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tuneSpec.h"

#define FIXED_INCOME_BASE 36000.0

/* Trim one trailing newline in place. */
static void trimLine(char* line) {
    size_t len = strlen(line);

    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
        line[len - 1] = '\0';
        len -= 1;
    }
}

/* Return the engine tax code for one label. */
static int taxCode(const char* name) {
    if (strcmp(name, "income") == 0) {
        return TAX_INCOME;
    }
    return TAX_CGT;
}

/* Grow one interval-group list. */
static int growIntervalGroups(IntervalGroupList* out, int need) {
    IntervalGroup* grown;

    grown = (IntervalGroup*)realloc(
        out->items,
        (size_t)need * sizeof(IntervalGroup)
    );
    if (grown == NULL) {
        return -1;
    }
    out->items = grown;
    return 0;
}

/* Grow one macro-group list. */
static int growMacroGroups(MacroGroupList* out, int need) {
    MacroGroup* grown;

    grown = (MacroGroup*)realloc(
        out->items,
        (size_t)need * sizeof(MacroGroup)
    );
    if (grown == NULL) {
        return -1;
    }
    out->items = grown;
    return 0;
}

/* Parse one comma-separated double list. */
static int parseDoubleList(const char* raw, double** outVals, int* outCount) {
    char* copy;
    char* tok;
    int count = 0;
    double* values = NULL;

    copy = (char*)malloc(strlen(raw) + 1);
    if (copy == NULL) {
        return -1;
    }
    strcpy(copy, raw);
    tok = strtok(copy, ",");
    while (tok != NULL) {
        double* grown = (double*)realloc(
            values,
            (size_t)(count + 1) * sizeof(double)
        );
        if (grown == NULL) {
            free(values);
            free(copy);
            return -1;
        }
        values = grown;
        values[count] = strtod(tok, NULL);
        count += 1;
        tok = strtok(NULL, ",");
    }
    free(copy);
    *outVals = values;
    *outCount = count;
    return 0;
}

/* Parse one comma-separated int list. */
static int parseIntList(
    const char* raw,
    int** outVals,
    int* outCount,
    int (*mapper)(const char*)
) {
    char* copy;
    char* tok;
    int count = 0;
    int* values = NULL;

    copy = (char*)malloc(strlen(raw) + 1);
    if (copy == NULL) {
        return -1;
    }
    strcpy(copy, raw);
    tok = strtok(copy, ",");
    while (tok != NULL) {
        int* grown = (int*)realloc(
            values,
            (size_t)(count + 1) * sizeof(int)
        );
        if (grown == NULL) {
            free(values);
            free(copy);
            return -1;
        }
        values = grown;
        values[count] = mapper == NULL ? atoi(tok) : mapper(tok);
        count += 1;
        tok = strtok(NULL, ",");
    }
    free(copy);
    *outVals = values;
    *outCount = count;
    return 0;
}

static int setDefaultIntAxis(int** vals, IntAxis* axis, int value) {
    *vals = (int*)malloc(sizeof(int));
    if (*vals == NULL) {
        return -1;
    }
    (*vals)[0] = value;
    axis->n = 1;
    axis->values = *vals;
    return 0;
}

static int setDefaultDoubleAxis(
    double** vals,
    DoubleAxis* axis,
    double value
) {
    *vals = (double*)malloc(sizeof(double));
    if (*vals == NULL) {
        return -1;
    }
    (*vals)[0] = value;
    axis->n = 1;
    axis->values = *vals;
    return 0;
}

int loadMeta(const char* path, TuneHostMeta* out) {
    FILE* fp;
    char line[2048];

    memset(out, 0, sizeof(*out));
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }

    while (fgets(line, sizeof(line), fp) != NULL) {
        char* eq;
        char* key;
        char* value;

        trimLine(line);
        if (line[0] == '\0') {
            continue;
        }
        eq = strchr(line, '=');
        if (eq == NULL) {
            fclose(fp);
            return -1;
        }
        *eq = '\0';
        key = line;
        value = eq + 1;

        if (strcmp(key, "ticker") == 0) {
            snprintf(out->ticker, sizeof(out->ticker), "%s", value);
        } else if (strcmp(key, "cacheRoot") == 0) {
            snprintf(out->cacheRoot, sizeof(out->cacheRoot), "%s", value);
        } else if (strcmp(key, "outDir") == 0) {
            snprintf(out->outDir, sizeof(out->outDir), "%s", value);
        } else if (strcmp(key, "resultsCsvPath") == 0) {
            snprintf(
                out->resultsCsvPath,
                sizeof(out->resultsCsvPath),
                "%s",
                value
            );
        } else if (strcmp(key, "bestRowCsvPath") == 0) {
            snprintf(
                out->bestRowCsvPath,
                sizeof(out->bestRowCsvPath),
                "%s",
                value
            );
        } else if (strcmp(key, "statsRowCsvPath") == 0) {
            snprintf(
                out->statsRowCsvPath,
                sizeof(out->statsRowCsvPath),
                "%s",
                value
            );
        } else if (strcmp(key, "anchorMs") == 0) {
            out->anchorMs = (int64_t)atoll(value);
        } else if (strcmp(key, "totalDays") == 0) {
            out->totalDays = atoi(value);
        } else if (strcmp(key, "primerDays") == 0) {
            out->primerDays = atoi(value);
        } else if (strcmp(key, "trainingDays") == 0) {
            out->trainingDays = atoi(value);
        } else if (strcmp(key, "tunerDays") == 0) {
            out->tunerDays = atoi(value);
        } else if (strcmp(key, "holdoutDays") == 0) {
            out->holdoutDays = atoi(value);
        } else if (strcmp(key, "feeRate") == 0) {
            out->feeRate = strtod(value, NULL);
        } else if (strcmp(key, "seedQuote") == 0) {
            out->seedQuote = strtod(value, NULL);
        } else if (strcmp(key, "dailyClusterPath") == 0) {
            snprintf(
                out->dailyClusterPath,
                sizeof(out->dailyClusterPath),
                "%s",
                value
            );
        } else if (strcmp(key, "dailyClusterModelPath") == 0) {
            snprintf(
                out->dailyClusterModelPath,
                sizeof(out->dailyClusterModelPath),
                "%s",
                value
            );
        }
    }

    fclose(fp);
    return 0;
}

int loadIntervalGroups(const char* path, IntervalGroupList* out) {
    FILE* fp;
    char line[512];

    memset(out, 0, sizeof(*out));
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }

    while (fgets(line, sizeof(line), fp) != NULL) {
        IntervalGroup item;

        trimLine(line);
        if (line[0] == '\0') {
            continue;
        }
        if (growIntervalGroups(out, out->count + 1) != 0) {
            fclose(fp);
            return -1;
        }
        if (
            sscanf(
                line,
                "%31[^,],%d,%d,%d",
                item.interval,
                &item.p1,
                &item.p2,
                &item.p3
            ) != 4
        ) {
            fclose(fp);
            return -1;
        }
        out->items[out->count] = item;
        out->count += 1;
    }

    fclose(fp);
    return 0;
}

int loadMacroGroups(const char* path, MacroGroupList* out) {
    FILE* fp;
    char line[1024];

    memset(out, 0, sizeof(*out));
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }

    while (fgets(line, sizeof(line), fp) != NULL) {
        MacroGroup item;

        trimLine(line);
        if (line[0] == '\0') {
            continue;
        }
        if (growMacroGroups(out, out->count + 1) != 0) {
            fclose(fp);
            return -1;
        }
        if (
            sscanf(
                line,
                "%31[^,],%d,%lf,%lf,%lf,%lf,%d,%d,%d,%d,%lf,%lf,%lf,%lf",
                item.macroInterval,
                &item.macroDynWin,
                &item.macroDynZMin,
                &item.macroDynZMax,
                &item.macroDynPctMin,
                &item.macroDynPctMax,
                &item.macroP1,
                &item.macroP3,
                &item.macroGradPeriod,
                &item.macroGradWinDays,
                &item.macroGradZMin,
                &item.macroGradZMax,
                &item.macroGradMultMin,
                &item.macroGradMultMax
            ) != 14
        ) {
            fclose(fp);
            return -1;
        }
        out->items[out->count] = item;
        out->count += 1;
    }

    fclose(fp);
    return 0;
}

int loadAxes(const char* path, ParsedAxes* out) {
    FILE* fp;
    char line[2048];

    memset(out, 0, sizeof(*out));
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    if (
        setDefaultDoubleAxis(
            &out->seedAssetPctVals,
            &out->axes.seedAssetPct,
            1.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->dailyStrongSellMultVals,
            &out->axes.dailyStrongSellMult,
            0.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->dailyStrongTargetPctVals,
            &out->axes.dailyStrongTargetPct,
            0.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->dailyBridgeDaysVals,
            &out->axes.dailyBridgeDays,
            0.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->dailyDownBuyMultVals,
            &out->axes.dailyDownBuyMult,
            0.4
        ) != 0
        || setDefaultDoubleAxis(
            &out->dailyCrabAssetCapPctVals,
            &out->axes.dailyCrabAssetCapPct,
            1.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->dailyLockTargetPctVals,
            &out->axes.dailyLockTargetPct,
            0.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->dailyLockGainPctVals,
            &out->axes.dailyLockGainPct,
            5.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->dailyLockNearHighPctVals,
            &out->axes.dailyLockNearHighPct,
            35.0
        ) != 0
        || setDefaultIntAxis(
            &out->dailyLockMaxDaysVals,
            &out->axes.dailyLockMaxDays,
            60
        ) != 0
        || setDefaultDoubleAxis(
            &out->postUltraCoastTargetPctVals,
            &out->axes.postUltraCoastTargetPct,
            1.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->postUltraGivebackPctVals,
            &out->axes.postUltraGivebackPct,
            0.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->postUltraReaccumPctVals,
            &out->axes.postUltraReaccumPct,
            0.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->postUltraDoubleTopPctVals,
            &out->axes.postUltraDoubleTopPct,
            0.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->postUltraMaxDaysVals,
            &out->axes.postUltraMaxDays,
            0.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->postUltraLockMinAssetPctVals,
            &out->axes.postUltraLockMinAssetPct,
            0.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->postUltraLockMaxAssetPctVals,
            &out->axes.postUltraLockMaxAssetPct,
            1.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->postUltraLockGivebackPctVals,
            &out->axes.postUltraLockGivebackPct,
            0.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->postUltraLockReaccumPctVals,
            &out->axes.postUltraLockReaccumPct,
            0.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->postUltraLockDoubleTopPctVals,
            &out->axes.postUltraLockDoubleTopPct,
            0.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->postUltraLockMaxDaysVals,
            &out->axes.postUltraLockMaxDays,
            0.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->macroSellRelaxPctVals,
            &out->axes.macroSellRelaxPct,
            0.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->annualIncomeBaseVals,
            &out->axes.annualIncomeBase,
            FIXED_INCOME_BASE
        ) != 0
        || setDefaultDoubleAxis(
            &out->peakLockCapPctVals,
            &out->axes.peakLockCapPct,
            1.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->peakLockUnlockGainPctVals,
            &out->axes.peakLockUnlockGainPct,
            25.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->peakLockReentryStepPctVals,
            &out->axes.peakLockReentryStepPct,
            0.15
        ) != 0
        || setDefaultDoubleAxis(
            &out->peakLockArmGainPctVals,
            &out->axes.peakLockArmGainPct,
            15.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->peakLockGivebackPctVals,
            &out->axes.peakLockGivebackPct,
            4.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->peakLockMaxDaysVals,
            &out->axes.peakLockMaxDays,
            120.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->peakLockEdgeDrawPctVals,
            &out->axes.peakLockEdgeDrawPct,
            5.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->peakLockEdgeSlopeDaysVals,
            &out->axes.peakLockEdgeSlopeDays,
            7.0
        ) != 0
        || setDefaultIntAxis(
            &out->peakLockRequireEdgeRiskVals,
            &out->axes.peakLockRequireEdgeRisk,
            1
        ) != 0
        || setDefaultDoubleAxis(
            &out->peakLockMaDaysVals,
            &out->axes.peakLockMaDays,
            30.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->peakLockKpVals,
            &out->axes.peakLockKp,
            6.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->peakLockKiVals,
            &out->axes.peakLockKi,
            0.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->peakLockKdVals,
            &out->axes.peakLockKd,
            0.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->peakLockIntegralDecayVals,
            &out->axes.peakLockIntegralDecay,
            0.985
        ) != 0
        || setDefaultDoubleAxis(
            &out->peakLockEntryThresholdVals,
            &out->axes.peakLockEntryThreshold,
            0.25
        ) != 0
        || setDefaultDoubleAxis(
            &out->peakLockExitThresholdVals,
            &out->axes.peakLockExitThreshold,
            0.05
        ) != 0
        || setDefaultIntAxis(
            &out->peakLockConfirmBarsVals,
            &out->axes.peakLockConfirmBars,
            6
        ) != 0
        || setDefaultDoubleAxis(
            &out->peakLockReleaseTargetPctVals,
            &out->axes.peakLockReleaseTargetPct,
            0.0
        ) != 0
        || setDefaultDoubleAxis(
            &out->peakLockUltraGraceDaysVals,
            &out->axes.peakLockUltraGraceDays,
            0.0
        ) != 0
    ) {
        fclose(fp);
        freeAxes(out);
        return -1;
    }

    while (fgets(line, sizeof(line), fp) != NULL) {
        char* eq;
        char* key;
        char* value;
        int count;

        trimLine(line);
        if (line[0] == '\0') {
            continue;
        }
        eq = strchr(line, '=');
        if (eq == NULL) {
            fclose(fp);
            return -1;
        }
        *eq = '\0';
        key = line;
        value = eq + 1;

        if (strcmp(key, "grad1BuyZMin") == 0) {
            if (parseDoubleList(value, &out->grad1BuyZMinVals, &count) != 0) {
                fclose(fp);
                return -1;
            }
            out->axes.grad1BuyZMin.n = count;
            out->axes.grad1BuyZMin.values = out->grad1BuyZMinVals;
        } else if (strcmp(key, "grad1SellZMin") == 0) {
            if (
                parseDoubleList(value, &out->grad1SellZMinVals, &count) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.grad1SellZMin.n = count;
            out->axes.grad1SellZMin.values = out->grad1SellZMinVals;
        } else if (strcmp(key, "grad1BuyWinDays") == 0) {
            if (
                parseIntList(
                    value,
                    &out->grad1BuyWinDaysVals,
                    &count,
                    NULL
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.grad1BuyWinDays.n = count;
            out->axes.grad1BuyWinDays.values = out->grad1BuyWinDaysVals;
        } else if (strcmp(key, "grad1SellWinDays") == 0) {
            if (
                parseIntList(
                    value,
                    &out->grad1SellWinDaysVals,
                    &count,
                    NULL
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.grad1SellWinDays.n = count;
            out->axes.grad1SellWinDays.values = out->grad1SellWinDaysVals;
        } else if (strcmp(key, "phaseBuy") == 0) {
            if (
                parseIntList(value, &out->phaseBuyVals, &count, NULL) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.phaseBuy.n = count;
            out->axes.phaseBuy.values = out->phaseBuyVals;
        } else if (strcmp(key, "phaseSell") == 0) {
            if (
                parseIntList(value, &out->phaseSellVals, &count, NULL) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.phaseSell.n = count;
            out->axes.phaseSell.values = out->phaseSellVals;
        } else if (strcmp(key, "finalPortionPct") == 0) {
            if (
                parseDoubleList(
                    value,
                    &out->finalPortionPctVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.finalPortionPct.n = count;
            out->axes.finalPortionPct.values = out->finalPortionPctVals;
        } else if (strcmp(key, "cooldown") == 0) {
            if (
                parseIntList(value, &out->cooldownVals, &count, NULL) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.cooldown.n = count;
            out->axes.cooldown.values = out->cooldownVals;
        } else if (strcmp(key, "taxMode") == 0) {
            if (
                parseIntList(value, &out->taxModeVals, &count, taxCode) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.taxMode.n = count;
            out->axes.taxMode.values = out->taxModeVals;
        } else if (strcmp(key, "seedAssetPct") == 0) {
            free(out->seedAssetPctVals);
            if (
                parseDoubleList(value, &out->seedAssetPctVals, &count)
                != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.seedAssetPct.n = count;
            out->axes.seedAssetPct.values = out->seedAssetPctVals;
        } else if (strcmp(key, "dailyStrongSellMult") == 0) {
            free(out->dailyStrongSellMultVals);
            if (
                parseDoubleList(
                    value,
                    &out->dailyStrongSellMultVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.dailyStrongSellMult.n = count;
            out->axes.dailyStrongSellMult.values =
                out->dailyStrongSellMultVals;
        } else if (strcmp(key, "dailyStrongTargetPct") == 0) {
            free(out->dailyStrongTargetPctVals);
            if (
                parseDoubleList(
                    value,
                    &out->dailyStrongTargetPctVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.dailyStrongTargetPct.n = count;
            out->axes.dailyStrongTargetPct.values =
                out->dailyStrongTargetPctVals;
        } else if (strcmp(key, "dailyBridgeDays") == 0) {
            free(out->dailyBridgeDaysVals);
            if (
                parseDoubleList(value, &out->dailyBridgeDaysVals, &count)
                != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.dailyBridgeDays.n = count;
            out->axes.dailyBridgeDays.values = out->dailyBridgeDaysVals;
        } else if (strcmp(key, "dailyDownBuyMult") == 0) {
            free(out->dailyDownBuyMultVals);
            if (
                parseDoubleList(
                    value,
                    &out->dailyDownBuyMultVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.dailyDownBuyMult.n = count;
            out->axes.dailyDownBuyMult.values = out->dailyDownBuyMultVals;
        } else if (strcmp(key, "dailyCrabAssetCapPct") == 0) {
            free(out->dailyCrabAssetCapPctVals);
            if (
                parseDoubleList(
                    value,
                    &out->dailyCrabAssetCapPctVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.dailyCrabAssetCapPct.n = count;
            out->axes.dailyCrabAssetCapPct.values =
                out->dailyCrabAssetCapPctVals;
        } else if (strcmp(key, "dailyLockTargetPct") == 0) {
            free(out->dailyLockTargetPctVals);
            if (
                parseDoubleList(
                    value,
                    &out->dailyLockTargetPctVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.dailyLockTargetPct.n = count;
            out->axes.dailyLockTargetPct.values =
                out->dailyLockTargetPctVals;
        } else if (strcmp(key, "dailyLockGainPct") == 0) {
            free(out->dailyLockGainPctVals);
            if (
                parseDoubleList(value, &out->dailyLockGainPctVals, &count)
                != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.dailyLockGainPct.n = count;
            out->axes.dailyLockGainPct.values = out->dailyLockGainPctVals;
        } else if (strcmp(key, "dailyLockNearHighPct") == 0) {
            free(out->dailyLockNearHighPctVals);
            if (
                parseDoubleList(
                    value,
                    &out->dailyLockNearHighPctVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.dailyLockNearHighPct.n = count;
            out->axes.dailyLockNearHighPct.values =
                out->dailyLockNearHighPctVals;
        } else if (strcmp(key, "dailyLockMaxDays") == 0) {
            free(out->dailyLockMaxDaysVals);
            if (
                parseIntList(
                    value,
                    &out->dailyLockMaxDaysVals,
                    &count,
                    NULL
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.dailyLockMaxDays.n = count;
            out->axes.dailyLockMaxDays.values = out->dailyLockMaxDaysVals;
        } else if (strcmp(key, "postUltraCoastTargetPct") == 0) {
            free(out->postUltraCoastTargetPctVals);
            if (
                parseDoubleList(
                    value,
                    &out->postUltraCoastTargetPctVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.postUltraCoastTargetPct.n = count;
            out->axes.postUltraCoastTargetPct.values =
                out->postUltraCoastTargetPctVals;
        } else if (strcmp(key, "postUltraGivebackPct") == 0) {
            free(out->postUltraGivebackPctVals);
            if (
                parseDoubleList(
                    value,
                    &out->postUltraGivebackPctVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.postUltraGivebackPct.n = count;
            out->axes.postUltraGivebackPct.values =
                out->postUltraGivebackPctVals;
        } else if (strcmp(key, "postUltraReaccumPct") == 0) {
            free(out->postUltraReaccumPctVals);
            if (
                parseDoubleList(
                    value,
                    &out->postUltraReaccumPctVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.postUltraReaccumPct.n = count;
            out->axes.postUltraReaccumPct.values =
                out->postUltraReaccumPctVals;
        } else if (strcmp(key, "postUltraDoubleTopPct") == 0) {
            free(out->postUltraDoubleTopPctVals);
            if (
                parseDoubleList(
                    value,
                    &out->postUltraDoubleTopPctVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.postUltraDoubleTopPct.n = count;
            out->axes.postUltraDoubleTopPct.values =
                out->postUltraDoubleTopPctVals;
        } else if (strcmp(key, "postUltraMaxDays") == 0) {
            free(out->postUltraMaxDaysVals);
            if (
                parseDoubleList(value, &out->postUltraMaxDaysVals, &count)
                != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.postUltraMaxDays.n = count;
            out->axes.postUltraMaxDays.values = out->postUltraMaxDaysVals;
        } else if (strcmp(key, "postUltraLockMinAssetPct") == 0) {
            free(out->postUltraLockMinAssetPctVals);
            if (
                parseDoubleList(
                    value,
                    &out->postUltraLockMinAssetPctVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.postUltraLockMinAssetPct.n = count;
            out->axes.postUltraLockMinAssetPct.values =
                out->postUltraLockMinAssetPctVals;
        } else if (strcmp(key, "postUltraLockMaxAssetPct") == 0) {
            free(out->postUltraLockMaxAssetPctVals);
            if (
                parseDoubleList(
                    value,
                    &out->postUltraLockMaxAssetPctVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.postUltraLockMaxAssetPct.n = count;
            out->axes.postUltraLockMaxAssetPct.values =
                out->postUltraLockMaxAssetPctVals;
        } else if (strcmp(key, "postUltraLockGivebackPct") == 0) {
            free(out->postUltraLockGivebackPctVals);
            if (
                parseDoubleList(
                    value,
                    &out->postUltraLockGivebackPctVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.postUltraLockGivebackPct.n = count;
            out->axes.postUltraLockGivebackPct.values =
                out->postUltraLockGivebackPctVals;
        } else if (strcmp(key, "postUltraLockReaccumPct") == 0) {
            free(out->postUltraLockReaccumPctVals);
            if (
                parseDoubleList(
                    value,
                    &out->postUltraLockReaccumPctVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.postUltraLockReaccumPct.n = count;
            out->axes.postUltraLockReaccumPct.values =
                out->postUltraLockReaccumPctVals;
        } else if (strcmp(key, "postUltraLockDoubleTopPct") == 0) {
            free(out->postUltraLockDoubleTopPctVals);
            if (
                parseDoubleList(
                    value,
                    &out->postUltraLockDoubleTopPctVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.postUltraLockDoubleTopPct.n = count;
            out->axes.postUltraLockDoubleTopPct.values =
                out->postUltraLockDoubleTopPctVals;
        } else if (strcmp(key, "postUltraLockMaxDays") == 0) {
            free(out->postUltraLockMaxDaysVals);
            if (
                parseDoubleList(
                    value,
                    &out->postUltraLockMaxDaysVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.postUltraLockMaxDays.n = count;
            out->axes.postUltraLockMaxDays.values =
                out->postUltraLockMaxDaysVals;
        } else if (strcmp(key, "macroSellRelaxPct") == 0) {
            free(out->macroSellRelaxPctVals);
            if (
                parseDoubleList(
                    value,
                    &out->macroSellRelaxPctVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.macroSellRelaxPct.n = count;
            out->axes.macroSellRelaxPct.values =
                out->macroSellRelaxPctVals;
        } else if (strcmp(key, "peakLockCapPct") == 0) {
            free(out->peakLockCapPctVals);
            if (
                parseDoubleList(value, &out->peakLockCapPctVals, &count)
                != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.peakLockCapPct.n = count;
            out->axes.peakLockCapPct.values = out->peakLockCapPctVals;
        } else if (strcmp(key, "peakLockUnlockGainPct") == 0) {
            free(out->peakLockUnlockGainPctVals);
            if (
                parseDoubleList(
                    value,
                    &out->peakLockUnlockGainPctVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.peakLockUnlockGainPct.n = count;
            out->axes.peakLockUnlockGainPct.values =
                out->peakLockUnlockGainPctVals;
        } else if (strcmp(key, "peakLockReentryStepPct") == 0) {
            free(out->peakLockReentryStepPctVals);
            if (
                parseDoubleList(
                    value,
                    &out->peakLockReentryStepPctVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.peakLockReentryStepPct.n = count;
            out->axes.peakLockReentryStepPct.values =
                out->peakLockReentryStepPctVals;
        } else if (strcmp(key, "peakLockArmGainPct") == 0) {
            free(out->peakLockArmGainPctVals);
            if (
                parseDoubleList(value, &out->peakLockArmGainPctVals, &count)
                != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.peakLockArmGainPct.n = count;
            out->axes.peakLockArmGainPct.values = out->peakLockArmGainPctVals;
        } else if (strcmp(key, "peakLockGivebackPct") == 0) {
            free(out->peakLockGivebackPctVals);
            if (
                parseDoubleList(value, &out->peakLockGivebackPctVals, &count)
                != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.peakLockGivebackPct.n = count;
            out->axes.peakLockGivebackPct.values =
                out->peakLockGivebackPctVals;
        } else if (strcmp(key, "peakLockMaxDays") == 0) {
            free(out->peakLockMaxDaysVals);
            if (
                parseDoubleList(value, &out->peakLockMaxDaysVals, &count)
                != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.peakLockMaxDays.n = count;
            out->axes.peakLockMaxDays.values = out->peakLockMaxDaysVals;
        } else if (strcmp(key, "peakLockEdgeDrawPct") == 0) {
            free(out->peakLockEdgeDrawPctVals);
            if (
                parseDoubleList(value, &out->peakLockEdgeDrawPctVals, &count)
                != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.peakLockEdgeDrawPct.n = count;
            out->axes.peakLockEdgeDrawPct.values =
                out->peakLockEdgeDrawPctVals;
        } else if (strcmp(key, "peakLockEdgeSlopeDays") == 0) {
            free(out->peakLockEdgeSlopeDaysVals);
            if (
                parseDoubleList(
                    value,
                    &out->peakLockEdgeSlopeDaysVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.peakLockEdgeSlopeDays.n = count;
            out->axes.peakLockEdgeSlopeDays.values =
                out->peakLockEdgeSlopeDaysVals;
        } else if (strcmp(key, "peakLockRequireEdgeRisk") == 0) {
            free(out->peakLockRequireEdgeRiskVals);
            if (
                parseIntList(
                    value,
                    &out->peakLockRequireEdgeRiskVals,
                    &count,
                    NULL
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.peakLockRequireEdgeRisk.n = count;
            out->axes.peakLockRequireEdgeRisk.values =
                out->peakLockRequireEdgeRiskVals;
        } else if (strcmp(key, "peakLockMaDays") == 0) {
            free(out->peakLockMaDaysVals);
            if (
                parseDoubleList(value, &out->peakLockMaDaysVals, &count)
                != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.peakLockMaDays.n = count;
            out->axes.peakLockMaDays.values = out->peakLockMaDaysVals;
        } else if (strcmp(key, "peakLockKp") == 0) {
            free(out->peakLockKpVals);
            if (
                parseDoubleList(value, &out->peakLockKpVals, &count) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.peakLockKp.n = count;
            out->axes.peakLockKp.values = out->peakLockKpVals;
        } else if (strcmp(key, "peakLockKi") == 0) {
            free(out->peakLockKiVals);
            if (
                parseDoubleList(value, &out->peakLockKiVals, &count) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.peakLockKi.n = count;
            out->axes.peakLockKi.values = out->peakLockKiVals;
        } else if (strcmp(key, "peakLockKd") == 0) {
            free(out->peakLockKdVals);
            if (
                parseDoubleList(value, &out->peakLockKdVals, &count) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.peakLockKd.n = count;
            out->axes.peakLockKd.values = out->peakLockKdVals;
        } else if (strcmp(key, "peakLockIntegralDecay") == 0) {
            free(out->peakLockIntegralDecayVals);
            if (
                parseDoubleList(
                    value,
                    &out->peakLockIntegralDecayVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.peakLockIntegralDecay.n = count;
            out->axes.peakLockIntegralDecay.values =
                out->peakLockIntegralDecayVals;
        } else if (strcmp(key, "peakLockEntryThreshold") == 0) {
            free(out->peakLockEntryThresholdVals);
            if (
                parseDoubleList(
                    value,
                    &out->peakLockEntryThresholdVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.peakLockEntryThreshold.n = count;
            out->axes.peakLockEntryThreshold.values =
                out->peakLockEntryThresholdVals;
        } else if (strcmp(key, "peakLockExitThreshold") == 0) {
            free(out->peakLockExitThresholdVals);
            if (
                parseDoubleList(
                    value,
                    &out->peakLockExitThresholdVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.peakLockExitThreshold.n = count;
            out->axes.peakLockExitThreshold.values =
                out->peakLockExitThresholdVals;
        } else if (strcmp(key, "peakLockConfirmBars") == 0) {
            free(out->peakLockConfirmBarsVals);
            if (
                parseIntList(
                    value,
                    &out->peakLockConfirmBarsVals,
                    &count,
                    NULL
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.peakLockConfirmBars.n = count;
            out->axes.peakLockConfirmBars.values =
                out->peakLockConfirmBarsVals;
        } else if (strcmp(key, "peakLockReleaseTargetPct") == 0) {
            free(out->peakLockReleaseTargetPctVals);
            if (
                parseDoubleList(
                    value,
                    &out->peakLockReleaseTargetPctVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.peakLockReleaseTargetPct.n = count;
            out->axes.peakLockReleaseTargetPct.values =
                out->peakLockReleaseTargetPctVals;
        } else if (strcmp(key, "peakLockUltraGraceDays") == 0) {
            free(out->peakLockUltraGraceDaysVals);
            if (
                parseDoubleList(
                    value,
                    &out->peakLockUltraGraceDaysVals,
                    &count
                ) != 0
            ) {
                fclose(fp);
                return -1;
            }
            out->axes.peakLockUltraGraceDays.n = count;
            out->axes.peakLockUltraGraceDays.values =
                out->peakLockUltraGraceDaysVals;
        }
    }

    fclose(fp);
    return 0;
}

void freeIntervalGroups(IntervalGroupList* groups) {
    free(groups->items);
    groups->items = NULL;
    groups->count = 0;
}

void freeMacroGroups(MacroGroupList* groups) {
    free(groups->items);
    groups->items = NULL;
    groups->count = 0;
}

void freeAxes(ParsedAxes* axes) {
    free(axes->grad1BuyZMinVals);
    free(axes->grad1SellZMinVals);
    free(axes->grad1BuyWinDaysVals);
    free(axes->grad1SellWinDaysVals);
    free(axes->phaseBuyVals);
    free(axes->phaseSellVals);
    free(axes->finalPortionPctVals);
    free(axes->cooldownVals);
    free(axes->taxModeVals);
    free(axes->seedAssetPctVals);
    free(axes->dailyStrongSellMultVals);
    free(axes->dailyStrongTargetPctVals);
    free(axes->dailyBridgeDaysVals);
    free(axes->dailyDownBuyMultVals);
    free(axes->dailyCrabAssetCapPctVals);
    free(axes->dailyLockTargetPctVals);
    free(axes->dailyLockGainPctVals);
    free(axes->dailyLockNearHighPctVals);
    free(axes->dailyLockMaxDaysVals);
    free(axes->postUltraCoastTargetPctVals);
    free(axes->postUltraGivebackPctVals);
    free(axes->postUltraReaccumPctVals);
    free(axes->postUltraDoubleTopPctVals);
    free(axes->postUltraMaxDaysVals);
    free(axes->postUltraLockMinAssetPctVals);
    free(axes->postUltraLockMaxAssetPctVals);
    free(axes->postUltraLockGivebackPctVals);
    free(axes->postUltraLockReaccumPctVals);
    free(axes->postUltraLockDoubleTopPctVals);
    free(axes->postUltraLockMaxDaysVals);
    free(axes->macroSellRelaxPctVals);
    free(axes->annualIncomeBaseVals);
    free(axes->peakLockCapPctVals);
    free(axes->peakLockUnlockGainPctVals);
    free(axes->peakLockReentryStepPctVals);
    free(axes->peakLockArmGainPctVals);
    free(axes->peakLockGivebackPctVals);
    free(axes->peakLockMaxDaysVals);
    free(axes->peakLockEdgeDrawPctVals);
    free(axes->peakLockEdgeSlopeDaysVals);
    free(axes->peakLockRequireEdgeRiskVals);
    free(axes->peakLockMaDaysVals);
    free(axes->peakLockKpVals);
    free(axes->peakLockKiVals);
    free(axes->peakLockKdVals);
    free(axes->peakLockIntegralDecayVals);
    free(axes->peakLockEntryThresholdVals);
    free(axes->peakLockExitThresholdVals);
    free(axes->peakLockConfirmBarsVals);
    free(axes->peakLockReleaseTargetPctVals);
    free(axes->peakLockUltraGraceDaysVals);
    memset(axes, 0, sizeof(*axes));
}
