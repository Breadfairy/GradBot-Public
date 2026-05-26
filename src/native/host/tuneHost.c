#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include "klineCsv.h"
#include "tuneSpec.h"

typedef struct{
    char interval[32];
    KlineSoa raw;
} KlineCacheEntry;

typedef struct{
    int count;
    KlineCacheEntry* items;
} KlineCache;

typedef struct{
    int has;
    EvalRow row;
    IntervalGroup intervalGroup;
    MacroGroup macroGroup;
    uint64_t comboIdx;
    double score;
} WinnerState;

typedef struct{
    int n;
    int cap;
    int64_t* closeMs;
    double* close;
    int* cluster;
    double* ret30;
    double* nearHigh;
} DailyPostureRaw;

typedef struct{
    int* cluster;
    double* ret30;
    double* nearHigh;
} DailyPostureAligned;

typedef struct{
    char interval[32];
    int windowBars;
    int periodFast;
    int periodMid;
    int periodSlow;
    int clusterCount;
    int featureCount;
    int pcaCount;
    int* featureIds;
    int* clusterRemap;
    int remapCount;
    double* center;
    double* scale;
    double* pcaMean;
    double* pcaComponents;
    double* centroids;
} ClusterModel;

#define DAY_MS (24LL * 60LL * 60LL * 1000LL)
#define FIXED_INCOME_BASE 36000.0
/* Return wall-clock seconds. */
static double nowSecs(void) {
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + ((double)tv.tv_usec / 1000000.0);
}

/* Return wall-clock UTC milliseconds. */
static int64_t nowMs(void) {
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return (
        ((int64_t)tv.tv_sec * 1000LL)
        + ((int64_t)tv.tv_usec / 1000LL)
    );
}

static void trimToken(char* token) {
    size_t len = strlen(token);

    while (len > 0 && (token[len - 1] == '\n' || token[len - 1] == '\r')) {
        token[len - 1] = '\0';
        len -= 1;
    }
}

static int growDaily(DailyPostureRaw* daily, int need) {
    int64_t* closeMs;
    double* close;
    int* cluster;
    double* ret30;
    double* nearHigh;
    int newCap = daily->cap <= 0 ? 512 : daily->cap * 2;

    while (newCap < need) {
        newCap *= 2;
    }
    closeMs = (int64_t*)realloc(
        daily->closeMs,
        (size_t)newCap * sizeof(int64_t)
    );
    close = (double*)realloc(daily->close, (size_t)newCap * sizeof(double));
    cluster = (int*)realloc(daily->cluster, (size_t)newCap * sizeof(int));
    ret30 = (double*)realloc(daily->ret30, (size_t)newCap * sizeof(double));
    nearHigh = (double*)realloc(
        daily->nearHigh,
        (size_t)newCap * sizeof(double)
    );
    if (
        closeMs == NULL
        || close == NULL
        || cluster == NULL
        || ret30 == NULL
        || nearHigh == NULL
    ) {
        free(closeMs);
        free(close);
        free(cluster);
        free(ret30);
        free(nearHigh);
        return -1;
    }
    daily->closeMs = closeMs;
    daily->close = close;
    daily->cluster = cluster;
    daily->ret30 = ret30;
    daily->nearHigh = nearHigh;
    daily->cap = newCap;
    return 0;
}

static int headerIndex(char* header, const char* name) {
    char* tok;
    int idx = 0;

    tok = strtok(header, ",");
    while (tok != NULL) {
        trimToken(tok);
        if (strcmp(tok, name) == 0) {
            return idx;
        }
        idx += 1;
        tok = strtok(NULL, ",");
    }
    return -1;
}

static void parseDailyLine(
    char* line,
    int closeMsIdx,
    int closeIdx,
    int clusterIdx,
    int64_t* closeMsOut,
    double* closeOut,
    int* clusterOut
) {
    char* start = line;
    char* end = line;
    char saved;
    int col = 0;

    *closeMsOut = 0;
    *closeOut = 0.0;
    *clusterOut = -1;

    while (1) {
        end = start;
        while (
            *end != '\0'
            && *end != ','
            && *end != '\n'
            && *end != '\r'
        ) {
            end += 1;
        }
        saved = *end;
        *end = '\0';
        if (col == closeMsIdx) {
            *closeMsOut = (int64_t)atoll(start);
        } else if (col == closeIdx) {
            *closeOut = strtod(start, NULL);
        } else if (col == clusterIdx) {
            *clusterOut = atoi(start);
        }
        *end = saved;
        if (saved != ',') {
            break;
        }
        start = end + 1;
        col += 1;
    }
}

static int loadDailyPostureCsv(
    const char* path,
    DailyPostureRaw* daily
) {
    FILE* fp;
    char line[8192];
    char headerA[8192];
    char headerB[8192];
    char headerC[8192];
    int closeMsIdx;
    int closeIdx;
    int clusterIdx;

    memset(daily, 0, sizeof(*daily));
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    fp = fopen(path, "r");
    if (fp == NULL || fgets(line, sizeof(line), fp) == NULL) {
        if (fp != NULL) {
            fclose(fp);
        }
        return -1;
    }
    snprintf(headerA, sizeof(headerA), "%s", line);
    snprintf(headerB, sizeof(headerB), "%s", line);
    snprintf(headerC, sizeof(headerC), "%s", line);
    closeMsIdx = headerIndex(headerA, "closeMs");
    closeIdx = headerIndex(headerB, "close");
    clusterIdx = headerIndex(headerC, "cluster");
    if (closeMsIdx < 0 || closeIdx < 0 || clusterIdx < 0) {
        fclose(fp);
        return -1;
    }

    while (fgets(line, sizeof(line), fp) != NULL) {
        int cluster = -1;
        int64_t closeMs = 0;
        double close = 0.0;

        parseDailyLine(
            line,
            closeMsIdx,
            closeIdx,
            clusterIdx,
            &closeMs,
            &close,
            &cluster
        );
        if (cluster < 0) {
            continue;
        }
        if (daily->n >= daily->cap && growDaily(daily, daily->n + 1) != 0) {
            fclose(fp);
            return -1;
        }
        daily->closeMs[daily->n] = closeMs;
        daily->close[daily->n] = close;
        daily->cluster[daily->n] = cluster;
        daily->n += 1;
    }
    fclose(fp);
    return 0;
}

static int parseModelDoubleList(
    const char* raw,
    double** outVals,
    int* outCount
) {
    char* copy;
    char* tok;
    double* vals = NULL;
    int count = 0;

    copy = (char*)malloc(strlen(raw) + 1);
    if (copy == NULL) {
        return -1;
    }
    strcpy(copy, raw);
    tok = strtok(copy, ",");
    while (tok != NULL && tok[0] != '\0') {
        double* grown = (double*)realloc(
            vals,
            (size_t)(count + 1) * sizeof(double)
        );
        if (grown == NULL) {
            free(vals);
            free(copy);
            return -1;
        }
        vals = grown;
        vals[count] = strtod(tok, NULL);
        count += 1;
        tok = strtok(NULL, ",");
    }
    free(copy);
    *outVals = vals;
    *outCount = count;
    return 0;
}

static int parseModelIntList(
    const char* raw,
    int** outVals,
    int* outCount
) {
    char* copy;
    char* tok;
    int* vals = NULL;
    int count = 0;

    copy = (char*)malloc(strlen(raw) + 1);
    if (copy == NULL) {
        return -1;
    }
    strcpy(copy, raw);
    tok = strtok(copy, ",");
    while (tok != NULL && tok[0] != '\0') {
        int* grown = (int*)realloc(
            vals,
            (size_t)(count + 1) * sizeof(int)
        );
        if (grown == NULL) {
            free(vals);
            free(copy);
            return -1;
        }
        vals = grown;
        vals[count] = atoi(tok);
        count += 1;
        tok = strtok(NULL, ",");
    }
    free(copy);
    *outVals = vals;
    *outCount = count;
    return 0;
}

static int loadClusterModel(const char* path, ClusterModel* model) {
    FILE* fp;
    char line[65536];
    int count;

    memset(model, 0, sizeof(*model));
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    while (fgets(line, sizeof(line), fp) != NULL) {
        char* eq;
        char* key;
        char* value;

        trimToken(line);
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
        if (strcmp(key, "interval") == 0) {
            snprintf(model->interval, sizeof(model->interval), "%s", value);
        } else if (strcmp(key, "windowBars") == 0) {
            model->windowBars = atoi(value);
        } else if (strcmp(key, "periodFast") == 0) {
            model->periodFast = atoi(value);
        } else if (strcmp(key, "periodMid") == 0) {
            model->periodMid = atoi(value);
        } else if (strcmp(key, "periodSlow") == 0) {
            model->periodSlow = atoi(value);
        } else if (strcmp(key, "clusterCount") == 0) {
            model->clusterCount = atoi(value);
        } else if (strcmp(key, "featureCount") == 0) {
            model->featureCount = atoi(value);
        } else if (strcmp(key, "pcaCount") == 0) {
            model->pcaCount = atoi(value);
        } else if (strcmp(key, "featureIds") == 0) {
            if (parseModelIntList(value, &model->featureIds, &count) != 0) {
                fclose(fp);
                return -1;
            }
        } else if (strcmp(key, "clusterRemap") == 0) {
            if (
                parseModelIntList(value, &model->clusterRemap, &count) != 0
            ) {
                fclose(fp);
                return -1;
            }
            model->remapCount = count;
        } else if (strcmp(key, "center") == 0) {
            if (parseModelDoubleList(value, &model->center, &count) != 0) {
                fclose(fp);
                return -1;
            }
        } else if (strcmp(key, "scale") == 0) {
            if (parseModelDoubleList(value, &model->scale, &count) != 0) {
                fclose(fp);
                return -1;
            }
        } else if (strcmp(key, "pcaMean") == 0) {
            if (parseModelDoubleList(value, &model->pcaMean, &count) != 0) {
                fclose(fp);
                return -1;
            }
        } else if (strcmp(key, "pcaComponents") == 0) {
            if (
                parseModelDoubleList(value, &model->pcaComponents, &count)
                != 0
            ) {
                fclose(fp);
                return -1;
            }
        } else if (strcmp(key, "centroids") == 0) {
            if (parseModelDoubleList(value, &model->centroids, &count) != 0) {
                fclose(fp);
                return -1;
            }
        }
    }
    fclose(fp);
    return (
        model->featureIds != NULL
        && model->center != NULL
        && model->scale != NULL
        && model->pcaMean != NULL
        && model->pcaComponents != NULL
        && model->centroids != NULL
        && model->featureCount > 0
        && model->clusterCount > 0
        && model->pcaCount > 0
        ? 0
        : -1
    );
}

static void freeClusterModel(ClusterModel* model) {
    free(model->featureIds);
    free(model->clusterRemap);
    free(model->center);
    free(model->scale);
    free(model->pcaMean);
    free(model->pcaComponents);
    free(model->centroids);
    memset(model, 0, sizeof(*model));
}

static double safePct(double num, double den) {
    return fabs(den) > 1e-12 ? (num / den) * 100.0 : NAN;
}

static double retPct(const double* values, int index, int bars) {
    double prev;

    if (index < bars) {
        return NAN;
    }
    prev = values[index - bars];
    return prev > 0.0 ? ((values[index] / prev) - 1.0) * 100.0 : NAN;
}

static double rollingMaxValue(const double* values, int index, int window) {
    int i;
    double high;

    if (index + 1 < window) {
        return NAN;
    }
    high = values[index - window + 1];
    for (i = index - window + 2; i <= index; i++) {
        if (values[i] > high) {
            high = values[i];
        }
    }
    return high;
}

static double rollingMinValue(const double* values, int index, int window) {
    int i;
    double low;

    if (index + 1 < window) {
        return NAN;
    }
    low = values[index - window + 1];
    for (i = index - window + 2; i <= index; i++) {
        if (values[i] < low) {
            low = values[i];
        }
    }
    return low;
}

static double rollingMeanValue(const double* values, int index, int window) {
    int i;
    double sum = 0.0;

    if (index + 1 < window) {
        return NAN;
    }
    for (i = index - window + 1; i <= index; i++) {
        if (!isfinite(values[i])) {
            return NAN;
        }
        sum += values[i];
    }
    return sum / (double)window;
}

static double rollingStdValue(const double* values, int index, int window) {
    int i;
    double sum = 0.0;
    double sum2 = 0.0;
    double mean;
    double var;

    if (index + 1 < window) {
        return NAN;
    }
    for (i = index - window + 1; i <= index; i++) {
        if (!isfinite(values[i])) {
            return NAN;
        }
        sum += values[i];
        sum2 += values[i] * values[i];
    }
    mean = sum / (double)window;
    var = (sum2 / (double)window) - (mean * mean);
    return sqrt(fmax(var, 0.0));
}

static double priorRollingZValue(
    const double* values,
    int index,
    int window
) {
    int i;
    double sum = 0.0;
    double sum2 = 0.0;
    double mean;
    double std;

    if (index < window) {
        return NAN;
    }
    for (i = index - window; i < index; i++) {
        if (!isfinite(values[i])) {
            return NAN;
        }
        sum += values[i];
        sum2 += values[i] * values[i];
    }
    mean = sum / (double)window;
    std = sqrt(fmax((sum2 / (double)window) - (mean * mean), 0.0));
    if (std <= 1e-12) {
        return NAN;
    }
    return (values[index] - mean) / std;
}

static double clusterFeatureValue(
    const KlineSoa* raw,
    const ClusterModel* model,
    const double* emaFast,
    const double* emaMid,
    const double* emaSlow,
    const int* trend,
    const double* logRet,
    const double* ret1,
    const double* bodyAbsPct,
    const double* rangePct,
    const double* logVolume,
    int index,
    int featureId
) {
    double close = raw->close[index];
    double open = raw->open[index];
    double high = raw->high[index];
    double low = raw->low[index];
    double highRoll = rollingMaxValue(raw->high, index, model->windowBars);
    double lowRoll = rollingMinValue(raw->low, index, model->windowBars);
    double upperRef = open > close ? open : close;
    double lowerRef = open < close ? open : close;
    double sumAbs;

    switch (featureId) {
        case 0:
            return safePct(close - emaFast[index], close);
        case 1:
            return safePct(close - emaMid[index], close);
        case 2:
            return safePct(close - emaSlow[index], close);
        case 3:
            return safePct(emaFast[index] - emaMid[index], close);
        case 4:
            return safePct(emaMid[index] - emaSlow[index], close);
        case 5:
            return safePct(emaFast[index] - emaSlow[index], close);
        case 6:
            if (index <= 0) {
                return NAN;
            }
            return safePct(emaFast[index] - emaFast[index - 1], emaFast[index]);
        case 7:
            if (index <= 0) {
                return NAN;
            }
            return safePct(emaMid[index] - emaMid[index - 1], emaMid[index]);
        case 8:
            if (index <= 0) {
                return NAN;
            }
            return safePct(emaSlow[index] - emaSlow[index - 1], emaSlow[index]);
        case 9:
            return (double)trend[index];
        case 10:
            return safePct(highRoll - close, close);
        case 11:
            return safePct(close - lowRoll, close);
        case 12:
            return safePct(highRoll - lowRoll, close);
        case 13:
            return rollingStdValue(logRet, index, model->windowBars);
        case 14:
            return retPct(raw->close, index, 1);
        case 15:
            return retPct(raw->close, index, 3);
        case 16:
            return retPct(raw->close, index, 6);
        case 17:
            return retPct(raw->close, index, 12);
        case 18:
            return retPct(raw->close, index, 24);
        case 19:
            sumAbs = rollingMeanValue(ret1, index, model->windowBars);
            if (!isfinite(sumAbs)) {
                return NAN;
            }
            sumAbs *= (double)model->windowBars;
            return (
                sumAbs > 1e-12
                ? fabs(retPct(raw->close, index, 24)) / sumAbs
                : NAN
            );
        case 20:
            return safePct(close - open, close);
        case 21:
            return safePct(fabs(close - open), close);
        case 22:
            return safePct(high - upperRef, close);
        case 23:
            return safePct(lowerRef - low, close);
        case 24:
            return rollingMeanValue(bodyAbsPct, index, model->windowBars);
        case 25:
            return rollingMeanValue(rangePct, index, model->windowBars);
        case 26:
            return priorRollingZValue(logVolume, index, 168);
        default:
            return NAN;
    }
}

static int buildDailyPostureFromModel(
    const KlineSoa* raw,
    const ClusterModel* model,
    DailyPostureRaw* daily
) {
    double* emaFast;
    double* emaMid;
    double* emaSlow;
    double* logRet;
    double* ret1;
    double* bodyAbsPct;
    double* rangePct;
    double* logVolume;
    double* feature;
    double* scaled;
    double* modelX;
    int* trend;
    int i;
    int j;
    int k;

    memset(daily, 0, sizeof(*daily));
    emaFast = (double*)malloc((size_t)raw->n * sizeof(double));
    emaMid = (double*)malloc((size_t)raw->n * sizeof(double));
    emaSlow = (double*)malloc((size_t)raw->n * sizeof(double));
    logRet = (double*)malloc((size_t)raw->n * sizeof(double));
    ret1 = (double*)malloc((size_t)raw->n * sizeof(double));
    bodyAbsPct = (double*)malloc((size_t)raw->n * sizeof(double));
    rangePct = (double*)malloc((size_t)raw->n * sizeof(double));
    logVolume = (double*)malloc((size_t)raw->n * sizeof(double));
    trend = (int*)malloc((size_t)raw->n * sizeof(int));
    feature = (double*)malloc((size_t)model->featureCount * sizeof(double));
    scaled = (double*)malloc((size_t)model->featureCount * sizeof(double));
    modelX = (double*)malloc((size_t)model->pcaCount * sizeof(double));
    if (
        emaFast == NULL
        || emaMid == NULL
        || emaSlow == NULL
        || logRet == NULL
        || ret1 == NULL
        || bodyAbsPct == NULL
        || rangePct == NULL
        || logVolume == NULL
        || trend == NULL
        || feature == NULL
        || scaled == NULL
        || modelX == NULL
    ) {
        free(emaFast);
        free(emaMid);
        free(emaSlow);
        free(logRet);
        free(ret1);
        free(bodyAbsPct);
        free(rangePct);
        free(logVolume);
        free(trend);
        free(feature);
        free(scaled);
        free(modelX);
        return -1;
    }

    emaLpf(raw->close, raw->n, model->periodFast, emaFast);
    emaLpf(raw->close, raw->n, model->periodMid, emaMid);
    emaLpf(raw->close, raw->n, model->periodSlow, emaSlow);
    trendCodes(emaFast, emaMid, emaSlow, raw->n, trend);
    for (i = 0; i < raw->n; i++) {
        double candleRange = raw->high[i] - raw->low[i];
        double candleBody = raw->close[i] - raw->open[i];

        logRet[i] = (
            i > 0 && raw->close[i - 1] > 0.0
            ? log(raw->close[i] / raw->close[i - 1])
            : NAN
        );
        ret1[i] = fabs(retPct(raw->close, i, 1));
        bodyAbsPct[i] = safePct(fabs(candleBody), raw->close[i]);
        rangePct[i] = safePct(candleRange, raw->close[i]);
        logVolume[i] = log1p(raw->volume[i]);
    }

    if (growDaily(daily, raw->n) != 0) {
        free(emaFast);
        free(emaMid);
        free(emaSlow);
        free(logRet);
        free(ret1);
        free(bodyAbsPct);
        free(rangePct);
        free(logVolume);
        free(trend);
        free(feature);
        free(scaled);
        free(modelX);
        return -1;
    }
    daily->n = raw->n;
    for (i = 0; i < raw->n; i++) {
        int valid = 1;
        int bestCluster = -1;
        double bestDist = INFINITY;

        daily->closeMs[i] = raw->closeTime[i];
        daily->close[i] = raw->close[i];
        daily->cluster[i] = -1;
        for (j = 0; j < model->featureCount; j++) {
            feature[j] = clusterFeatureValue(
                raw,
                model,
                emaFast,
                emaMid,
                emaSlow,
                trend,
                logRet,
                ret1,
                bodyAbsPct,
                rangePct,
                logVolume,
                i,
                model->featureIds[j]
            );
            if (!isfinite(feature[j]) || fabs(model->scale[j]) <= 1e-12) {
                valid = 0;
                break;
            }
            scaled[j] = (feature[j] - model->center[j]) / model->scale[j];
        }
        if (!valid) {
            continue;
        }
        for (k = 0; k < model->pcaCount; k++) {
            double sum = 0.0;

            for (j = 0; j < model->featureCount; j++) {
                sum += (
                    (scaled[j] - model->pcaMean[j])
                    * model->pcaComponents[
                        (k * model->featureCount) + j
                    ]
                );
            }
            modelX[k] = sum;
        }
        for (k = 0; k < model->clusterCount; k++) {
            double dist = 0.0;

            for (j = 0; j < model->pcaCount; j++) {
                double d = modelX[j]
                    - model->centroids[(k * model->pcaCount) + j];

                dist += d * d;
            }
            if (dist < bestDist) {
                bestDist = dist;
                bestCluster = k;
            }
        }
        if (
            model->clusterRemap != NULL
            && bestCluster >= 0
            && bestCluster < model->remapCount
        ) {
            bestCluster = model->clusterRemap[bestCluster];
        }
        daily->cluster[i] = bestCluster;
    }

    free(emaFast);
    free(emaMid);
    free(emaSlow);
    free(logRet);
    free(ret1);
    free(bodyAbsPct);
    free(rangePct);
    free(logVolume);
    free(trend);
    free(feature);
    free(scaled);
    free(modelX);
    return 0;
}

static void computeDailyPosture(DailyPostureRaw* daily) {
    int i;
    int j;

    for (i = 0; i < daily->n; i++) {
        double high = daily->close[i];

        daily->ret30[i] = (
            i >= 30 && daily->close[i - 30] > 0.0
            ? ((daily->close[i] / daily->close[i - 30]) - 1.0) * 100.0
            : 0.0
        );
        for (j = i - 59; j <= i; j++) {
            if (j >= 0 && daily->close[j] > high) {
                high = daily->close[j];
            }
        }
        daily->nearHigh[i] = (
            daily->close[i] > 0.0
            ? ((high / daily->close[i]) - 1.0) * 100.0
            : 0.0
        );
    }
}

static int alignDailyPosture(
    const DailyPostureRaw* daily,
    const KlineSoa* micro,
    DailyPostureAligned* out
) {
    int i;
    int pos = 0;

    memset(out, 0, sizeof(*out));
    if (daily == NULL || daily->n <= 0) {
        return 0;
    }
    out->cluster = (int*)malloc((size_t)micro->n * sizeof(int));
    out->ret30 = (double*)malloc((size_t)micro->n * sizeof(double));
    out->nearHigh = (double*)malloc((size_t)micro->n * sizeof(double));
    if (out->cluster == NULL || out->ret30 == NULL || out->nearHigh == NULL) {
        free(out->cluster);
        free(out->ret30);
        free(out->nearHigh);
        return -1;
    }
    for (i = 0; i < micro->n; i++) {
        while (
            pos + 1 < daily->n
            && daily->closeMs[pos + 1] <= micro->openTime[i]
        ) {
            pos += 1;
        }
        if (daily->closeMs[pos] <= micro->openTime[i]) {
            out->cluster[i] = daily->cluster[pos];
            out->ret30[i] = daily->ret30[pos];
            out->nearHigh[i] = daily->nearHigh[pos];
        } else {
            out->cluster[i] = -1;
            out->ret30[i] = 0.0;
            out->nearHigh[i] = 0.0;
        }
    }
    return 0;
}

static void freeDailyPostureRaw(DailyPostureRaw* daily) {
    free(daily->closeMs);
    free(daily->close);
    free(daily->cluster);
    free(daily->ret30);
    free(daily->nearHigh);
}

static void freeDailyPostureAligned(DailyPostureAligned* aligned) {
    free(aligned->cluster);
    free(aligned->ret30);
    free(aligned->nearHigh);
}

/* Trim one raw kline series to the requested tune window in place. */
static void windowKlines(
    KlineSoa* raw,
    int64_t anchorMs,
    int totalDays,
    int holdoutDays
) {
    int64_t curMs = anchorMs > 0 ? anchorMs : nowMs();
    int64_t startMs = curMs - ((int64_t)totalDays * DAY_MS);
    int64_t endMs = curMs - ((int64_t)holdoutDays * DAY_MS);
    int start = 0;
    int end = raw->n;
    int keep;

    while (start < raw->n && raw->openTime[start] < startMs) {
        start += 1;
    }
    while (end > start && raw->openTime[end - 1] >= endMs) {
        end -= 1;
    }

    keep = end - start;
    if (keep <= 0) {
        raw->n = 0;
        return;
    }
    if (start > 0) {
        memmove(
            raw->openTime,
            raw->openTime + start,
            (size_t)keep * sizeof(int64_t)
        );
        memmove(
            raw->open,
            raw->open + start,
            (size_t)keep * sizeof(double)
        );
        memmove(
            raw->high,
            raw->high + start,
            (size_t)keep * sizeof(double)
        );
        memmove(
            raw->low,
            raw->low + start,
            (size_t)keep * sizeof(double)
        );
        memmove(
            raw->close,
            raw->close + start,
            (size_t)keep * sizeof(double)
        );
        memmove(
            raw->volume,
            raw->volume + start,
            (size_t)keep * sizeof(double)
        );
        memmove(
            raw->closeTime,
            raw->closeTime + start,
            (size_t)keep * sizeof(int64_t)
        );
        memmove(
            raw->quoteVolume,
            raw->quoteVolume + start,
            (size_t)keep * sizeof(double)
        );
        memmove(
            raw->tradeCount,
            raw->tradeCount + start,
            (size_t)keep * sizeof(int)
        );
        memmove(
            raw->takerBase,
            raw->takerBase + start,
            (size_t)keep * sizeof(double)
        );
        memmove(
            raw->takerQuote,
            raw->takerQuote + start,
            (size_t)keep * sizeof(double)
        );
    }
    raw->n = keep;
}

/* Return the lower-case cache file path for one interval. */
static void cachePath(
    const char* cacheRoot,
    const char* ticker,
    const char* interval,
    char* out,
    size_t outLen
) {
    char lower[64];
    size_t i;
    size_t n = strlen(ticker);

    for (i = 0; i < n && i + 1 < sizeof(lower); i++) {
        char ch = ticker[i];

        if (ch >= 'A' && ch <= 'Z') {
            lower[i] = (char)(ch - 'A' + 'a');
        } else {
            lower[i] = ch;
        }
    }
    lower[i] = '\0';
    snprintf(
        out,
        outLen,
        "%s/%s/%s_%s.csv",
        cacheRoot,
        ticker,
        lower,
        interval
    );
}

/* Grow one interval cache list. */
static int growCache(KlineCache* cache, int need) {
    KlineCacheEntry* grown;

    grown = (KlineCacheEntry*)realloc(
        cache->items,
        (size_t)need * sizeof(KlineCacheEntry)
    );
    if (grown == NULL) {
        return -1;
    }
    cache->items = grown;
    return 0;
}

/* Load or reuse one raw kline series by interval. */
static KlineSoa* cacheLoad(
    KlineCache* cache,
    const char* cacheRoot,
    const char* ticker,
    const char* interval,
    int64_t anchorMs,
    int totalDays,
    int holdoutDays
) {
    int i;
    char path[1024];

    for (i = 0; i < cache->count; i++) {
        if (strcmp(cache->items[i].interval, interval) == 0) {
            return &cache->items[i].raw;
        }
    }

    if (growCache(cache, cache->count + 1) != 0) {
        return NULL;
    }
    memset(&cache->items[cache->count], 0, sizeof(KlineCacheEntry));
    snprintf(
        cache->items[cache->count].interval,
        sizeof(cache->items[cache->count].interval),
        "%s",
        interval
    );
    cachePath(cacheRoot, ticker, interval, path, sizeof(path));
    if (loadKlineCsv(path, &cache->items[cache->count].raw) != 0) {
        return NULL;
    }
    windowKlines(
        &cache->items[cache->count].raw,
        anchorMs,
        totalDays,
        holdoutDays
    );
    cache->count += 1;
    return &cache->items[cache->count - 1].raw;
}

/* Release all cached raw kline arrays. */
static void freeCache(KlineCache* cache) {
    int i;

    for (i = 0; i < cache->count; i++) {
        freeKlines(&cache->items[i].raw);
    }
    free(cache->items);
    cache->items = NULL;
    cache->count = 0;
}

/* Return the rounded metric value used in CSV output. */
static double roundMetric(double value) {
    if (!isfinite(value)) {
        return value;
    }
    return round(value * 1000000.0) / 1000000.0;
}

/* Return the CSV tax label for one code. */
static const char* taxName(int code) {
    if (code == TAX_INCOME) {
        return "income";
    }
    return "cgt";
}

/* Return edge vs benchmark under the configured tax mode. */
static double edgeVsBench(const TuneParams* param, const EvalRow* row) {
    double grossEdge = row->simValue - row->benchValue;
    double netEdge = row->simPostTax - row->benchPostTax;

    if (param->taxMode == TAX_INCOME) {
        return grossEdge;
    }
    return netEdge;
}

/* Write the standard tuner CSV header. */
static int writeHeader(FILE* fp) {
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
            "MACRO_GRAD_WIN_DAYS,"
            "MACRO_GRAD_Z_MIN,MACRO_GRAD_Z_MAX,"
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

/* Compare two doubles for ascending qsort order. */
static int compareDouble(const void* aPtr, const void* bPtr) {
    double a = *(const double*)aPtr;
    double b = *(const double*)bPtr;

    if (a < b) {
        return -1;
    }
    if (a > b) {
        return 1;
    }
    return 0;
}

/* Write one full tuner row. */
static int writeRow(
    FILE* fp,
    const TuneHostMeta* meta,
    const IntervalGroup* intervalGroup,
    const MacroGroup* macroGroup,
    const TuneParams* param,
    const EvalRow* row
) {
    double grossEdge = row->simValue - row->benchValue;
    double netEdge = row->simPostTax - row->benchPostTax;
    double edge = edgeVsBench(param, row);
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
            intervalGroup->interval,
            meta->totalDays,
            intervalGroup->p1,
            intervalGroup->p2,
            intervalGroup->p3,
            param->grad1BuyZMin,
            param->grad1SellZMin,
            param->grad1BuyWinDays,
            param->grad1SellWinDays,
            param->phaseBuy,
            param->phaseSell,
            param->finalPortionPct,
            param->cooldown,
            macroGroup->macroInterval,
            macroGroup->macroP1,
            macroGroup->macroGradPeriod,
            macroGroup->macroP3,
            macroGroup->macroDynWin,
            macroGroup->macroDynZMin,
            macroGroup->macroDynZMax,
            macroGroup->macroDynPctMin,
            macroGroup->macroDynPctMax,
            macroGroup->macroGradWinDays,
            macroGroup->macroGradZMin,
            macroGroup->macroGradZMax,
            macroGroup->macroGradMultMin,
            macroGroup->macroGradMultMax,
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
            roundMetric(grossEdge),
            roundMetric(netEdge),
            roundMetric(edge),
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

/* Decode one combo index into one full TuneParams row. */
static void decodeParam(
    const TuneParams* base,
    const TuneAxes* axes,
    uint64_t comboIdx,
    TuneParams* out
) {
    uint64_t axisIdx = comboIdx;

    *out = *base;
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
    out->phaseBuy = axes->phaseBuy.values[
        axisIdx % (uint64_t)axes->phaseBuy.n
    ];
    axisIdx /= (uint64_t)axes->phaseBuy.n;
    out->phaseSell = axes->phaseSell.values[
        axisIdx % (uint64_t)axes->phaseSell.n
    ];
    axisIdx /= (uint64_t)axes->phaseSell.n;
    out->finalPortionPct = axes->finalPortionPct.values[
        axisIdx % (uint64_t)axes->finalPortionPct.n
    ];
    axisIdx /= (uint64_t)axes->finalPortionPct.n;
    out->cooldown = axes->cooldown.values[
        axisIdx % (uint64_t)axes->cooldown.n
    ];
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
    out->annualIncomeBase = FIXED_INCOME_BASE;
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

/* Return the median bars-per-day estimate for one raw kline series. */
static double barsPerDay(const KlineSoa* raw) {
    double* diffs;
    double med;
    int i;
    int count;

    if (raw->n <= 1) {
        return 96.0;
    }
    diffs = (double*)malloc((size_t)(raw->n - 1) * sizeof(double));
    if (diffs == NULL) {
        return 96.0;
    }
    count = 0;
    for (i = 1; i < raw->n; i++) {
        double diff = (double)(raw->openTime[i] - raw->openTime[i - 1]);

        if (diff > 0.0) {
            diffs[count] = diff;
            count += 1;
        }
    }
    if (count == 0) {
        free(diffs);
        return 96.0;
    }
    qsort(
        diffs,
        (size_t)count,
        sizeof(double),
        compareDouble
    );
    med = diffs[count / 2];
    free(diffs);
    if (med <= 0.0) {
        return 96.0;
    }
    return (24.0 * 60.0 * 60.0 * 1000.0) / med;
}

/* Return the start index for one interval group. */
static int startIdx(
    const KlineSoa* raw,
    const IntervalGroup* group,
    int primerDays
) {
    int maxPeriod = group->p1;
    double bpd;

    if (group->p2 > maxPeriod) {
        maxPeriod = group->p2;
    }
    if (group->p3 > maxPeriod) {
        maxPeriod = group->p3;
    }
    bpd = barsPerDay(raw);
    return (maxPeriod * 2) + (int)round((double)primerDays * bpd);
}

/* Build one base TuneParams row from fixed host metadata. */
static void baseParam(
    const TuneHostMeta* meta,
    const ParsedAxes* axes,
    TuneParams* out
) {
    memset(out, 0, sizeof(*out));
    out->grad1BuyZMin = axes->grad1BuyZMinVals[0];
    out->grad1SellZMin = axes->grad1SellZMinVals[0];
    out->grad1BuyWinDays = axes->grad1BuyWinDaysVals[0];
    out->grad1SellWinDays = axes->grad1SellWinDaysVals[0];
    out->phaseBuy = axes->phaseBuyVals[0];
    out->phaseSell = axes->phaseSellVals[0];
    out->finalPortionPct = axes->finalPortionPctVals[0];
    out->cooldown = axes->cooldownVals[0];
    out->feeRate = meta->feeRate;
    out->seedQuote = meta->seedQuote;
    out->taxMode = axes->taxModeVals[0];
    out->seedAssetPct = axes->seedAssetPctVals[0];
    out->annualIncomeBase = FIXED_INCOME_BASE;
    out->dailyStrongSellMult = axes->dailyStrongSellMultVals[0];
    out->dailyStrongTargetPct = axes->dailyStrongTargetPctVals[0];
    out->dailyBridgeDays = axes->dailyBridgeDaysVals[0];
    out->dailyDownBuyMult = axes->dailyDownBuyMultVals[0];
    out->dailyCrabAssetCapPct = axes->dailyCrabAssetCapPctVals[0];
    out->dailyLockTargetPct = axes->dailyLockTargetPctVals[0];
    out->dailyLockGainPct = axes->dailyLockGainPctVals[0];
    out->dailyLockNearHighPct = axes->dailyLockNearHighPctVals[0];
    out->dailyLockMaxDays = axes->dailyLockMaxDaysVals[0];
    out->postUltraCoastTargetPct = axes->postUltraCoastTargetPctVals[0];
    out->postUltraGivebackPct = axes->postUltraGivebackPctVals[0];
    out->postUltraReaccumPct = axes->postUltraReaccumPctVals[0];
    out->postUltraDoubleTopPct = axes->postUltraDoubleTopPctVals[0];
    out->postUltraMaxDays = axes->postUltraMaxDaysVals[0];
    out->postUltraLockMinAssetPct =
        axes->postUltraLockMinAssetPctVals[0];
    out->postUltraLockMaxAssetPct =
        axes->postUltraLockMaxAssetPctVals[0];
    out->postUltraLockGivebackPct = axes->postUltraLockGivebackPctVals[0];
    out->postUltraLockReaccumPct = axes->postUltraLockReaccumPctVals[0];
    out->postUltraLockDoubleTopPct =
        axes->postUltraLockDoubleTopPctVals[0];
    out->postUltraLockMaxDays = axes->postUltraLockMaxDaysVals[0];
    out->macroSellRelaxPct = axes->macroSellRelaxPctVals[0];
    out->peakLockCapPct = axes->peakLockCapPctVals[0];
    out->peakLockUnlockGainPct = axes->peakLockUnlockGainPctVals[0];
    out->peakLockReentryStepPct = axes->peakLockReentryStepPctVals[0];
    out->peakLockArmGainPct = axes->peakLockArmGainPctVals[0];
    out->peakLockGivebackPct = axes->peakLockGivebackPctVals[0];
    out->peakLockMaxDays = axes->peakLockMaxDaysVals[0];
    out->peakLockEdgeDrawPct = axes->peakLockEdgeDrawPctVals[0];
    out->peakLockEdgeSlopeDays = axes->peakLockEdgeSlopeDaysVals[0];
    out->peakLockRequireEdgeRisk = axes->peakLockRequireEdgeRiskVals[0];
    out->peakLockMaDays = axes->peakLockMaDaysVals[0];
    out->peakLockKp = axes->peakLockKpVals[0];
    out->peakLockKi = axes->peakLockKiVals[0];
    out->peakLockKd = axes->peakLockKdVals[0];
    out->peakLockIntegralDecay = axes->peakLockIntegralDecayVals[0];
    out->peakLockEntryThreshold = axes->peakLockEntryThresholdVals[0];
    out->peakLockExitThreshold = axes->peakLockExitThresholdVals[0];
    out->peakLockConfirmBars = axes->peakLockConfirmBarsVals[0];
    out->peakLockReleaseTargetPct = axes->peakLockReleaseTargetPctVals[0];
    out->peakLockUltraGraceDays = axes->peakLockUltraGraceDaysVals[0];
}

/* Write one single-row winner CSV. */
static int writeWinnerCsv(
    const char* path,
    const TuneHostMeta* meta,
    const WinnerState* winner,
    const TuneAxes* axes,
    const TuneParams* base
) {
    FILE* fp;
    TuneParams param;
    char tmpPath[1024];

    snprintf(tmpPath, sizeof(tmpPath), "%s.tmp", path);
    fp = fopen(tmpPath, "w");
    if (fp == NULL) {
        return -1;
    }
    if (writeHeader(fp) != 0) {
        fclose(fp);
        return -1;
    }
    decodeParam(base, axes, winner->comboIdx, &param);
    if (
        writeRow(
            fp,
            meta,
            &winner->intervalGroup,
            &winner->macroGroup,
            &param,
            &winner->row
        ) != 0
    ) {
        fclose(fp);
        return -1;
    }
    fclose(fp);
    if (rename(tmpPath, path) != 0) {
        return -1;
    }
    return 0;
}

int main(int argc, char** argv) {
    TuneHostMeta meta;
    IntervalGroupList intervalGroups;
    MacroGroupList macroGroups;
    ParsedAxes axes;
    KlineCache microCache = {0};
    KlineCache macroCache = {0};
    KlineCache postureCache = {0};
    DailyPostureRaw dailyRaw = {0};
    ClusterModel clusterModel = {0};
    TuneParams base;
    WinnerState best = {0};
    WinnerState stats = {0};
    double startSecs;
    double endSecs;
    uint64_t totalCombos;
    uint64_t innerCombos;
    uint64_t completed = 0;
    int i;
    int j;
    char metaPath[1024];
    char intervalPath[1024];
    char macroPath[1024];
    char axesPath[1024];
    char resultsTmpPath[1024];

    if (argc != 2) {
        fprintf(stderr, "usage: %s <spec_dir>\n", argv[0]);
        return 1;
    }

    snprintf(metaPath, sizeof(metaPath), "%s/meta.txt", argv[1]);
    snprintf(
        intervalPath,
        sizeof(intervalPath),
        "%s/interval_groups.csv",
        argv[1]
    );
    snprintf(
        macroPath,
        sizeof(macroPath),
        "%s/macro_groups.csv",
        argv[1]
    );
    snprintf(axesPath, sizeof(axesPath), "%s/axes.txt", argv[1]);

    if (
        loadMeta(metaPath, &meta) != 0
        || loadIntervalGroups(intervalPath, &intervalGroups) != 0
        || loadMacroGroups(macroPath, &macroGroups) != 0
        || loadAxes(axesPath, &axes) != 0
    ) {
        return 1;
    }
    if (meta.dailyClusterModelPath[0] != '\0') {
        KlineSoa* postureRaw;

        if (loadClusterModel(meta.dailyClusterModelPath, &clusterModel) != 0) {
            return 1;
        }
        postureRaw = cacheLoad(
            &postureCache,
            meta.cacheRoot,
            meta.ticker,
            clusterModel.interval,
            meta.anchorMs,
            meta.totalDays,
            meta.holdoutDays
        );
        if (
            postureRaw == NULL
            || buildDailyPostureFromModel(
                postureRaw,
                &clusterModel,
                &dailyRaw
            ) != 0
        ) {
            return 1;
        }
    } else if (loadDailyPostureCsv(meta.dailyClusterPath, &dailyRaw) != 0) {
        return 1;
    }
    computeDailyPosture(&dailyRaw);
    snprintf(
        resultsTmpPath,
        sizeof(resultsTmpPath),
        "%s.tmp",
        meta.resultsCsvPath
    );

    innerCombos = tuneAxesCount(&axes.axes);
    totalCombos = (uint64_t)intervalGroups.count
        * (uint64_t)macroGroups.count
        * innerCombos;
    baseParam(&meta, &axes, &base);
    startSecs = nowSecs();

    for (i = 0; i < intervalGroups.count; i++) {
            KlineSoa* microRaw = cacheLoad(
                &microCache,
                meta.cacheRoot,
                meta.ticker,
                intervalGroups.items[i].interval,
                meta.anchorMs,
                meta.totalDays,
                meta.holdoutDays
            );
            MicroSoa micro;
            DailyPostureAligned dailyAligned;
            int begin;

            if (microRaw == NULL) {
                return 1;
            }
            if (alignDailyPosture(&dailyRaw, microRaw, &dailyAligned) != 0) {
                return 1;
            }
            begin = startIdx(
                &microRaw[0],
                &intervalGroups.items[i],
                meta.primerDays + meta.trainingDays
            );
            micro.n = microRaw->n;
            micro.p1 = intervalGroups.items[i].p1;
            micro.p2 = intervalGroups.items[i].p2;
            micro.p3 = intervalGroups.items[i].p3;
            micro.ts = microRaw->openTime;
            micro.opens = microRaw->open;
            micro.highs = microRaw->high;
            micro.lows = microRaw->low;
            micro.closes = microRaw->close;
            micro.volumes = microRaw->volume;
            micro.dailyCluster = dailyAligned.cluster;
            micro.dailyRet30 = dailyAligned.ret30;
            micro.dailyNearHigh = dailyAligned.nearHigh;

            for (j = 0; j < macroGroups.count; j++) {
                KlineSoa* macroRaw = cacheLoad(
                    &macroCache,
                    meta.cacheRoot,
                    meta.ticker,
                    macroGroups.items[j].macroInterval,
                    meta.anchorMs,
                    meta.totalDays,
                    meta.holdoutDays
                );
                MacroSoa macro;
                TuneGroupMeta groupMeta;
                TuneRunOptions options;
                TuneRunResult result;

                if (macroRaw == NULL) {
                    return 1;
                }

                macro.n = macroRaw->n;
                macro.p1 = macroGroups.items[j].macroP1;
                macro.p3 = macroGroups.items[j].macroP3;
                macro.dynWinDays = (double)macroGroups.items[j].macroDynWin;
                macro.dynZMin = macroGroups.items[j].macroDynZMin;
                macro.dynZMax = macroGroups.items[j].macroDynZMax;
                macro.dynPctMax = macroGroups.items[j].macroDynPctMax;
                macro.dynPctMin = macroGroups.items[j].macroDynPctMin;
                macro.gradWinDays =
                    (double)macroGroups.items[j].macroGradWinDays;
                macro.gradZMin = macroGroups.items[j].macroGradZMin;
                macro.gradZMax = macroGroups.items[j].macroGradZMax;
                macro.gradMultMin = macroGroups.items[j].macroGradMultMin;
                macro.gradMultMax = macroGroups.items[j].macroGradMultMax;
                macro.ts = macroRaw->closeTime;
                macro.closes = macroRaw->close;
                macro.p2 = macroGroups.items[j].macroGradPeriod;

                memset(&groupMeta, 0, sizeof(groupMeta));
                groupMeta.ticker = meta.ticker;
                groupMeta.interval = intervalGroups.items[i].interval;
                groupMeta.days = meta.totalDays;
                groupMeta.p1 = intervalGroups.items[i].p1;
                groupMeta.p2 = intervalGroups.items[i].p2;
                groupMeta.p3 = intervalGroups.items[i].p3;
                groupMeta.macroInterval = macroGroups.items[j].macroInterval;
                groupMeta.macroP1 = macroGroups.items[j].macroP1;
                groupMeta.macroP3 = macroGroups.items[j].macroP3;
                groupMeta.macroGradPeriod =
                    macroGroups.items[j].macroGradPeriod;
                groupMeta.macroDynWinDays = macroGroups.items[j].macroDynWin;
                groupMeta.macroDynZMin = macroGroups.items[j].macroDynZMin;
                groupMeta.macroDynZMax = macroGroups.items[j].macroDynZMax;
                groupMeta.macroDynPctMin =
                    macroGroups.items[j].macroDynPctMin;
                groupMeta.macroDynPctMax =
                    macroGroups.items[j].macroDynPctMax;
                groupMeta.macroGradWinDays =
                    macroGroups.items[j].macroGradWinDays;
                groupMeta.macroGradZMin =
                    macroGroups.items[j].macroGradZMin;
                groupMeta.macroGradZMax =
                    macroGroups.items[j].macroGradZMax;
                groupMeta.macroGradMultMin =
                    macroGroups.items[j].macroGradMultMin;
                groupMeta.macroGradMultMax =
                    macroGroups.items[j].macroGradMultMax;
                memset(&options, 0, sizeof(options));
                options.csvPath = resultsTmpPath;
                options.appendCsv = (completed > 0);
                options.doneCount = completed;
                options.totalCount = totalCombos;
                options.startSecs = startSecs;
                options.progressRows = 4096;
                options.progressSecs = 0.25;

                if (
                    runTuneGroupBatch(
                        &micro,
                        &macro,
                        begin,
                        &base,
                        &axes.axes,
                        &groupMeta,
                        &options,
                        &result
                    ) != 0
                ) {
                    return 1;
                }

                if (!best.has || result.bestGrossPct > best.score) {
                    best.has = 1;
                    best.row = result.bestRow;
                    best.intervalGroup = intervalGroups.items[i];
                    best.macroGroup = macroGroups.items[j];
                    best.comboIdx = result.bestComboIdx;
                    best.score = result.bestGrossPct;
                }
                if (!stats.has || result.statsScore > stats.score) {
                    stats.has = 1;
                    stats.row = result.statsRow;
                    stats.intervalGroup = intervalGroups.items[i];
                    stats.macroGroup = macroGroups.items[j];
                    stats.comboIdx = result.statsComboIdx;
                    stats.score = result.statsScore;
                }
                completed += result.evalCount;
            }
            freeDailyPostureAligned(&dailyAligned);
    }

    endSecs = nowSecs();
    fprintf(
        stderr,
        "[host] total evaluations: %llu\n",
        (unsigned long long)completed
    );
    fprintf(stderr, "[host] elapsed seconds: %.3f\n", endSecs - startSecs);
    if (!best.has) {
        return 1;
    }
    if (rename(resultsTmpPath, meta.resultsCsvPath) != 0) {
        return 1;
    }
    if (
        writeWinnerCsv(
            meta.bestRowCsvPath,
            &meta,
            &best,
            &axes.axes,
            &base
        ) != 0
    ) {
        return 1;
    }
    if (
        writeWinnerCsv(
            meta.statsRowCsvPath,
            &meta,
            stats.has ? &stats : &best,
            &axes.axes,
            &base
        ) != 0
    ) {
        return 1;
    }

    freeCache(&microCache);
    freeCache(&macroCache);
    freeCache(&postureCache);
    freeIntervalGroups(&intervalGroups);
    freeMacroGroups(&macroGroups);
    freeAxes(&axes);
    freeDailyPostureRaw(&dailyRaw);
    freeClusterModel(&clusterModel);
    return 0;
}
