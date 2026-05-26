#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "klineCsv.h"

/* Grow one kline SoA to fit at least need rows. */
static int growKlines(KlineSoa* out, int need) {
    int cap = out->cap <= 0 ? 1024 : out->cap;
    int64_t* openTimePtr;
    double* openPtr;
    double* highPtr;
    double* lowPtr;
    double* closePtr;
    double* volumePtr;
    int64_t* closeTimePtr;
    double* quoteVolumePtr;
    int* tradeCountPtr;
    double* takerBasePtr;
    double* takerQuotePtr;

    while (cap < need) {
        cap *= 2;
    }

    openTimePtr = (int64_t*)realloc(
        out->openTime,
        (size_t)cap * sizeof(int64_t)
    );
    if (openTimePtr == NULL) {
        return -1;
    }
    out->openTime = openTimePtr;

    openPtr = (double*)realloc(out->open, (size_t)cap * sizeof(double));
    if (openPtr == NULL) {
        return -1;
    }
    out->open = openPtr;

    highPtr = (double*)realloc(out->high, (size_t)cap * sizeof(double));
    if (highPtr == NULL) {
        return -1;
    }
    out->high = highPtr;

    lowPtr = (double*)realloc(out->low, (size_t)cap * sizeof(double));
    if (lowPtr == NULL) {
        return -1;
    }
    out->low = lowPtr;

    closePtr = (double*)realloc(out->close, (size_t)cap * sizeof(double));
    if (closePtr == NULL) {
        return -1;
    }
    out->close = closePtr;

    volumePtr = (double*)realloc(
        out->volume,
        (size_t)cap * sizeof(double)
    );
    if (volumePtr == NULL) {
        return -1;
    }
    out->volume = volumePtr;

    closeTimePtr = (int64_t*)realloc(
        out->closeTime,
        (size_t)cap * sizeof(int64_t)
    );
    if (closeTimePtr == NULL) {
        return -1;
    }
    out->closeTime = closeTimePtr;

    quoteVolumePtr = (double*)realloc(
        out->quoteVolume,
        (size_t)cap * sizeof(double)
    );
    if (quoteVolumePtr == NULL) {
        return -1;
    }
    out->quoteVolume = quoteVolumePtr;

    tradeCountPtr = (int*)realloc(
        out->tradeCount,
        (size_t)cap * sizeof(int)
    );
    if (tradeCountPtr == NULL) {
        return -1;
    }
    out->tradeCount = tradeCountPtr;

    takerBasePtr = (double*)realloc(
        out->takerBase,
        (size_t)cap * sizeof(double)
    );
    if (takerBasePtr == NULL) {
        return -1;
    }
    out->takerBase = takerBasePtr;

    takerQuotePtr = (double*)realloc(
        out->takerQuote,
        (size_t)cap * sizeof(double)
    );
    if (takerQuotePtr == NULL) {
        return -1;
    }
    out->takerQuote = takerQuotePtr;

    out->cap = cap;
    return 0;
}

/* Append one parsed CSV row. */
static int pushKline(
    KlineSoa* out,
    int64_t openTime,
    double open,
    double high,
    double low,
    double close,
    double volume,
    int64_t closeTime,
    double quoteVolume,
    int tradeCount,
    double takerBase,
    double takerQuote
) {
    int idx = out->n;

    if (growKlines(out, idx + 1) != 0) {
        return -1;
    }
    out->openTime[idx] = openTime;
    out->open[idx] = open;
    out->high[idx] = high;
    out->low[idx] = low;
    out->close[idx] = close;
    out->volume[idx] = volume;
    out->closeTime[idx] = closeTime;
    out->quoteVolume[idx] = quoteVolume;
    out->tradeCount[idx] = tradeCount;
    out->takerBase[idx] = takerBase;
    out->takerQuote[idx] = takerQuote;
    out->n += 1;
    return 0;
}

/* Load one cached Binance kline CSV into SoA columns. */
int loadKlineCsv(const char* path, KlineSoa* out) {
    FILE* fp;
    char line[1024];

    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }

    while (fgets(line, sizeof(line), fp) != NULL) {
        long long openTime;
        double open;
        double high;
        double low;
        double close;
        double volume;
        long long closeTime;
        double quoteVolume;
        int tradeCount;
        double takerBase;
        double takerQuote;
        int parsed;

        parsed = sscanf(
            line,
            "%lld,%lf,%lf,%lf,%lf,%lf,%lld,%lf,%d,%lf,%lf",
            &openTime,
            &open,
            &high,
            &low,
            &close,
            &volume,
            &closeTime,
            &quoteVolume,
            &tradeCount,
            &takerBase,
            &takerQuote
        );
        if (parsed != 11) {
            fclose(fp);
            return -1;
        }
        if (
            pushKline(
                out,
                (int64_t)openTime,
                open,
                high,
                low,
                close,
                volume,
                (int64_t)closeTime,
                quoteVolume,
                tradeCount,
                takerBase,
                takerQuote
            ) != 0
        ) {
            fclose(fp);
            return -1;
        }
    }

    fclose(fp);
    return 0;
}

/* Free one owned kline SoA. */
void freeKlines(KlineSoa* klines) {
    if (klines == NULL) {
        return;
    }
    free(klines->openTime);
    free(klines->open);
    free(klines->high);
    free(klines->low);
    free(klines->close);
    free(klines->volume);
    free(klines->closeTime);
    free(klines->quoteVolume);
    free(klines->tradeCount);
    free(klines->takerBase);
    free(klines->takerQuote);
    memset(klines, 0, sizeof(*klines));
}
