#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include "binanceRest.h"
#include "httpTls.h"

#define BINANCE_HOST "data-api.binance.vision"
#define BINANCE_PORT "443"
#define KLINE_LIMIT 1000

/* Return current UTC milliseconds. */
static int64_t nowMs(void) {
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return (
        ((int64_t)tv.tv_sec * 1000)
        + ((int64_t)tv.tv_usec / 1000)
    );
}

/* Map one canonical interval onto milliseconds. */
static int64_t intervalMs(const char* interval) {
    if (strcmp(interval, "1m") == 0) {
        return 60LL * 1000LL;
    }
    if (strcmp(interval, "5m") == 0) {
        return 5LL * 60LL * 1000LL;
    }
    if (strcmp(interval, "15m") == 0) {
        return 15LL * 60LL * 1000LL;
    }
    if (strcmp(interval, "30m") == 0) {
        return 30LL * 60LL * 1000LL;
    }
    if (strcmp(interval, "1h") == 0) {
        return 60LL * 60LL * 1000LL;
    }
    if (strcmp(interval, "2h") == 0) {
        return 2LL * 60LL * 60LL * 1000LL;
    }
    if (strcmp(interval, "4h") == 0) {
        return 4LL * 60LL * 60LL * 1000LL;
    }
    if (strcmp(interval, "6h") == 0) {
        return 6LL * 60LL * 60LL * 1000LL;
    }
    if (strcmp(interval, "8h") == 0) {
        return 8LL * 60LL * 60LL * 1000LL;
    }
    if (strcmp(interval, "12h") == 0) {
        return 12LL * 60LL * 60LL * 1000LL;
    }
    if (strcmp(interval, "1d") == 0) {
        return 24LL * 60LL * 60LL * 1000LL;
    }
    return 0;
}

/* Normalize one user interval string onto a Binance interval token. */
const char* normInterval(const char* interval) {
    if (strcmp(interval, "1m") == 0) {
        return "1m";
    }
    if (strcmp(interval, "1min") == 0) {
        return "1m";
    }
    if (strcmp(interval, "1minute") == 0) {
        return "1m";
    }
    if (strcmp(interval, "5m") == 0) {
        return "5m";
    }
    if (strcmp(interval, "5min") == 0) {
        return "5m";
    }
    if (strcmp(interval, "15m") == 0) {
        return "15m";
    }
    if (strcmp(interval, "15min") == 0) {
        return "15m";
    }
    if (strcmp(interval, "30m") == 0) {
        return "30m";
    }
    if (strcmp(interval, "30min") == 0) {
        return "30m";
    }
    if (strcmp(interval, "1h") == 0) {
        return "1h";
    }
    if (strcmp(interval, "1hr") == 0) {
        return "1h";
    }
    if (strcmp(interval, "1hour") == 0) {
        return "1h";
    }
    if (strcmp(interval, "60m") == 0) {
        return "1h";
    }
    if (strcmp(interval, "2h") == 0) {
        return "2h";
    }
    if (strcmp(interval, "2hr") == 0) {
        return "2h";
    }
    if (strcmp(interval, "4h") == 0) {
        return "4h";
    }
    if (strcmp(interval, "4hr") == 0) {
        return "4h";
    }
    if (strcmp(interval, "6h") == 0) {
        return "6h";
    }
    if (strcmp(interval, "6hr") == 0) {
        return "6h";
    }
    if (strcmp(interval, "8h") == 0) {
        return "8h";
    }
    if (strcmp(interval, "8hr") == 0) {
        return "8h";
    }
    if (strcmp(interval, "12h") == 0) {
        return "12h";
    }
    if (strcmp(interval, "12hr") == 0) {
        return "12h";
    }
    if (strcmp(interval, "1d") == 0) {
        return "1d";
    }
    if (strcmp(interval, "1day") == 0) {
        return "1d";
    }
    return NULL;
}

/* Skip any JSON whitespace. */
static const char* skipWs(const char* cur) {
    while (*cur != '\0' && isspace((unsigned char)*cur)) {
        cur += 1;
    }
    return cur;
}

/* Consume one expected delimiter. */
static int eatChar(const char** cur, char want) {
    *cur = skipWs(*cur);
    if (**cur != want) {
        return -1;
    }
    *cur += 1;
    return 0;
}

/* Parse one quoted or plain integer token. */
static int takeInt64(const char** cur, int64_t* out) {
    char* endPtr;
    const char* start;

    *cur = skipWs(*cur);
    start = *cur;
    if (*start == '"') {
        start += 1;
    }
    *out = strtoll(start, &endPtr, 10);
    if (**cur == '"') {
        if (*endPtr != '"') {
            return -1;
        }
        endPtr += 1;
    }
    *cur = endPtr;
    return 0;
}

/* Parse one quoted or plain float token. */
static int takeDouble(const char** cur, double* out) {
    char* endPtr;
    const char* start;

    *cur = skipWs(*cur);
    start = *cur;
    if (*start == '"') {
        start += 1;
    }
    *out = strtod(start, &endPtr);
    if (**cur == '"') {
        if (*endPtr != '"') {
            return -1;
        }
        endPtr += 1;
    }
    *cur = endPtr;
    return 0;
}

/* Skip one quoted or plain scalar token. */
static int skipScalar(const char** cur) {
    *cur = skipWs(*cur);
    if (**cur == '"') {
        *cur += 1;
        while (**cur != '\0' && **cur != '"') {
            *cur += 1;
        }
        if (**cur != '"') {
            return -1;
        }
        *cur += 1;
        return 0;
    }
    while (**cur != '\0' && **cur != ',' && **cur != ']') {
        *cur += 1;
    }
    return 0;
}

/* Grow all SoA columns together. */
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

    volumePtr = (double*)realloc(out->volume, (size_t)cap * sizeof(double));
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

/* Append one parsed kline row into the SoA storage. */
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

/* Parse one Binance klines JSON body into struct-of-arrays columns. */
static int parseKlines(const char* body, KlineSoa* out) {
    const char* cur = skipWs(body);

    if (eatChar(&cur, '[') != 0) {
        return -1;
    }
    cur = skipWs(cur);
    while (*cur != '\0' && *cur != ']') {
        int64_t openTime;
        int64_t closeTime;
        double open;
        double high;
        double low;
        double close;
        double volume;
        double quoteVolume;
        double takerBase;
        double takerQuote;
        int64_t tradeCount;

        if (eatChar(&cur, '[') != 0) {
            return -1;
        }
        if (takeInt64(&cur, &openTime) != 0 || eatChar(&cur, ',') != 0) {
            return -1;
        }
        if (takeDouble(&cur, &open) != 0 || eatChar(&cur, ',') != 0) {
            return -1;
        }
        if (takeDouble(&cur, &high) != 0 || eatChar(&cur, ',') != 0) {
            return -1;
        }
        if (takeDouble(&cur, &low) != 0 || eatChar(&cur, ',') != 0) {
            return -1;
        }
        if (takeDouble(&cur, &close) != 0 || eatChar(&cur, ',') != 0) {
            return -1;
        }
        if (takeDouble(&cur, &volume) != 0 || eatChar(&cur, ',') != 0) {
            return -1;
        }
        if (takeInt64(&cur, &closeTime) != 0 || eatChar(&cur, ',') != 0) {
            return -1;
        }
        if (takeDouble(&cur, &quoteVolume) != 0 || eatChar(&cur, ',') != 0) {
            return -1;
        }
        if (takeInt64(&cur, &tradeCount) != 0 || eatChar(&cur, ',') != 0) {
            return -1;
        }
        if (takeDouble(&cur, &takerBase) != 0 || eatChar(&cur, ',') != 0) {
            return -1;
        }
        if (takeDouble(&cur, &takerQuote) != 0 || eatChar(&cur, ',') != 0) {
            return -1;
        }
        if (skipScalar(&cur) != 0 || eatChar(&cur, ']') != 0) {
            return -1;
        }
        if (
            pushKline(
                out,
                openTime,
                open,
                high,
                low,
                close,
                volume,
                closeTime,
                quoteVolume,
                (int)tradeCount,
                takerBase,
                takerQuote
            ) != 0
        ) {
            return -1;
        }
        cur = skipWs(cur);
        if (*cur == ',') {
            cur += 1;
            cur = skipWs(cur);
        }
    }
    return eatChar(&cur, ']');
}

/* Append one fetched page while avoiding duplicate open times. */
static int appendPage(KlineSoa* out, const KlineSoa* page) {
    int i;

    for (i = 0; i < page->n; i++) {
        if (out->n > 0 && out->openTime[out->n - 1] == page->openTime[i]) {
            continue;
        }
        if (
            pushKline(
                out,
                page->openTime[i],
                page->open[i],
                page->high[i],
                page->low[i],
                page->close[i],
                page->volume[i],
                page->closeTime[i],
                page->quoteVolume[i],
                page->tradeCount[i],
                page->takerBase[i],
                page->takerQuote[i]
            ) != 0
        ) {
            return -1;
        }
    }
    return 0;
}

/* Fetch one exact klines page from Binance over HTTPS. */
static int fetchPage(
    const char* symbol,
    const char* interval,
    int64_t startMs,
    int64_t endMs,
    KlineSoa* out
) {
    ByteBuf body = {0};
    char path[512];
    int rc;

    snprintf(
        path,
        sizeof(path),
        "/api/v3/klines?symbol=%s&interval=%s&startTime=%lld"
        "&endTime=%lld&limit=%d",
        symbol,
        interval,
        (long long)startMs,
        (long long)endMs,
        KLINE_LIMIT
    );
    rc = httpGet(BINANCE_HOST, BINANCE_PORT, path, &body);
    if (rc != 0) {
        return -1;
    }
    rc = parseKlines(body.data, out);
    freeBuf(&body);
    return rc;
}

/* Fetch a rolling window of Binance klines directly into SoA columns. */
int fetchKlines(
    const char* symbol,
    const char* interval,
    int days,
    KlineSoa* out
) {
    const char* norm = normInterval(interval);
    int64_t barMs;
    int64_t endMs = nowMs();
    int64_t startMs;
    int64_t nextStart;

    if (norm == NULL || days <= 0) {
        return -1;
    }
    barMs = intervalMs(norm);
    if (barMs <= 0) {
        return -1;
    }
    startMs = endMs - ((int64_t)days * 24LL * 60LL * 60LL * 1000LL);
    nextStart = startMs;

    while (nextStart <= endMs) {
        KlineSoa page = {0};

        if (fetchPage(symbol, norm, nextStart, endMs, &page) != 0) {
            freeKlines(&page);
            return -1;
        }
        if (page.n == 0) {
            freeKlines(&page);
            return 0;
        }
        if (appendPage(out, &page) != 0) {
            freeKlines(&page);
            return -1;
        }
        nextStart = page.openTime[page.n - 1] + barMs;
        if (page.n < KLINE_LIMIT) {
            freeKlines(&page);
            return 0;
        }
        freeKlines(&page);
        usleep(100000);
    }

    return 0;
}

/* Write one kline SoA to the on-disk cache CSV format. */
int writeKlines(
    const char* path,
    const KlineSoa* klines
) {
    FILE* fp;
    char tmpPath[1024];
    int i;

    snprintf(tmpPath, sizeof(tmpPath), "%s.tmp", path);
    fp = fopen(tmpPath, "w");
    if (fp == NULL) {
        return -1;
    }
    for (i = 0; i < klines->n; i++) {
        if (
            fprintf(
                fp,
                "%lld,%.8f,%.8f,%.8f,%.8f,%.8f,%lld,%.8f,%d,%.8f,"
                "%.8f,0\n",
                (long long)klines->openTime[i],
                klines->open[i],
                klines->high[i],
                klines->low[i],
                klines->close[i],
                klines->volume[i],
                (long long)klines->closeTime[i],
                klines->quoteVolume[i],
                klines->tradeCount[i],
                klines->takerBase[i],
                klines->takerQuote[i]
            ) < 0
        ) {
            fclose(fp);
            unlink(tmpPath);
            return -1;
        }
    }
    fclose(fp);
    if (rename(tmpPath, path) != 0) {
        unlink(tmpPath);
        return -1;
    }
    return 0;
}
