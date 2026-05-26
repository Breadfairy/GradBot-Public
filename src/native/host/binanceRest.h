#ifndef BINANCE_REST_H
#define BINANCE_REST_H

#include <stdint.h>

typedef struct{
    int n;
    int cap;
    int64_t* openTime;
    double* open;
    double* high;
    double* low;
    double* close;
    double* volume;
    int64_t* closeTime;
    double* quoteVolume;
    int* tradeCount;
    double* takerBase;
    double* takerQuote;
} KlineSoa;

const char* normInterval(const char* interval);

int fetchKlines(
    const char* symbol,
    const char* interval,
    int days,
    KlineSoa* out
);

int writeKlines(
    const char* path,
    const KlineSoa* klines
);

void freeKlines(KlineSoa* klines);

#endif
