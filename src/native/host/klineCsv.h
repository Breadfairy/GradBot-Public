#ifndef KLINE_CSV_H
#define KLINE_CSV_H

#include "binanceRest.h"

int loadKlineCsv(const char* path, KlineSoa* out);

#endif
