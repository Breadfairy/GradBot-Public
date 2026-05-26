#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "binanceRest.h"

/* Create one directory tree segment if missing. */
static int ensureDir(const char* path) {
    char tmp[1024];
    char* cur;

    snprintf(tmp, sizeof(tmp), "%s", path);
    for (cur = tmp + 1; *cur != '\0'; cur++) {
        if (*cur != '/') {
            continue;
        }
        *cur = '\0';
        mkdir(tmp, 0755);
        *cur = '/';
    }
    if (mkdir(tmp, 0755) != 0) {
        /* Best-effort: existing directories are fine. */
    }
    return 0;
}

/* Build the standard cache path for one ticker and interval. */
static void cachePath(
    const char* root,
    const char* symbol,
    const char* interval,
    char* out,
    size_t outLen
) {
    char lower[64];
    size_t i;
    size_t n = strlen(symbol);

    for (i = 0; i < n && i + 1 < sizeof(lower); i++) {
        lower[i] = (char)tolower((unsigned char)symbol[i]);
    }
    lower[i] = '\0';
    snprintf(
        out,
        outLen,
        "%s/%s/%s_%s.csv",
        root,
        symbol,
        lower,
        interval
    );
}

int main(int argc, char** argv) {
    KlineSoa klines = {0};
    const char* symbol;
    const char* interval;
    const char* root;
    int days;
    char dirPath[1024];
    char filePath[1024];

    if (argc != 5) {
        fprintf(
            stderr,
            "usage: %s <SYMBOL> <INTERVAL> <DAYS> <CACHE_ROOT>\n",
            argv[0]
        );
        return 1;
    }

    symbol = argv[1];
    interval = normInterval(argv[2]);
    days = atoi(argv[3]);
    root = argv[4];
    if (interval == NULL || days <= 0) {
        return 1;
    }

    snprintf(dirPath, sizeof(dirPath), "%s/%s", root, symbol);
    ensureDir(dirPath);
    cachePath(root, symbol, interval, filePath, sizeof(filePath));

    if (fetchKlines(symbol, interval, days, &klines) != 0) {
        freeKlines(&klines);
        return 1;
    }
    if (writeKlines(filePath, &klines) != 0) {
        freeKlines(&klines);
        return 1;
    }

    printf(
        "[host] wrote %d rows to %s\n",
        klines.n,
        filePath
    );
    freeKlines(&klines);
    return 0;
}
