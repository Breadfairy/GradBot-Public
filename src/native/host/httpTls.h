#ifndef HTTP_TLS_H
#define HTTP_TLS_H

#include <stddef.h>

typedef struct{
    char* data;
    size_t size;
} ByteBuf;

int httpGet(
    const char* host,
    const char* port,
    const char* path,
    ByteBuf* out
);

void freeBuf(ByteBuf* buf);

#endif
