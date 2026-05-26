#include <ctype.h>
#include <netdb.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <unistd.h>
#include "httpTls.h"

/* Release one owned byte buffer. */
void freeBuf(ByteBuf* buf) {
    if (buf == NULL) {
        return;
    }
    free(buf->data);
    buf->data = NULL;
    buf->size = 0;
}

/* Grow one output buffer by the requested byte count. */
static int growBuf(ByteBuf* buf, size_t addBytes) {
    char* grown;
    size_t need;

    need = buf->size + addBytes + 1;
    grown = (char*)realloc(buf->data, need);
    if (grown == NULL) {
        return -1;
    }
    buf->data = grown;
    return 0;
}

/* Append one raw byte span to the destination buffer. */
static int appendBuf(ByteBuf* buf, const char* src, size_t srcLen) {
    if (growBuf(buf, srcLen) != 0) {
        return -1;
    }
    memcpy(buf->data + buf->size, src, srcLen);
    buf->size += srcLen;
    buf->data[buf->size] = '\0';
    return 0;
}

/* Open one TCP socket using Beej-style getaddrinfo iteration. */
static int openSock(const char* host, const char* port) {
    struct addrinfo hints;
    struct addrinfo* info = NULL;
    struct addrinfo* cur;
    int fd = -1;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if (getaddrinfo(host, port, &hints, &info) != 0) {
        return -1;
    }

    for (cur = info; cur != NULL; cur = cur->ai_next) {
        fd = socket(cur->ai_family, cur->ai_socktype, cur->ai_protocol);
        if (fd < 0) {
            continue;
        }
        if (connect(fd, cur->ai_addr, cur->ai_addrlen) == 0) {
            break;
        }
        close(fd);
        fd = -1;
    }

    freeaddrinfo(info);
    return fd;
}

/* Find one header block terminator. */
static const char* headerEnd(const char* data, size_t size) {
    size_t i;

    if (size < 4) {
        return NULL;
    }
    for (i = 0; i + 3 < size; i++) {
        if (
            data[i] == '\r'
            && data[i + 1] == '\n'
            && data[i + 2] == '\r'
            && data[i + 3] == '\n'
        ) {
            return data + i;
        }
    }
    return NULL;
}

/* Return non-zero if one header exists in the raw header block. */
static int hasHeader(
    const char* headers,
    size_t size,
    const char* needle
) {
    size_t needLen = strlen(needle);
    size_t i;

    if (needLen == 0 || size < needLen) {
        return 0;
    }
    for (i = 0; i + needLen <= size; i++) {
        if (strncasecmp(headers + i, needle, needLen) == 0) {
            return 1;
        }
    }
    return 0;
}

/* Decode one HTTP chunked body into a flat output buffer. */
static int decodeChunked(
    const char* src,
    size_t size,
    ByteBuf* out
) {
    const char* cur = src;
    const char* end = src + size;

    while (cur < end) {
        const char* lineEnd;
        char sizeBuf[32];
        size_t lineLen;
        unsigned long chunkLen;

        lineEnd = strstr(cur, "\r\n");
        if (lineEnd == NULL || lineEnd > end) {
            return -1;
        }
        lineLen = (size_t)(lineEnd - cur);
        if (lineLen == 0 || lineLen >= sizeof(sizeBuf)) {
            return -1;
        }
        memcpy(sizeBuf, cur, lineLen);
        sizeBuf[lineLen] = '\0';
        chunkLen = strtoul(sizeBuf, NULL, 16);
        cur = lineEnd + 2;
        if (chunkLen == 0) {
            return 0;
        }
        if ((size_t)(end - cur) < chunkLen + 2) {
            return -1;
        }
        if (appendBuf(out, cur, (size_t)chunkLen) != 0) {
            return -1;
        }
        cur += chunkLen;
        if (cur[0] != '\r' || cur[1] != '\n') {
            return -1;
        }
        cur += 2;
    }

    return 0;
}

/* Read one TLS socket fully into memory. */
static int readTls(SSL* ssl, ByteBuf* out) {
    char chunk[8192];

    for (;;) {
        int got = SSL_read(ssl, chunk, (int)sizeof(chunk));

        if (got > 0) {
            if (appendBuf(out, chunk, (size_t)got) != 0) {
                return -1;
            }
            continue;
        }
        if (got == 0) {
            return 0;
        }
        if (SSL_get_error(ssl, got) != SSL_ERROR_ZERO_RETURN) {
            return -1;
        }
        return 0;
    }
}

/* Build one TLS-backed HTTP GET request body. */
int httpGet(
    const char* host,
    const char* port,
    const char* path,
    ByteBuf* out
) {
    SSL_CTX* ctx = NULL;
    SSL* ssl = NULL;
    ByteBuf raw = {0};
    const char* hdrEnd;
    const char* body;
    size_t hdrSize;
    size_t bodySize;
    int fd = -1;
    int rc = -1;
    char req[2048];

    OPENSSL_init_ssl(0, NULL);
    fd = openSock(host, port);
    if (fd < 0) {
        return -1;
    }

    ctx = SSL_CTX_new(TLS_client_method());
    if (ctx == NULL) {
        goto cleanup;
    }
    SSL_CTX_set_default_verify_paths(ctx);
    SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, NULL);

    ssl = SSL_new(ctx);
    if (ssl == NULL) {
        goto cleanup;
    }
    SSL_set_fd(ssl, fd);
    SSL_set_tlsext_host_name(ssl, host);
    SSL_set1_host(ssl, host);

    if (SSL_connect(ssl) != 1) {
        goto cleanup;
    }
    if (SSL_get_verify_result(ssl) != X509_V_OK) {
        goto cleanup;
    }

    snprintf(
        req,
        sizeof(req),
        "GET %s HTTP/1.1\r\n"
        "Host: %s\r\n"
        "User-Agent: gradbot-c-host/1\r\n"
        "Accept: application/json\r\n"
        "Accept-Encoding: identity\r\n"
        "Connection: close\r\n\r\n",
        path,
        host
    );
    if (SSL_write(ssl, req, (int)strlen(req)) <= 0) {
        goto cleanup;
    }
    if (readTls(ssl, &raw) != 0) {
        goto cleanup;
    }

    hdrEnd = headerEnd(raw.data, raw.size);
    if (hdrEnd == NULL) {
        goto cleanup;
    }
    hdrSize = (size_t)(hdrEnd - raw.data);
    if (raw.size < 12 || strncmp(raw.data, "HTTP/1.1 200", 12) != 0) {
        goto cleanup;
    }
    body = hdrEnd + 4;
    bodySize = raw.size - (size_t)(body - raw.data);

    freeBuf(out);
    if (hasHeader(raw.data, hdrSize, "Transfer-Encoding: chunked")) {
        if (decodeChunked(body, bodySize, out) != 0) {
            goto cleanup;
        }
    } else if (appendBuf(out, body, bodySize) != 0) {
        goto cleanup;
    }

    rc = 0;

cleanup:
    freeBuf(&raw);
    if (ssl != NULL) {
        SSL_shutdown(ssl);
        SSL_free(ssl);
    }
    if (ctx != NULL) {
        SSL_CTX_free(ctx);
    }
    if (fd >= 0) {
        close(fd);
    }
    if (rc != 0) {
        freeBuf(out);
    }
    ERR_clear_error();
    return rc;
}
