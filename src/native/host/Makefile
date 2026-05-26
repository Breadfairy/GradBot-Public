CC = cc
CFLAGS = -O3 -Wall -Wextra -pedantic -std=c99
ROOT = ../../..
BUILD = $(ROOT)/build/native
OBJDIR = $(BUILD)/obj/host
ENGINE_DIR = ../engine
OBJ = \
	$(OBJDIR)/binanceCli.o \
	$(OBJDIR)/binanceRest.o \
	$(OBJDIR)/httpTls.o \
	$(OBJDIR)/klineCsv.o
BIN = $(BUILD)/gradbot_binance
TUNE_OBJ = \
	$(OBJDIR)/tuneHost.o \
	$(OBJDIR)/tuneSpec.o \
	$(OBJDIR)/klineCsv.o \
	$(OBJDIR)/engine_batch.o \
	$(OBJDIR)/engine_core.o
TUNE_BIN = $(BUILD)/gradbot_tune

OPENSSL_PREFIX := $(shell brew --prefix openssl@3 2>/dev/null)

ifeq ($(OPENSSL_PREFIX),)
SSL_CFLAGS =
SSL_LDFLAGS = -lssl -lcrypto
else
SSL_CFLAGS = -I$(OPENSSL_PREFIX)/include
SSL_LDFLAGS = -L$(OPENSSL_PREFIX)/lib -lssl -lcrypto
endif

all: $(BIN) $(TUNE_BIN)

gradbot_binance: $(BIN)

gradbot_tune: $(TUNE_BIN)

$(BIN): $(OBJ)
	$(CC) $(OBJ) $(SSL_LDFLAGS) -o $(BIN)

$(TUNE_BIN): $(TUNE_OBJ)
	$(CC) $(TUNE_OBJ) -lm -o $(TUNE_BIN)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BUILD):
	mkdir -p $(BUILD)

$(OBJDIR)/binanceCli.o: binanceCli.c binanceRest.h | $(OBJDIR) $(BUILD)
	$(CC) $(CFLAGS) $(SSL_CFLAGS) -c binanceCli.c -o $(OBJDIR)/binanceCli.o

$(OBJDIR)/binanceRest.o: binanceRest.c binanceRest.h httpTls.h | $(OBJDIR) $(BUILD)
	$(CC) $(CFLAGS) $(SSL_CFLAGS) -c binanceRest.c -o $(OBJDIR)/binanceRest.o

$(OBJDIR)/httpTls.o: httpTls.c httpTls.h | $(OBJDIR) $(BUILD)
	$(CC) $(CFLAGS) $(SSL_CFLAGS) -c httpTls.c -o $(OBJDIR)/httpTls.o

$(OBJDIR)/tuneHost.o: tuneHost.c tuneSpec.h klineCsv.h $(ENGINE_DIR)/engine.h | $(OBJDIR) $(BUILD)
	$(CC) $(CFLAGS) -I$(ENGINE_DIR) -c tuneHost.c -o $(OBJDIR)/tuneHost.o

$(OBJDIR)/tuneSpec.o: tuneSpec.c tuneSpec.h $(ENGINE_DIR)/engine.h | $(OBJDIR) $(BUILD)
	$(CC) $(CFLAGS) -I$(ENGINE_DIR) -c tuneSpec.c -o $(OBJDIR)/tuneSpec.o

$(OBJDIR)/klineCsv.o: klineCsv.c klineCsv.h binanceRest.h | $(OBJDIR) $(BUILD)
	$(CC) $(CFLAGS) -c klineCsv.c -o $(OBJDIR)/klineCsv.o

$(OBJDIR)/engine_batch.o: $(ENGINE_DIR)/batch.c $(ENGINE_DIR)/engine.h | $(OBJDIR) $(BUILD)
	$(CC) $(CFLAGS) -I$(ENGINE_DIR) -c $(ENGINE_DIR)/batch.c -o $(OBJDIR)/engine_batch.o

$(OBJDIR)/engine_core.o: $(ENGINE_DIR)/engine.c $(ENGINE_DIR)/engine.h | $(OBJDIR) $(BUILD)
	$(CC) $(CFLAGS) -I$(ENGINE_DIR) -c $(ENGINE_DIR)/engine.c -o $(OBJDIR)/engine_core.o

clean:
	rm -f $(BIN) $(OBJ) $(TUNE_BIN) $(TUNE_OBJ)
