CC = cc
CFLAGS = -O3 -Wall -Wextra -pedantic -std=c99 -fPIC
ROOT = ../../..
BUILD = $(ROOT)/build/native
OBJDIR = $(BUILD)/obj/engine
OBJ = $(OBJDIR)/batch.o $(OBJDIR)/engine.o
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
LIB = $(BUILD)/libgradbot_engine.dylib
SHARED = -dynamiclib
else
LIB = $(BUILD)/libgradbot_engine.so
SHARED = -shared
endif

all: $(LIB)

$(LIB): $(OBJ)
	$(CC) $(OBJ) $(SHARED) -lm -o $(LIB)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BUILD):
	mkdir -p $(BUILD)

$(OBJDIR)/batch.o: batch.c engine.h | $(OBJDIR) $(BUILD)
	$(CC) $(CFLAGS) batch.c -c -o $(OBJDIR)/batch.o

$(OBJDIR)/engine.o: engine.c engine.h | $(OBJDIR) $(BUILD)
	$(CC) $(CFLAGS) engine.c -c -o $(OBJDIR)/engine.o

clean:
	rm -f $(LIB) $(OBJ)
