CC ?= cc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -pedantic

APP = retrofutureai
TEST = test_inference

.PHONY: all test clean

all: $(APP)

$(APP): src/main.c src/inference.c src/inference.h
	$(CC) $(CFLAGS) -Isrc src/main.c src/inference.c -o $(APP)

$(TEST): tests/test_inference.c src/inference.c src/inference.h
	$(CC) $(CFLAGS) -Isrc tests/test_inference.c src/inference.c -o $(TEST)

test: $(TEST)
	./$(TEST)

clean:
	rm -f $(APP) $(TEST)
