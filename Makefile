CC ?= cc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -pedantic

APP = retrofutureai
TEST = test_inference

.PHONY: all test clean

all: $(APP)

$(APP): src/main.c src/inference.c include/retrofutureai.h
	$(CC) $(CFLAGS) -Iinclude src/main.c src/inference.c -o $(APP)

$(TEST): tests/test_inference.c src/inference.c include/retrofutureai.h
	$(CC) $(CFLAGS) -Iinclude tests/test_inference.c src/inference.c -o $(TEST)

test: $(TEST)
	./$(TEST)

clean:
	rm -f $(APP) $(TEST)

