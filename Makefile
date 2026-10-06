CC ?= cc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -pedantic

APP = retrofutureai
TEST_INFERENCE = test_inference
TEST_DENSE = test_dense
TEST_RELU = test_relu

.PHONY: all test clean

all: $(APP)

$(APP): src/main.c src/inference.c include/retrofutureai.h
	$(CC) $(CFLAGS) -Iinclude src/main.c src/inference.c -o $(APP)

$(TEST_INFERENCE): tests/test_inference.c src/inference.c include/retrofutureai.h
	$(CC) $(CFLAGS) -Iinclude tests/test_inference.c src/inference.c -o $(TEST_INFERENCE)

$(TEST_DENSE): tests/test_dense.c src/runtime/dense.c include/retrofutureai.h
	$(CC) $(CFLAGS) -Iinclude tests/test_dense.c src/runtime/dense.c -o $(TEST_DENSE)

$(TEST_RELU): tests/test_relu.c src/runtime/relu.c include/retrofutureai.h
	$(CC) $(CFLAGS) -Iinclude tests/test_relu.c src/runtime/relu.c -o $(TEST_RELU)

test: $(TEST_INFERENCE) $(TEST_DENSE) $(TEST_RELU)
	./$(TEST_INFERENCE)
	./$(TEST_DENSE)
	./$(TEST_RELU)

clean:
	rm -f $(APP) $(TEST_INFERENCE) $(TEST_DENSE) $(TEST_RELU)

