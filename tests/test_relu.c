#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "retrofutureai.h"

static void test_negative_values_become_zero(void)
{
    int32_t values[] = { -10, -1, -42 };

    rf_relu_i32(values, 3);

    assert(values[0] == 0);
    assert(values[1] == 0);
    assert(values[2] == 0);
}

static void test_positive_values_unchanged(void)
{
    int32_t values[] = { 1, 7, 12345 };

    rf_relu_i32(values, 3);

    assert(values[0] == 1);
    assert(values[1] == 7);
    assert(values[2] == 12345);
}

static void test_zero_unchanged(void)
{
    int32_t values[] = { 0 };

    rf_relu_i32(values, 1);

    assert(values[0] == 0);
}

static void test_mixed_values(void)
{
    int32_t values[] = { -5, 0, 3, -1, 9 };

    rf_relu_i32(values, 5);

    assert(values[0] == 0);
    assert(values[1] == 0);
    assert(values[2] == 3);
    assert(values[3] == 0);
    assert(values[4] == 9);
}

int main(void)
{
    test_negative_values_become_zero();
    test_positive_values_unchanged();
    test_zero_unchanged();
    test_mixed_values();

    puts("all ReLU tests passed");
    return 0;
}

