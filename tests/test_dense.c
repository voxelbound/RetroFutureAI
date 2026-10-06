#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "retrofutureai.h"

static void test_single_output(void)
{
    const int8_t input[] = { 1, 2, 3 };
    const int8_t weights[] = { 4, 5, 6 };
    const int32_t bias[] = { 7 };
    int32_t output[1];

    rf_dense_i8(input, weights, bias, output, 3, 1);

    assert(output[0] == 39);
}

static void test_multiple_outputs(void)
{
    const int8_t input[] = { 2, -1 };

    const int8_t weights[] = {
         3,  4,
        -2,  5
    };

    const int32_t bias[] = {
        1,
        -3
    };

    int32_t output[2];

    rf_dense_i8(input, weights, bias, output, 2, 2);

    assert(output[0] == 3);
    assert(output[1] == -12);
}

static void test_zero_input(void)
{
    const int8_t input[] = { 0, 0, 0 };
    const int8_t weights[] = {
        10, -20, 30
    };
    const int32_t bias[] = {
        42
    };

    int32_t output[1];

    rf_dense_i8(input, weights, bias, output, 3, 1);

    assert(output[0] == 42);
}

static void test_mixed_signs(void)
{
    const int8_t input[] = { -3, 2, -1 };

    const int8_t weights[] = {
        -4, 5, 6
    };

    const int32_t bias[] = {
        -2
    };

    int32_t output[1];

    rf_dense_i8(input, weights, bias, output, 3, 1);

    assert(output[0] == 14);
}

int main(void)
{
    test_single_output();
    test_multiple_outputs();
    test_zero_input();
    test_mixed_signs();

    puts("all dense tests passed");
    return 0;
}

