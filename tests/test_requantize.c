#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "retrofutureai.h"

static void test_shift_zero_preserves_in_range_values(void)
{
    const int32_t input[] = { -10, 0, 42, 127 };
    int8_t output[4];

    rf_requantize_i32_to_i8(input, output, 4, 0);

    assert(output[0] == -10);
    assert(output[1] == 0);
    assert(output[2] == 42);
    assert(output[3] == 127);
}

static void test_positive_values_are_truncated(void)
{
    const int32_t input[] = { 7, 8, 15, 16 };
    int8_t output[4];

    rf_requantize_i32_to_i8(input, output, 4, 2);

    assert(output[0] == 1);
    assert(output[1] == 2);
    assert(output[2] == 3);
    assert(output[3] == 4);
}

static void test_negative_values_truncate_toward_zero(void)
{
    const int32_t input[] = { -7, -8, -15, -16 };
    int8_t output[4];

    rf_requantize_i32_to_i8(input, output, 4, 2);

    assert(output[0] == -1);
    assert(output[1] == -2);
    assert(output[2] == -3);
    assert(output[3] == -4);
}

static void test_positive_values_are_clamped(void)
{
    const int32_t input[] = { 127, 128, 1000 };
    int8_t output[3];

    rf_requantize_i32_to_i8(input, output, 3, 0);

    assert(output[0] == 127);
    assert(output[1] == 127);
    assert(output[2] == 127);
}

static void test_negative_values_are_clamped(void)
{
    const int32_t input[] = { -128, -129, -1000 };
    int8_t output[3];

    rf_requantize_i32_to_i8(input, output, 3, 0);

    assert(output[0] == -128);
    assert(output[1] == -128);
    assert(output[2] == -128);
}

static void test_shift_then_clamp(void)
{
    const int32_t input[] = { 1024, -1024 };
    int8_t output[2];

    rf_requantize_i32_to_i8(input, output, 2, 2);

    assert(output[0] == 127);
    assert(output[1] == -128);
}

int main(void)
{
    test_shift_zero_preserves_in_range_values();
    test_positive_values_are_truncated();
    test_negative_values_truncate_toward_zero();
    test_positive_values_are_clamped();
    test_negative_values_are_clamped();
    test_shift_then_clamp();

    puts("all requantization tests passed");
    return 0;
}

