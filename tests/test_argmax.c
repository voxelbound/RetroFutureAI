#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "retrofutureai.h"

static void test_single_value(void)
{
    const int32_t values[] = { 42 };

    assert(rf_argmax_i32(values, 1) == 0);
}

static void test_maximum_in_middle(void)
{
    const int32_t values[] = { 10, 20, 50, 30, 40 };

    assert(rf_argmax_i32(values, 5) == 2);
}

static void test_negative_values(void)
{
    const int32_t values[] = { -20, -3, -10, -50 };

    assert(rf_argmax_i32(values, 4) == 1);
}

static void test_first_maximum_wins_tie(void)
{
    const int32_t values[] = { 5, 12, 7, 12, 3 };

    assert(rf_argmax_i32(values, 5) == 1);
}

static void test_maximum_at_end(void)
{
    const int32_t values[] = { 1, 2, 3, 99 };

    assert(rf_argmax_i32(values, 4) == 3);
}

static void test_empty_input_returns_minus_one(void)
{
    assert(rf_argmax_i32(NULL, 0) == -1);
}

int main(void)
{
    test_single_value();
    test_maximum_in_middle();
    test_negative_values();
    test_first_maximum_wins_tie();
    test_maximum_at_end();
    test_empty_input_returns_minus_one();

    puts("all argmax tests passed");
    return 0;
}
