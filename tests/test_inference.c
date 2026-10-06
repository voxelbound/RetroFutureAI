#include <assert.h>
#include <stdio.h>
#include <stdint.h>

#include "retrofutureai.h"

static void test_signal_vector(void)
{
    const int8_t input[RF_INPUTS] = { 8, 2, -3, 1 };
    rf_result_t result = rf_infer(input);

    assert(result.best_class == 0);
    assert(result.score[0] == 77);
}

static void test_noise_vector(void)
{
    const int8_t input[RF_INPUTS] = { -2, 7, 5, -1 };
    rf_result_t result = rf_infer(input);

    assert(result.best_class == 1);
}

static void test_idle_vector(void)
{
    const int8_t input[RF_INPUTS] = { 0, -1, 1, 8 };
    rf_result_t result = rf_infer(input);

    assert(result.best_class == 2);
}

static void test_class_names(void)
{
    assert(rf_class_name(0) != 0);
    assert(rf_class_name(99) != 0);
}

int main(void)
{
    test_signal_vector();
    test_noise_vector();
    test_idle_vector();
    test_class_names();

    puts("all inference tests passed");
    return 0;
}
