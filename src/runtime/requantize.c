#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#include "retrofutureai.h"

void rf_requantize_i32_to_i8(
    const int32_t *input,
    int8_t *output,
    size_t count,
    unsigned int shift)
{
    size_t i;
    int32_t divisor;

    if (shift >= 31) {
        divisor = INT32_MAX;
    } else {
        divisor = (int32_t)(1U << shift);
    }

    for (i = 0; i < count; ++i) {
        int32_t value;

        if (shift == 0) {
            value = input[i];
        } else {
            value = input[i] / divisor;
        }

        if (value > INT8_MAX) {
            value = INT8_MAX;
        } else if (value < INT8_MIN) {
            value = INT8_MIN;
        }

        output[i] = (int8_t)value;
    }
}

