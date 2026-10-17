#include <stddef.h>
#include <stdint.h>

#include "retrofutureai.h"

/*
 * Apply ReLU in place to int32 accumulator values.
 *
 * ReLU(x) = max(0, x). The operation stays in int32 because requantization to
 * int8 is a separate, explicit step in the inference pipeline.
 */
void rf_relu_i32(int32_t *values, size_t count)
{
    size_t i;

    for (i = 0; i < count; ++i) {
        if (values[i] < 0) {
            values[i] = 0;
        }
    }
}
