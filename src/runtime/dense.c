#include <stddef.h>
#include <stdint.h>

#include "retrofutureai.h"

void rf_dense_i8(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *bias,
    int32_t *output,
    size_t input_size,
    size_t output_size)
{
    size_t o;
    size_t i;

    for (o = 0; o < output_size; ++o) {
        int32_t acc = bias[o];
        const int8_t *row = &weights[o * input_size];

        for (i = 0; i < input_size; ++i) {
            acc += (int32_t)input[i] * (int32_t)row[i];
        }

        output[o] = acc;
    }
}

