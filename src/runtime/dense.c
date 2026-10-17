#include <stddef.h>
#include <stdint.h>

#include "retrofutureai.h"

/*
 * Portable reference implementation of an int8 fully connected layer.
 *
 * The deliberately simple nested loops make the arithmetic and memory layout
 * easy to compare with future m68k-specific and assembly implementations.
 */
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

        /*
         * Each output neuron owns one contiguous row of input_size weights.
         * This keeps the inner multiply-accumulate loop sequential in memory.
         */
        const int8_t *row = &weights[o * input_size];

        for (i = 0; i < input_size; ++i) {
            /*
             * Explicit casts make it clear that multiplication contributes to
             * a 32-bit accumulator rather than remaining an 8-bit operation.
             */
            acc += (int32_t)input[i] * (int32_t)row[i];
        }

        output[o] = acc;
    }
}
