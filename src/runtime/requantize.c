#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#include "retrofutureai.h"

/*
 * Reduce int32 accumulator values back to int8 activations.
 *
 * Scaling uses division rather than a signed right shift so negative values
 * have explicit truncation-toward-zero semantics under C99. Future optimized
 * implementations must reproduce the same results exactly.
 */
void rf_requantize_i32_to_i8(
    const int32_t *input,
    int8_t *output,
    size_t count,
    unsigned int shift)
{
    size_t i;
    int32_t divisor;

    /*
     * Normal operation uses 2^shift. The >=31 case mirrors the current runtime
     * contract without attempting a shift that cannot be represented safely
     * in a positive signed int32 value.
     */
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
            /*
             * Signed integer division in C99 truncates toward zero, which is
             * the rounding behavior used by the reference runtime.
             */
            value = input[i] / divisor;
        }

        /* Saturate before narrowing to int8 to avoid wraparound. */
        if (value > INT8_MAX) {
            value = INT8_MAX;
        } else if (value < INT8_MIN) {
            value = INT8_MIN;
        }

        output[i] = (int8_t)value;
    }
}
