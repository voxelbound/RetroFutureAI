#include <stddef.h>
#include <stdint.h>

#include "retrofutureai.h"

/*
 * Find the index of the largest int32 value.
 *
 * The comparison uses '>' rather than '>=' deliberately, so equal values keep
 * the first maximum. That makes tie behavior deterministic across targets.
 */
int rf_argmax_i32(const int32_t *values, size_t count)
{
    size_t i;
    size_t best;

    if (count == 0) {
        return -1;
    }

    best = 0;

    for (i = 1; i < count; ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }

    return (int)best;
}
