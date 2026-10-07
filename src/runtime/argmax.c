#include <stddef.h>
#include <stdint.h>

#include "retrofutureai.h"

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

