#include <stddef.h>
#include <stdint.h>

#include "retrofutureai.h"

void rf_relu_i32(int32_t *values, size_t count)
{
    size_t i;

    for (i = 0; i < count; ++i) {
        if (values[i] < 0) {
            values[i] = 0;
        }
    }
}

