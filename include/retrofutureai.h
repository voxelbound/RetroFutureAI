#ifndef RETROFUTUREAI_H
#define RETROFUTUREAI_H

#include <stddef.h>
#include <stdint.h>

#define RF_INPUTS 4
#define RF_CLASSES 3

typedef struct {
    int32_t score[RF_CLASSES];
    int best_class;
} rf_result_t;

void rf_dense_i8(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *bias,
    int32_t *output,
    size_t input_size,
    size_t output_size);

void rf_relu_i32(int32_t *values, size_t count);
rf_result_t rf_infer(const int8_t input[RF_INPUTS]);
const char *rf_class_name(int class_id);

#endif

