#ifndef RETROFUTUREAI_H
#define RETROFUTUREAI_H

#include <stddef.h>
#include <stdint.h>

#define RF_INPUTS 4
#define RF_CLASSES 3

/*
 * Result returned by the current synthetic classifier.
 *
 * score contains one signed 32-bit score per class. best_class is the index
 * of the largest score, or -1 if no class could be selected.
 */
typedef struct {
    int32_t score[RF_CLASSES];
    int best_class;
} rf_result_t;

/**
 * Compute a fully connected layer using signed 8-bit inputs and weights.
 *
 * Each output is calculated as:
 *
 *     bias[o] + sum(input[i] * weight[o, i])
 *
 * Products and sums are accumulated in signed 32-bit integers. Weights are
 * stored output-row first and passed as one contiguous array.
 *
 * @param input       Input vector containing input_size int8 values.
 * @param weights     Flattened [output_size][input_size] int8 weight matrix.
 * @param bias        Bias vector containing output_size int32 values.
 * @param output      Destination for output_size int32 accumulator values.
 * @param input_size  Number of input values per output.
 * @param output_size Number of outputs to calculate.
 */
void rf_dense_i8(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *bias,
    int32_t *output,
    size_t input_size,
    size_t output_size);

/**
 * Apply ReLU to signed 32-bit values in place.
 *
 * Negative values are replaced by zero. Zero and positive values are left
 * unchanged.
 *
 * @param values Array of values to modify.
 * @param count  Number of elements in values.
 */
void rf_relu_i32(int32_t *values, size_t count);

/**
 * Convert signed 32-bit accumulator values to signed 8-bit activations.
 *
 * Values are scaled by division with 2^shift for normal shifts, using C99
 * integer division and therefore truncation toward zero. Results are then
 * clamped to the signed int8 range [-128, 127].
 *
 * The exact rounding and saturation behavior is part of the runtime contract
 * and must be preserved by future optimized implementations.
 *
 * @param input  Source int32 values.
 * @param output Destination int8 values.
 * @param count  Number of values to convert.
 * @param shift  Power-of-two scaling exponent.
 */
void rf_requantize_i32_to_i8(
    const int32_t *input,
    int8_t *output,
    size_t count,
    unsigned int shift);

/**
 * Return the index of the largest signed 32-bit value.
 *
 * If several values share the maximum, the first one wins. An empty input
 * returns -1.
 *
 * @param values Input vector.
 * @param count  Number of values in the vector.
 * @return Index of the first maximum value, or -1 when count is zero.
 */
int rf_argmax_i32(const int32_t *values, size_t count);

/**
 * Run the current synthetic 4-input, 3-class classifier.
 *
 * @param input Four signed 8-bit input values.
 * @return Class scores together with the selected class index.
 */
rf_result_t rf_infer(const int8_t input[RF_INPUTS]);

/**
 * Return the display name for a synthetic demo class.
 *
 * @param class_id Class index.
 * @return Class name, or "unknown" for an invalid index.
 */
const char *rf_class_name(int class_id);

#endif
