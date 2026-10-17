#include <stdint.h>

#include "retrofutureai.h"

/*
 * Tiny synthetic linear classifier.
 *
 * The model is deliberately compiled into the binary. This keeps the first
 * experiment focused on the inference path rather than file formats or model
 * loading.
 *
 * This is a hand-written demo model, not a trained neural network. Its purpose
 * is to exercise the runtime API before learned weights are introduced.
 */
static const int8_t WEIGHTS[RF_CLASSES][RF_INPUTS] = {
    {  8,  3, -2,  1 },  /* signal */
    { -4,  6,  5, -3 },  /* noise  */
    {  1, -2,  2,  7 }   /* idle   */
};

static const int32_t BIAS[RF_CLASSES] = {
    0, 0, 0
};

static const char *CLASS_NAMES[RF_CLASSES] = {
    "signal",
    "noise",
    "idle"
};

/*
 * Run the current synthetic classifier.
 *
 * The dense operator produces one int32 score per class. Argmax then converts
 * those raw scores into the selected class index.
 */
rf_result_t rf_infer(const int8_t input[RF_INPUTS])
{
    rf_result_t result;

    rf_dense_i8(
        input,
        &WEIGHTS[0][0],
        BIAS,
        result.score,
        RF_INPUTS,
        RF_CLASSES);

    result.best_class = rf_argmax_i32(result.score, RF_CLASSES);

    return result;
}

/*
 * Convert a class index from the synthetic demo into a printable name.
 */
const char *rf_class_name(int class_id)
{
    if (class_id < 0 || class_id >= RF_CLASSES) {
        return "unknown";
    }

    return CLASS_NAMES[class_id];
}
