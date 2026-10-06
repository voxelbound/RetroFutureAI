#include "retrofutureai.h"

/*
 * Tiny synthetic linear classifier.
 *
 * The model is deliberately compiled into the binary. This keeps the first
 * experiment focused on the inference path rather than file formats or model
 * loading.
 */
static const int8_t WEIGHTS[RF_CLASSES][RF_INPUTS] = {
    {  8,  3, -2,  1 },  /* signal */
    { -4,  6,  5, -3 },  /* noise  */
    {  1, -2,  2,  7 }   /* idle   */
};

static const int16_t BIAS[RF_CLASSES] = {
    0, 0, 0
};

static const char *CLASS_NAMES[RF_CLASSES] = {
    "signal",
    "noise",
    "idle"
};

rf_result_t rf_infer(const int8_t input[RF_INPUTS])
{
    rf_result_t result;
    int c;
    int i;

    for (c = 0; c < RF_CLASSES; ++c) {
        int32_t acc = BIAS[c];

        for (i = 0; i < RF_INPUTS; ++i) {
            acc += (int32_t)input[i] * (int32_t)WEIGHTS[c][i];
        }

        result.score[c] = acc;
    }

    result.best_class = 0;
    for (c = 1; c < RF_CLASSES; ++c) {
        if (result.score[c] > result.score[result.best_class]) {
            result.best_class = c;
        }
    }

    return result;
}

const char *rf_class_name(int class_id)
{
    if (class_id < 0 || class_id >= RF_CLASSES) {
        return "unknown";
    }

    return CLASS_NAMES[class_id];
}
