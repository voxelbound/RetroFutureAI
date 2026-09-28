#ifndef RETROFUTUREAI_INFERENCE_H
#define RETROFUTUREAI_INFERENCE_H

#include <stdint.h>

#define RF_INPUTS 4
#define RF_CLASSES 3

typedef struct {
    int32_t score[RF_CLASSES];
    int best_class;
} rf_result_t;

rf_result_t rf_infer(const int8_t input[RF_INPUTS]);
const char *rf_class_name(int class_id);

#endif
