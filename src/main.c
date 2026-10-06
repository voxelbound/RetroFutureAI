#include <stdio.h>
#include <stdint.h>

#include "retrofutureai.h"

int main(void)
{
    const int8_t input[RF_INPUTS] = { 8, 2, -3, 1 };
    rf_result_t result = rf_infer(input);
    int i;

    printf("RetroFutureAI fixed-point inference demo\n");
    printf("input: [%d, %d, %d, %d]\n",
           input[0], input[1], input[2], input[3]);

    printf("scores: [");
    for (i = 0; i < RF_CLASSES; ++i) {
        printf("%ld%s",
               (long)result.score[i],
               (i == RF_CLASSES - 1) ? "" : ", ");
    }
    printf("]\n");

    printf("class: %s\n", rf_class_name(result.best_class));

    return 0;
}
