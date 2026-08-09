#pragma once
/* Ported verbatim from context/mnist_conv_dense_int8_avr.c. */

#include <stdint.h>

static inline int8_t requantize(float acc_scaled, int8_t zero_point) {
    int32_t q = (int32_t)(acc_scaled + (acc_scaled >= 0.0f ? 0.5f : -0.5f)) + zero_point;
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    return (int8_t)q;
}
