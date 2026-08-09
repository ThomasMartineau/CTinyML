#pragma once
/* Phase 5 stub: prototype only. Phase 6 replaces this with the real,
 * tested implementation ported from context/mnist_conv_dense_int8_avr.c
 * (round via +-0.5f, saturate to [-128,127]). */

#include <stdint.h>

int8_t requantize(float acc_scaled, int8_t zero_point);
