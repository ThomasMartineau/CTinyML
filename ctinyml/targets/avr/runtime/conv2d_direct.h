#pragma once
/* Phase 5 stub: prototype only. Phase 6 replaces this with the real,
 * tested implementation ported near-verbatim from
 * context/mnist_conv_dense_int8_avr.c (shared runtime-parameterized
 * kernel, direct conv, zero-pad via bounds-check continue). */

#include <stdint.h>

void conv2d_int8(
    const int8_t *input, uint8_t in_ch, uint8_t in_h, uint8_t in_w, int8_t input_zp,
    const int8_t *weight_pgm, const int32_t *bias_pgm, const float *requant_pgm,
    uint8_t out_ch, uint8_t k, uint8_t stride, uint8_t pad,
    int8_t *output, uint8_t out_h, uint8_t out_w, int8_t out_zp, uint8_t relu);
