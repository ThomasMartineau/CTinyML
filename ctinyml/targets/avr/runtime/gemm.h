#pragma once
/* Phase 5 stub: prototype only. Phase 6 replaces this with the real,
 * tested implementation ported near-verbatim from
 * context/mnist_conv_dense_int8_avr.c, extended with a skip_requant
 * flag -> int32_t* output path for the final layer (output is `void *`
 * so the same function serves both the requantized int8_t* path and the
 * raw-accumulator int32_t* path, matching the `relu` flag's style). */

#include <stdint.h>

void dense_int8(
    const int8_t *input, uint16_t in_dim, int8_t input_zp,
    const int8_t *weight_pgm, const int32_t *bias_pgm, const float *requant_pgm,
    uint16_t out_dim, void *output, int8_t out_zp, uint8_t relu, uint8_t skip_requant);
