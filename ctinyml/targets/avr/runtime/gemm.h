#pragma once
/* Ported near-verbatim from context/mnist_conv_dense_int8_avr.c, extended
 * with a skip_requant flag -> int32_t* output path for the final layer
 * feeding ArgMax directly on raw accumulators (output is `void *` so the
 * same function serves both paths, matching the `relu` flag's style --
 * "prefer the flag: keeps one function", per the parent plan).
 *
 * When skip_requant is set, the float-scale/clamp/round path doesn't run
 * at all: the raw int32 accumulator (acc + bias) is written directly,
 * and relu/out_zp/requant_pgm are ignored (this model's final Gemm never
 * has clamp_min0 set, so relu=1 with skip_requant=1 never actually
 * arises here, but the precedence -- skip_requant wins outright -- is
 * still well-defined if it ever did).
 */

#include <stdint.h>

#include "pgm_read_f32.h"
#include "qmath.h"

static inline void dense_int8(
    const int8_t *input, uint16_t in_dim, int8_t input_zp,
    const int8_t *weight_pgm, const int32_t *bias_pgm, const float *requant_pgm,
    uint16_t out_dim, void *output, int8_t out_zp, uint8_t relu, uint8_t skip_requant)
{
    for (uint16_t j = 0; j < out_dim; ++j) {
        int32_t acc = 0;
        for (uint16_t i = 0; i < in_dim; ++i) {
            int8_t w = (int8_t)pgm_read_byte(&weight_pgm[(uint32_t)j * in_dim + i]);
            acc += (int32_t)w * (int32_t)(input[i] - input_zp);
        }
        int32_t bias = pgm_read_i32(&bias_pgm[j]);
        acc += bias;

        if (skip_requant) {
            ((int32_t *)output)[j] = acc;
            continue;
        }

        float scaled = (float)acc * pgm_read_f32(&requant_pgm[j]);
        if (relu && scaled < 0.0f) scaled = 0.0f;
        ((int8_t *)output)[j] = requantize(scaled, out_zp);
    }
}
