#pragma once
/* Ported near-verbatim from context/mnist_conv_dense_int8_avr.c: direct
 * nested-loop convolution, no im2col (AVR has no cache to exploit, and
 * im2col's memory cost -- a duplicated, expanded input buffer -- would
 * be pure waste here). Zero-padding is a bounds check inside the loop,
 * never a materialized padded buffer.
 *
 * input:  int8_t[in_ch][in_h][in_w], flat, row-major
 * output: int8_t[out_ch][out_h][out_w], flat, row-major
 * relu:   fold ReLU into requantization (clip negative pre-scale to 0)
 */

#include <stdint.h>

#include "pgm_read_f32.h"
#include "qmath.h"

static inline void conv2d_int8(
    const int8_t *input, uint8_t in_ch, uint8_t in_h, uint8_t in_w, int8_t input_zp,
    const int8_t *weight_pgm, const int32_t *bias_pgm, const float *requant_pgm,
    uint8_t out_ch, uint8_t k, uint8_t stride, uint8_t pad,
    int8_t *output, uint8_t out_h, uint8_t out_w, int8_t out_zp, uint8_t relu)
{
    for (uint8_t oc = 0; oc < out_ch; ++oc) {
        int32_t bias = pgm_read_i32(&bias_pgm[oc]);
        float requant_mult = pgm_read_f32(&requant_pgm[oc]);

        for (uint8_t oy = 0; oy < out_h; ++oy) {
            for (uint8_t ox = 0; ox < out_w; ++ox) {
                int32_t acc = 0;
                int16_t iy0 = (int16_t)oy * stride - pad;
                int16_t ix0 = (int16_t)ox * stride - pad;

                for (uint8_t ic = 0; ic < in_ch; ++ic) {
                    for (uint8_t ky = 0; ky < k; ++ky) {
                        int16_t iy = iy0 + ky;
                        if (iy < 0 || iy >= in_h) continue; /* zero-pad via skip, no padded buffer */

                        for (uint8_t kx = 0; kx < k; ++kx) {
                            int16_t ix = ix0 + kx;
                            if (ix < 0 || ix >= in_w) continue;

                            int8_t in_val = input[((uint16_t)ic * in_h + iy) * in_w + ix];
                            uint16_t widx = (((uint16_t)oc * in_ch + ic) * k + ky) * k + kx;
                            int8_t w = (int8_t)pgm_read_byte(&weight_pgm[widx]);

                            acc += (int32_t)w * (int32_t)(in_val - input_zp);
                        }
                    }
                }
                acc += bias;

                float scaled = (float)acc * requant_mult;
                if (relu && scaled < 0.0f) scaled = 0.0f;
                output[((uint16_t)oc * out_h + oy) * out_w + ox] = requantize(scaled, out_zp);
            }
        }
    }
}
