#pragma once
/* Byte-for-byte copy of context/mnist_conv_dense_int8_avr.c's
 * conv2d_int8()/dense_int8() function bodies (and the pgm_read_i32/
 * pgm_read_f32/requantize helpers they call), for
 * tests/test_runtime.py's regression test only -- NOT shipped runtime
 * code. The only change from the original source is swapping the
 * AVR-only `#include <avr/pgmspace.h>` for a host-portable
 * PROGMEM/pgm_read_* shim (no separate flash address space on host),
 * so this can be compiled standalone here to prove the real port in
 * ctinyml/targets/avr/runtime/ produces bit-identical output.
 *
 * If context/mnist_conv_dense_int8_avr.c's kernel logic ever changes,
 * this fixture needs updating to match -- it is a deliberate, tracked
 * duplicate, not an independent implementation.
 */

#include <stdint.h>
#include <string.h>

#define PROGMEM

static inline int32_t pgm_read_i32(const int32_t *p) {
    return *p;
}

static inline float pgm_read_f32(const float *p) {
    return *p;
}

static inline uint8_t pgm_read_byte(const void *p) {
    return *(const uint8_t *)p;
}

static inline int8_t requantize(float acc_scaled, int8_t zero_point) {
    int32_t q = (int32_t)(acc_scaled + (acc_scaled >= 0.0f ? 0.5f : -0.5f)) + zero_point;
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

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

static inline void dense_int8(
    const int8_t *input, uint16_t in_dim, int8_t input_zp,
    const int8_t *weight_pgm, const int32_t *bias_pgm, const float *requant_pgm,
    uint16_t out_dim, int8_t *output, int8_t out_zp, uint8_t relu)
{
    for (uint16_t j = 0; j < out_dim; ++j) {
        int32_t acc = 0;
        for (uint16_t i = 0; i < in_dim; ++i) {
            int8_t w = (int8_t)pgm_read_byte(&weight_pgm[(uint32_t)j * in_dim + i]);
            acc += (int32_t)w * (int32_t)(input[i] - input_zp);
        }
        int32_t bias = pgm_read_i32(&bias_pgm[j]);
        acc += bias;

        float scaled = (float)acc * pgm_read_f32(&requant_pgm[j]);
        if (relu && scaled < 0.0f) scaled = 0.0f;
        output[j] = requantize(scaled, out_zp);
    }
}
