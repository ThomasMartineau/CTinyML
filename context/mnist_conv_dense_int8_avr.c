/*
 * Quantized int8 inference for AVR: Conv1 -> ReLU -> Conv2 -> ReLU ->
 * flatten -> Dense1 -> ReLU -> Dense2 -> logits.
 *
 * Architecture (28x28 input):
 *   Conv1:  1ch -> 8ch,  3x3, stride 2, pad 1   -> 8x14x14
 *   Conv2:  8ch -> 8ch,  3x3, stride 2, pad 1   -> 8x7x7 (= 392 flat)
 *   Dense1: 392 -> 128
 *   Dense2: 128 -> 10
 *
 * Design choices, carried over from the dense-only sketch:
 *   - No im2col. Direct nested-loop convolution reads PROGMEM weights
 *     and accumulates in int32_t. AVR has no cache to exploit, so
 *     im2col's usual speed advantage doesn't apply here, but its
 *     memory cost (a duplicated, expanded input buffer) very much
 *     would.
 *   - Zero-padding is a bounds check inside the loop, never a
 *     materialized padded buffer.
 *   - Requantization multiplier (input_scale * weight_scale /
 *     output_scale) is precomputed OFFLINE in Python and stored as a
 *     single float per output channel/neuron. AVR does float
 *     multiply in software already; a runtime division on top of
 *     that would be pure waste.
 *   - Floats read from PROGMEM go through pgm_read_dword + memcpy,
 *     NOT pgm_read_byte. Reading a 4-byte float with pgm_read_byte
 *     (as seen in the onnx2c AVR output) silently truncates to one
 *     byte -- a correctness bug, not just an inefficiency.
 *   - weight_zero_point == 0 throughout (matches the model's export),
 *     so no weight-side zero-point correction term is needed.
 */

#include <avr/pgmspace.h>
#include <stdint.h>
#include <string.h>

/* ---------------- Architecture-specific dimensions ---------------- */

#define IMG_H 28
#define IMG_W 28

#define CONV1_IN_CH   1
#define CONV1_OUT_CH  8
#define CONV1_K       3
#define CONV1_STRIDE  2
#define CONV1_PAD     1
#define CONV1_OUT_H   14
#define CONV1_OUT_W   14

#define CONV2_IN_CH   8
#define CONV2_OUT_CH  8
#define CONV2_K       3
#define CONV2_STRIDE  2
#define CONV2_PAD     1
#define CONV2_OUT_H   7
#define CONV2_OUT_W   7

#define DENSE1_IN   (CONV2_OUT_CH * CONV2_OUT_H * CONV2_OUT_W) /* 392 */
#define DENSE1_OUT  128
#define DENSE2_IN   128
#define DENSE2_OUT  10

/* ---------------- Weight/bias/scale storage (flash) ---------------- */
/* Layout: weight[out_ch][in_ch][ky][kx] for conv, weight[out][in] for dense.
 * Values below are placeholders -- generate these from your trained,
 * pruned, calibrated model. */

static const int8_t conv1_weight[CONV1_OUT_CH * CONV1_IN_CH * CONV1_K * CONV1_K] PROGMEM = { /* ... */ };
static const int32_t conv1_bias[CONV1_OUT_CH] PROGMEM = { /* ... */ };
static const float conv1_requant[CONV1_OUT_CH] PROGMEM = { /* ... */ }; /* per-out-channel */

static const int8_t conv2_weight[CONV2_OUT_CH * CONV2_IN_CH * CONV2_K * CONV2_K] PROGMEM = { /* ... */ };
static const int32_t conv2_bias[CONV2_OUT_CH] PROGMEM = { /* ... */ };
static const float conv2_requant[CONV2_OUT_CH] PROGMEM = { /* ... */ };

static const int8_t dense1_weight[DENSE1_OUT * DENSE1_IN] PROGMEM = { /* ... */ };
static const int32_t dense1_bias[DENSE1_OUT] PROGMEM = { /* ... */ };
static const float dense1_requant[DENSE1_OUT] PROGMEM = { /* ... */ };

static const int8_t dense2_weight[DENSE2_OUT * DENSE2_IN] PROGMEM = { /* ... */ };
static const int32_t dense2_bias[DENSE2_OUT] PROGMEM = { /* ... */ };
static const float dense2_requant[DENSE2_OUT] PROGMEM = { /* ... */ };

/* ---------------- PROGMEM helpers ---------------- */

static inline int32_t pgm_read_i32(const int32_t *p) {
    return (int32_t)pgm_read_dword(p);
}

/* avr-libc has no pgm_read_float(); reinterpret the 4 bytes correctly
 * instead of truncating with pgm_read_byte(). */
static inline float pgm_read_f32(const float *p) {
    uint32_t bits = pgm_read_dword((const uint32_t *)p);
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline int8_t requantize(float acc_scaled, int8_t zero_point) {
    int32_t q = (int32_t)(acc_scaled + (acc_scaled >= 0.0f ? 0.5f : -0.5f)) + zero_point;
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

/* ---------------- Direct quantized convolution ---------------- */
/*
 * input:  int8_t[in_ch][in_h][in_w], flat, row-major
 * output: int8_t[out_ch][out_h][out_w], flat, row-major
 * relu:   fold ReLU into requantization (clip negative pre-scale to 0)
 */
static void conv2d_int8(
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

/* ---------------- Dense (fully-connected) layer ---------------- */

static void dense_int8(
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

/* ---------------- Forward pass ---------------- */
/*
 * SRAM budget: two ping-pong buffers sized to the largest activation
 * that needs to persist across a layer boundary (Conv1's output,
 * 8*14*14 = 1568 bytes), plus the final 10-byte logits.
 * Total: ~3.2 KB static, well inside the 8 KB SRAM budget, with no
 * malloc and no per-layer buffer growth.
 */
static int8_t buf_a[CONV1_OUT_CH * CONV1_OUT_H * CONV1_OUT_W]; /* 1568 B, reused for dense1 input view */
static int8_t buf_b[CONV1_OUT_CH * CONV1_OUT_H * CONV1_OUT_W]; /* 1568 B, sized for the larger of conv2-out/dense1-out */

void model_forward(const int8_t input[IMG_H * IMG_W], int8_t logits[DENSE2_OUT],
                    int8_t input_zp, int8_t mid_zp, int8_t out_zp)
{
    /* Conv1: input -> buf_a (8x14x14 = 1568) */
    conv2d_int8(input, CONV1_IN_CH, IMG_H, IMG_W, input_zp,
                conv1_weight, conv1_bias, conv1_requant,
                CONV1_OUT_CH, CONV1_K, CONV1_STRIDE, CONV1_PAD,
                buf_a, CONV1_OUT_H, CONV1_OUT_W, mid_zp, /*relu=*/1);

    /* Conv2: buf_a -> buf_b (8x7x7 = 392) */
    conv2d_int8(buf_a, CONV2_IN_CH, CONV1_OUT_H, CONV1_OUT_W, mid_zp,
                conv2_weight, conv2_bias, conv2_requant,
                CONV2_OUT_CH, CONV2_K, CONV2_STRIDE, CONV2_PAD,
                buf_b, CONV2_OUT_H, CONV2_OUT_W, mid_zp, /*relu=*/1);

    /* Flatten is a no-op: buf_b is already (out_ch, out_h, out_w)
     * row-major, matching PyTorch's channel-major flatten order. */

    /* Dense1: buf_b (392) -> buf_a (128, reused) */
    dense_int8(buf_b, DENSE1_IN, mid_zp,
               dense1_weight, dense1_bias, dense1_requant,
               DENSE1_OUT, buf_a, mid_zp, /*relu=*/1);

    /* Dense2: buf_a (128) -> logits (10), no ReLU on the output layer */
    dense_int8(buf_a, DENSE2_IN, mid_zp,
               dense2_weight, dense2_bias, dense2_requant,
               DENSE2_OUT, logits, out_zp, /*relu=*/0);
}
