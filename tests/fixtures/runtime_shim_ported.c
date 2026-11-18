/* Test-only shim: the real runtime kernels are declared `static inline`
 * (correct for a single-translation-unit embedded build), which means
 * they have no external linkage -- a compiled .dll won't expose them as
 * callable symbols. Including the header here pulls the implementation
 * into this one translation unit; these non-static wrappers (with
 * external linkage, exported via -Wl,--export-all-symbols) forward
 * straight through so tests/test_runtime.py can call them via ctypes.
 */

#include <stdint.h>

#include "conv2d_direct.h"
#include "gemm.h"
#include "argmax.h"
#include "qmath.h"
#include "pgm_read_f32.h"

void test_conv2d_int8(
    const int8_t *input, uint8_t in_ch, uint8_t in_h, uint8_t in_w, int8_t input_zp,
    const int8_t *weight_pgm, const int32_t *bias_pgm, const float *requant_pgm,
    uint8_t out_ch, uint8_t k, uint8_t stride, uint8_t pad,
    int8_t *output, uint8_t out_h, uint8_t out_w, int8_t out_zp, uint8_t relu)
{
    conv2d_int8(input, in_ch, in_h, in_w, input_zp, weight_pgm, bias_pgm, requant_pgm,
                out_ch, k, stride, pad, output, out_h, out_w, out_zp, relu);
}

void test_dense_int8(
    const int8_t *input, uint16_t in_dim, int8_t input_zp,
    const int8_t *weight_pgm, const int32_t *bias_pgm, const float *requant_pgm,
    uint16_t out_dim, void *output, int8_t out_zp, uint8_t relu, uint8_t skip_requant)
{
    dense_int8(input, in_dim, input_zp, weight_pgm, bias_pgm, requant_pgm,
               out_dim, output, out_zp, relu, skip_requant);
}

uint8_t test_argmax_i32(const int32_t *logits, uint16_t n) {
    return argmax_i32(logits, n);
}

int8_t test_requantize(float acc_scaled, int8_t zero_point) {
    return requantize(acc_scaled, zero_point);
}

int32_t test_pgm_read_i32(const int32_t *p) {
    return pgm_read_i32(p);
}

float test_pgm_read_f32(const float *p) {
    return pgm_read_f32(p);
}
