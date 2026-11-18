/* Same purpose as runtime_shim_ported.c, wrapping
 * reference_kernels_verbatim.h instead -- the byte-for-byte copy of
 * context/mnist_conv_dense_int8_avr.c's kernels, so
 * tests/test_runtime.py's regression test can call it via ctypes too.
 * dense_int8() here has the ORIGINAL (pre-skip_requant) signature.
 */

#include <stdint.h>

#include "reference_kernels_verbatim.h"

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
    uint16_t out_dim, int8_t *output, int8_t out_zp, uint8_t relu)
{
    dense_int8(input, in_dim, input_zp, weight_pgm, bias_pgm, requant_pgm,
               out_dim, output, out_zp, relu);
}
