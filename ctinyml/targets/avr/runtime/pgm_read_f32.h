#pragma once
/* Ported near-verbatim from context/mnist_conv_dense_int8_avr.c
 * (pgm_read_dword + memcpy, never pgm_read_byte on floats -- reading a
 * 4-byte float with pgm_read_byte, as seen in the onnx2c AVR output,
 * silently truncates to one byte, a correctness bug not just an
 * inefficiency).
 *
 * Also owns the PROGMEM/pgm_read_* host-portability shim: on AVR these
 * come from <avr/pgmspace.h>; on host (used for this project's own
 * standalone unit tests -- see tests/test_runtime.py -- since no
 * avr-gcc toolchain is available in this environment), PROGMEM is a
 * no-op and pgm_read_* are plain dereferences, since there's no
 * separate flash address space to worry about. The #ifdef __AVR__
 * branch below is therefore untested in this environment; verified by
 * manual line-by-line comparison against the reference file instead --
 * re-verify with a real avr-gcc build before flashing hardware.
 */

#include <stdint.h>
#include <string.h>

#ifdef __AVR__
#include <avr/pgmspace.h>

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

#else

#ifndef PROGMEM
#define PROGMEM
#endif

static inline uint8_t pgm_read_byte(const void *p) {
    return *(const uint8_t *)p;
}

static inline int32_t pgm_read_i32(const int32_t *p) {
    return *p;
}

static inline float pgm_read_f32(const float *p) {
    return *p;
}

#endif
