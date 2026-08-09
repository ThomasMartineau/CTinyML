#pragma once
/* Phase 5 stub: prototype only. Phase 6 replaces this with the real,
 * tested implementation ported from context/mnist_conv_dense_int8_avr.c
 * (pgm_read_dword + memcpy, never pgm_read_byte on floats).
 *
 * Also owns the PROGMEM/pgm_read_* host-portability shim: on AVR these
 * come from <avr/pgmspace.h>; on host (this file's only job right now --
 * letting generated model.c host-compile for Phase 5's verification),
 * PROGMEM is a no-op and pgm_read_* are plain dereferences, since there's
 * no separate flash address space to worry about.
 */

#include <stdint.h>

#ifdef __AVR__
#include <avr/pgmspace.h>
#else
#ifndef PROGMEM
#define PROGMEM
#endif
#endif

int32_t pgm_read_i32(const int32_t *p);
float pgm_read_f32(const float *p);
