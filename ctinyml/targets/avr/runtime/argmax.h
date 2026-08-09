#pragma once
/* Phase 5 stub: prototype only. Phase 6 replaces this with the real,
 * tested implementation -- new, trivial int32[N] -> uint8 index,
 * consuming dense_int8()'s skip_requant output directly. */

#include <stdint.h>

uint8_t argmax_i32(const int32_t *logits, uint16_t n);
