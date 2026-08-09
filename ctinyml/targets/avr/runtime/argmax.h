#pragma once
/* New -- no reference-file source (it terminates at raw logits). Trivial
 * int32[N] -> uint8 index, consuming dense_int8()'s skip_requant output
 * directly. First-occurrence tie-break, matching numpy.argmax's default
 * and this model's select_last_index=0. */

#include <stdint.h>

static inline uint8_t argmax_i32(const int32_t *logits, uint16_t n) {
    uint16_t best_idx = 0;
    int32_t best_val = logits[0];
    for (uint16_t i = 1; i < n; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = i;
        }
    }
    return (uint8_t)best_idx;
}
