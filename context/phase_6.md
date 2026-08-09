# Subplan: Phase 6 — AVR device runtime library

## Context

Phases 0–5 are implemented and green (74 passing tests). Phase 5's codegen (`ctinyml/targets/avr/codegen/`) already generates a real `model.c` for the quantized MNIST graph and verified it compiles against **stub** runtime headers (prototypes only, no bodies) in `ctinyml/targets/avr/runtime/*.h`. Phase 6 fills those five headers in for real — porting `conv2d_int8`/`dense_int8`/`requantize`/`pgm_read_i32`/`pgm_read_f32` near-verbatim from `context/mnist_conv_dense_int8_avr.c`, plus a new `argmax_i32` — with standalone host-side unit tests, per the parent plan's Phase 6 section.

**This subplan is explicitly informed by `context/known_issues.md`'s Phase 0–5 retrospective.** Concretely, three lessons carry forward:

1. **The Phase 5 stub headers are a contract, not a draft.** Phase 5's `emit_conv2d.py`/`emit_gemm.py`/`emit_argmax.py` generate call-sites against *exact* pinned signatures (parameter order, types, the `void *output`/`skip_requant` convention for `dense_int8`), and Phase 5's own test suite (`test_generate_model_c_compiles_against_stub_headers`) only checked those signatures against empty-bodied stubs. Phase 6 must fill in bodies for **exactly** those declared prototypes — no signature drift — and the full Phase 0–5 test suite gets re-run at the end specifically to catch any accidental divergence (this is the direct descendant of the Phase 3 lesson: a signature/shape assumption that looks right in isolation can silently break a downstream consumer that was never re-verified against it).
2. **Toolchain gaps get handled the way Phase 5 handled them: explicitly deferred, not silently skipped.** `avr-gcc` still isn't installed. Host `gcc` (MinGW-w64) is confirmed available and is what Phase 5's compile-check already used. The **real** `pgm_read_dword`-based AVR path (guarded `#ifdef __AVR__`) genuinely cannot be exercised in this environment — it gets written verbatim from the reference file and manually cross-checked line-by-line, explicitly flagged in the plan and in code comments as untested-here, not quietly assumed correct.
3. **A new, Phase-6-specific toolchain gotcha, worth planning around now rather than discovering mid-implementation (same spirit as the Phase 0 case-insensitivity surprise):** the reference file's kernels (and this port) are declared `static` — correct for a single-translation-unit embedded build, but that means they have no external linkage, so a compiled `.o`/`.dll` won't expose `conv2d_int8` etc. as callable symbols for host-side testing. MinGW-built DLLs also don't auto-export symbols the way ELF `.so`s do. Both are solved together: small **test-only shim `.c` files** `#include` the real header (pulling the `static` implementation into that translation unit) and define thin *non-static* wrapper functions ctypes can call, compiled with `-shared -Wl,--export-all-symbols`.

**How the required "verbatim regression" test actually works, since the reference file can't be compiled as-is:** `context/mnist_conv_dense_int8_avr.c` (a) unconditionally `#include`s `<avr/pgmspace.h>` (host-uncompilable) and (b) has empty placeholder weight arrays (`{ /* ... */ }`) — there's no real numeric baseline to run. So "confirm identical output to the original file compiled standalone" is implemented as: a test-only fixture (`tests/fixtures/reference_kernels_verbatim.h`) containing the reference file's `conv2d_int8`/`dense_int8` function bodies copied **byte-for-byte**, with only the AVR-only include swapped for the same host-portable `PROGMEM`/`pgm_read_*` shim Phase 5 already established — then both that fixture and the real port are compiled into separate DLLs, fed identical synthetic weight/bias/requant data (the reference file's own 8-channel `CONV1`/`CONV2` shapes, not this project's 6-channel model), and asserted to produce bit-identical output. This is what actually catches a transcription slip, which is the acceptance criterion's real intent.

**Design decisions already effectively locked by earlier phases (confirmed, not re-opened here):**
- Float-multiply requantization (not fixed-point `M0`/shift) — `quantize.py` already computes `requant_mult` as a plain float array and `emit_weights.py` already emits it as `float ... PROGMEM`, matching the parent plan's stated recommendation.
- `dense_int8`'s `skip_requant` semantics: when set, the function writes the raw `int32_t` accumulator (`acc + bias`) directly, ignoring `relu`/`out_zp`/`requant_pgm` entirely — no float-scale/clamp/round path runs at all for that case, matching "ArgMax consumes raw int32 accumulators directly." (Never exercised with `relu=1` simultaneously in this model — final Gemm never has `clamp_min0` — so that combination is documented but not a design fork.)

---

## Files to add/change

```
ctinyml/targets/avr/runtime/
  pgm_read_f32.h                       # REAL — pgm_read_i32/pgm_read_f32: #ifdef __AVR__ real pgm_read_dword path (verbatim) + host fallback (Phase 5's shim, kept)
  qmath.h                              # REAL — requantize()
  conv2d_direct.h                      # REAL — conv2d_int8()
  gemm.h                               # REAL — dense_int8(), skip_requant branch
  argmax.h                             # REAL — argmax_i32() (new, no reference-file source)
tests/
  fixtures/
    reference_kernels_verbatim.h       # NEW — byte-for-byte copy of the reference file's conv2d_int8/dense_int8 bodies (host-portable include swap only), test-only
    runtime_shim_ported.c              # NEW — non-static ctypes-callable wrappers around the real runtime/*.h
    runtime_shim_reference.c           # NEW — same wrappers around reference_kernels_verbatim.h
  test_runtime.py                      # NEW
```

## Runtime header implementations

**`pgm_read_f32.h`:**
```c
#ifdef __AVR__
#include <avr/pgmspace.h>
static inline int32_t pgm_read_i32(const int32_t *p) { return (int32_t)pgm_read_dword(p); }
static inline float pgm_read_f32(const float *p) {
    uint32_t bits = pgm_read_dword((const uint32_t *)p);
    float f; memcpy(&f, &bits, sizeof(f)); return f;
}
#else
/* Phase 5's host-portability shim, unchanged: no separate flash address
 * space on host, so PROGMEM is a no-op and pgm_read_* are plain reads. */
#define PROGMEM
static inline int32_t pgm_read_i32(const int32_t *p) { return *p; }
static inline float pgm_read_f32(const float *p) { return *p; }
#endif
```

**`qmath.h`:** `requantize()` lifted verbatim (round via `±0.5f`, saturate `[-128,127]`).

**`conv2d_direct.h`:** `conv2d_int8()` lifted near-verbatim — direct nested-loop, bounds-check-`continue` zero-padding (no padded buffer), int32 accumulate with `(in_val - input_zp)`, per-output-channel bias + `pgm_read_f32`'d requant multiplier, `relu` flag folds the clamp into requantization exactly as in the reference file.

**`gemm.h`:** `dense_int8()` lifted near-verbatim, extended with `uint8_t skip_requant` + `void *output`: `if (skip_requant) { ((int32_t*)output)[j] = acc; } else { ((int8_t*)output)[j] = requantize(scaled, out_zp); }`.

**`argmax.h`:** new — linear scan for the max, first-occurrence tie-break (matches `numpy.argmax`'s default and this model's `select_last_index=0`), returns `uint8_t`.

## Test harness (`tests/test_runtime.py`)

Module-scoped fixtures compile+load two DLLs once per test session (via host `gcc -shared -Wl,--export-all-symbols`, skipping with a clear reason if `gcc` isn't found — same pattern as Phase 5's compile-check):
- `ported_lib` — `runtime_shim_ported.c` (`#include`s the real headers).
- `reference_lib` — `runtime_shim_reference.c` (`#include`s `reference_kernels_verbatim.h`).

ctypes `argtypes`/`restype` declared precisely per the pinned signatures (`ctypes.c_int8`, `c_uint8`, `c_int32`, `c_float`, `POINTER(...)`) — float32 throughout, never `c_double`, to avoid a float32-vs-Python-float64 precision mismatch making an exact-equality assertion flaky (the Phase 4 lesson generalized: don't trust default type coercion without checking). All kernel outputs here are integers (`int8`/`int32`), so exact `==` assertions are safe and used throughout — no float tolerance needed anywhere in this file.

**Hand-computed tests** (`ported_lib`), two cases for `conv2d_int8` covering different things:
- *Case A* — stride=1, pad=0 (no boundary math), 5×5 input `1..25`, 3×3 all-ones kernel, `input_zp=0`, `requant_mult=0.5`, `out_zp=0`: exercises plain rounding (several exact `.5` cases) without saturation. Hand-computed expected `3×3` output: `[[32,36,41],[54,59,63],[77,81,86]]`.
- *Case B* — stride=2, pad=1 (the real model's actual config), 4×4 input `1..16`, 3×3 all-ones kernel, `input_zp=2`, `out_zp=3`, `requant_mult=1.0`: exercises zero-point handling on both sides plus the stride-2/pad-1 boundary math (corner/edge/full windows), with padding taps contributing **zero** (bounds-check skip, not `0 - input_zp`) — the correct quantized-conv zero-padding semantics. Hand-computed expected `2×2` output: `[[9,21],[48,84]]`.
- `dense_int8` — two cases sharing input `[1,2,3,4]`, `input_zp=1`, weight rows `[1,1,1,1]`/`[2,0,0,0]`, bias `[0,-100]`: (a) normal path, `relu=1` → `[6, 0]` (second channel's negative pre-scale clamped by ReLU); (b) `skip_requant=1` → raw `int32` `[6, -100]`, unclamped.
- `argmax_i32([10, 50, -3, 7, 50], n=5) == 1` (first occurrence of the max).
- `requantize()` directly: half-away-from-zero rounding (`2.5→3`, `-2.5→-3`), saturation (`200.0→127`, `-200.0→-128`), zero-point offset (`10.4,zp=5→15`).
- `pgm_read_i32`/`pgm_read_f32` (host path): trivial round-trip reads.

**Regression test** (`ported_lib` vs `reference_lib`): reference file's own `CONV1` shape (`1→8ch, 3×3, s2, p1`, `28×28→14×14`) with synthetic deterministic weight/bias/requant arrays (e.g. `np.arange`-derived, seeded) fed identically to both; assert bit-identical `int8` output arrays. Repeated for `CONV2` (`8→8ch`) and for `dense_int8`.

**Bonus integration check** (not in the parent plan's literal acceptance text, but directly enabled by this phase landing and cheap to add): compile the real Phase 5 `generate_model_c(...)` output together with the real runtime headers into one DLL, call `model_forward()` via ctypes on a real quantized MNIST input, assert the returned class is in `[0, 9]`. Bridges Phase 5 and Phase 6 concretely before Phase 7's full accuracy validation.

**AVR-real path:** `#ifdef __AVR__` branch in `pgm_read_f32.h` — written, verified by manual line-by-line comparison against the reference file, explicitly **not** exercised by any test here (no `avr-gcc`). Noted as a follow-up once the toolchain is available or Phase 6 hardware bring-up happens.

---

## Verification

1. `conda run -n ctinyml pytest -v` — all Phase 0–5 tests (re-run in full, per lesson #1 above — confirms no signature drift broke Phase 5's generated call-sites) plus the new Phase 6 tests green.
2. Manual sanity: run the bonus integration DLL call directly and print the returned class index.
3. Manual line-by-line diff of `qmath.h`/`conv2d_direct.h`/`gemm.h`'s core logic against `context/mnist_conv_dense_int8_avr.c`.
4. Append a `context/known_issues.md` entry for anything found during implementation (established pattern from every prior phase).

Once green, this gates the start of Phase 7 (End-to-end validation) per the parent plan's sequential execution order — not part of this subplan.
