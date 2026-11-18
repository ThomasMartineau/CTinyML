# CTinyML — Implementation Plan (ONNX→C, AVR-first)

**Target:** Arduino Mega 2560 (ATmega2560) — 248 KB usable flash, 8 KB SRAM, no FPU, no dynamic allocation.
**Input model:** `mnist_classifier.onnx` — (originally Div(norm) →; now a float [0, 1] input, see note below) Conv(6,1,3,3,s2,p1) → ReLU → Conv(6,6,3,3,s2,p1) → ReLU → Reshape → Gemm(10,294) → Reshape → ArgMax.
**Confirmed architecture decisions** (see summary below) are locked; this plan sequences implementation for Claude Code to execute against, phase by phase, each with its own acceptance test before moving on.

---

## Naming & target extensibility

Repo/package is **CTinyML**, not AVR-specific — the ATmega2560 is the first target, not the only one. To keep that door open without over-engineering for a target we're not building yet, the module layout below draws one line: everything through the buffer planner (frontend on `onnx_ir`, structural passes, the closed IR, quantization) is target-agnostic and operates on the graph representation alone; only codegen and the runtime library know about AVR specifics (register width, `PROGMEM`, `pgm_read_*`, no-FPU fixed-point). Those live under a `targets/avr/` namespace so a second target later is an additional sibling directory, not a refactor of the front half of the pipeline. No abstract "target interface" is being built yet — that's premature until there's a second real target to generalize from — but the directory boundary keeps the seam honest from day one.

---

## Reference implementation: `mnist_conv_dense_int8_avr.c`

Before Phase 5/6 begin, Claude Code should read this file first — it's a hand-written, already-reasoned-through sketch of exactly the runtime this project is building, and it settles several implementation details that would otherwise need re-deriving from scratch. Treat it as the structural ground truth for the AVR runtime library, not just inspiration.

**What it already gets right, and where each piece lands in the module layout:**

| Pattern in the reference file | Lands in |
|---|---|
| `pgm_read_i32` / corrected `pgm_read_f32` (`pgm_read_dword` + `memcpy`, never `pgm_read_byte` on floats) | `targets/avr/runtime/pgm_read_f32.h` — verbatim |
| `requantize()` — round-to-nearest via `+0.5f`/`-0.5f`, saturate to `[-128,127]` | `targets/avr/runtime/qmath.h` |
| `conv2d_int8()` — direct nested-loop conv, zero-pad via bounds-check `continue` (no padded buffer), int32 accumulate, per-output-channel bias + requant multiplier, `relu` flag folds clamp into requantization | `targets/avr/runtime/conv2d_direct.h` |
| `dense_int8()` — same accumulate/requant/relu pattern for Gemm | `targets/avr/runtime/gemm.h` |
| `buf_a` / `buf_b` static ping-pong sized to the largest cross-layer activation, no malloc | This *is* Phase 4's buffer planner output — the reference file hand-computed what the planner needs to derive automatically |
| `model_forward()` — top-level call sequence through the buffers | This *is* what `assemble.py` (Phase 5) needs to generate |

**Important structural point — this changes how codegen should work:** the reference file uses **one shared, runtime-parameterized kernel function per op type** (`conv2d_int8`, `dense_int8`), called multiple times with different pointers/dimensions as *arguments* — not a separately monomorphized, fully-unrolled function per layer. This is a better default than the "compile-time `#define` shapes, monomorphized per layer" framing from the earlier design discussion, for two concrete reasons specific to this project: (1) flash is the tighter budget (248 KB) and a shared function reduces code size versus duplicating near-identical loop bodies per layer; (2) with only two conv layers and one Gemm here, the runtime-parameter overhead per call is negligible next to the accumulate loop itself. **Supersedes the earlier monomorphization framing — adopt the reference file's shared-function pattern as the canonical runtime kernel design.** Phase 5 and Phase 6 below are written to reflect this.

**Two gaps to fill, since the reference file predates the ArgMax-terminated, 2-conv/1-gemm architecture:**
1. `dense_int8` in the reference file always requantizes to int8. Phase 3's `skip_requant` decision (final Gemm feeds ArgMax directly on int32 accumulators) needs a variant — either a `skip_requant` flag on `dense_int8` that writes to an `int32_t*` output instead of `int8_t*`, or a separate thin wrapper. Prefer the flag: keeps one function, matches the existing `relu` flag's style.
2. `argmax.h` doesn't exist in the reference file yet (it terminates at raw logits) — new, trivial, per Phase 6 below.

**Templating approach for codegen (Phase 5):** given the runtime kernels are shared functions, not per-layer generated code, `emit_conv2d.py` / `emit_gemm.py` don't need a general templating engine (Jinja2, etc.) — their job is narrow: emit the PROGMEM weight/bias/multiplier arrays for that layer, and emit one call-site line into the shared kernel with the right pointers, dimensions, and flags, mirroring `model_forward()`'s existing style line-for-line. Plain Python f-strings against a fixed call-site format (matching the reference file's own layout) are sufficient — pulling in a template engine would be solving a problem this project doesn't have. `assemble.py` stitches these call-sites into `model_forward()` in topological order exactly as the reference file demonstrates by hand.

---

## 0. Locked design decisions (context for every phase)

- Flat arrays, compile-time `#define` shapes, monomorphized per layer — no generic tensor struct at runtime.
- Direct convolution (no im2col).
- ReLU fused into the preceding Conv/Gemm IR node as a `clamp_min0` flag — not a separate IR op.
- `Div` (input normalization) folded into layer-1's quantization scale at compile time — not emitted as a runtime op.
- `Reshape` nodes are compile-time no-ops (aliasing, no data movement) when the flatten is row-major-compatible.
- Final Gemm feeding `ArgMax` skips int8 requantization — ArgMax consumes raw int32 accumulators directly (monotonic under shared per-tensor affine output scale).
- Per-channel requantization multipliers (M0, shift) precomputed offline in Python.
- Weights in `PROGMEM` as int8; float scalars via corrected `pgm_read_dword` + `memcpy` (never raw `pgm_read_byte` on floats).
- CSR sparse weight encoding evaluated only above a weight-count threshold (skip both conv layers, evaluate on the Gemm/fc layer only).
- Liveness/refcount-based buffer planner, ping-pong static buffers sized to the largest intermediate activation.

---

## Phase 0 — Repo scaffold

**Tasks**
- Create `ctinyml/` package with the module layout below (stub files, no logic yet).
- Add `onnx-ir` (PyPI: `onnx-ir`) as a dependency alongside `onnx` — it owns graph loading and the Phase 1–2 passes; see Phases 1–2 below for why.
- Set up `pytest`, a `tests/` dir, and a `fixtures/` dir to hold the uploaded ONNX model plus any hand-built minimal graphs for unit tests.
- Add a `tools/` dir for a CLI entrypoint: `python -m ctinyml compile model.onnx --target avr -o build/`. The `--target` flag is required from day one (default-and-only value `avr` for now) so adding a target later is additive, not a breaking CLI change.

```
ctinyml/
  closed_ir.py              # target-agnostic — our narrow post-fusion IR (Phase 3 onward)
  frontend/
    load.py                 # onnx_ir.load() wrapper + static-shape/op-set validation
  passes/                   # target-agnostic
    onnx_ir_passes/         # operate on onnx_ir.Graph (Phase 2, pre-conversion)
      fold_div_into_quant.py
      drop_noop_reshape.py
      fuse_relu.py
    convert_to_closed_ir.py # onnx_ir.Graph -> closed_ir.Graph, the Phase 2/3 boundary
    quantize.py              # operate on closed_ir.Graph (Phase 3 onward)
    drop_final_requant.py
    buffer_planner.py
    sparsity_analysis.py
  layout.py                 # target-agnostic (shape/stride macro derivation)
  targets/
    avr/
      codegen/
        emit_weights.py
        emit_conv2d.py
        emit_gemm.py
        emit_argmax.py
        assemble.py
      runtime/
        pgm_read_f32.h
        qmath.h
        conv2d_direct.h
        gemm.h
        argmax.h
tests/
  fixtures/
tools/
  cli.py
```

**Acceptance:** `pytest` runs (0 tests, green), CLI stub prints usage, `import onnx_ir` succeeds.

---

## Phase 1 — ONNX frontend (via `onnx_ir`)

**Tasks**
- `frontend/load.py`: `onnx_ir.load()` the model file directly — no hand-rolled protobuf parsing. This also resolves the external-data split (`.onnx` + `.onnx.data`) that caused friction earlier; `onnx_ir` handles that natively rather than us staging files under an expected filename by hand.
- Run `onnx.shape_inference` (still via the `onnx` package, ahead of `onnx_ir.from_proto`, since shape inference itself isn't something `onnx_ir` performs) and assert every intermediate tensor has a fully static shape (batch dim included — reject dynamic batch; we know it's always 1 on-device).
- Validate op set against the supported list: `{Div, Conv, Relu, Reshape, Gemm, ArgMax}` by walking `graph.all_nodes()` / iterating `node.op_type` on the loaded `onnx_ir.Graph`. Anything else → hard error naming the unsupported op.

**Acceptance:** Load `mnist_classifier.onnx` (or the consolidated `mnist_classifier_full.onnx`), assert the `onnx_ir.Graph` has 9 nodes in the exact topological order from the earlier graph dump, all shapes static. Unit test with a hand-built 2-node ONNX graph (Conv+Relu) as a second fixture, independent of the MNIST model, to avoid the frontend tests depending on external-data weights being present.

---

## Phase 2 — Structural passes (via `onnx_ir.passes`, pre-quantization)

Each of the three passes below is a graph-rewrite operation — remove or replace a node, rewire producers/consumers — which is exactly what `onnx_ir`'s documented transformation patterns ("replace a node block with a new block," "replace all downstream uses of a value") and its `PassBase`/`Sequential` runner exist for. Implement each as an `onnx_ir.passes.PassBase` subclass operating directly on the `onnx_ir.Graph` from Phase 1, run via `onnx_ir.passes.Sequential`, rather than hand-rolling topological rewrite logic — the library's `DoublyLinkedSet`-backed node container already guarantees safe mutation during iteration, which is exactly the class of bug (dangling references mid-rewrite) this kind of pass is prone to if written from scratch.

Order matters — each pass should still be independently testable on a small synthetic graph, not just the full MNIST graph.

1. **`drop_noop_reshape.py`** — remove `Reshape` nodes where output shape is a row-major-compatible flatten/unflatten of the input; rewire producer→consumer directly. Assert (don't silently skip) if a Reshape is *not* eligible — we want a loud failure if a future model needs a real data-movement reshape, not a silent miscompile.
2. **`fuse_relu.py`** — for every `Relu` whose sole input producer is `Conv` or `Gemm`, delete the `Relu` node and set `clamp_min0=True` on the producer (as an `onnx_ir` node attribute, carried through to Phase 3's conversion). Assert every `Relu` in the graph gets fused (fail loudly on an unfused one — that signals a graph shape this compiler doesn't support yet, e.g. branching).
3. **`fold_div_into_quant.py`** — detect the `Div(input, scalar)` pattern at the graph input, record the scalar as `input_prescale` graph metadata, remove the node, rewire. This pass only *records* the value; actual folding into layer-1's quant scale happens in Phase 3.

**Phase 2→3 boundary — `convert_to_closed_ir.py`:** after the three passes above, the `onnx_ir.Graph` is down to 4 nodes (`Conv(clamp_min0) → Conv(clamp_min0) → Gemm → ArgMax`) and has no further need for `onnx_ir`'s generality (subgraphs, control flow, multi-device sharding, the full ONNX op spec) — none of which this compiler's backend will ever touch. This single conversion function walks the reduced `onnx_ir.Graph` once and builds our own minimal `closed_ir.Graph` (`Tensor`/`Op`/`Graph` dataclasses as defined earlier: flat `attrs` dict, quantization metadata living directly on `Tensor`). Everything from Phase 3 onward operates on `closed_ir.Graph`, not `onnx_ir.Graph` — this is a deliberate, load-bearing boundary: upstream stays general-purpose and battle-tested, downstream stays narrow and easy to reason about for quantization/buffer-planning/codegen.

**Acceptance:** Run the three `onnx_ir` passes + conversion on the MNIST graph; result is a 4-node `closed_ir.Graph` matching the structure above, with `input_prescale` recorded on the graph metadata. Snapshot test the resulting `closed_ir.Graph` (serialize to a readable text form and diff against a checked-in golden file) so any future pass-ordering regression is caught immediately.

---

## Phase 3 — Quantization

**Tasks**

- **Float reference interpreter (build this first, share it with Phase 7):** a small numpy-based interpreter that runs the 4-op IR (post-Phase-2: `Conv(clamp_min0) → Conv(clamp_min0) → Gemm → ArgMax`) forward in float32. This is needed twice — once here to collect activation calibration statistics, once in Phase 7 to produce the golden float baseline the C implementation gets compared against — so build it once as shared infrastructure rather than duplicating the forward-pass logic in two places.

- **Calibration data interface:** `quantize.py` accepts calibration data as a plain iterable of preprocessed input arrays (e.g. a numpy array of shape `[N, 1, 28, 28]`, or any generator yielding individual/batched inputs) — no custom reader class or `get_next()` protocol. This deliberately doesn't mirror ONNX Runtime's `CalibrationDataReader`: that shape exists to stream data across ORT's C++ FFI boundary, which doesn't apply here — the calibration set is a few hundred MNIST images running through an in-process Python interpreter, so a `for batch in calibration_data:` loop over any iterable is the whole interface.

- **Weight quantization** — no calibration data needed, computed directly from the static weight tensors already in the IR:
  - Per-channel scale for each Conv/Gemm output channel, **symmetric** (zero-point fixed at 0) — matches the reference file's convention and needs no zero-point correction term in the MAC loop (`acc += w * (in_val - input_zp)` only subtracts on the activation side).
  - int8 weight values; int32 bias (`bias_scale = input_scale × weight_scale`, standard convention).

- **Activation quantization** — requires calibration, via the float interpreter above:
  - Run every calibration input through the float interpreter, tracking running min/max per activation tensor (the input, and each Conv/Gemm's pre-quantized output) across all batches.
  - Derive scale/zero-point per tensor from the aggregated min/max range, **asymmetric** (zero-point free, not pinned to 0). This matters specifically because every activation in this graph is post-ReLU except the input — non-negative values quantized symmetrically waste half the int8 range (-128 to -1 never used). Asymmetric activation quant + symmetric weight quant is the standard split and is what the reference file's `input_zp`/`mid_zp`/`out_zp` parameters already assume.
  - Default calibration statistic is plain running min/max — simplest, deterministic, matches "correctness over theory." Only reach for percentile clipping or entropy/KL-divergence calibration (ONNX Runtime's default for larger models) if Phase 7's ≥95% argmax-match acceptance test doesn't clear with min/max; that's an escalation path, not a starting assumption, for a model this small.

- Fold `input_prescale` (recorded in Phase 2) into layer-1's input scale so no runtime division ever occurs.

- Requantization multiplier per output channel: **plain float `acc * requant_mult`**, matching the reference file's `conv1_requant`/`conv2_requant` arrays and `requantize()` function — not a fixed-point `(M0, shift)` pair. Reconciling with Phase 6: AVR does float multiply in software regardless, so a fixed-point path saves nothing here and only adds a second numeric path to validate; keep fixed-point as a later optimization to revisit only if profiling shows it's worth it.

- `drop_final_requant.py`: mark the last Gemm (the one feeding ArgMax) as `skip_requant=True` — its int32 accumulator is emitted directly as the op's output, no scale/zero-point needed for that tensor at all.

**Acceptance:** Unit tests on synthetic per-channel weight tensors verifying reconstructed float multipliers are correct within a tight tolerance across a range of scales (including small and near-1.0 cases). Separately, a calibration test: feed a small synthetic calibration set with a known min/max through the float interpreter, verify the derived activation scale/zero-point exactly reproduces that range at the int8 boundaries (0 and 255 for asymmetric, i.e. the calibrated min maps to qmin and max maps to qmax). Run the float interpreter itself against a hand-computed 2×2 input through one conv layer as a sanity check before trusting it for calibration or Phase 7 validation.

---

## Phase 4 — Buffer planner

**Tasks**
- `buffer_planner.py`: liveness analysis over the (now 4-node, quantized) IR — for each tensor, compute first-def/last-use op index.
- Ping-pong static allocation: two fixed buffers sized to the largest intermediate activation across the whole graph; assign each intermediate tensor to buffer A or B alternating by op, verified against liveness so no buffer is overwritten while still live.
- Emit total SRAM budget report (weights are PROGMEM/flash, so this is purely activation buffers + any small int32 accumulator scratch) and hard-fail compilation if it exceeds the 8 KB budget (leave headroom — target well under, since stack/globals need room too; flag if usage exceeds e.g. 6 KB as a warning even if technically under 8 KB).

**Acceptance:** Run on the MNIST graph, manually verify by hand the two largest activations (14×14×6 = 1176 bytes post-conv1, 7×7×6 = 294 bytes post-conv2) are what sizes the ping-pong buffers, and that the planner doesn't allocate a third buffer.

---

## Phase 5 — AVR codegen backend (`targets/avr/codegen/`)

**Tasks**
- `layout.py`: emit `#define` shape macros per tensor (`CONV1_OUT_CH`, `CONV1_OUT_H`, etc. — purely descriptive/documentation and for sizing static buffer declarations, *not* for monomorphizing per-layer kernel code), and the ping-pong buffer declarations from Phase 4, matching `buf_a` / `buf_b` in the reference file.
- `emit_weights.py`: emit int8 weight arrays and int32 bias arrays as `const ... PROGMEM`, plus per-channel `(M0, shift)`-derived requant multiplier tables (skip the multiplier table for the final Gemm per `skip_requant`) — same declaration style as `conv1_weight` / `conv1_bias` / `conv1_requant` in the reference file.
- `emit_conv2d.py`: emit **one call-site line per conv layer into the shared `conv2d_int8()` kernel** (reference file signature), passing that layer's dims, PROGMEM pointers, `input_zp`/`out_zp`, and `relu`/`clamp_min0` flag as arguments — not a separate generated function per layer. **Priority correctness item carries over unchanged:** unit test the stride-2/pad-1 boundary math explicitly (output index 0 and output index (dim-1) for both conv layers' actual dimensions, 28→14 and 14→7) against the shared kernel in `conv2d_direct.h`.
- `emit_gemm.py`: same pattern — call-site into the shared `dense_int8()`, with the `skip_requant` flag wired through for the final layer (writing to an `int32_t*` output rather than `int8_t*`, per the gap noted above).
- `emit_argmax.py`: call site into `argmax.h` operating on the final layer's int32 output.
- `assemble.py`: stitch the call-sites into `model_forward()` in topological order — this is Phase 5's core deliverable, and should read as a generated version of the reference file's `model_forward()` function, just with the real layer count/shapes/pointers substituted in via the IR rather than hand-written.

**Acceptance:** Generated `model.c` compiles under `avr-gcc -mmcu=atmega2560` against the real weight arrays (now available). Diff the generated `model_forward()` call sequence against the reference file's structure by eye — it should be recognizably the same shape, just data-driven. Check flash-size estimate from `avr-size` against the 248 KB budget.

---

## Phase 6 — AVR device runtime library (`targets/avr/runtime/`, hand-written, not generated)

**Tasks — this phase is largely porting the reference file, not designing from scratch:**
- `pgm_read_f32.h`: lift `pgm_read_i32` and the corrected `pgm_read_f32` directly from the reference file.
- `qmath.h`: lift `requantize()` directly (round via `±0.5f`, saturate to `[-128,127]`). Note it currently takes a pre-scaled float (`acc_scaled = acc * requant_mult`) rather than a fixed-point `(M0, shift)` pair — decide during this phase whether to keep the float-multiply path as-is (simpler, matches the reference file, and AVR does float multiply in software either way per the file's own header comment) or switch to integer fixed-point `(acc * M0) >> shift` (avoids the float unit entirely). The reference file's own reasoning favors keeping float since the multiply is already software-emulated — recommend keeping it and treating fixed-point as an optimization to revisit only if profiling shows it's worth it, not a default.
- `conv2d_direct.h`: lift `conv2d_int8()` near-verbatim (shared runtime-parameterized kernel, not per-layer unrolled — see above).
- `gemm.h`: lift `dense_int8()` near-verbatim, extended with the `skip_requant` flag → `int32_t*` output path for the final layer.
- `argmax.h`: new — trivial `int32[N] → uint8` index, consuming `dense_int8`'s `skip_requant` output directly.

**Acceptance:** Standalone unit tests for `conv2d_direct.h` and `gemm.h` compiled and run on the host (x86, not AVR) against hand-computed small examples (e.g. a 1-channel 5×5 input, known 3×3 kernel, verify output matches a numpy reference) before ever touching AVR-specific code. Since this phase is mostly a port, also add a direct regression test: feed the reference file's own conv1/conv2 shapes through the ported `conv2d_int8()` and confirm identical output to the original file compiled standalone — this catches any transcription slip during the port. This isolates kernel-logic bugs from AVR-toolchain/memory-model issues.

---

## Phase 7 — End-to-end validation

**Tasks**
- Full pipeline run producing `model.c` against the real weights (`mnist_classifier_full.onnx`). Cross-check against the **Phase 3 float reference interpreter** (not a fresh PyTorch/ORT comparison — reuse the same interpreter calibration used, so any mismatch is attributable to quantization itself rather than to two independently-written forward passes disagreeing) on a handful of real MNIST digit images: compare predicted class (post-ArgMax) for exact match, and compare pre-quantization float logits vs. dequantized int32 accumulators for reasonable numerical closeness (not exact, due to quantization).
- On-device smoke test on actual ATmega2560 hardware if available: flash, feed a known digit, confirm predicted class over serial.

**Acceptance:** ≥95% match rate on a small held-out MNIST batch between the C implementation's argmax and the float interpreter's argmax (some divergence expected from quantization at digit-decision boundaries).

---

## Phase 8 — Stretch: CSR sparsity for the Gemm layer

Deferred until Phases 1–7 are solid. `sparsity_analysis.py` computes the break-even nonzero ratio for the fc layer only (2940 weights) — CSR row-ptr/col-idx overhead vs. dense storage savings — and only emits CSR-encoded weights if the pruned model clears that threshold.

---

## Suggested execution order for Claude Code

Sequential, each phase gated on its own tests passing before starting the next:
**0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8.**

Real weight data is now available (`mnist_classifier_full.onnx`, weights embedded, no external-data split needed) — Phases 3, 5, and 7 are unblocked and can run against real values from the start rather than synthetic placeholders. Phase 8 (CSR sparsity) remains a stretch goal and — per the weight inspection above — won't do anything on *this* model specifically, since all three weight tensors are 100% dense; keep it for toolkit generality against future unstructured-pruned models.

**Before starting Phase 5 or 6, read `mnist_conv_dense_int8_avr.c` in full.** It is not background context — it is the concrete target shape for both phases' output, and the "Reference implementation" section above maps each of its functions to the module that either lifts it verbatim or generates equivalent call-sites into it.
