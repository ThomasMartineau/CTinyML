# Subplan: Phase 5 — AVR codegen backend

## Context

Phases 0–4 are implemented and green. `ctinyml/passes/buffer_planner.py` leaves the fully-quantized MNIST `closed_ir.Graph` with every activation tensor assigned to one of two 1176-byte ping-pong buffers (`conv2d`→buffer 0, `conv2d_1`→buffer 1, `linear`→buffer 0), every weight/bias quantized per-channel, and per-op `requant_mult` arrays. Phase 5 turns that into an actual `model.c`: `#define` shape/zero-point macros, `PROGMEM` weight/bias/requant arrays, and a `model_forward()` function built from call-sites into the (Phase-6-owned) shared runtime kernels — reading as a generated version of `context/mnist_conv_dense_int8_avr.c`'s own `model_forward()`, per the parent plan's explicit acceptance bullet.

**Toolchain check (done now, changes the plan):** `avr-gcc`/`avr-size` are **not installed** in this environment (confirmed: not on `PATH`). Per your choice, verification uses **host gcc instead** (MinGW-W64 13.2.0, confirmed present) against **minimal stub runtime headers** — function prototypes only, no real kernel bodies (that's Phase 6's job) — to check the generated `model.c` is syntactically/type-correct (right pointer types, array sizes, argument counts) without needing the AVR cross-compiler. The literal `avr-gcc -mmcu=atmega2560` + `avr-size` flash-budget check from the parent plan's acceptance text is **explicitly deferred**, not silently dropped — revisit once the toolchain is installed or Phase 6 lands.

**Op-dispatch architecture decision (made mid-implementation, changes the module layout below):** the first draft of `emit_layout.py` had an `if op.op_type == "Conv": ... elif == "Gemm": ...` block — the same *shape* of dispatch already present in `interpreter.py`'s `match op.op_type` from Phase 3. Rather than let that pattern re-multiply across `emit_layout.py`, `emit_conv2d.py`/`emit_gemm.py`/`emit_argmax.py`, and `assemble.py` independently, this phase introduces **per-op-type registries** (mirroring onnxruntime's kernel-registry pattern), split across the same target-agnostic/AVR-specific boundary the rest of the project already respects:

- **`ctinyml/ops.py` (new, target-agnostic):** one `Kernel` class per op type (`ConvKernel`, `GemmKernel`, `ArgMaxKernel`), each owning that op's shape derivation and float32 forward evaluation, in a `KERNELS: dict[str, Kernel]` registry. `interpreter.py`'s `run()` becomes `KERNELS[op.op_type].forward(op, inputs)` instead of a `match` statement. The actual numeric bodies (what were `interpreter._conv2d`/`_gemm`/`_argmax`) **move into `ops.py`** (as the kernels' `forward` implementations) so `ops.py` has no dependency on `interpreter.py` — `interpreter.py` depends on `ops.py`, not the other way around, avoiding a circular import between "the registry" and "the thing that used to own the functions." `test_interpreter.py`'s existing hand-computed tests get a one-line import change (`from ctinyml.ops import ...` instead of `from ctinyml.interpreter import ...`); their assertions are untouched.
- **`targets/avr/codegen/` (new, AVR-specific):** a parallel `Emitter` per op type (`ConvEmitter` in `emit_conv2d.py`, `GemmEmitter` in `emit_gemm.py`, `ArgMaxEmitter` in `emit_argmax.py` — same filenames Phase 0 already scaffolded, now holding a class instead of a bare function), each owning **both** `.layout_macros(op, ctx)` (the `#define` block) **and** `.call_site(op, ctx)` (the C call-site line) for that op type — directly what you asked for: emission and layout colocated per operator. A small `EMITTERS: dict[str, Emitter]` registry (`targets/avr/codegen/registry.py`) is built from all three and used by both `emit_layout.py` (loops ops, calls `.layout_macros`) and `assemble.py` (loops ops, calls `.call_site`) — one lookup site each, not a branch re-derived in every file.

What stays as a flat dataclass, deliberately: `closed_ir.Op`/`Tensor` themselves are **not** subclassed. They're built generically by `convert_to_closed_ir.py` and mutated generically by `buffer_planner.py` (which never branches on `op_type` at all) — forcing them into a class hierarchy would reintroduce a type-dispatch at construction time and blur the target-agnostic/AVR boundary (an `Op` subclass can't cleanly own both an AVR-specific `.call_site()` and the target-agnostic interpreter's `.forward()` without one layer leaking into the other). The registries live *beside* the IR, keyed by the IR's plain `op_type` string, not *inside* it.

**Real signatures pinned down now** (Phase 6 will implement these for real; Phase 5 needs the exact call-site shape to generate code against):
- `conv2d_int8(...)` — used as-is, verbatim from the reference file's signature; no changes needed since every `Conv` in this graph requantizes.
- `dense_int8(...)` — needs the `skip_requant` variant the parent plan flags as a gap: adding a `uint8_t skip_requant` flag and widening `output` to `void *` (cast internally to `int8_t*` or `int32_t*` depending on the flag) — "prefer the flag: keeps one function, matches the existing `relu` flag's style," per the parent plan. `requant_pgm`/`out_zp`/`relu` are simply unused (pass `NULL`/`0`) when `skip_requant=1`.
- `argmax_i32(const int32_t *logits, uint16_t n) -> uint8_t` — new, trivial, per the parent plan's Phase 6 gap note.
- `model_forward()`'s signature is **simplified from the reference file's**, not copied verbatim: the reference file takes `input_zp`/`mid_zp`/`out_zp` as *runtime parameters* because it was hand-written as a generic template before calibration values existed. This compiler has real calibrated zero-points at compile time for every activation tensor (`input`, `conv2d`, `conv2d_1` — three independent values, not two shared ones like the reference file's simplified `mid_zp`), so they're baked in as `#define`s instead of threaded through as parameters. `model_forward()` also returns `uint8_t` directly (the predicted class) rather than writing to a `logits[]` out-parameter, since this graph's IR already ends at `ArgMax`, unlike the reference file's raw-logits-terminated example. Net signature: `uint8_t model_forward(const int8_t *input);`. The overall **call sequence shape** (`conv2d_int8 → conv2d_int8 → dense_int8 → argmax`, ping-pong buffer reuse) still matches the reference file exactly — that's what the acceptance bullet's eyeball diff is checking.

**Bug found while designing this phase, fixed (done):** `quantize.py`'s `_calibrate_activations` collected `tensors_by_name` from *every* op's outputs, including `ArgMax`'s — so the final `pred` tensor (a discrete class index, never meant to be quantized) was spuriously getting `.scale`/`.zero_point` set and `.dtype` overwritten from `INT64` to `INT8`. Fixed with a named, extensible `NON_CALIBRATED_OPS = {"ArgMax"}` set (not an inline op-type check) so a future non-calibrated op type is a one-line addition. Verified: `graph.outputs[0].dtype == "INT64"`, `.scale is None` after `quantize()`. Needs a regression test added to `test_quantize.py` and an entry appended to `context/known_issues.md`.

**Resolving a real inconsistency between Phase 0's file tree and Phase 5's task text:** Phase 0's stub tree places `layout.py` at the top level of `ctinyml/`, annotated *"target-agnostic (shape/stride macro derivation)"*. Phase 5's task list says *"`layout.py`: emit `#define` shape macros"* — but emitting C preprocessor text is not target-agnostic, and directly contradicts the plan's own stated architecture ("everything through the buffer planner ... is target-agnostic ... only codegen and the runtime library know about AVR specifics"). Resolving in favor of the explicit architecture principle: `ctinyml/layout.py` stays target-agnostic (pure-Python shape-formula helpers + macro-name derivation), and AVR-specific `#define`/buffer-declaration text emission lives in `targets/avr/codegen/emit_layout.py`, which uses `layout.py`'s helpers plus the `ops.py`/`EMITTERS` registries described above.

**Explicitly out of scope for this subplan:** wiring `tools/cli.py`'s `compile` command to actually invoke this pipeline. That needs a calibration-data argument/interface the CLI doesn't have yet and the parent plan's Phase 5 task list doesn't mention CLI completion at all — `assemble.generate_model_c(...)` is tested directly, the same way every prior phase's top-level function has been.

---

## Files to add/change

```
ctinyml/
  ops.py                              # NEW — target-agnostic Kernel classes + KERNELS registry
  interpreter.py                      # REFACTOR — run() dispatches via ops.KERNELS; numeric bodies move into ops.py
  layout.py                           # REAL (was stub) — target-agnostic shape-formula + naming helpers
  passes/quantize.py                  # FIX (done) — NON_CALIBRATED_OPS excludes ArgMax from calibration
  targets/avr/codegen/
    errors.py                         # DONE — CodegenError
    base.py                           # NEW — Emitter Protocol, LayoutContext dataclass
    registry.py                       # NEW — EMITTERS dict, built from emit_conv2d/emit_gemm/emit_argmax
    emit_layout.py                    # NEW — orchestrates #define macros + buf_a/buf_b decls via EMITTERS
    emit_weights.py                   # REAL — PROGMEM weight/bias/requant arrays (plain function, no polymorphism needed)
    emit_conv2d.py                    # REAL — ConvEmitter class (layout_macros + call_site)
    emit_gemm.py                      # REAL — GemmEmitter class (skip_requant-aware)
    emit_argmax.py                    # REAL — ArgMaxEmitter class
    assemble.py                       # REAL — stitches everything into a full model.c string
  targets/avr/runtime/
    pgm_read_f32.h                    # Phase-5 minimal stub: prototypes + host-portability PROGMEM shim
    qmath.h                           # Phase-5 minimal stub: requantize() prototype
    conv2d_direct.h                   # Phase-5 minimal stub: conv2d_int8() prototype
    gemm.h                            # Phase-5 minimal stub: dense_int8() prototype (skip_requant variant)
    argmax.h                          # Phase-5 minimal stub: argmax_i32() prototype
tests/
  test_interpreter.py                 # UPDATE — import _conv2d/_gemm/_argmax from ctinyml.ops instead
  test_quantize.py                    # + regression test for the ArgMax-calibration fix
  test_layout.py                      # NEW
  test_codegen.py                     # NEW
```

All 5 runtime headers get a clear `/* Phase 5 stub: prototype only. Phase 6 replaces this with the real, tested implementation ported from context/mnist_conv_dense_int8_avr.c. */` banner so it's unambiguous they're not the final artifact.

## `ctinyml/ops.py` (target-agnostic)

```python
class Kernel(Protocol):
    op_type: str
    def forward(self, op: closed_ir.Op, inputs: list[np.ndarray]) -> np.ndarray: ...

class ConvKernel:
    op_type = "Conv"
    def output_shape(self, op: closed_ir.Op) -> tuple[int, int, int, int]:
        """Recomputes (n, out_ch, out_h, out_w) from op.inputs/attrs via
        layout.conv_output_dim -- the target-agnostic half of the
        "stride-2/pad-1 boundary math" cross-check; emit_conv2d.py's
        ConvEmitter additionally checks stride/pad symmetry, which is
        an AVR-kernel-signature limitation, not a target-agnostic one."""
    def forward(self, op, inputs) -> np.ndarray: ...  # body = today's interpreter._conv2d

class GemmKernel:
    op_type = "Gemm"
    def forward(self, op, inputs) -> np.ndarray: ...  # body = today's interpreter._gemm

class ArgMaxKernel:
    op_type = "ArgMax"
    def forward(self, op, inputs) -> np.ndarray: ...  # body = today's interpreter._argmax (incl. the size-1-dim squeeze fix)

KERNELS: dict[str, Kernel] = {"Conv": ConvKernel(), "Gemm": GemmKernel(), "ArgMax": ArgMaxKernel()}
```

`interpreter.py`'s `run()` shrinks to the graph-walking loop only: build `inputs`, `kernel = KERNELS.get(op.op_type)` (raise `ValueError` naming the op_type if missing, same as today), `out = kernel.forward(op, inputs)`. Its public signature/behavior (`run(graph, input_array) -> dict[str, np.ndarray]`) is unchanged, so `quantize.py`'s `_calibrate_activations` (its only caller) needs no changes.

## `ctinyml/layout.py` (target-agnostic, unchanged from before)

```python
def conv_output_dim(in_dim: int, k: int, stride: int, pad: int) -> int:
    return (in_dim + 2 * pad - k) // stride + 1

def macro_names(graph: closed_ir.Graph) -> dict[str, str]:
    """op.name -> "CONV1"/"GEMM1"-style prefix, positional counter per op_type."""
```

## `targets/avr/codegen/base.py`

```python
@dataclass
class LayoutContext:
    macro_names: dict[str, str]                       # op.name -> "CONV1" / "GEMM1"
    zero_point_macros: dict[str, str]                  # tensor.name -> "INPUT_ZP" / "CONV2D_ZP" / ...
    tensor_ref: Callable[[closed_ir.Tensor], str]      # tensor -> C expr ("input"/"buf_a"/"(int32_t*)buf_a")

class Emitter(Protocol):
    op_type: str
    def layout_macros(self, op: closed_ir.Op, ctx: LayoutContext) -> list[str]: ...
    def call_site(self, op: closed_ir.Op, ctx: LayoutContext) -> str: ...
```

## `targets/avr/codegen/emit_conv2d.py` (`ConvEmitter`) / `emit_gemm.py` (`GemmEmitter`) / `emit_argmax.py` (`ArgMaxEmitter`)

Each owns, for its op type:
- `layout_macros`: the `#define {PREFIX}_*` block (Conv: `IN_CH/OUT_CH/K/STRIDE/PAD/IN_H/IN_W/OUT_H/OUT_W`, asserting square kernel + symmetric stride/pad against `ops.KERNELS["Conv"].output_shape(op)` — `CodegenError` on mismatch, this is the stride-2/pad-1 boundary check, verified as `28→14`/`14→7`; Gemm: `IN/OUT` from the weight's `(out,in)` shape, asserting `transA==0, transB==1`; ArgMax: no macros, empty list) plus any zero-point `#define` for that op's requantized output.
- `call_site`: one f-string-formatted call-site line against the pinned signatures — `conv2d_int8(...)` / `dense_int8(...)` (passing `skip_requant=1, out_zp=0, relu=0, requant_pgm=NULL` when `op.attrs.get("skip_requant")`) / `argmax_i32(...)`. No template engine, matching the parent plan's explicit guidance.

## `targets/avr/codegen/registry.py`

```python
EMITTERS: dict[str, Emitter] = {"Conv": ConvEmitter(), "Gemm": GemmEmitter(), "ArgMax": ArgMaxEmitter()}
```

## `targets/avr/codegen/emit_layout.py`

Orchestrates: `IMG_H`/`IMG_W` + input `#define ..._ZP`, then loops `graph.ops` calling `EMITTERS[op.op_type].layout_macros(op, ctx)` (building `ctx.zero_point_macros`/`macro_names` incrementally as it goes), then the `buf_a`/`buf_b` declarations sized from `BufferPlan.buffer_size`. Builds and returns the `LayoutContext` (its `tensor_ref` closure: `"input"` for the graph input, `"buf_a"`/`"buf_b"` for `tensor.buffer == 0/1`, cast to `(int32_t*)` when `tensor.dtype == "INT32"`) alongside the emitted text.

## `targets/avr/codegen/emit_weights.py`

Stays a plain function (no per-op-type polymorphism needed — flattening a quantized `Tensor.data` into a C array initializer is identical logic regardless of whether the tensor came from a Conv or a Gemm, only the size differs). Per `Conv`/`Gemm` op: `static const int8_t {name}_weight[N] PROGMEM = {...};`, `static const int32_t {name}_bias[N] PROGMEM = {...};`, and (skipped when `op.attrs.get("skip_requant")`) `static const float {name}_requant[N] PROGMEM = {...};` — values from `Tensor.data.flatten()` (already row-major).

## `targets/avr/codegen/assemble.py`

```python
def generate_model_c(graph: closed_ir.Graph, buffer_plan: BufferPlan) -> str:
    """emit_layout.generate(...) + emit_weights(...) + one
    EMITTERS[op.op_type].call_site(op, ctx) per op (graph.ops order,
    already topological), closing with
    `uint8_t model_forward(const int8_t *input) { ...; return <argmax call-site>; }`."""
```

---

## Tests

### `tests/test_interpreter.py` (updated, not rewritten)
Only the import line changes (`from ctinyml.ops import _conv2d, _gemm, _argmax` — or however the refactor exposes them for direct testing, e.g. as free functions `ops.py` re-exports or as methods; whichever keeps every existing assertion in `test_conv2d_hand_computed`/`test_gemm_hand_computed`/`test_argmax_hand_computed`/`test_full_graph_forward_mnist` untouched). Full re-run required to confirm zero regression from the refactor.

### `tests/test_quantize.py` (+1 test)
- `test_quantize_excludes_argmax_output` — after `quantize()` on the real MNIST graph, `graph.outputs[0].dtype == "INT64"` and `.scale is None`.

### `tests/test_layout.py`
- `test_conv_output_dim_matches_real_model` — `conv_output_dim(28, 3, 2, 1) == 14`, `conv_output_dim(14, 3, 2, 1) == 7`.
- `test_macro_names_on_mnist` — `{"node_conv2d": "CONV1", "node_conv2d_1": "CONV2", "node_linear": "GEMM1"}`.

### `tests/test_codegen.py`
Full pipeline helper (`load_model → run_structural_passes → convert_to_closed_ir → drop_final_requant → quantize → buffer_planner.plan`), same pattern as `test_buffer_planner.py`.

- `test_emit_layout_macros_on_mnist` — generated text contains `#define CONV1_OUT_H 14`, `#define CONV2_OUT_H 7`, `#define GEMM1_IN 294`, `#define GEMM1_OUT 10`, buffer declarations sized `1176`.
- `test_conv_emitter_rejects_asymmetric_stride` — synthetic `Op` with `strides=(2,1)` → `ConvEmitter().layout_macros(...)` raises `CodegenError`.
- `test_emit_weights_values_match_quantized_tensor` — generated conv1 weight array's parsed int values equal `conv1_op.inputs[1].data.flatten().tolist()`.
- `test_gemm_emitter_skip_requant_call_site` — the generated `dense_int8(...)` line for the final Gemm contains the skip-requant flag/cast to `int32_t*`.
- `test_generate_model_c_structure` — `generate_model_c(...)` output contains, in order, two `conv2d_int8(` calls, one `dense_int8(` call, one `argmax_i32(` call, ending in `uint8_t model_forward(const int8_t *input)` that `return`s the argmax result.
- `test_generate_model_c_compiles_against_stub_headers` — **the key verification test**: writes the Phase-5 stub headers + generated `model.c` to `tmp_path`, invokes host `gcc -c -std=c11 -Wall -Wextra model.c -I<tmp_path> -o model.o`, asserts exit code 0 and no warnings.

---

## Verification

1. `conda run -n ctinyml pytest -v` — all Phase 0–4 tests (including the refactored `test_interpreter.py`) plus the new Phase 5 tests green.
2. Manual sanity — same as before, via `assemble.generate_model_c(...)` — eyeball the output against `context/mnist_conv_dense_int8_avr.c`'s `model_forward()`: recognizably the same call shape (two `conv2d_int8`, one `dense_int8`, ping-pong `buf_a`/`buf_b` reuse), just data-driven.
3. `gcc -c -std=c11 model.c -I ctinyml/targets/avr/runtime -o model.o` — manual re-run of the host-compile check.
4. Append the `quantize.py` ArgMax-calibration bug fix to `context/known_issues.md`, matching the existing Phase 0–4 entries' format.

Once green, this gates the start of Phase 6 (AVR device runtime library — the real, tested kernel implementations these stub headers currently only declare) per the parent plan's sequential execution order — not part of this subplan.
