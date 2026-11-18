# Subplan: Phase 3 — Quantization

## Context

Phases 0–2 are implemented and green (`conda run -n ctinyml pytest` → 40 passed). `ctinyml/passes/convert_to_closed_ir.py` turns the MNIST model into a 4-op `closed_ir.Graph`: `Conv(clamp_min0) → Conv(clamp_min0) → Gemm → ArgMax`, with `input_prescale == 255.0` recorded and weight/bias data already sitting on each `Tensor.data` as float32 numpy arrays.

Phase 3, per `context/implementation_plan.md`, quantizes that graph to int8: a shared float32 reference interpreter (to calibrate activation ranges, and reused unchanged by Phase 7 later), per-channel symmetric weight quantization, calibration-driven asymmetric activation quantization, per-channel float requant multipliers, and marking the final Gemm to skip requantization since it feeds `ArgMax` directly on raw int32 accumulators. This subplan implements all of that plus its pytest coverage.

**Ground-truth check against `context/mnist_conv_dense_int8_avr.c`** (re-read in full, since the parent plan calls it "structural ground truth" and this phase is where its numeric conventions actually get exercised): `requantize()` clamps to **`[-128, 127]`** and all zero-points (`input_zp`, `mid_zp`, `out_zp`) are typed `int8_t`. This directly **contradicts** one sentence in the parent plan's Phase 3 acceptance bullet — "reproduces that range at the int8 boundaries (**0 and 255** for asymmetric)" — which is uint8 phrasing that doesn't match either the reference file or the rest of the plan's own "int8 weight values" / "weight_zero_point == 0" language. Resolving this in favor of the reference file (the file the parent plan explicitly designates as ground truth) and the internally-consistent rest of the spec: **activation quantization targets signed int8, qmin = -128, qmax = 127**, not 0/255. The acceptance test below checks `-128`/`127`, not `0`/`255`.

**Design resolution — `input_prescale` and calibration data domain:** Phase 2 already removed the `Div` node and rewired `Conv1` to consume the graph's raw input directly. So the closed IR's declared input *is* the pre-Div, raw-pixel-domain tensor — exactly what "preprocessed input arrays" in the calibration interface means: the same arrays you'd have fed the *original* 9-node ONNX model's `input` tensor (roughly `[0, 255]` for this model), not a separately-normalized `[0,1]` array. Calibrating directly on that domain, using the float interpreter (which has no `Div` op anymore either, by construction), *is* "folding `input_prescale` into layer-1's input scale" — there's no separate multiply-by-`input_prescale` formula to write; it falls out of calibrating on the correct domain. `graph.input_prescale` itself ends up unused by the quantization math — noting that now so it isn't mistaken for a bug later.

---

## Files to add/change

```
ctinyml/
  interpreter.py                     # NEW — float32 reference interpreter (Conv/Gemm/ArgMax), shared with Phase 7
  passes/
    errors.py                        # + ClosedIRPassError (alongside existing StructuralPassError)
    drop_final_requant.py            # REAL — mark the final Gemm skip_requant=True
    quantize.py                      # REAL — calibration + weight/bias/activation quantization
tests/
  test_interpreter.py                # NEW
  test_quantize.py                   # NEW
```

## `ctinyml/interpreter.py`

Pure numpy, float32 throughout, target-agnostic (operates on `closed_ir.Graph`, no `onnx_ir` dependency — this file is shared as-is by Phase 7).

```python
def run(graph: closed_ir.Graph, input_array: np.ndarray) -> dict[str, np.ndarray]:
    """Forward-evaluate `graph` in float32 for one input. Returns every
    activation tensor's computed value keyed by name: the graph input
    plus each op's output(s) — exactly the tensor set quantize.py needs
    to calibrate."""
```

Op dispatch table (`Conv`, `Gemm`, `ArgMax` — the only op types that survive Phase 2):
- `_conv2d(x, weight, bias, strides, pads, clamp_min0)`: zero-pad with `np.pad`, then `numpy.lib.stride_tricks.sliding_window_view` + `np.einsum('ncyxkl,ockl->noyx', windows, weight)` + bias — vectorized, and mathematically identical to the reference file's bounds-check-skip direct convolution (padding with zeros and windowing contributes the same zero terms that the bounds check skips). Applies the `clamp_min0` attr (fused ReLU) to the result before returning.
- `_gemm(a, b, c, alpha, beta, transA, transB, clamp_min0)`: standard `alpha * (A @ B) + beta * C` with optional transposes; same `clamp_min0` handling.
- `_argmax(x, axis, keepdims, select_last_index)`: `np.argmax`, with `select_last_index` handled by reversing along `axis` before arg-maxing and correcting the index (not exercised by the real graph, which uses `select_last_index=0`, but implemented for correctness/generality).

**Hand-computed sanity check (done now, to pin down the expected values before writing the test):** input 4×4 `1..16` row-major, 3×3 all-ones kernel, `stride=2, pad=1` — the exact `k/s/p` the real model uses, chosen so this test doubles as the "stride-2/pad-1 boundary math" check Phase 5 will also need. Computed by hand:
```
input:            output (bias=0):
 1  2  3  4        14  30
 5  6  7  8   ->
 9 10 11 12        57  99
13 14 15 16
```
(corner `(0,0)` sums 4 in-bounds cells `1+2+5+6=14`; edge and full-window cells computed the same way — `(0,1)`→`30`, `(1,0)`→`57`, `(1,1)` has the full 3×3 window → `99`.) This is embedded directly as the expected array in `test_conv2d_hand_computed`.

## `ctinyml/passes/errors.py` — add

```python
class ClosedIRPassError(ValueError):
    """Raised when a closed_ir pass (drop_final_requant, quantize) can't
    safely process the graph."""
```

## `ctinyml/passes/drop_final_requant.py`

```python
def apply(graph: closed_ir.Graph) -> closed_ir.Graph:
    """Find the Gemm feeding ArgMax; set attrs['skip_requant'] = True."""
```
Walks `graph.ops` for the `ArgMax` op, finds its producer by tensor identity (`op.outputs[0] is argmax.inputs[0]`), asserts that producer is a `Gemm` (`ClosedIRPassError` otherwise — same loud-failure stance as Phase 2), sets the flag. Mutates in place, returns the graph for chaining (matches `run_structural_passes`'s style).

## `ctinyml/passes/quantize.py`

```python
QMIN, QMAX = -128, 127          # signed int8, matches requantize()'s clamp in the reference file
QRANGE = QMAX - QMIN            # 255

def quantize(graph: closed_ir.Graph, calibration_data: Iterable[np.ndarray]) -> closed_ir.Graph:
    """Call after drop_final_requant.apply(graph). Mutates graph in place:
    calibrates + quantizes every activation tensor, quantizes weights/
    biases per-output-channel, computes per-channel requant multipliers.
    Leaves any op already marked skip_requant with an unquantized int32
    output instead."""
```

Internals:
- `_calibrate_activations(graph, calibration_data)`: builds `name -> Tensor` for every non-initializer tensor (graph input + each op's outputs — already deduplicated by object identity from `convert_to_closed_ir`, so setting `.scale`/`.zero_point` once is visible everywhere that tensor is referenced). Runs `interpreter.run(graph, x)` per calibration sample, tracks running `(lo, hi)` per name (seeded `+inf/-inf`), then sets `.scale`/`.zero_point`/`.dtype = "INT8"` via `_asymmetric_qparams`.
- `_asymmetric_qparams(lo, hi) -> (scale, zero_point)`: widen the interval to include `0` (standard practice so the additive identity is exactly representable), guard `hi - lo` against a zero-width interval (`max(hi - lo, 1e-8)`), `scale = (hi - lo) / QRANGE`, `zero_point = clip(round(QMIN - lo / scale), QMIN, QMAX)`. **Pure function, unit-tested directly** against the acceptance criterion: calibrated `lo` maps to `QMIN`, `hi` maps to `QMAX`.
- `_quantize_weight_and_bias(op)`: per-channel (axis 0) symmetric weight scale `= max(abs(weight), axis=(1,2,3 or 1)) / 127` (zero-guarded), `int8` weight data via round-half-away-from-zero + clip to `[-127, 127]`; bias scale `= input_tensor.scale * weight_scale` (per-channel), `int32` bias data. Overwrites `.data`, `.scale`, `.zero_point = 0`, `.dtype` on the weight/bias `Tensor` objects in place.
- `_requant_multiplier(input_scale, weight_scale, output_scale) -> np.ndarray`: `input_scale * weight_scale / output_scale`, per-channel array. **Pure function, unit-tested directly** across small/near-1.0 scale triples.
- Main loop over `graph.ops`: for each `Conv`/`Gemm`, quantize weight+bias; if `op.attrs.get("skip_requant")`, reset that op's output tensor to `scale=None, zero_point=None, dtype="INT32"` (raw accumulator) instead of computing a requant multiplier; otherwise compute `_requant_multiplier(...)` from the op's (already-calibrated) input/output tensor scales and store it as `op.attrs["requant_mult"]`.

---

## Tests

### `tests/test_interpreter.py`
- `test_conv2d_hand_computed` — the 4×4/3×3/stride-2/pad-1 example above; also a `bias` variant and a `clamp_min0` variant (large negative bias → all-zero output) to exercise both code paths.
- `test_gemm_hand_computed` — small hand-computed `x=[1,2,3,4]`, `transB=1` weight, verified by hand.
- `test_argmax_hand_computed` — trivial `[10, 2] -> 0`.
- `test_full_graph_forward_mnist` — `interpreter.run()` on the real MNIST closed IR (via the same `load_model → run_structural_passes → convert_to_closed_ir` helper `test_closed_ir.py` already uses) with a fixed synthetic input; asserts it runs end-to-end and the `ArgMax` output is a valid class index in `[0, 9]`. Sanity/plumbing check only — no ground truth to compare against yet (that's Phase 7, deliberately, per the parent plan's "not a fresh PyTorch/ORT comparison" stance).

### `tests/test_quantize.py`
- `test_requant_multiplier_various_scales` — parametrized `(input_scale, weight_scale, output_scale)` triples spanning ~`1e-4` to near-`1.0`; asserts `_requant_multiplier` matches direct computation within tight tolerance.
- `test_asymmetric_qparams_reproduces_calibrated_range` — the corrected acceptance test: given a known `(lo, hi)`, asserts `round(lo / scale) + zero_point == QMIN` and `round(hi / scale) + zero_point == QMAX`.
- `test_weight_quantization_symmetric_per_channel` — synthetic per-channel weight tensor with deliberately varied magnitudes (~`0.01`, ~`1.0`, ~`100.0` — "small and near-1.0 cases" from the acceptance bullet); asserts `zero_point == 0`, `dtype == int8`, values in `[-127, 127]`, and dequantized (`int8 * scale`) values within a tolerance of `scale/2` per channel of the originals.
- `test_drop_final_requant_marks_gemm` — on the real MNIST closed IR, asserts only the `Gemm` op gets `skip_requant=True`, not either `Conv`.
- `test_calibrate_activations_on_synthetic_graph` — small synthetic `closed_ir.Graph` (reuse `synthetic.conv_relu_graph` converted through the existing Phase 1/2 pipeline) with a tiny calibration set of known combined min/max; asserts the resulting input tensor's `.scale`/`.zero_point` match `_asymmetric_qparams` applied to that known range — integration-level check that calibration's running min/max tracking and tensor lookup are wired correctly, not just the pure function.
- `test_quantize_end_to_end_mnist` — full pipeline (`load_model → run_structural_passes → convert_to_closed_ir → drop_final_requant.apply → quantize` with an 8-sample synthetic `[0,255]`-range calibration set, seeded `np.random.default_rng`, plus an all-zeros and all-255 sample to guarantee non-degenerate ranges). Asserts: `Conv1`/`Conv2` weight `Tensor`s are `int8` with length-6 per-channel scale arrays, bias `Tensor`s are `int32`; both have `attrs["requant_mult"]` of length 6; the `Gemm`'s weight/bias are quantized (length-10 scale array) but `attrs` has no `requant_mult` and its output tensor has `scale is None`, `dtype == "INT32"`; the graph input tensor got `.scale`/`.zero_point` set (dtype `INT8`).

---

## Verification

1. `conda run -n ctinyml pytest -v` — all Phase 0–2 tests plus the new Phase 3 tests green.
2. Manual sanity, mirroring prior phases:
   ```
   conda run -n ctinyml python -c "
   import numpy as np
   from ctinyml.frontend.load import load_model
   from ctinyml.passes import run_structural_passes
   from ctinyml.passes.convert_to_closed_ir import convert
   from ctinyml.passes.drop_final_requant import apply as drop_final_requant
   from ctinyml.passes.quantize import quantize

   g = convert(run_structural_passes(load_model('tests/fixtures/mnist/mnist_classifier.onnx')))
   drop_final_requant(g)
   rng = np.random.default_rng(0)
   calib = [rng.uniform(0, 255, size=(1,1,28,28)).astype(np.float32) for _ in range(8)]
   quantize(g, calib)
   for op in g.ops:
       print(op.op_type, op.attrs.get('skip_requant', False), op.inputs[1].dtype, op.inputs[1].scale.shape if op.inputs[1].scale is not None else None)
   "
   ```
   → expect `Conv False INT8 (6,)`, `Conv False INT8 (6,)`, `Gemm True INT8 (10,)`.

Once green, this gates the start of Phase 4 (Buffer planner) per the parent plan's sequential execution order — not part of this subplan.
