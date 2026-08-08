# Subplan: Phase 4 — Buffer planner

## Context

Phases 0–3 are implemented and green (`conda run -n ctinyml pytest` → 59 passed). `ctinyml/passes/quantize.py` (run after `drop_final_requant.apply`) leaves the MNIST `closed_ir.Graph` fully quantized: `Conv(clamp_min0) → Conv(clamp_min0) → Gemm(skip_requant) → ArgMax`, with every weight/bias/activation `Tensor` carrying its final int8/int32 `.dtype`.

Phase 4, per `context/implementation_plan.md`, does liveness analysis over that graph and assigns each intermediate activation tensor to one of two fixed-size, byte-addressed SRAM "ping-pong" buffers — no `malloc`, sized once at compile time to the largest activation in the graph — then reports the total SRAM footprint against the ATmega2560's 8 KB budget (hard-fail over budget, soft-warn above 6 KB to leave headroom for stack/globals). This subplan implements that as `ctinyml/passes/buffer_planner.py` plus its pytest coverage.

**Grounded against the actual MNIST closed IR** (re-checked against `tests/fixtures/golden/mnist_closed_ir.txt` from Phase 2, plus Phase 3's known quantized dtypes — these are the concrete numbers the tests below assert against, not estimates):

| Tensor | Producer | Consumer | Shape | Quantized dtype | Bytes |
|---|---|---|---|---|---|
| `conv2d` | `Conv` (op 0) | `Conv` (op 1) | `(1,6,14,14)` | `INT8` | **1176** |
| `conv2d_1` | `Conv` (op 1) | `Gemm` (op 2) | `(1,6,7,7)` | `INT8` | **294** |
| `linear` | `Gemm` (op 2, `skip_requant`) | `ArgMax` (op 3) | `(1,10)` | `INT32` (raw accumulator) | **40** |

`input` (graph input) and `pred` (graph output, the final class index) are excluded from buffer planning — they're caller-supplied/caller-written, exactly like `model_forward`'s `input`/`logits` parameters in `context/mnist_conv_dense_int8_avr.c`, never part of `buf_a`/`buf_b`.

**Design resolution — uniform buffer sizing, not per-lane sizing:** the parent plan's Phase 4 task list says "two fixed buffers sized to the largest intermediate activation across the whole graph" (singular largest, both buffers), and the reference file's own `buf_a`/`buf_b` are declared with the *same* size (`CONV1_OUT_CH*CONV1_OUT_H*CONV1_OUT_W` for both), not independently right-sized per buffer. Adopting that: **both** buffers are sized to `max(nbytes) = 1176`, not 1176/294 separately — simpler static allocation, matches the reference file exactly, and is what the acceptance bullet's "the two largest activations ... are what sizes the ping-pong buffers" is checking for (that both those numbers are correctly discovered by the liveness scan), not that they separately size two different-sized arrays. Total SRAM = `2 × 1176 = 2352` bytes — comfortably under both the 6 KB warning line and the 8 KB hard budget. `linear`'s 40 bytes reuses one of the same two buffers (byte-region reinterpreted as `int32_t*` by codegen later) rather than needing a third — this is exactly what "the planner doesn't allocate a third buffer" in the acceptance text is verifying.

---

## Files to add/change

```
ctinyml/
  closed_ir.py                  # + Tensor.buffer: int | None = None (which of the 2 buffers, if any)
  passes/
    buffer_planner.py           # REAL — liveness analysis + ping-pong assignment + SRAM report
tests/
  test_buffer_planner.py        # NEW
```

`closed_ir.Graph.to_text()` is deliberately left untouched (doesn't reference `.buffer`) — no golden-file regeneration needed; Phase 2/3's existing tests are unaffected since they never call `buffer_planner.plan()`.

## `ctinyml/passes/buffer_planner.py`

```python
BUFFER_BUDGET_BYTES = 8192   # 8 KB SRAM, ATmega2560 hard limit
BUFFER_WARN_BYTES = 6144     # 6 KB soft threshold — leaves headroom for stack/globals

@dataclass
class BufferPlan:
    buffer_size: int              # bytes, uniform across both buffers
    num_buffers: int               # always 2 for this ping-pong strategy
    total_sram_bytes: int          # num_buffers * buffer_size
    assignments: dict[str, int]    # tensor name -> buffer index (0 or 1)
    warning: str | None            # set if over BUFFER_WARN_BYTES, still under budget

def plan(graph: closed_ir.Graph) -> BufferPlan:
    """Call after quantize.quantize(graph) -- sizing is dtype-dependent
    (int8 vs int32), so this must see final quantized dtypes."""
```

Internals:
- **Discover internal tensors**: every op output *not* also in `graph.outputs` — `{t for op in graph.ops for t in op.outputs} - set(graph.outputs)`. (Weights/biases never appear here since they're only ever op *inputs*, never outputs; the graph input is excluded by construction since it's never an op output either.) For the MNIST graph this yields exactly `{conv2d, conv2d_1, linear}`.
- **Byte size**: `nbytes(t) = prod(t.shape) * itemsize(t.dtype)`, with `itemsize` mapping `"INT8"->1, "INT32"->4` (defensively `"FLOAT"->4, "INT64"->8` too, though those shouldn't appear on internal tensors post-quantization).
- **Liveness**: `def_index` = index of the producing op; `last_use_index` = max index among ops whose `inputs` contain that tensor (by identity — `closed_ir` tensors are already deduplicated object-per-value from `convert_to_closed_ir`, so this is a simple `is`/`in` scan, no separate use-def table needed).
- **Greedy 2-slot ping-pong assignment**, processed in `graph.ops` order (already topological): before assigning a newly-defined tensor, free any buffer slot whose occupant's `last_use_index < current_op_index` (i.e. free starting the op *after* its last use, not during it — during the op that consumes tensor X and produces tensor Y, both X's buffer and Y's buffer are legitimately live simultaneously, which is exactly what 2 buffers cover for a strict producer→consumer chain). If no slot is free when one is needed, **raise `ClosedIRPassError`** — this is the general safety net for any future graph shape needing 3+ simultaneously-live tensors (branching), not something this specific linear MNIST graph will ever trigger, but a real, tested failure mode (mirrors Phase 2's `fuse_relu` branching guard).
- **Size + budget report**: `buffer_size = max(nbytes(t) for t in internal_tensors)`; `total = 2 * buffer_size`; raise `ClosedIRPassError` if `total > BUFFER_BUDGET_BYTES`; set `.warning` (don't raise) if `total > BUFFER_WARN_BYTES`.
- Mutates each internal `Tensor.buffer` in place to its assigned index (0/1) — consistent with how `quantize.py` already annotates `Tensor` objects in place — and also returns the `BufferPlan` report for the caller/tests to inspect.

Reuses `ClosedIRPassError` from `ctinyml/passes/errors.py` (added in Phase 3) — no new exception type needed.

---

## Tests (`tests/test_buffer_planner.py`)

Full-pipeline helper mirroring `test_quantize.py`'s `_mnist_closed_ir()` but carried one step further (`drop_final_requant` + `quantize` + `plan`), since sizing is dtype-dependent and must run post-quantization.

- `test_plan_on_mnist_matches_hand_computed_sizes` — asserts `buffer_size == 1176`, `num_buffers == 2`, `total_sram_bytes == 2352`, `warning is None`; and that the three internal tensors are discovered by name (`conv2d`, `conv2d_1`, `linear`) with no others.
- `test_plan_on_mnist_alternates_buffers_and_uses_only_two` — asserts `assignments["conv2d"] != assignments["conv2d_1"]` (they're simultaneously live during op 1, must differ) and every assignment value is in `{0, 1}` — the explicit "the planner doesn't allocate a third buffer" check from the parent plan's acceptance bullet.
- `test_plan_sets_tensor_buffer_field` — asserts the actual `Tensor.buffer` attribute on `graph.ops[0].outputs[0]` (etc.) matches `assignments`, not just the returned report.
- `test_raises_when_third_buffer_needed` — synthetic hand-built `closed_ir.Graph` (plain dataclasses, no `onnx`/`onnx_ir` needed — `buffer_planner` only touches `closed_ir`) with genuine fan-out: op0 defines `T0`; op1 consumes `T0`, defines `T1`; op2 consumes **both** `T0` and `T1`, defines `T2`. At op2, `T0` and `T1` are both still live (both are op2's inputs) while `T2` also needs a slot — 3 simultaneous tensors, 2 buffers → `ClosedIRPassError`.
- `test_hard_fails_over_sram_budget` — synthetic 2-op chain with a deliberately oversized shape (e.g. one activation tensor `> 8192` bytes) → `ClosedIRPassError` mentioning the budget.
- `test_warns_between_soft_and_hard_budget` — synthetic chain sized so `total_sram_bytes` lands between 6 KB and 8 KB → no exception, `plan_result.warning is not None`.

---

## Verification

1. `conda run -n ctinyml pytest -v` — all Phase 0–3 tests plus the new Phase 4 tests green.
2. Manual sanity, mirroring prior phases:
   ```
   conda run -n ctinyml python -c "
   import numpy as np
   from ctinyml.frontend.load import load_model
   from ctinyml.passes import run_structural_passes
   from ctinyml.passes.convert_to_closed_ir import convert
   from ctinyml.passes.drop_final_requant import apply as drop_final_requant
   from ctinyml.passes.quantize import quantize
   from ctinyml.passes.buffer_planner import plan

   g = convert(run_structural_passes(load_model('tests/fixtures/mnist/mnist_classifier.onnx')))
   drop_final_requant(g)
   rng = np.random.default_rng(0)
   calib = [rng.uniform(0, 255, size=(1,1,28,28)).astype(np.float32) for _ in range(8)]
   quantize(g, calib)
   result = plan(g)
   print(result.buffer_size, result.num_buffers, result.total_sram_bytes, result.assignments, result.warning)
   "
   ```
   → expect `1176 2 2352 {'conv2d': 0, 'conv2d_1': 1, 'linear': 0} None`.

Once green, this gates the start of Phase 5 (AVR codegen backend) per the parent plan's sequential execution order — not part of this subplan.
