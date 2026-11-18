# Subplan: Phase 2 — Structural passes (via `onnx_ir.passes`, pre-quantization)

## Context

Phase 0 (repo/package scaffold) and Phase 1 (ONNX frontend) are implemented and green (`conda run -n ctinyml pytest` → 31 passed). `ctinyml/frontend/load.py` loads `mnist_classifier.onnx` into a validated 9-node `onnx_ir.Model`: `Div → Conv → Relu → Conv → Relu → Reshape → Gemm → Reshape → ArgMax`.

Phase 2, per `context/implementation_plan.md`, reduces that to a 4-node graph (`Conv(clamp_min0) → Conv(clamp_min0) → Gemm → ArgMax`) via three `onnx_ir.passes.PassBase` rewrites, then converts the result into our own narrow `closed_ir.Graph` — the Phase 2→3 boundary, after which nothing downstream touches `onnx_ir` again. This subplan covers exactly that: the three structural passes, the closed IR dataclasses (currently a stub), the conversion function, and pytest coverage including the golden-snapshot acceptance test the parent plan calls for.

**API grounding done directly against the real model and the installed `onnx_ir` 0.2.1** (in the `ctinyml` conda env) to remove guesswork before writing code:
- `onnx_ir.passes.InPlacePass` is the right base (matches `IdentityEliminationPass`'s pattern in `onnx_ir/passes/common/identity_elimination.py` — the library's own canonical "remove a node, rewire" example): implement `call(self, model) -> PassResult`, mutate in place, return `PassResult(model, modified=True)`.
- Rewiring pattern (lifted straight from that reference pass): `ir.convenience.replace_all_uses_with(old_value, new_value, replace_graph_outputs=True)` then `graph.remove(node, safe=True)`. `safe=True` verifies no remaining users before removal — exactly the "assert loudly, don't silently miscompile" property every pass here needs.
- Verified against the real graph:
  - **Div**: `inputs = [graph_input('input'), initializer('val_0', scalar 255.0)]`; `graph.inputs[0] is div.inputs[0]` → `True`, confirming the first input *is* the literal graph input, not an intermediate. `val_0.const_value.numpy()` → `255.0`.
  - **Reshape** (`view`): `[1,6,7,7] → [1,294]`, `allowzero=1` (inert here — no `0` in the target shape, so my initial worry about `allowzero` gating eligibility was wrong; dropped that check). **Reshape** (`val_6`): `[1,10] → [10]`, shape input `[-1]`.
  - **Relu**: `relu.inputs[0].producer()` → the preceding `Conv` node object directly (method call, not property); `list(relu.outputs[0].uses())` → exactly one use.
  - **Conv** attrs: `group, auto_pad, dilations, strides, pads`. **Gemm** attrs: `transA=0, transB=1, alpha=1.0, beta=1.0`. **ArgMax** attrs: `axis=0, keepdims=0, select_last_index=0`.
  - `graph.meta` is a plain dict-like `MetadataStore` for arbitrary in-memory Python values (as opposed to `metadata_props`, which is the serializable `str→str` ONNX-spec field) — the right place for `input_prescale`, since it never needs to round-trip to a `.onnx` file.
  - `node.attributes.add(ir.AttrInt64("clamp_min0", 1))` — confirmed working for adding a non-ONNX-spec custom attribute (fine; we never re-serialize this IR back through the ONNX checker).
  - `ir.Attr.value` is a generic accessor that works for every attribute type (`int`, `tuple[int,...]`, `float`, `str`, ...) — the one method `convert_to_closed_ir.py` needs to flatten `node.attributes` into a plain dict.
  - `onnx_ir.tape.Tape` (`tape.op(op_type, inputs, attributes=...)`) builds `onnx_ir` graphs directly in Python, bypassing the ONNX checker entirely — the right tool for the one deliberately-invalid fixture below (a mismatched-element-count Reshape), since `onnx.checker` would reject that graph before we ever got to test our own pass's defensive assertion.

**Design resolution worth recording:** "rewire producer→consumer directly" for `drop_noop_reshape` means the consumer's input `Value` ends up being the *pre-reshape* tensor object, keeping its pre-reshape shape (e.g. Gemm's input wCanonically becomes `relu_1` with shape `(1,6,7,7)`, not `(1,294)`). That's intentional, not a bug to fix here — it matches the locked decision "flat arrays, ... no generic tensor struct at runtime": nothing downstream cares about rank, only total element count (`K` for Gemm = `prod(shape)` = 294 either way). Buffer sizing/codegen (Phases 4–5) will derive `K` from `prod(shape)`, not from shape rank. Noting this now so it isn't re-litigated during Phase 5.

---

## Files to add/change

```
ctinyml/
  closed_ir.py                              # REAL — Tensor/Op/Graph dataclasses (currently a stub)
  passes/
    __init__.py                             # REAL — StructuralPassError, run_structural_passes()
    onnx_ir_passes/
      drop_noop_reshape.py                  # REAL
      fuse_relu.py                          # REAL
      fold_div_into_quant.py                # REAL
    convert_to_closed_ir.py                 # REAL
tests/
  fixtures/
    synthetic.py                            # extended with new builder functions (below)
    golden/
      mnist_closed_ir.txt                   # checked-in snapshot, generated once during implementation
  test_passes.py                            # new — one test class per pass + Sequential end-to-end
  test_closed_ir.py                         # new — conversion + golden snapshot
```

## `ctinyml/closed_ir.py` — dataclasses

```python
@dataclass
class Tensor:
    name: str
    shape: tuple[int, ...]
    dtype: str                          # e.g. "FLOAT" (str(onnx_ir.DataType member))
    data: np.ndarray | None = None      # populated for initializers only
    is_initializer: bool = False
    scale: float | None = None          # Phase 3 quantization metadata (unset here)
    zero_point: int | None = None       # Phase 3 quantization metadata (unset here)

@dataclass
class Op:
    op_type: str
    name: str
    inputs: list[Tensor]
    outputs: list[Tensor]
    attrs: dict[str, Any] = field(default_factory=dict)   # flat, e.g. {"strides": (2,2), "clamp_min0": 1}

@dataclass
class Graph:
    ops: list[Op]
    inputs: list[Tensor]
    outputs: list[Tensor]
    input_prescale: float | None = None  # folded from onnx_ir Graph.meta["input_prescale"]

    def to_text(self) -> str: ...        # deterministic pretty-printer for the golden-snapshot test
```

`to_text()`: stable, whitespace-normalized multi-line dump — graph-level `input_prescale`, then each op as one block (`op_type`, `name`, sorted `attrs`, each input/output as `name: shape dtype`). Round any floats to 6 decimals for determinism. Deliberately **omits initializer `data` arrays** (shapes/dtypes only) so the golden file stays small and readable, not a weight dump.

## Pass implementations (`ctinyml/passes/onnx_ir_passes/`)

Shared exception, defined once in `ctinyml/passes/__init__.py`:
```python
class StructuralPassError(ValueError):
    """Raised when a structural pass can't safely rewrite the graph."""
```
All three passes raise this (not a silent skip) on anything outside the one shape they're built for — consistent with Phase 1's `FrontendError` precedent.

1. **`drop_noop_reshape.py`** — `DropNoopReshapePass(ir.passes.InPlacePass)`:
   - For every `Reshape` node: assert `prod(node.inputs[0].shape) == prod(node.outputs[0].shape)` (both static, guaranteed by Phase 1) — else `StructuralPassError` naming the node. (Per ONNX spec this can't actually fail for a valid graph; it's a defensive invariant the rest of the pipeline's "aliasing, no data movement" assumption depends on, and it's what the loud-failure unit test below exercises via a hand-built invalid graph.)
   - Rewire: `replace_all_uses_with(reshape.outputs[0], reshape.inputs[0], replace_graph_outputs=True)`, then `graph.remove(reshape, safe=True)`.
   - Drop the now-dangling shape-tensor initializer (`node.inputs[1]`) if it has no remaining uses.
   - Postcondition: no `Reshape` nodes remain in the graph — else `StructuralPassError` (residual-reshape safety net).

2. **`fuse_relu.py`** — `FuseReluPass(ir.passes.InPlacePass)`:
   - Snapshot `relus = [n for n in graph.all_nodes() if n.op_type == "Relu"]` first (don't mutate while iterating `all_nodes()`).
   - For each: eligible iff `producer = relu.inputs[0].producer()` is not `None`, `producer.op_type in {"Conv", "Gemm"}`, **and** `len(list(producer.outputs[0].uses())) == 1` (sole consumer — guards the branching case explicitly; fusing into a producer with other consumers would silently clamp *their* view too).
   - If eligible: `producer.attributes.add(ir.AttrInt64("clamp_min0", 1))`; `replace_all_uses_with(relu.outputs[0], producer.outputs[0], replace_graph_outputs=True)`; `graph.remove(relu, safe=True)`.
   - Postcondition: no `Relu` nodes remain — else `StructuralPassError` naming the unfused node(s) (this is where a branching-input graph gets caught, per the parent plan's explicit "fail loudly ... e.g. branching").

3. **`fold_div_into_quant.py`** — `FoldDivIntoQuantPass(ir.passes.InPlacePass)`:
   - Find the `Div` node whose `inputs[0] is graph.inputs[0]` and whose `inputs[1]` is a scalar (`shape == ()`) initializer. Exactly one expected — `StructuralPassError` if none found (this compiler currently only targets this fixed architecture, so "no Div" is a hard error, not a silent no-op — consistent with the other two passes).
   - `graph.meta["input_prescale"] = float(div.inputs[1].const_value.numpy())`.
   - `replace_all_uses_with(div.outputs[0], div.inputs[0], replace_graph_outputs=True)`; `graph.remove(div, safe=True)`; drop the now-unused scalar initializer.

## `ctinyml/passes/__init__.py` — orchestration

```python
STRUCTURAL_PASSES = onnx_ir.passes.Sequential(
    DropNoopReshapePass(), FuseReluPass(), FoldDivIntoQuantPass(),
)  # order per parent plan: reshape, then relu-fuse, then div-fold

def run_structural_passes(model: onnx_ir.Model) -> onnx_ir.Model:
    return STRUCTURAL_PASSES(model).model
```
Kept here (not inside `convert_to_closed_ir.py`) so `convert_to_closed_ir.convert()` stays a pure conversion function taking an already-reduced model, per the parent plan's description of it as a separate boundary step, and so a later CLI wiring (Phase 5+) can call `run_structural_passes` directly.

## `ctinyml/passes/convert_to_closed_ir.py`

```python
def convert(model: onnx_ir.Model) -> closed_ir.Graph:
    ...
```
Walks `model.graph.all_nodes()` once; builds a `Tensor` per distinct `Value` (cached by `id(value)` so shared values — e.g. a weight reused nowhere here, but general-purpose — aren't duplicated), populating `.data` from `value.const_value.numpy()` when present. Builds one `Op` per node with `attrs = {k: v.value for k, v in node.attributes.items()}`. Reads `input_prescale` from `model.graph.meta.get("input_prescale")`.

---

## Tests

### `tests/fixtures/synthetic.py` — new builders (alongside Phase 1's existing three)

- `div_relu_graph(tmp_path)` — `Div(input, scalar_initializer) → Relu`, for `fold_div_into_quant` unit tests. Built with `onnx.helper` + `onnx.checker` (valid ONNX, same style as Phase 1's fixtures).
- `conv_relu_branching_graph(tmp_path)` — `Conv` output feeds **both** a `Relu` and a second consumer (e.g. a parallel `Identity` to a second graph output), so `fuse_relu`'s sole-consumer guard fires. Valid ONNX, `onnx.helper` + checker.
- `reshape_bad_shape_graph()` — **not** an `onnx.helper` builder: built directly with `onnx_ir.tape.Tape` (bypasses `onnx.checker`, which would otherwise reject a Reshape with mismatched element counts before the test ever reaches our pass). Returns an `onnx_ir.Model` object directly, not a file path — the one fixture in this file that isn't round-tripped through disk.
- Reuse Phase 1's `conv_relu_graph(tmp_path)` directly for `fuse_relu`'s happy-path test — no need to duplicate it.

### `tests/test_passes.py`

- `test_drop_noop_reshape_removes_node` — build a small `Conv → Reshape(flatten) → Gemm`-shaped synthetic graph (new builder `conv_reshape_gemm_graph`, or reuse the MNIST fixture's first Reshape in isolation — prefer a small synthetic one for speed/independence), run `DropNoopReshapePass`, assert no `Reshape` node remains and the downstream consumer's input `Value` is now the pre-reshape producer's output (identity check).
- `test_drop_noop_reshape_rejects_mismatched_shape` — `reshape_bad_shape_graph()` → `DropNoopReshapePass` raises `StructuralPassError`.
- `test_fuse_relu_happy_path` — Phase 1's `conv_relu_graph` fixture → after the pass, `Relu` gone, `Conv` has `clamp_min0=1`, graph output aliases `Conv`'s output.
- `test_fuse_relu_rejects_branching` — `conv_relu_branching_graph` → `StructuralPassError`.
- `test_fold_div_into_quant_happy_path` — `div_relu_graph` → `graph.meta["input_prescale"]` matches the fixture's scalar, `Div` gone, `Relu`'s input is the graph input directly.
- `test_fold_div_into_quant_requires_div` — Phase 1's `conv_relu_graph` fixture (no `Div` present) → `StructuralPassError`.
- `test_structural_passes_on_mnist` — `load_model(MNIST_FIXTURE)` → `run_structural_passes(model)` → assert `[n.op_type for n in model.graph.all_nodes()] == ["Conv", "Conv", "Gemm", "ArgMax"]`, both `Conv`s have `clamp_min0 == 1`, `model.graph.meta["input_prescale"] == 255.0`. This is the parent plan's Phase 2 acceptance bullet, checked directly.

### `tests/test_closed_ir.py`

- `test_convert_mnist_structure` — run the full pipeline (`load_model` → `run_structural_passes` → `convert_to_closed_ir.convert`) on the MNIST fixture; assert `len(graph.ops) == 4`, op types in order, `graph.input_prescale == 255.0`, first two ops have `attrs["clamp_min0"] == 1`, weight `Tensor`s have `.data` populated with the right `.shape`.
- `test_golden_snapshot` — same pipeline → `graph.to_text()` → compare against checked-in `tests/fixtures/golden/mnist_closed_ir.txt` (exact string equality; on mismatch pytest's assert diff is enough, no custom diff needed). The golden file itself is generated once by running the real pipeline during implementation and reviewed by eye before checking in — it's the "any future pass-ordering regression is caught immediately" acceptance criterion from the parent plan.

---

## Verification

1. `conda run -n ctinyml pytest -v` — all Phase 0/1 tests plus the new Phase 2 tests green.
2. Manual sanity, mirroring Phase 1's style:
   `conda run -n ctinyml python -c "from ctinyml.frontend.load import load_model; from ctinyml.passes import run_structural_passes; from ctinyml.passes.convert_to_closed_ir import convert; g = convert(run_structural_passes(load_model('tests/fixtures/mnist/mnist_classifier.onnx'))); print([op.op_type for op in g.ops]); print(g.input_prescale)"`
   → expect `['Conv', 'Conv', 'Gemm', 'ArgMax']` and `255.0`.
3. Eyeball the generated `tests/fixtures/golden/mnist_closed_ir.txt` once before committing it, to confirm it's actually readable/reviewable (the parent plan's stated purpose for the snapshot).

Once green, this gates the start of Phase 3 (Quantization) per the parent plan's sequential execution order — not part of this subplan.
# Subplan: Phase 2 — Structural passes (via `onnx_ir.passes`, pre-quantization)

## Context

Phase 0 (repo/package scaffold) and Phase 1 (ONNX frontend) are implemented and green (`conda run -n ctinyml pytest` → 31 passed). `ctinyml/frontend/load.py` loads `mnist_classifier.onnx` into a validated 9-node `onnx_ir.Model`: `Div → Conv → Relu → Conv → Relu → Reshape → Gemm → Reshape → ArgMax`.

Phase 2, per `context/implementation_plan.md`, reduces that to a 4-node graph (`Conv(clamp_min0) → Conv(clamp_min0) → Gemm → ArgMax`) via three `onnx_ir.passes.PassBase` rewrites, then converts the result into our own narrow `closed_ir.Graph` — the Phase 2→3 boundary, after which nothing downstream touches `onnx_ir` again. This subplan covers exactly that: the three structural passes, the closed IR dataclasses (currently a stub), the conversion function, and pytest coverage including the golden-snapshot acceptance test the parent plan calls for.

**API grounding done directly against the real model and the installed `onnx_ir` 0.2.1** (in the `ctinyml` conda env) to remove guesswork before writing code:
- `onnx_ir.passes.InPlacePass` is the right base (matches `IdentityEliminationPass`'s pattern in `onnx_ir/passes/common/identity_elimination.py` — the library's own canonical "remove a node, rewire" example): implement `call(self, model) -> PassResult`, mutate in place, return `PassResult(model, modified=True)`.
- Rewiring pattern (lifted straight from that reference pass): `ir.convenience.replace_all_uses_with(old_value, new_value, replace_graph_outputs=True)` then `graph.remove(node, safe=True)`. `safe=True` verifies no remaining users before removal — exactly the "assert loudly, don't silently miscompile" property every pass here needs.
- Verified against the real graph:
  - **Div**: `inputs = [graph_input('input'), initializer('val_0', scalar 255.0)]`; `graph.inputs[0] is div.inputs[0]` → `True`, confirming the first input *is* the literal graph input, not an intermediate. `val_0.const_value.numpy()` → `255.0`.
  - **Reshape** (`view`): `[1,6,7,7] → [1,294]`, `allowzero=1` (inert here — no `0` in the target shape, so my initial worry about `allowzero` gating eligibility was wrong; dropped that check). **Reshape** (`val_6`): `[1,10] → [10]`, shape input `[-1]`.
  - **Relu**: `relu.inputs[0].producer()` → the preceding `Conv` node object directly (method call, not property); `list(relu.outputs[0].uses())` → exactly one use.
  - **Conv** attrs: `group, auto_pad, dilations, strides, pads`. **Gemm** attrs: `transA=0, transB=1, alpha=1.0, beta=1.0`. **ArgMax** attrs: `axis=0, keepdims=0, select_last_index=0`.
  - `graph.meta` is a plain dict-like `MetadataStore` for arbitrary in-memory Python values (as opposed to `metadata_props`, which is the serializable `str→str` ONNX-spec field) — the right place for `input_prescale`, since it never needs to round-trip to a `.onnx` file.
  - `node.attributes.add(ir.AttrInt64("clamp_min0", 1))` — confirmed working for adding a non-ONNX-spec custom attribute (fine; we never re-serialize this IR back through the ONNX checker).
  - `ir.Attr.value` is a generic accessor that works for every attribute type (`int`, `tuple[int,...]`, `float`, `str`, ...) — the one method `convert_to_closed_ir.py` needs to flatten `node.attributes` into a plain dict.
  - `onnx_ir.tape.Tape` (`tape.op(op_type, inputs, attributes=...)`) builds `onnx_ir` graphs directly in Python, bypassing the ONNX checker entirely — the right tool for the one deliberately-invalid fixture below (a mismatched-element-count Reshape), since `onnx.checker` would reject that graph before we ever got to test our own pass's defensive assertion.

**Design resolution worth recording:** "rewire producer→consumer directly" for `drop_noop_reshape` means the consumer's input `Value` ends up being the *pre-reshape* tensor object, keeping its pre-reshape shape (e.g. Gemm's input wCanonically becomes `relu_1` with shape `(1,6,7,7)`, not `(1,294)`). That's intentional, not a bug to fix here — it matches the locked decision "flat arrays, ... no generic tensor struct at runtime": nothing downstream cares about rank, only total element count (`K` for Gemm = `prod(shape)` = 294 either way). Buffer sizing/codegen (Phases 4–5) will derive `K` from `prod(shape)`, not from shape rank. Noting this now so it isn't re-litigated during Phase 5.

---

## Files to add/change

```
ctinyml/
  closed_ir.py                              # REAL — Tensor/Op/Graph dataclasses (currently a stub)
  passes/
    __init__.py                             # REAL — StructuralPassError, run_structural_passes()
    onnx_ir_passes/
      drop_noop_reshape.py                  # REAL
      fuse_relu.py                          # REAL
      fold_div_into_quant.py                # REAL
    convert_to_closed_ir.py                 # REAL
tests/
  fixtures/
    synthetic.py                            # extended with new builder functions (below)
    golden/
      mnist_closed_ir.txt                   # checked-in snapshot, generated once during implementation
  test_passes.py                            # new — one test class per pass + Sequential end-to-end
  test_closed_ir.py                         # new — conversion + golden snapshot
```

## `ctinyml/closed_ir.py` — dataclasses

```python
@dataclass
class Tensor:
    name: str
    shape: tuple[int, ...]
    dtype: str                          # e.g. "FLOAT" (str(onnx_ir.DataType member))
    data: np.ndarray | None = None      # populated for initializers only
    is_initializer: bool = False
    scale: float | None = None          # Phase 3 quantization metadata (unset here)
    zero_point: int | None = None       # Phase 3 quantization metadata (unset here)

@dataclass
class Op:
    op_type: str
    name: str
    inputs: list[Tensor]
    outputs: list[Tensor]
    attrs: dict[str, Any] = field(default_factory=dict)   # flat, e.g. {"strides": (2,2), "clamp_min0": 1}

@dataclass
class Graph:
    ops: list[Op]
    inputs: list[Tensor]
    outputs: list[Tensor]
    input_prescale: float | None = None  # folded from onnx_ir Graph.meta["input_prescale"]

    def to_text(self) -> str: ...        # deterministic pretty-printer for the golden-snapshot test
```

`to_text()`: stable, whitespace-normalized multi-line dump — graph-level `input_prescale`, then each op as one block (`op_type`, `name`, sorted `attrs`, each input/output as `name: shape dtype`). Round any floats to 6 decimals for determinism. Deliberately **omits initializer `data` arrays** (shapes/dtypes only) so the golden file stays small and readable, not a weight dump.

## Pass implementations (`ctinyml/passes/onnx_ir_passes/`)

Shared exception, defined once in `ctinyml/passes/__init__.py`:
```python
class StructuralPassError(ValueError):
    """Raised when a structural pass can't safely rewrite the graph."""
```
All three passes raise this (not a silent skip) on anything outside the one shape they're built for — consistent with Phase 1's `FrontendError` precedent.

1. **`drop_noop_reshape.py`** — `DropNoopReshapePass(ir.passes.InPlacePass)`:
   - For every `Reshape` node: assert `prod(node.inputs[0].shape) == prod(node.outputs[0].shape)` (both static, guaranteed by Phase 1) — else `StructuralPassError` naming the node. (Per ONNX spec this can't actually fail for a valid graph; it's a defensive invariant the rest of the pipeline's "aliasing, no data movement" assumption depends on, and it's what the loud-failure unit test below exercises via a hand-built invalid graph.)
   - Rewire: `replace_all_uses_with(reshape.outputs[0], reshape.inputs[0], replace_graph_outputs=True)`, then `graph.remove(reshape, safe=True)`.
   - Drop the now-dangling shape-tensor initializer (`node.inputs[1]`) if it has no remaining uses.
   - Postcondition: no `Reshape` nodes remain in the graph — else `StructuralPassError` (residual-reshape safety net).

2. **`fuse_relu.py`** — `FuseReluPass(ir.passes.InPlacePass)`:
   - Snapshot `relus = [n for n in graph.all_nodes() if n.op_type == "Relu"]` first (don't mutate while iterating `all_nodes()`).
   - For each: eligible iff `producer = relu.inputs[0].producer()` is not `None`, `producer.op_type in {"Conv", "Gemm"}`, **and** `len(list(producer.outputs[0].uses())) == 1` (sole consumer — guards the branching case explicitly; fusing into a producer with other consumers would silently clamp *their* view too).
   - If eligible: `producer.attributes.add(ir.AttrInt64("clamp_min0", 1))`; `replace_all_uses_with(relu.outputs[0], producer.outputs[0], replace_graph_outputs=True)`; `graph.remove(relu, safe=True)`.
   - Postcondition: no `Relu` nodes remain — else `StructuralPassError` naming the unfused node(s) (this is where a branching-input graph gets caught, per the parent plan's explicit "fail loudly ... e.g. branching").

3. **`fold_div_into_quant.py`** — `FoldDivIntoQuantPass(ir.passes.InPlacePass)`:
   - Find the `Div` node whose `inputs[0] is graph.inputs[0]` and whose `inputs[1]` is a scalar (`shape == ()`) initializer. Exactly one expected — `StructuralPassError` if none found (this compiler currently only targets this fixed architecture, so "no Div" is a hard error, not a silent no-op — consistent with the other two passes).
   - `graph.meta["input_prescale"] = float(div.inputs[1].const_value.numpy())`.
   - `replace_all_uses_with(div.outputs[0], div.inputs[0], replace_graph_outputs=True)`; `graph.remove(div, safe=True)`; drop the now-unused scalar initializer.

## `ctinyml/passes/__init__.py` — orchestration

```python
STRUCTURAL_PASSES = onnx_ir.passes.Sequential(
    DropNoopReshapePass(), FuseReluPass(), FoldDivIntoQuantPass(),
)  # order per parent plan: reshape, then relu-fuse, then div-fold

def run_structural_passes(model: onnx_ir.Model) -> onnx_ir.Model:
    return STRUCTURAL_PASSES(model).model
```
Kept here (not inside `convert_to_closed_ir.py`) so `convert_to_closed_ir.convert()` stays a pure conversion function taking an already-reduced model, per the parent plan's description of it as a separate boundary step, and so a later CLI wiring (Phase 5+) can call `run_structural_passes` directly.

## `ctinyml/passes/convert_to_closed_ir.py`

```python
def convert(model: onnx_ir.Model) -> closed_ir.Graph:
    ...
```
Walks `model.graph.all_nodes()` once; builds a `Tensor` per distinct `Value` (cached by `id(value)` so shared values — e.g. a weight reused nowhere here, but general-purpose — aren't duplicated), populating `.data` from `value.const_value.numpy()` when present. Builds one `Op` per node with `attrs = {k: v.value for k, v in node.attributes.items()}`. Reads `input_prescale` from `model.graph.meta.get("input_prescale")`.

---

## Tests

### `tests/fixtures/synthetic.py` — new builders (alongside Phase 1's existing three)

- `div_relu_graph(tmp_path)` — `Div(input, scalar_initializer) → Relu`, for `fold_div_into_quant` unit tests. Built with `onnx.helper` + `onnx.checker` (valid ONNX, same style as Phase 1's fixtures).
- `conv_relu_branching_graph(tmp_path)` — `Conv` output feeds **both** a `Relu` and a second consumer (e.g. a parallel `Identity` to a second graph output), so `fuse_relu`'s sole-consumer guard fires. Valid ONNX, `onnx.helper` + checker.
- `reshape_bad_shape_graph()` — **not** an `onnx.helper` builder: built directly with `onnx_ir.tape.Tape` (bypasses `onnx.checker`, which would otherwise reject a Reshape with mismatched element counts before the test ever reaches our pass). Returns an `onnx_ir.Model` object directly, not a file path — the one fixture in this file that isn't round-tripped through disk.
- Reuse Phase 1's `conv_relu_graph(tmp_path)` directly for `fuse_relu`'s happy-path test — no need to duplicate it.

### `tests/test_passes.py`

- `test_drop_noop_reshape_removes_node` — build a small `Conv → Reshape(flatten) → Gemm`-shaped synthetic graph (new builder `conv_reshape_gemm_graph`, or reuse the MNIST fixture's first Reshape in isolation — prefer a small synthetic one for speed/independence), run `DropNoopReshapePass`, assert no `Reshape` node remains and the downstream consumer's input `Value` is now the pre-reshape producer's output (identity check).
- `test_drop_noop_reshape_rejects_mismatched_shape` — `reshape_bad_shape_graph()` → `DropNoopReshapePass` raises `StructuralPassError`.
- `test_fuse_relu_happy_path` — Phase 1's `conv_relu_graph` fixture → after the pass, `Relu` gone, `Conv` has `clamp_min0=1`, graph output aliases `Conv`'s output.
- `test_fuse_relu_rejects_branching` — `conv_relu_branching_graph` → `StructuralPassError`.
- `test_fold_div_into_quant_happy_path` — `div_relu_graph` → `graph.meta["input_prescale"]` matches the fixture's scalar, `Div` gone, `Relu`'s input is the graph input directly.
- `test_fold_div_into_quant_requires_div` — Phase 1's `conv_relu_graph` fixture (no `Div` present) → `StructuralPassError`.
- `test_structural_passes_on_mnist` — `load_model(MNIST_FIXTURE)` → `run_structural_passes(model)` → assert `[n.op_type for n in model.graph.all_nodes()] == ["Conv", "Conv", "Gemm", "ArgMax"]`, both `Conv`s have `clamp_min0 == 1`, `model.graph.meta["input_prescale"] == 255.0`. This is the parent plan's Phase 2 acceptance bullet, checked directly.

### `tests/test_closed_ir.py`

- `test_convert_mnist_structure` — run the full pipeline (`load_model` → `run_structural_passes` → `convert_to_closed_ir.convert`) on the MNIST fixture; assert `len(graph.ops) == 4`, op types in order, `graph.input_prescale == 255.0`, first two ops have `attrs["clamp_min0"] == 1`, weight `Tensor`s have `.data` populated with the right `.shape`.
- `test_golden_snapshot` — same pipeline → `graph.to_text()` → compare against checked-in `tests/fixtures/golden/mnist_closed_ir.txt` (exact string equality; on mismatch pytest's assert diff is enough, no custom diff needed). The golden file itself is generated once by running the real pipeline during implementation and reviewed by eye before checking in — it's the "any future pass-ordering regression is caught immediately" acceptance criterion from the parent plan.

---

## Verification

1. `conda run -n ctinyml pytest -v` — all Phase 0/1 tests plus the new Phase 2 tests green.
2. Manual sanity, mirroring Phase 1's style:
   `conda run -n ctinyml python -c "from ctinyml.frontend.load import load_model; from ctinyml.passes import run_structural_passes; from ctinyml.passes.convert_to_closed_ir import convert; g = convert(run_structural_passes(load_model('tests/fixtures/mnist/mnist_classifier.onnx'))); print([op.op_type for op in g.ops]); print(g.input_prescale)"`
   → expect `['Conv', 'Conv', 'Gemm', 'ArgMax']` and `255.0`.
3. Eyeball the generated `tests/fixtures/golden/mnist_closed_ir.txt` once before committing it, to confirm it's actually readable/reviewable (the parent plan's stated purpose for the snapshot).

Once green, this gates the start of Phase 3 (Quantization) per the parent plan's sequential execution order — not part of this subplan.
