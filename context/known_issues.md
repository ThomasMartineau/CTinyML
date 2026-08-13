# Known Issues — Phases 0–7

A retrospective log of bugs and gotchas actually hit while *implementing* Phases 0–7, as opposed to `context/phase_*.md`, which are archived copies of the plan documents (intent, written before implementation). Kept separate and up to date so the same mistakes aren't rediscovered in Phase 8+.

---

## Phase 0 — Windows case-insensitivity vs. setuptools case-sensitive package discovery

**Symptom:** `pip install -e .` reported success, but `import ctinyml` failed with `ModuleNotFoundError` immediately after.

**Root cause:** A pre-existing empty `CTinyML/` directory (from repo creation) collided with the intended `ctinyml/` package name. Windows' filesystem path resolution is case-insensitive, so writing files into `ctinyml/...` silently landed inside the existing `CTinyML/` directory — but setuptools' `packages.find` pattern matching (`include = ["ctinyml*"]`) is case-sensitive internally, so it never matched the physically-`CTinyML`-cased directory. The generated editable-install finder's `MAPPING` dict ended up containing only `tools`, not `ctinyml`.

**Fix:** Two-step directory rename (`CTinyML` → temp name → `ctinyml`) to force a true case change on NTFS — a direct same-name-different-case rename is a no-op on a case-insensitive filesystem. Re-ran `pip install -e .` afterward; `MAPPING` then correctly included `ctinyml`.

---

## Phase 1 — no implementation bugs

Nothing broke, but one thing worth recording as "confirmed, not assumed": `onnx_ir.load()` does **not** run ONNX shape inference itself. The frontend needed the explicit three-step pipeline `onnx.load(path) → onnx.shape_inference.infer_shapes(proto) → onnx_ir.from_proto(proto)` — the more obvious-looking `onnx_ir.load()` shortcut would have left intermediate tensor shapes unresolved for any model that doesn't already embed them.

---

## Phase 2 — no implementation bugs

Reading `onnx_ir.passes.common.identity_elimination.IdentityEliminationPass` in the installed library (its rewire-then-remove pattern: `replace_all_uses_with` then `graph.remove(node, safe=True)`) before writing any of `drop_noop_reshape`/`fuse_relu`/`fold_div_into_quant` front-loaded the risk here — all three passes worked on the first real run against the MNIST graph.

One design decision made here *looked* safe at the time but turned out to be the root cause of both Phase 3 bugs below: `drop_noop_reshape`'s "rewire producer→consumer directly" leaves the downstream consumer holding a tensor with its **pre-reshape** shape/rank (e.g. Gemm's input ends up `(1,6,7,7)` instead of `(1,294)`). This is correct and harmless for `Gemm`, which only cares about total element count — but it is not a universally safe assumption for every op type.

---

## Phase 3, bug 1 — ArgMax rank-sensitivity

**Symptom:** `interpreter.run()` on the real MNIST closed IR failed with `TypeError: only 0-dimensional arrays can be converted to Python scalars` (surfaced by `test_full_graph_forward_mnist`).

**Root cause:** Phase 2's `drop_noop_reshape` had removed a *second* `Reshape` that used to squeeze `(1,10) → (10,)` immediately before `ArgMax` in the original ONNX graph. `ArgMax`'s `axis=0` attribute was authored against that squeezed rank-1 shape. With the reshape gone, the same `axis=0` was being applied to the real rank-2 `(1,10)` tensor — silently computing an argmax over the size-1 *batch* axis instead of the 10-class axis, producing a garbage all-zeros result instead of raising outright. This is precisely the gap in Phase 2's "only element count matters, not rank" assumption: true for `Gemm` (rank-agnostic), false for `ArgMax`, whose `axis` attribute *is* rank-sensitive.

**Fix (`ctinyml/interpreter.py`):** squeeze all size-1 dimensions from `ArgMax`'s input before applying the stored `axis` attribute. Safe specifically because batch=1 is a hard-locked invariant of this compiler (Phase 1's frontend rejects any other batch size), so squeezing never discards real information — it exactly undoes the effect of the dropped reshape.

## Phase 3, bug 2 — Gemm input needed explicit flattening

**Symptom:** First run of `interpreter.run()` on the MNIST graph raised `ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0 ... (size 294 is different from 7)` inside `_gemm`.

**Root cause:** Same structural cause as bug 1 — `Gemm`'s input tensor also kept its pre-reshape shape `(1,6,7,7)` instead of `(1,294)` post `drop_noop_reshape`.

**Fix:** Reshape `Gemm`'s input to `(batch, -1)` before the matmul, in `interpreter.py`'s op dispatch. This one *was* anticipated by the Phase 2 design note above (Gemm's `K` derives from `prod(shape)`, not rank) — but Phase 2 never had an actual numeric consumer of that shape to exercise it, so it still needed a real code fix once Phase 3's interpreter ran the graph end-to-end for the first time.

## Phase 3, discrepancy — parent plan's own acceptance text was self-inconsistent

`context/implementation_plan.md`'s Phase 3 acceptance bullet describes the asymmetric-quantization boundary check as "0 and 255" — unsigned uint8 phrasing. This contradicts the same document's own "int8 weight values" / `weight_zero_point == 0` locked decisions, and the reference file `context/mnist_conv_dense_int8_avr.c` (`requantize()` clamps to `[-128, 127]`, all zero-points typed `int8_t`). Resolved in favor of the reference file — the plan's own designated "structural ground truth" — and documented explicitly in the Phase 3 plan rather than silently picking one. `QMIN`/`QMAX` in `ctinyml/passes/quantize.py` are `-128`/`127`.

---

## Phase 4, bug 1 — `closed_ir.Tensor` value-equality vs. identity

**Symptom:** None observed at runtime — **caught by self-review before any test was run against it**, not by a failing test.

**Root cause:** `closed_ir.Tensor` is a plain `@dataclass`, so `==` and `in` (on a list) do field-by-field *value* comparison, not identity comparison — and one field (`.data`) can hold a numpy array. Comparing two `Tensor`s whose earlier fields already matched would eventually compare `.data` arrays with `==`, which returns an array rather than a bool, raising `ValueError: truth value of an array is ambiguous...` when Python tries to fold the field tuple into one equality result. The first draft of `buffer_planner.py` used `tensor in internal` / `tensor in op.inputs` checks; these happened not to crash only because an earlier field (name/shape) always differed first and short-circuited the tuple comparison before reaching `.data` — fragile, and semantically the wrong check regardless (value-equality was never actually the intent; two structurally-identical tensors should not be treated as "the same tensor").

**Fix:** Rewrote `buffer_planner.py` to track tensors by `id()` identity throughout (`internal_ids = {id(t) for t in internal}`, `any(inp is tensor for inp in op.inputs)`, etc.), matching the pattern `convert_to_closed_ir.py` already used for its `id(value)`-keyed tensor cache.

## Phase 4, bug 2 — own test's construction was wrong

**Symptom:** `test_raises_when_third_buffer_needed` failed with `Failed: DID NOT RAISE ClosedIRPassError`.

**Root cause:** The test's first draft made the 3-way fan-in tensor (the one meant to force a 3rd simultaneously-live buffer) the graph's own declared *output* — which `buffer_planner.py` correctly excludes from planning entirely (graph outputs are caller-written, like `model_forward`'s `logits` parameter in the reference file, never assigned an SRAM slot). So the intended conflict never actually occurred; the implementation was correct, the test fixture wasn't exercising it.

**Fix:** Extended the synthetic graph with a 4th op so the fan-in point (`t2`) was consumed by a later op instead of being the graph's final output — making it a genuine internal tensor subject to buffer planning.

---

## Phase 5, bug — `quantize.py` was spuriously quantizing `ArgMax`'s output

**Symptom:** None observed via a failing test — caught during Phase 5 design review, before any codegen code needed to trust `Tensor.dtype`.

**Root cause:** `quantize.py`'s `_calibrate_activations` collected `tensors_by_name` from *every* op's outputs, including `ArgMax`'s. The final `pred` tensor (a discrete `int64` class index, never meant to be quantized) was getting `.scale`/`.zero_point` set and `.dtype` overwritten to `INT8`. Harmless to Phases 3/4's own tests (nothing downstream consumed `pred`'s quantization metadata), but Phase 5's codegen needs `.dtype` to be trustworthy to know it's emitting an `ArgMax` call, not a requantized activation.

**Fix:** Added a named, extensible `NON_CALIBRATED_OPS = {"ArgMax"}` set in `quantize.py` (not an inline `if op.op_type == "ArgMax"` check), so a future non-calibrated op type is a one-line addition. Regression test: `test_quantize_excludes_argmax_output` in `test_quantize.py`.

## Phase 5, design flaw caught before implementation — a static per-op-type "is terminal" flag doesn't generalize

**What almost shipped:** an `IS_TERMINAL: bool` class attribute on each AVR codegen `Emitter` (`True` for `ArgMaxEmitter`, `False` for `ConvEmitter`/`GemmEmitter`), used by `assemble.py` to decide which op's `call_site` gets wrapped in `return ...;` for `model_forward()`.

**Why it was wrong:** "is this op the graph's final op" is a fact about *graph topology* (which op happens to produce `graph.outputs[-1]` in this particular model), not a fact about the *op type*. A hypothetical regression variant ending in a raw `Gemm` (no `ArgMax` at all) would have `GemmEmitter.IS_TERMINAL` hardcoded to `False` — wrong for that graph, and `assemble.py` would still need *some* op-type-specific branch to handle it, defeating the point of the registry.

**Fix (caught via a reviewer question before writing the code):** moved "where does this op's output go" into `tensor_ref` (built once per graph in `emit_layout.py`, already graph-topology-aware) — it resolves any tensor that's one of `graph.outputs` to the fixed `out_class` output-parameter name, regardless of which op produced it. Every `Emitter.call_site` became a uniform void statement (`ArgMaxEmitter`'s is `*out_class = argmax_i32(...);` instead of a bare return-expression), so `assemble.py` has no per-op-type or per-emitter special-casing at all. `model_forward()`'s signature reverted to `void model_forward(const int8_t *input, uint8_t *out_class)` (an output parameter, matching the reference file's own convention) rather than the initially-simpler-looking `uint8_t model_forward(const int8_t *input)` return-value form, which was the actual root of the problem — it only worked because it implicitly assumed the terminal op always produces a single scalar.

## Phase 5, architecture change touching already-shipped Phase 3 code — kernel/emitter registries

Not a bug, but worth recording since it retroactively modified `interpreter.py`: the first draft of `emit_layout.py` had an `if op.op_type == "Conv": ... elif == "Gemm": ...` block, the same shape of dispatch already present in `interpreter.py`'s `match op.op_type` from Phase 3. Rather than let that pattern re-multiply across every codegen module, introduced per-op-type registries mirroring onnxruntime's kernel-registry pattern: `ctinyml/ops.py`'s target-agnostic `KERNELS` registry (owns shape derivation + float forward eval; `interpreter.py`'s numeric bodies moved here so `ops.py` has no dependency on `interpreter.py`, avoiding a circular import) and `targets/avr/codegen/registry.py`'s AVR-specific `EMITTERS` registry (owns `#define` macros + C call-sites, colocated per op type). `closed_ir.Op`/`Tensor` themselves stay flat dataclasses, not subclassed — the registries live beside the IR, keyed by its plain `op_type` string, not inside it. Full regression run confirmed zero behavior change from the `interpreter.py` refactor.

---

## Phase 6 — no implementation bugs

Every hand-computed kernel test (`conv2d_int8` boundary/rounding/zero-point cases, `dense_int8` both output paths, `argmax_i32`, `requantize` rounding/saturation, `pgm_read_*` round-trips), the byte-for-byte regression test against a verbatim copy of the reference file's kernels (both `CONV1`'s and `CONV2`'s actual shapes, plus `dense_int8`), and the bonus end-to-end DLL check (real Phase 5 codegen output linked against the real Phase 6 kernels) all passed on the first run. The three anticipated risks from this phase's own plan — signature drift against Phase 5's pinned stub contract, the `avr-gcc` toolchain gap, and MinGW not auto-exporting DLL symbols for `static` kernel functions — were all real and all handled as planned (full Phase 0–5 re-run confirmed no drift; host-gcc substitution plus an explicit "untested here" flag on the `#ifdef __AVR__` path; test-only non-static shim wrappers compiled with `-Wl,--export-all-symbols`). Planning for known failure modes ahead of time, informed by the Phase 0–5 retrospective, is presumably why none of them actually caused a mid-implementation surprise this time.

---

## Phase 7, bug (major) — `input_prescale` was recorded but never actually applied anywhere

**Symptom:** Phase 7's primary acceptance test (argmax match rate between the C model and the float interpreter, on real MNIST images) came in at 93% — just under the 95% gate. That alone looked like ordinary quantization noise. But the *diagnostic* true-label-accuracy check told the real story: the float interpreter itself only got **55%** right against ground truth, and the C path **50%** — against the notebook's own reported **97.8%** test accuracy for this exact trained model. Something was fundamentally broken, not just borderline.

**How it was isolated (bisection, not guesswork):**
1. Ruled out bad calibration/eval data first: checked the fetched MNIST's first-20 labels and per-class counts against the well-known canonical MNIST ordering/distribution — matched exactly, so the `liac-arff` parser wasn't scrambling anything.
2. Installed `onnxruntime` (diagnostic-only, never added as a project dependency) and ran the **original, untouched** 9-node ONNX graph on the same real images: **94%** accuracy — confirming the trained weights/graph were fine, and the bug was somewhere in *our* pipeline.
3. Reconstructed an ORT-loadable graph directly from the `closed_ir.Graph` (post `run_structural_passes` + `convert_to_closed_ir`, with `clamp_min0` re-expanded back to explicit `Relu` nodes and a `Reshape` re-inserted before `Gemm` so ORT's stricter rank-2 requirement was satisfied) and ran it through ORT: **55%** — matching the float interpreter's own number exactly, on a completely independent implementation. That pinned the bug to something *in* the closed IR's data/structure itself, not in either consumer's arithmetic.
4. Confirmed weight *values* survived `convert_to_closed_ir` bit-for-bit (`max diff: 0.0` against the original proto's initializers) — ruled out a weight-extraction bug.
5. Manually divided the raw eval images by `graph.input_prescale` (255.0) before feeding them to the float interpreter: accuracy jumped to **94%**, matching ORT on the original graph exactly. That located it precisely.

**Root cause:** `fold_div_into_quant.py` (Phase 2) correctly *records* `input_prescale = 255.0` and structurally removes the `Div` node, rewiring `Conv1` to consume the graph's raw `[0,255]`-domain input directly — intentional, since the point was to never run a division on-device. But **nothing downstream ever actually accounted for it**. Phase 3's original design reasoning (both in `quantize.py`'s docstring and the Phase 3 subplan) argued that "calibrating directly on the raw domain is what folds `input_prescale` into layer-1's input scale — there's no separate multiply-by-`input_prescale` step needed." That reasoning was **wrong**: calibrating on the raw domain correctly gives the graph input tensor its own scale (needed so `quantize_input` correctly encodes real raw pixel bytes) — but the *trained weights* still expect the *normalized* `[0,1]` domain, and nothing ever reconciled the two. Concretely, both the float interpreter (feeding raw `[0,255]` values straight into `Conv1`, 255× too large) and the quantized requantization math (`requant_mult = input_scale_raw * weight_scale / output_scale`, using the raw-domain scale directly) were silently computing a different function than the one the network was trained on.

**Fix — two coordinated changes, matching the two consumers of the graph's raw input:**
- `ctinyml/interpreter.py`: `run()` now divides the raw input by `graph.input_prescale` *before* it flows into any op's computation, while still recording the **raw**, undivided value under the input tensor's own name in the returned dict (that raw value is what `quantize.py` needs to calibrate the input tensor's own scale against — it's what the real device's int8 input actually represents). Guarded so graphs without an `input_prescale` (e.g. synthetic test fixtures) are unaffected.
- `ctinyml/passes/quantize.py`: new `_effective_input_scale(graph, graph_input, input_tensor)` — returns `input_tensor.scale / graph.input_prescale` specifically when `input_tensor` *is* the graph's own declared input (and `input_prescale` is set), else `input_tensor.scale` unchanged. Used in place of the raw tensor scale everywhere `quantize()` computes bias scale and the requant multiplier, so the *only* op actually touching the raw input reconstructs the value its weights were trained against — still without ever dividing on-device, since the fold happens once, offline, into the multiplier constant.

**Result:** argmax match rate 93% → **99%**; logit-closeness mean absolute error 17.05 → **0.08** (correlation was already 0.9999 either way — float and C were consistent *with each other* throughout, which is exactly why this bug was invisible to every purely-synthetic test in Phases 3–6); true-label accuracy 55%/50% → **94%/93%** (float/C), now in line with the notebook's 97.8% given `n=100` sampling variance. Regression tests added: `test_run_applies_input_prescale_to_computation_but_records_raw_value` / `test_run_without_input_prescale_uses_raw_value_directly` (`test_interpreter.py`), `test_effective_input_scale_folds_prescale_for_the_graph_input` and siblings (`test_quantize.py`).

**Why every earlier phase's tests missed this:** Phases 3–6 exclusively used *synthetic* calibration/eval data (`rng.uniform(0, 255, ...)`) and never checked predictions against real ground-truth labels — only structural properties (dtypes, shapes, "produces a valid class index 0-9") and self-consistency between the float interpreter and the C kernels, which this bug doesn't break (both were wrong *together*, consistently). A quantization pipeline can be perfectly internally self-consistent — float reference and quantized C implementation agreeing with each other to four nines — while both being confidently wrong relative to the model they were supposedly compiling. **Self-consistency between your own reference and your own implementation is not accuracy validation; only checking against an independent source of truth (real labeled data, or an independent tool like ORT) can catch a bug that both sides share.** This is precisely why Phase 7 exists as a separate, later phase in the parent plan rather than being folded into Phase 3 — but it's also why Phase 7 needs to be run seriously, with real data and real accuracy checks, not deferred or synthetic-ized for convenience.

---

## Lessons for Phase 5/6 (AVR codegen + runtime)

Both phases generate C code directly from this same `closed_ir.Graph`, so these apply there too:

- **Shape/rank metadata on a `closed_ir.Tensor` is not reliable post-`drop_noop_reshape` for anything rank-sensitive** — only `prod(shape)` (total element count) is trustworthy. Any future codegen touching an op with a rank-sensitive attribute (the way `ArgMax`'s `axis` is) needs the same squeeze-or-recompute treatment applied in `interpreter.py`, not a literal read of `op.attrs["axis"]` against the tensor's stored shape.
- **Never compare `closed_ir.Tensor`/`Op` objects with `==` or Python's `in` operator on a list of them** — always compare by `is`/`id()`. The dataclasses' default value-equality is both semantically wrong for IR nodes and unsafe once `.data` (a numpy array) is populated.
- **A fact about *this graph's shape* (e.g. "which op is terminal") is not a fact about an *op type*** — don't encode it as a static per-op-type/per-Emitter class attribute; derive it from the graph (topology, tensor identity) at codegen time instead.
- **A design decision that eliminates a runtime operation (folding a division into an offline constant) still has to be accounted for mathematically *somewhere*** — "we don't need to do X at runtime" is not the same claim as "X's effect is automatically preserved." Verify the arithmetic explicitly (as `_effective_input_scale`'s docstring now does), don't just reason informally that it must work out.
