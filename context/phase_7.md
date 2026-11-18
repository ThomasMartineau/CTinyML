# Subplan: Phase 7 — End-to-end validation

## Context

Phases 0–6 are implemented and green (91 passing tests). Every phase so far has used **synthetic** calibration/test data (random-uniform `[0,255]` noise) — fine for validating plumbing, but Phase 7's whole point is a *real* accuracy claim, so this phase re-quantizes with real MNIST images and measures real agreement between the C implementation and the Phase 3 float reference interpreter.

**Real MNIST data source, per your pointer to `mnist_model.ipynb`:** the notebook already loads it — `sklearn.datasets.fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')`, `mnist_784` preserving original ordering so `X[:60000]`/`X[60000:]` are the standard train/test split. Pixel values come back as `float64` in `[0,255]` — the same raw-pixel domain our closed IR's `input` tensor already expects (Phase 2 folded away the `/255.0` div). `scikit-learn` is installed in the `gaze` conda env (used to run the notebook), **not** in `ctinyml`. Rather than add `scikit-learn`/network access as an ongoing `ctinyml` dependency, this is a **one-time data-prep step**: run a small script once via `gaze` to pull a modest real MNIST sample and save it as a checked-in fixture (`tests/fixtures/mnist_holdout.npz`) — same pattern as `tests/fixtures/mnist/mnist_classifier.onnx` already being a checked-in binary asset. `ctinyml`'s own test run stays offline and dependency-light afterward.

**Hardware smoke test — confirmed out of scope for this subplan** (your answer): no ATmega2560 connected here, and `avr-gcc` still isn't installed (Phase 5/6 gap, unchanged). Explicitly deferred, not silently dropped, matching how Phase 5/6 handled the same toolchain gap — noted as a follow-up once real hardware + `avr-gcc` are available.

**A real implementation gotcha, worth flagging now rather than discovering mid-build:** `quantize.py` mutates a `closed_ir.Graph`'s `Tensor.data` **in place** (float weights become int8). The Phase 3 float interpreter needs float weights; codegen needs quantized ones. So this phase needs **two independent `closed_ir.Graph` trees** from the same structurally-reduced `onnx_ir.Model` — `convert_to_closed_ir.convert(model)` already builds a fresh `Tensor`/`Op`/`Graph` object tree per call (its `tensor_cache` is a local, reset each call), so calling it **twice** on the same reduced model gives one graph to keep float (interpreter baseline) and one to quantize (codegen) — no aliasing. Structurally identical, object-independent.

**How the "dequantized int32 accumulator vs. float logits" comparison actually gets built:** the generated `model_forward()` only returns the final class byte — it was never designed to expose intermediate logits (that's correct; not reopening Phase 5's codegen contract for this). Instead, this reuses Phase 6's own `tests/fixtures/runtime_shim_ported.c` (already exposes `test_conv2d_int8`/`test_dense_int8`/`test_argmax_i32` as ctypes-callable wrappers around the real kernels) to manually orchestrate `conv1 → conv2 → gemm(skip_requant)` in Python, capturing the raw `int32` output directly — no new C code, reusing exactly what Phase 6 already built and tested. Dequantized via `input_scale * weight_scale[c]` (the same formula `_requant_multiplier` uses, just applied post-hoc here since the final Gemm never got one computed).

**Acceptance, per the parent plan:** ≥95% match rate between the C implementation's argmax and the float interpreter's argmax on the held-out batch — this is the hard pass/fail gate. The logit-closeness comparison and true-label accuracy are **reported diagnostics with generous sanity tolerances**, not additional hard gates — the parent plan lists them as "Task"s to do/report, and explicitly hedges closeness as "not exact, due to quantization."

---

## Files to add/change

```
ctinyml/
  passes/quantize.py                    # + quantize_input(image, tensor) -> np.ndarray (reusable: raw float image -> int8 via an already-calibrated tensor's scale/zero_point)
tests/
  fixtures/
    dll_compile.py                      # NEW — small shared compile+load helper, factored out of test_runtime.py's _compile_dll (now 2 call sites)
    prepare_mnist_holdout.py            # NEW — one-time prep script (run via `gaze`, not part of pytest): fetch_openml -> tests/fixtures/mnist_holdout.npz
    mnist_holdout.npz                   # NEW — checked-in fixture: 128 real calibration images + 100 real held-out eval images+labels, raw [0,255] float32
  test_runtime.py                       # MINOR REFACTOR — import _compile_dll/GCC from the new shared helper instead of a local copy
  test_quantize.py                      # + small direct unit test for quantize_input
  test_e2e_validation.py                # NEW — Phase 7's own tests
```

## `quantize_input` (`ctinyml/passes/quantize.py`)

```python
def quantize_input(image: np.ndarray, tensor: closed_ir.Tensor) -> np.ndarray:
    """Quantize a raw float image to int8 using tensor's already-calibrated
    scale/zero_point (e.g. graph.inputs[0] after quantize() has run).
    Reusable beyond this test -- any future inference/deployment path
    needs exactly this."""
    q = _round_half_away_from_zero(image / tensor.scale) + tensor.zero_point
    return np.clip(q, QMIN, QMAX).astype(np.int8)
```

## `tests/fixtures/prepare_mnist_holdout.py` (one-time script, not a pytest test)

```python
from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X, y = mnist.data, mnist.target.astype(np.int64)
X_train, y_train = X[:60000], y[:60000]
X_test, y_test = X[60000:], y[60000:]

rng = np.random.default_rng(0)
calib_idx = rng.choice(len(X_train), size=128, replace=False)
eval_idx = rng.choice(len(X_test), size=100, replace=False)

np.savez_compressed(
    "tests/fixtures/mnist_holdout.npz",
    calibration_images=X_train[calib_idx].reshape(-1, 1, 28, 28).astype(np.float32),
    eval_images=X_test[eval_idx].reshape(-1, 1, 28, 28).astype(np.float32),
    eval_labels=y_test[eval_idx],
)
```
Run once via `conda run -n gaze python tests/fixtures/prepare_mnist_holdout.py` during implementation (needs network on first run; `fetch_openml` caches after). If that fails (no network, no cache), implementation pauses to ask how to proceed rather than guessing. Result: a small (~100–300 KB compressed) checked-in fixture; `ctinyml`'s own env/tests never need `scikit-learn` or network access.

## `tests/fixtures/dll_compile.py`

```python
GCC = shutil.which("gcc")

def compile_dll(build_dir: Path, source: Path, include_dirs: list[Path], dll_name: str) -> Path:
    """Compile `source` (+ include_dirs) into a shared lib with
    -Wl,--export-all-symbols (needed for `static` kernel functions --
    see Phase 6). Asserts a clean compile."""
```
`test_runtime.py`'s existing `_compile_dll`/`GCC` become imports from here — the exact function, just no longer duplicated once Phase 7 needs the same thing.

## `tests/test_e2e_validation.py`

Module-scoped fixture: loads `mnist_holdout.npz`; builds `float_graph` and `quant_graph` from one `run_structural_passes(load_model(...))` call (per the aliasing note above); quantizes `quant_graph` with the **real** calibration images; runs `drop_final_requant` + `buffer_planner.plan`; generates `model.c` via Phase 5's `assemble.generate_model_c`; compiles+loads **two** DLLs via the shared `compile_dll` helper — the full generated `model_forward()`, and Phase 6's existing `runtime_shim_ported.c` (for the per-kernel orchestration below).

- `test_argmax_match_rate_meets_acceptance_threshold` — **the primary gate**. For each of the 100 eval images: `float_pred = argmax(interpreter.run(float_graph, image)[<gemm_output_name>]... )` — actually taken directly from the interpreter's own `ArgMax` output tensor, not recomputed; `c_pred` from calling the compiled `model_forward()` on `quantize_input(image, quant_graph.inputs[0])`. Assert `mean(c_pred == float_pred) >= 0.95`.
- `test_logit_closeness_diagnostic` — for each eval image, get the float pre-`ArgMax` logits (the `Gemm` op's output tensor from the float interpreter's result dict) and the real dequantized `int32` accumulator (manual `test_conv2d_int8 → test_conv2d_int8 → test_dense_int8(skip_requant)` orchestration via the shim DLL, using `quant_graph`'s actual quantized weight/bias arrays, dequantized by `input_scale * weight_scale[c]`). Reports mean absolute/relative error and a correlation coefficient; asserts a **generous** sanity bound (e.g. correlation ≥ 0.9) — diagnostic, not the acceptance gate.
- `test_true_label_accuracy_reported` — informational: how often float-argmax and C-argmax each match `eval_labels`; printed, with a loose smoke-test floor (e.g. ≥ 0.85 — well under the notebook's own reported ~97.8%, just enough to catch a badly broken pipeline) rather than a strict accuracy claim.

---

## Verification

1. `conda run -n gaze python tests/fixtures/prepare_mnist_holdout.py` — one-time; confirm `tests/fixtures/mnist_holdout.npz` written.
2. `conda run -n ctinyml pytest -v` — all Phase 0–6 tests (including the refactored `test_runtime.py`) plus the new Phase 7 tests green, in particular the ≥95% match-rate assertion.
3. Manually print the match-rate/closeness/true-label numbers from a direct run, sanity-check they're in a believable range (expect match-rate noticeably above 95%, true-label accuracy in the same neighborhood as the notebook's ~97.8%, given int8 quantization on an easy dataset like MNIST rarely flips a confident prediction).
4. Append a `context/known_issues.md` entry (established pattern every phase so far) — including if the real match-rate turns out lower than expected, since that would be a genuine finding, not just a process note.

Once green, this closes out the parent plan's software-validated scope for Phase 7 — the on-device hardware smoke test remains explicitly deferred (needs `avr-gcc` + physical hardware, neither available here), and Phase 8 (CSR sparsity, stretch goal) is next per the parent plan's sequential order if pursued at all.
