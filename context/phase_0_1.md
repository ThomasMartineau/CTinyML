# Subplan: Phase 0 (Repo scaffold) + Phase 1 (ONNX frontend) with pytest validation

## Context

`context/implementation_plan.md` lays out the full CTinyML ONNX→C compiler build, phases 0–8. This subplan covers only the first two gated phases — Phase 0 (repo/package scaffold) and Phase 1 (ONNX frontend via `onnx_ir`) — each ending in a pytest suite that must pass before the next phase starts, per the plan's "gated on its own tests passing" execution order.

Repo currently has **no** Python project scaffolding: no `ctinyml/`, `tests/`, `tools/`, or packaging file. It does have real assets ready to build against:
- `context/mnist_conv_dense_int8_avr.c` — the reference AVR runtime (used later, Phase 5/6, not this subplan).
- `models/1786138703/mnist_classifier.onnx` (+ external-data sidecar `.onnx.data`) — the real exported model from `mnist_model.ipynb`. Verified by direct load: **9 nodes**, exact op sequence `Div → Conv → Relu → Conv → Relu → Reshape → Gemm → Reshape → ArgMax`, all shapes fully static (`[1,1,28,28]` input → `[]` scalar ArgMax output), matching the plan's stated architecture exactly.
- Confirmed working `onnx_ir` API surface (installed in the `gaze` conda env, version 0.2.1) against this real model:
  - `onnx.load(path)` → `onnx.shape_inference.infer_shapes(proto)` → `onnx_ir.from_proto(proto)` → `onnx_ir.Model` — full pipeline tested end-to-end, works, and requires no extra external-data flags (external data resolves automatically relative to the model path).
  - `model.graph.all_nodes()`, `node.op_type`, `node.inputs`/`node.outputs`, `value.shape.is_static()`, `graph.inputs`/`graph.outputs` all behave as the parent plan assumes.
  - `onnx_ir.passes.PassBase` / `onnx_ir.passes.Sequential` exist (needed starting Phase 2, out of scope here, but confirms the plan's chosen library shape is real).

**Environment decision:** per user choice, set up a **new dedicated conda env named `ctinyml`** (Python 3.11, matching `gaze`'s interpreter) rather than reusing `gaze` or a plain venv. `gaze` remains untouched.

**Windows case-insensitivity note:** an empty, untracked `CTinyML/` directory already sits at repo root. Since Windows filesystems are case-insensitive, a sibling `ctinyml/` would collide with it. Resolution: reuse that existing empty directory as the `ctinyml/` package root (write files into it) rather than trying to create a separate directory — no git history to preserve since it's untracked and empty.

---

## Phase 0 — Repo scaffold

### 0.1 Conda environment

Create the dedicated env and install runtime + dev dependencies:

```
conda create -n ctinyml python=3.11 -y
conda run -n ctinyml pip install onnx onnx-ir numpy pytest
```

(Pin to the versions verified working if drift becomes an issue later: `onnx==1.22.0`, `onnx-ir==0.2.1`, `numpy==2.4.6` — but start unpinned since these are fresh installs, not reused from `gaze`.)

### 0.2 Packaging file — `pyproject.toml`

Minimal `setuptools`-backed project at repo root:
- `[project]` name `ctinyml`, dependencies `onnx`, `onnx-ir`, `numpy`.
- `[project.optional-dependencies] dev = ["pytest"]`.
- `[tool.pytest.ini_options] testpaths = ["tests"]`.
- Package discovery pointed at the `ctinyml/` directory (the repurposed `CTinyML/` dir — write `pyproject.toml`'s `[tool.setuptools.packages.find]` to include `ctinyml*`).

Install editable into the new env: `conda run -n ctinyml pip install -e ".[dev]"`.

### 0.3 Package/dir scaffold (stub files, no logic except Phase 1's `load.py`)

Exactly the tree from the parent plan's Phase 0 section, written into the reused `CTinyML/` (→ treated as `ctinyml/`) directory:

```
ctinyml/
  __init__.py
  __main__.py                       # thin: parses argv, delegates to CLI logic
  closed_ir.py                      # stub: module docstring only, Phase 3
  frontend/
    __init__.py
    load.py                         # REAL — Phase 1 implementation lands here
  passes/
    __init__.py
    onnx_ir_passes/
      __init__.py
      fold_div_into_quant.py        # stub
      drop_noop_reshape.py          # stub
      fuse_relu.py                  # stub
    convert_to_closed_ir.py         # stub
    quantize.py                     # stub
    drop_final_requant.py           # stub
    buffer_planner.py               # stub
    sparsity_analysis.py            # stub
  layout.py                         # stub
  targets/
    __init__.py
    avr/
      __init__.py
      codegen/
        __init__.py
        emit_weights.py             # stub
        emit_conv2d.py              # stub
        emit_gemm.py                # stub
        emit_argmax.py              # stub
        assemble.py                 # stub
      runtime/
        pgm_read_f32.h              # placeholder header w/ TODO comment (Phase 6)
        qmath.h                     # placeholder
        conv2d_direct.h             # placeholder
        gemm.h                      # placeholder
        argmax.h                    # placeholder
tests/
  __init__.py
  fixtures/
    mnist/
      mnist_classifier.onnx         # copied from models/1786138703/
      mnist_classifier.onnx.data    # copied external-data sidecar
    synthetic.py                    # builder functions for hand-built ONNX graphs (onnx.helper), not binary fixtures
  test_scaffold.py                  # Phase 0 tests
  test_frontend.py                  # Phase 1 tests
tools/
  cli.py                            # argparse: `compile <model.onnx> --target avr -o build/`
```

Stub `.py` files: module docstring stating which phase implements them + `pass` (nothing importable fails). Stub `.h` files: a `#pragma once` + `/* TODO: Phase 6 */` comment, non-functional placeholders.

**CLI wiring note:** `tools/cli.py` holds the actual argparse logic (`build_parser()`, `main()`) per the parent plan's file layout. `ctinyml/__main__.py` imports and calls it (`from tools.cli import main; main()`) so `python -m ctinyml ...` works when run from the repo root (repo root is importable as the working directory, same mechanism the editable install's `pip install -e .` leaves in place). `--target` is a required flag, only legal value `avr` for now, so a future second target is additive.

### 0.4 Phase 0 pytest — `tests/test_scaffold.py`

Validates the scaffold itself, not any real logic:
- `test_package_importable` — `import ctinyml` succeeds.
- `test_submodules_importable` — every stub module listed above imports without error (parametrized over the module list).
- `test_onnx_ir_importable` — `import onnx_ir` succeeds (guards the env setup, not just the package).
- `test_cli_help_exits_zero` — invoke `python -m ctinyml --help` (via `subprocess.run`) and assert exit code 0 and usage text printed.
- `test_cli_requires_target` — invoke `python -m ctinyml compile tests/fixtures/mnist/mnist_classifier.onnx` **without** `--target` and assert nonzero exit (argparse enforces the required flag).

**Acceptance:** `conda run -n ctinyml pytest` green, all of the above pass.

---

## Phase 1 — ONNX frontend (`ctinyml/frontend/load.py`)

### 1.1 Implementation

```python
import onnx
import onnx.shape_inference
import onnx_ir

SUPPORTED_OPS = {"Div", "Conv", "Relu", "Reshape", "Gemm", "ArgMax"}

class FrontendError(ValueError):
    """Raised for anything this compiler's frontend can't accept."""

def load_model(path) -> onnx_ir.Model:
    proto = onnx.load(str(path))
    inferred = onnx.shape_inference.infer_shapes(proto)
    model = onnx_ir.from_proto(inferred)
    _validate_static_shapes(model.graph)
    _validate_op_set(model.graph)
    return model

def _validate_static_shapes(graph) -> None:
    # every node's inputs/outputs (incl. graph inputs/outputs) must have
    # graph.inputs/outputs value.shape.is_static(); batch dim included —
    # reject dynamic batch explicitly with the tensor name in the error.

def _validate_op_set(graph) -> None:
    # walk graph.all_nodes(); any node.op_type not in SUPPORTED_OPS ->
    # FrontendError naming the op_type and node name.
```

This follows the parent plan's Phase 1 tasks precisely: `onnx.shape_inference` runs ahead of `onnx_ir.from_proto` (not `onnx_ir.load()` directly), external-data resolution is handled by `onnx.load` (already verified working against the real fixture with no extra flags), static-shape and op-set validation both hard-fail loudly rather than skip silently.

### 1.2 Fixtures

- `tests/fixtures/mnist/mnist_classifier.onnx` + `.onnx.data` — copy of `models/1786138703/mnist_classifier.onnx`, the real model. Used only for the full end-to-end structural test.
- `tests/fixtures/synthetic.py` — small builder functions using `onnx.helper` (no checked-in binaries, deterministic, easy to read/diff):
  - `conv_relu_graph()` → minimal 2-node `Conv → Relu` graph, static shapes, opset 18. Independent of MNIST weights, per the parent plan's explicit acceptance requirement.
  - `unsupported_op_graph()` → minimal graph containing one op outside `SUPPORTED_OPS` (e.g. `Sigmoid`) to exercise the op-set rejection path.
  - `dynamic_batch_graph()` → minimal graph with a symbolic/dynamic batch dim on the input, to exercise the static-shape rejection path.

### 1.3 Phase 1 pytest — `tests/test_frontend.py`

- `test_mnist_node_count_and_order` — load the real fixture, assert `model.graph.num_nodes() == 9` and `[n.op_type for n in model.graph.all_nodes()] == ["Div","Conv","Relu","Conv","Relu","Reshape","Gemm","Reshape","ArgMax"]`.
- `test_mnist_shapes_static` — assert every node's every input/output `Value.shape.is_static()` is `True`, and the graph input shape is exactly `[1,1,28,28]`.
- `test_conv_relu_synthetic_loads` — build `conv_relu_graph()`, round-trip it through `onnx.save`/temp file (or directly via `onnx_ir.from_proto` if `load_model` is refactored to accept a proto — decide based on whichever keeps `load_model`'s signature simplest; default to a temp-file round trip to exercise the real file-loading path end-to-end), assert it loads with no error and both ops appear.
- `test_unsupported_op_rejected` — `unsupported_op_graph()` → `load_model` raises `FrontendError`, and the exception message contains the offending op name (`"Sigmoid"`).
- `test_dynamic_batch_rejected` — `dynamic_batch_graph()` → `load_model` raises `FrontendError` mentioning the tensor with the dynamic dim.

**Acceptance:** `conda run -n ctinyml pytest` green, covering both the real MNIST model's exact 9-node topology/static-shapes and the three synthetic-graph edge cases (accept, reject-unsupported-op, reject-dynamic-shape) — matching the parent plan's Phase 1 acceptance bullet plus the two structural failure modes the loader is responsible for enforcing.

---

## Verification (end-to-end)

1. `conda create -n ctinyml python=3.11 -y && conda run -n ctinyml pip install -e ".[dev]"`
2. `conda run -n ctinyml pytest -v` — expect all Phase 0 + Phase 1 tests passing, nothing skipped.
3. `conda run -n ctinyml python -m ctinyml --help` — manual smoke check of CLI usage text.
4. Manual sanity: `conda run -n ctinyml python -c "from ctinyml.frontend.load import load_model; m = load_model('tests/fixtures/mnist/mnist_classifier.onnx'); print(m.graph.num_nodes())"` → prints `9`.

Once green, this gates the start of Phase 2 (structural passes) per the parent plan's sequential execution order — not part of this subplan.
