"""Phase 7 acceptance tests — end-to-end validation against real MNIST
images (tests/fixtures/mnist_holdout.npz, prepared by
tests/fixtures/prepare_mnist_holdout.py).

Primary acceptance gate (per the parent plan): >=95% match rate between
the compiled C model_forward()'s argmax and the Phase 3 float
interpreter's argmax on a held-out batch. The logit-closeness and
true-label-accuracy checks are reported diagnostics with generous
tolerances, not additional hard gates.
"""

import ctypes
from pathlib import Path

import numpy as np
import pytest

from ctinyml import interpreter
from ctinyml.frontend.load import load_model
from ctinyml.passes import run_structural_passes
from ctinyml.passes.buffer_planner import plan
from ctinyml.passes.convert_to_closed_ir import convert
from ctinyml.passes.drop_final_requant import apply as drop_final_requant
from ctinyml.passes.quantize import quantize, quantize_input
from ctinyml.targets.avr.codegen.assemble import MODEL_C_PRELUDE, generate_model_c
from tests.fixtures.dll_compile import (
    FIXTURES_DIR,
    GCC,
    RUNTIME_DIR,
    compile_dll,
    f32p,
    i8p,
    i32p,
)

MNIST_FIXTURE = FIXTURES_DIR / "mnist" / "mnist_classifier.onnx"
HOLDOUT_FIXTURE = FIXTURES_DIR / "mnist_holdout.npz"

MATCH_RATE_THRESHOLD = 0.95  # the parent plan's actual acceptance gate
LOGIT_CORRELATION_FLOOR = 0.9  # generous diagnostic sanity bound
TRUE_LABEL_ACCURACY_FLOOR = 0.85  # loose smoke-test floor, well under the notebook's ~97.8%


@pytest.fixture(scope="module")
def build_dir(tmp_path_factory):
    if GCC is None:
        pytest.skip("host gcc not available")
    return tmp_path_factory.mktemp("e2e_build")


@pytest.fixture(scope="module")
def holdout_data():
    if not HOLDOUT_FIXTURE.exists():
        pytest.skip(
            f"{HOLDOUT_FIXTURE} not found -- run "
            "`python tests/fixtures/prepare_mnist_holdout.py` once first."
        )
    data = np.load(HOLDOUT_FIXTURE)
    # Stored as (N, 1, 28, 28); insert the leading batch axis so
    # iterating yields (1, 1, 28, 28) per sample -- what every op in this
    # pipeline (interpreter, quantize_input, model_forward) expects.
    calibration_images = data["calibration_images"][:, np.newaxis]
    eval_images = data["eval_images"][:, np.newaxis]
    return calibration_images, eval_images, data["eval_labels"]


@pytest.fixture(scope="module")
def pipeline(build_dir, holdout_data):
    """Builds both graphs (float-only and quantized -- see the module
    docstring on why two independent closed_ir.Graph trees are needed),
    quantizes with the real calibration images, and compiles+loads both
    DLLs this phase needs."""
    calibration_images, _eval_images, _eval_labels = holdout_data

    model = run_structural_passes(load_model(MNIST_FIXTURE))
    float_graph = convert(model)
    quant_graph = convert(model)  # independent object tree -- convert()'s tensor_cache is per-call

    drop_final_requant(quant_graph)
    quantize(quant_graph, calibration_images)
    buffer_plan = plan(quant_graph)
    model_c = generate_model_c(quant_graph, buffer_plan)

    model_c_path: Path = build_dir / "model_e2e_validation.c"
    model_c_path.write_text(MODEL_C_PRELUDE + model_c)

    model_dll = compile_dll(build_dir, model_c_path, [RUNTIME_DIR], "model_e2e_validation.dll")
    model_lib = ctypes.CDLL(str(model_dll))
    model_lib.model_forward.argtypes = [
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    model_lib.model_forward.restype = None

    shim_dll = compile_dll(
        build_dir, FIXTURES_DIR / "runtime_shim_ported.c", [RUNTIME_DIR], "shim_e2e_validation.dll"
    )
    shim_lib = ctypes.CDLL(str(shim_dll))
    from tests.fixtures.dll_compile import CONV_ARGTYPES, DENSE_ARGTYPES_PORTED

    shim_lib.test_conv2d_int8.argtypes = CONV_ARGTYPES
    shim_lib.test_conv2d_int8.restype = None
    shim_lib.test_dense_int8.argtypes = DENSE_ARGTYPES_PORTED
    shim_lib.test_dense_int8.restype = None

    return float_graph, quant_graph, model_lib, shim_lib


def _c_predict(model_lib, quant_graph, image: np.ndarray) -> int:
    quantized = quantize_input(image, quant_graph.inputs[0]).reshape(-1)
    out_class = np.zeros(1, dtype=np.uint8)
    model_lib.model_forward(
        i8p(quantized), out_class.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    )
    return int(out_class[0])


def _float_predict(float_graph, image: np.ndarray) -> tuple[int, np.ndarray]:
    conv1, conv2, gemm, argmax_op = float_graph.ops
    values = interpreter.run(float_graph, image)
    (pred_tensor,) = argmax_op.outputs
    (logits_tensor,) = gemm.outputs
    return int(values[pred_tensor.name]), np.asarray(values[logits_tensor.name]).reshape(-1)


def _shim_raw_logits(shim_lib, quant_graph, image: np.ndarray) -> np.ndarray:
    """Manually orchestrates conv1 -> conv2 -> gemm(skip_requant) through
    the real compiled kernels (via runtime_shim_ported.c), capturing the
    raw int32 accumulator directly -- the generated model_forward()
    deliberately doesn't expose it (see module docstring)."""
    conv1, conv2, gemm, _argmax_op = quant_graph.ops
    quantized_image = quantize_input(image, quant_graph.inputs[0]).reshape(-1)

    def run_conv(op, x, in_ch, in_h, in_w, out_ch, out_h, out_w):
        weight, bias = op.inputs[1], op.inputs[2]
        out = np.zeros(out_ch * out_h * out_w, dtype=np.int8)
        strides = op.attrs["strides"]
        pads = op.attrs["pads"]
        shim_lib.test_conv2d_int8(
            i8p(x), in_ch, in_h, in_w, op.inputs[0].zero_point,
            i8p(weight.data.flatten()), i32p(bias.data), f32p(op.attrs["requant_mult"]),
            out_ch, weight.data.shape[-1], strides[0], pads[0],
            i8p(out), out_h, out_w, op.outputs[0].zero_point,
            1 if op.attrs.get("clamp_min0") else 0,
        )
        return out

    _, in_c, in_h, in_w = quant_graph.inputs[0].shape
    _, out_ch1, out_h1, out_w1 = conv1.outputs[0].shape
    a = run_conv(conv1, quantized_image, in_c, in_h, in_w, out_ch1, out_h1, out_w1)

    _, out_ch2, out_h2, out_w2 = conv2.outputs[0].shape
    b = run_conv(conv2, a, out_ch1, out_h1, out_w1, out_ch2, out_h2, out_w2)

    gemm_weight, gemm_bias = gemm.inputs[1], gemm.inputs[2]
    (out_dim, in_dim) = gemm_weight.shape
    raw_logits = np.zeros(out_dim, dtype=np.int32)
    shim_lib.test_dense_int8(
        i8p(b), in_dim, gemm.inputs[0].zero_point,
        i8p(gemm_weight.data.flatten()), i32p(gemm_bias.data), None,
        out_dim, ctypes.cast(i32p(raw_logits), ctypes.c_void_p), 0, 0, 1,
    )

    input_scale = gemm.inputs[0].scale
    weight_scale = gemm_weight.scale
    return raw_logits.astype(np.float64) * (input_scale * weight_scale)


def test_argmax_match_rate_meets_acceptance_threshold(pipeline, holdout_data):
    float_graph, quant_graph, model_lib, _shim_lib = pipeline
    _calib, eval_images, _eval_labels = holdout_data

    matches = []
    for image in eval_images:
        float_pred, _logits = _float_predict(float_graph, image)
        c_pred = _c_predict(model_lib, quant_graph, image)
        matches.append(c_pred == float_pred)

    match_rate = float(np.mean(matches))
    print(f"\nargmax match rate: {match_rate:.3f} ({sum(matches)}/{len(matches)})")
    assert match_rate >= MATCH_RATE_THRESHOLD


def test_logit_closeness_diagnostic(pipeline, holdout_data):
    float_graph, quant_graph, _model_lib, shim_lib = pipeline
    _calib, eval_images, _eval_labels = holdout_data

    float_logits_all = []
    dequant_logits_all = []
    for image in eval_images:
        _pred, float_logits = _float_predict(float_graph, image)
        dequant_logits = _shim_raw_logits(shim_lib, quant_graph, image)
        float_logits_all.append(float_logits)
        dequant_logits_all.append(dequant_logits)

    float_arr = np.concatenate(float_logits_all)
    dequant_arr = np.concatenate(dequant_logits_all)
    correlation = float(np.corrcoef(float_arr, dequant_arr)[0, 1])
    mae = float(np.mean(np.abs(float_arr - dequant_arr)))

    print(f"\nlogit closeness: correlation={correlation:.4f}, mean_abs_error={mae:.4f}")
    assert correlation >= LOGIT_CORRELATION_FLOOR


def test_true_label_accuracy_reported(pipeline, holdout_data):
    float_graph, quant_graph, model_lib, _shim_lib = pipeline
    _calib, eval_images, eval_labels = holdout_data

    float_correct = 0
    c_correct = 0
    for image, label in zip(eval_images, eval_labels, strict=True):
        float_pred, _logits = _float_predict(float_graph, image)
        c_pred = _c_predict(model_lib, quant_graph, image)
        float_correct += int(float_pred == label)
        c_correct += int(c_pred == label)

    n = len(eval_labels)
    float_acc = float_correct / n
    c_acc = c_correct / n
    print(f"\ntrue-label accuracy: float={float_acc:.3f}, C={c_acc:.3f} (n={n})")
    assert float_acc >= TRUE_LABEL_ACCURACY_FLOOR
    assert c_acc >= TRUE_LABEL_ACCURACY_FLOOR
