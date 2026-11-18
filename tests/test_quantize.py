"""Phase 3 acceptance tests — calibration + weight/activation quantization."""

from pathlib import Path

import numpy as np
import pytest

from ctinyml.frontend.load import load_model
from ctinyml.passes import run_structural_passes
from ctinyml.passes.convert_to_closed_ir import convert
from ctinyml.passes.drop_final_requant import apply as drop_final_requant
from ctinyml.passes.errors import ClosedIRPassError
from ctinyml.passes.onnx_ir_passes.fuse_relu import FuseReluPass
from ctinyml.passes.quantize import (
    INPUT_SCALE,
    INPUT_ZERO_POINT,
    QMAX,
    QMIN,
    _asymmetric_qparams,
    _calibrate_activations,
    _requant_multiplier,
    quantize,
    quantize_input,
)
from tests.fixtures import synthetic

MNIST_FIXTURE = Path(__file__).parent / "fixtures" / "mnist" / "mnist_classifier.onnx"


def _mnist_closed_ir():
    model = load_model(MNIST_FIXTURE)
    model = run_structural_passes(model)
    return convert(model)


# --- pure-function unit tests ---------------------------------------------


@pytest.mark.parametrize(
    "input_scale,weight_scale,output_scale",
    [
        (1.0, 0.01, 1.0),
        (0.5, 0.001, 2.0),
        (1e-4, 1e-4, 1e-4),
        (0.9, 0.99, 1.01),
        (123.4, 0.0002, 5.0),
    ],
)
def test_requant_multiplier_various_scales(input_scale, weight_scale, output_scale):
    weight_scale_arr = np.array([weight_scale, weight_scale * 2], dtype=np.float64)
    expected = input_scale * weight_scale_arr / output_scale
    actual = _requant_multiplier(input_scale, weight_scale_arr, output_scale)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_asymmetric_qparams_reproduces_calibrated_range():
    lo, hi = -2.0, 5.0
    scale, zero_point = _asymmetric_qparams(lo, hi)

    assert round(lo / scale) + zero_point == QMIN
    assert round(hi / scale) + zero_point == QMAX


def test_asymmetric_qparams_widens_to_include_zero():
    # An all-positive range (e.g. post-ReLU) should still map 0 exactly.
    scale, zero_point = _asymmetric_qparams(3.0, 10.0)
    assert round(0.0 / scale) + zero_point == QMIN  # lo was widened to 0


def test_asymmetric_qparams_guards_zero_width_interval():
    scale, zero_point = _asymmetric_qparams(0.0, 0.0)
    assert scale > 0
    assert QMIN <= zero_point <= QMAX


# --- weight quantization ---------------------------------------------------


def test_weight_quantization_symmetric_per_channel():
    from ctinyml import closed_ir

    rng = np.random.default_rng(0)
    # Deliberately varied per-channel magnitudes: small, near-1.0, large.
    magnitudes = np.array([0.01, 1.0, 100.0])
    weight_data = np.stack(
        [rng.uniform(-m, m, size=(2, 3, 3)).astype(np.float32) for m in magnitudes]
    )
    input_tensor = closed_ir.Tensor(name="in", shape=(1,), dtype="INT8", scale=1.0, zero_point=0)
    weight_tensor = closed_ir.Tensor(
        name="w", shape=weight_data.shape, dtype="FLOAT", data=weight_data, is_initializer=True
    )
    bias_tensor = closed_ir.Tensor(
        name="b",
        shape=(3,),
        dtype="FLOAT",
        data=np.zeros(3, dtype=np.float32),
        is_initializer=True,
    )
    op = closed_ir.Op(
        op_type="Conv", name="conv", inputs=[input_tensor, weight_tensor, bias_tensor], outputs=[]
    )

    from ctinyml.passes.quantize import _quantize_weight_and_bias

    weight_scale = _quantize_weight_and_bias(op, input_tensor.scale)

    assert weight_tensor.zero_point == 0
    assert weight_tensor.data.dtype == np.int8
    assert weight_tensor.data.min() >= -127
    assert weight_tensor.data.max() <= 127

    dequantized = weight_tensor.data.astype(np.float64) * weight_scale.reshape(-1, 1, 1, 1)
    for c in range(3):
        tol = weight_scale[c] / 2 + 1e-9
        assert np.max(np.abs(dequantized[c] - weight_data[c])) <= tol


# --- drop_final_requant ------------------------------------------------


def test_drop_final_requant_marks_gemm():
    graph = _mnist_closed_ir()
    drop_final_requant(graph)

    marked = [op for op in graph.ops if op.attrs.get("skip_requant")]
    assert len(marked) == 1
    assert marked[0].op_type == "Gemm"

    convs = [op for op in graph.ops if op.op_type == "Conv"]
    assert all(not op.attrs.get("skip_requant") for op in convs)


def test_drop_final_requant_requires_argmax():
    from ctinyml import closed_ir

    graph = closed_ir.Graph(ops=[], inputs=[], outputs=[])
    with pytest.raises(ClosedIRPassError, match="ArgMax"):
        drop_final_requant(graph)


# --- calibration integration -----------------------------------------------


def test_calibrate_activations_on_synthetic_graph(tmp_path):
    # Run only the pass this synthetic graph needs (fuse_relu) rather
    # than the full run_structural_passes.
    model = load_model(synthetic.conv_relu_graph(tmp_path))
    model = FuseReluPass()(model).model
    graph = convert(model)

    shape = graph.inputs[0].shape
    low_sample = np.full(shape, -3.0, dtype=np.float32)
    high_sample = np.full(shape, 5.0, dtype=np.float32)
    mid_sample = np.zeros(shape, dtype=np.float32)

    _calibrate_activations(graph, [low_sample, mid_sample, high_sample])

    # The graph input is not calibrated: it always gets the fixed [0, 1] -> int8
    # encoding (scale 1/255, zp -128), whatever range the data spans.
    assert graph.inputs[0].scale == pytest.approx(INPUT_SCALE) == pytest.approx(1.0 / 255.0)
    assert graph.inputs[0].zero_point == INPUT_ZERO_POINT == -128
    assert graph.inputs[0].dtype == "INT8"


def test_quantize_input_maps_uint8_pixels_to_pixel_minus_128(tmp_path):
    model = load_model(synthetic.conv_relu_graph(tmp_path))
    graph = convert(FuseReluPass()(model).model)
    _calibrate_activations(graph, [np.zeros(graph.inputs[0].shape, dtype=np.float32)])

    pixels = np.arange(256)
    quantized = quantize_input((pixels / 255.0).astype(np.float32), graph.inputs[0])

    np.testing.assert_array_equal(quantized, (pixels - 128).astype(np.int8))


# --- full pipeline on the real model ---------------------------------------


def test_quantize_end_to_end_mnist():
    graph = _mnist_closed_ir()
    drop_final_requant(graph)

    rng = np.random.default_rng(0)
    calibration_data = [
        rng.uniform(0, 1, size=(1, 1, 28, 28)).astype(np.float32) for _ in range(8)
    ]
    calibration_data.append(np.zeros((1, 1, 28, 28), dtype=np.float32))
    calibration_data.append(np.full((1, 1, 28, 28), 1.0, dtype=np.float32))

    quantize(graph, calibration_data)

    conv1, conv2, gemm, _argmax = graph.ops

    for conv in (conv1, conv2):
        weight, bias = conv.inputs[1], conv.inputs[2]
        assert weight.dtype == "INT8"
        assert weight.data.dtype == np.int8
        assert weight.scale.shape == (6,)
        assert bias.dtype == "INT32"
        assert bias.data.dtype == np.int32
        assert bias.scale.shape == (6,)
        assert conv.attrs["requant_mult"].shape == (6,)

    gemm_weight, gemm_bias = gemm.inputs[1], gemm.inputs[2]
    assert gemm_weight.dtype == "INT8"
    assert gemm_weight.scale.shape == (10,)
    assert gemm_bias.dtype == "INT32"
    assert gemm_bias.scale.shape == (10,)
    assert "requant_mult" not in gemm.attrs
    assert gemm.outputs[0].scale is None
    assert gemm.outputs[0].dtype == "INT32"

    input_tensor = graph.inputs[0]
    assert input_tensor.scale is not None
    assert input_tensor.zero_point is not None
    assert input_tensor.dtype == "INT8"


def test_quantize_excludes_argmax_output():
    # Regression test: ArgMax's output is a discrete class index (int64),
    # never meant to be calibrated/quantized -- see NON_CALIBRATED_OPS.
    graph = _mnist_closed_ir()
    drop_final_requant(graph)
    rng = np.random.default_rng(0)
    calibration_data = [
        rng.uniform(0, 1, size=(1, 1, 28, 28)).astype(np.float32) for _ in range(8)
    ]
    quantize(graph, calibration_data)

    (output_tensor,) = graph.outputs
    assert output_tensor.dtype == "INT64"
    assert output_tensor.scale is None
    assert output_tensor.zero_point is None


def test_quantize_input_reproduces_calibrated_boundaries():
    from ctinyml import closed_ir

    tensor = closed_ir.Tensor(name="in", shape=(1,), dtype="INT8")
    scale, zero_point = _asymmetric_qparams(-2.0, 5.0)
    tensor.scale, tensor.zero_point = scale, zero_point

    lo_q = quantize_input(np.array([-2.0], dtype=np.float32), tensor)
    hi_q = quantize_input(np.array([5.0], dtype=np.float32), tensor)

    assert int(lo_q[0]) == QMIN
    assert int(hi_q[0]) == QMAX


def test_quantize_input_requires_calibrated_tensor():
    from ctinyml import closed_ir

    tensor = closed_ir.Tensor(name="in", shape=(1,), dtype="FLOAT")
    with pytest.raises(ValueError, match="no calibrated"):
        quantize_input(np.array([1.0], dtype=np.float32), tensor)


def test_holdout_int8_images_match_the_compilers_input_encoding():
    # tests/fixtures/mnist_holdout.npz stores the exact bytes the board
    # consumes (`eval_images_int8`, = pixel - 128) next to the float [0, 1]
    # images. tools/serial_infer.py sends them as-is, so they must never
    # drift from the compiler's fixed input encoding.
    holdout = np.load(Path(__file__).parent / "fixtures" / "mnist_holdout.npz")
    graph = _mnist_closed_ir()
    quantize(graph, holdout["calibration_images"][:, np.newaxis])

    expected = quantize_input(holdout["eval_images"], graph.inputs[0])

    assert holdout["eval_images_int8"].dtype == np.int8
    np.testing.assert_array_equal(holdout["eval_images_int8"], expected)
