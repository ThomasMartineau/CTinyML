"""Phase 3 acceptance tests — float32 reference interpreter."""

from pathlib import Path

import numpy as np

from ctinyml import interpreter
from ctinyml.frontend.load import load_model
from ctinyml.passes import run_structural_passes
from ctinyml.passes.convert_to_closed_ir import convert
from ctinyml.ops import _argmax, _conv2d, _gemm

MNIST_FIXTURE = Path(__file__).parent / "fixtures" / "mnist" / "mnist_classifier.onnx"


def _mnist_closed_ir():
    model = load_model(MNIST_FIXTURE)
    model = run_structural_passes(model)
    return convert(model)


# --- hand-computed sanity checks -----------------------------------------

# 4x4 input, 1..16 row-major, 3x3 all-ones kernel, stride=2, pad=1 -- the
# same k/s/p the real model uses. Hand-computed expected output (bias=0):
#   corner (0,0): in-bounds cells {(0,0),(0,1),(1,0),(1,1)} = 1+2+5+6 = 14
#   (0,1): {(0,1),(0,2),(0,3),(1,1),(1,2),(1,3)} = 2+3+4+6+7+8 = 30
#   (1,0): {(1,0),(1,1),(2,0),(2,1),(3,0),(3,1)} = 5+6+9+10+13+14 = 57
#   (1,1): full 3x3 window = 6+7+8+10+11+12+14+15+16 = 99
_CONV_INPUT = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
_CONV_WEIGHT = np.ones((1, 1, 3, 3), dtype=np.float32)
_CONV_EXPECTED = np.array([[[[14.0, 30.0], [57.0, 99.0]]]], dtype=np.float32)


def test_conv2d_hand_computed():
    bias = np.zeros((1,), dtype=np.float32)
    out = _conv2d(_CONV_INPUT, _CONV_WEIGHT, bias, strides=(2, 2), pads=(1, 1, 1, 1), clamp_min0=False)
    np.testing.assert_array_equal(out, _CONV_EXPECTED)


def test_conv2d_hand_computed_with_bias():
    bias = np.array([1.0], dtype=np.float32)
    out = _conv2d(_CONV_INPUT, _CONV_WEIGHT, bias, strides=(2, 2), pads=(1, 1, 1, 1), clamp_min0=False)
    np.testing.assert_array_equal(out, _CONV_EXPECTED + 1.0)


def test_conv2d_clamp_min0_zeroes_negative_output():
    bias = np.array([-1000.0], dtype=np.float32)
    out = _conv2d(_CONV_INPUT, _CONV_WEIGHT, bias, strides=(2, 2), pads=(1, 1, 1, 1), clamp_min0=True)
    np.testing.assert_array_equal(out, np.zeros_like(_CONV_EXPECTED))


def test_gemm_hand_computed():
    a = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    weight = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    c = np.zeros((1, 2), dtype=np.float32)
    out = _gemm(a, weight, c, alpha=1.0, beta=1.0, transA=False, transB=True, clamp_min0=False)
    np.testing.assert_array_equal(out, [[10.0, 2.0]])


def test_argmax_hand_computed():
    out = _argmax(
        np.array([10.0, 2.0], dtype=np.float32), axis=0, keepdims=False, select_last_index=False
    )
    assert int(out) == 0


# --- full-graph plumbing sanity -------------------------------------------


def test_full_graph_forward_mnist():
    graph = _mnist_closed_ir()
    x = np.full((1, 1, 28, 28), 128.0, dtype=np.float32)
    values = interpreter.run(graph, x)

    (output_tensor,) = graph.outputs
    pred = values[output_tensor.name]
    assert 0 <= int(pred) <= 9


# --- input_prescale regression (see known_issues.md's Phase 7 entry) ------

# Minimal 1x1 conv graph (weight=[[2.0]], bias=0) so the effect of
# applying/not-applying input_prescale is trivial to verify by hand:
# output = weight * (raw_input / prescale).


def _single_conv_graph(input_prescale):
    from ctinyml import closed_ir

    input_tensor = closed_ir.Tensor(name="x", shape=(1, 1, 1, 1), dtype="FLOAT")
    weight = closed_ir.Tensor(
        name="w", shape=(1, 1, 1, 1), dtype="FLOAT",
        data=np.array([[[[2.0]]]], dtype=np.float32), is_initializer=True,
    )
    bias = closed_ir.Tensor(
        name="b", shape=(1,), dtype="FLOAT", data=np.zeros(1, dtype=np.float32), is_initializer=True,
    )
    output_tensor = closed_ir.Tensor(name="y", shape=(1, 1, 1, 1), dtype="FLOAT")
    op = closed_ir.Op(
        op_type="Conv", name="conv", inputs=[input_tensor, weight, bias], outputs=[output_tensor],
        attrs={"strides": (1, 1), "pads": (0, 0, 0, 0)},
    )
    return closed_ir.Graph(
        ops=[op], inputs=[input_tensor], outputs=[output_tensor], input_prescale=input_prescale
    ), input_tensor, output_tensor


def test_run_applies_input_prescale_to_computation_but_records_raw_value():
    graph, input_tensor, output_tensor = _single_conv_graph(input_prescale=255.0)
    raw = np.array([[[[255.0]]]], dtype=np.float32)

    values = interpreter.run(graph, raw)

    # The tensor recorded/returned under the input's own name stays raw
    # -- that's the domain quantize.py needs to calibrate the input
    # tensor's own scale/zero_point against.
    np.testing.assert_array_equal(values[input_tensor.name], raw)
    # But what actually flowed into the conv is raw/255 = 1.0, so the
    # output is 2*1.0 = 2.0, not 2*255 = 510.
    np.testing.assert_allclose(values[output_tensor.name], [[[[2.0]]]])


def test_run_without_input_prescale_uses_raw_value_directly():
    graph, _input_tensor, output_tensor = _single_conv_graph(input_prescale=None)
    raw = np.array([[[[255.0]]]], dtype=np.float32)

    values = interpreter.run(graph, raw)

    np.testing.assert_allclose(values[output_tensor.name], [[[[510.0]]]])
