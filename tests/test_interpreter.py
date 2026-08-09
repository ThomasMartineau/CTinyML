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
