"""Phase 5 acceptance tests — target-agnostic layout helpers."""

from pathlib import Path

from ctinyml import layout
from ctinyml.frontend.load import load_model
from ctinyml.passes import run_structural_passes
from ctinyml.passes.convert_to_closed_ir import convert

MNIST_FIXTURE = Path(__file__).parent / "fixtures" / "mnist" / "mnist_classifier.onnx"


def _mnist_closed_ir():
    model = load_model(MNIST_FIXTURE)
    model = run_structural_passes(model)
    return convert(model)


def test_conv_output_dim_matches_real_model():
    # The stride-2/pad-1 boundary math, checked against the real model's
    # actual dimensions.
    assert layout.conv_output_dim(28, 3, 2, 1) == 14
    assert layout.conv_output_dim(14, 3, 2, 1) == 7


def test_macro_names_on_mnist():
    graph = _mnist_closed_ir()
    names = layout.macro_names(graph)
    assert names == {
        "node_conv2d": "CONV1",
        "node_conv2d_1": "CONV2",
        "node_linear": "GEMM1",
    }
