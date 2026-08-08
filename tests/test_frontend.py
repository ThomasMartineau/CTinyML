"""Phase 1 acceptance tests — ONNX frontend (ctinyml.frontend.load)."""

from pathlib import Path

import pytest

from ctinyml.frontend.load import FrontendError, load_model
from tests.fixtures import synthetic

MNIST_FIXTURE = Path(__file__).parent / "fixtures" / "mnist" / "mnist_classifier.onnx"

EXPECTED_MNIST_OPS = [
    "Div",
    "Conv",
    "Relu",
    "Conv",
    "Relu",
    "Reshape",
    "Gemm",
    "Reshape",
    "ArgMax",
]


def test_mnist_node_count_and_order():
    model = load_model(MNIST_FIXTURE)
    assert model.graph.num_nodes() == 9
    assert [n.op_type for n in model.graph.all_nodes()] == EXPECTED_MNIST_OPS


def test_mnist_shapes_static():
    model = load_model(MNIST_FIXTURE)
    for node in model.graph.all_nodes():
        for value in (*node.inputs, *node.outputs):
            if value is None:
                continue
            assert value.shape is not None
            assert value.shape.is_static()

    (input_value,) = model.graph.inputs
    assert list(input_value.shape) == [1, 1, 28, 28]


def test_conv_relu_synthetic_loads(tmp_path):
    path = synthetic.conv_relu_graph(tmp_path)
    model = load_model(path)
    assert [n.op_type for n in model.graph.all_nodes()] == ["Conv", "Relu"]


def test_unsupported_op_rejected(tmp_path):
    path = synthetic.unsupported_op_graph(tmp_path)
    with pytest.raises(FrontendError, match="Sigmoid"):
        load_model(path)


def test_dynamic_batch_rejected(tmp_path):
    path = synthetic.dynamic_batch_graph(tmp_path)
    with pytest.raises(FrontendError, match="non-static shape"):
        load_model(path)
