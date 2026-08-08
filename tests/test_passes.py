"""Phase 2 acceptance tests — structural onnx_ir passes."""

from pathlib import Path

import pytest

from ctinyml.frontend.load import load_model
from ctinyml.passes import StructuralPassError, run_structural_passes
from ctinyml.passes.onnx_ir_passes.drop_noop_reshape import DropNoopReshapePass
from ctinyml.passes.onnx_ir_passes.fold_div_into_quant import FoldDivIntoQuantPass
from ctinyml.passes.onnx_ir_passes.fuse_relu import FuseReluPass
from tests.fixtures import synthetic

MNIST_FIXTURE = Path(__file__).parent / "fixtures" / "mnist" / "mnist_classifier.onnx"


# --- drop_noop_reshape ------------------------------------------------


def test_drop_noop_reshape_removes_node(tmp_path):
    model = load_model(synthetic.conv_reshape_gemm_graph(tmp_path))
    result = DropNoopReshapePass()(model)

    op_types = [n.op_type for n in result.model.graph.all_nodes()]
    assert op_types == ["Conv", "Gemm"]

    gemm = next(n for n in result.model.graph.all_nodes() if n.op_type == "Gemm")
    assert gemm.inputs[0].producer().op_type == "Conv"


def test_drop_noop_reshape_rejects_mismatched_shape():
    model = synthetic.reshape_bad_shape_graph()
    with pytest.raises(StructuralPassError, match="not a no-op-eligible"):
        DropNoopReshapePass()(model)


# --- fuse_relu ----------------------------------------------------------


def test_fuse_relu_happy_path(tmp_path):
    model = load_model(synthetic.conv_relu_graph(tmp_path))
    result = FuseReluPass()(model)

    op_types = [n.op_type for n in result.model.graph.all_nodes()]
    assert op_types == ["Conv"]

    conv = next(n for n in result.model.graph.all_nodes() if n.op_type == "Conv")
    assert conv.attributes.get_int("clamp_min0") == 1
    (graph_output,) = result.model.graph.outputs
    assert graph_output.producer() is conv


def test_fuse_relu_rejects_branching(tmp_path):
    model = load_model(synthetic.conv_relu_branching_graph(tmp_path))
    with pytest.raises(StructuralPassError, match="could not be fused"):
        FuseReluPass()(model)


# --- fold_div_into_quant -------------------------------------------------


def test_fold_div_into_quant_happy_path(tmp_path):
    model = load_model(synthetic.div_relu_graph(tmp_path))
    result = FoldDivIntoQuantPass()(model)

    op_types = [n.op_type for n in result.model.graph.all_nodes()]
    assert op_types == ["Relu"]
    assert result.model.graph.meta["input_prescale"] == 2.0

    relu = next(n for n in result.model.graph.all_nodes() if n.op_type == "Relu")
    assert relu.inputs[0] is result.model.graph.inputs[0]


def test_fold_div_into_quant_requires_div(tmp_path):
    model = load_model(synthetic.conv_relu_graph(tmp_path))
    with pytest.raises(StructuralPassError, match="No Div"):
        FoldDivIntoQuantPass()(model)


# --- end-to-end on the real model ----------------------------------------


def test_structural_passes_on_mnist():
    model = load_model(MNIST_FIXTURE)
    model = run_structural_passes(model)

    op_types = [n.op_type for n in model.graph.all_nodes()]
    assert op_types == ["Conv", "Conv", "Gemm", "ArgMax"]

    convs = [n for n in model.graph.all_nodes() if n.op_type == "Conv"]
    assert len(convs) == 2
    assert all(n.attributes.get_int("clamp_min0") == 1 for n in convs)

    assert model.graph.meta["input_prescale"] == 255.0
