"""Phase 2 acceptance tests — closed_ir conversion + golden snapshot."""

from pathlib import Path

from ctinyml.frontend.load import load_model
from ctinyml.passes import run_structural_passes
from ctinyml.passes.convert_to_closed_ir import convert

MNIST_FIXTURE = Path(__file__).parent / "fixtures" / "mnist" / "mnist_classifier.onnx"
GOLDEN_FIXTURE = Path(__file__).parent / "fixtures" / "golden" / "mnist_closed_ir.txt"


def _mnist_closed_ir():
    model = load_model(MNIST_FIXTURE)
    model = run_structural_passes(model)
    return convert(model)


def test_convert_mnist_structure():
    graph = _mnist_closed_ir()

    assert [op.op_type for op in graph.ops] == ["Conv", "Conv", "Gemm", "ArgMax"]
    assert graph.input_prescale == 255.0

    conv1, conv2, gemm, argmax = graph.ops
    assert conv1.attrs["clamp_min0"] == 1
    assert conv2.attrs["clamp_min0"] == 1

    weight = next(t for t in conv1.inputs if t.name == "net.block1.conv.weight")
    assert weight.is_initializer
    assert weight.shape == (6, 1, 3, 3)
    assert weight.data is not None
    assert weight.data.shape == (6, 1, 3, 3)


def test_golden_snapshot():
    graph = _mnist_closed_ir()
    actual = graph.to_text()
    expected = GOLDEN_FIXTURE.read_text()
    assert actual == expected
