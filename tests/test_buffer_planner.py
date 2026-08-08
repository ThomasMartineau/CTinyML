"""Phase 4 acceptance tests — liveness-based ping-pong buffer planner."""

from pathlib import Path

import numpy as np
import pytest

from ctinyml import closed_ir
from ctinyml.frontend.load import load_model
from ctinyml.passes import run_structural_passes
from ctinyml.passes.buffer_planner import (
    BUFFER_BUDGET_BYTES,
    BUFFER_WARN_BYTES,
    plan,
)
from ctinyml.passes.convert_to_closed_ir import convert
from ctinyml.passes.drop_final_requant import apply as drop_final_requant
from ctinyml.passes.errors import ClosedIRPassError
from ctinyml.passes.quantize import quantize

MNIST_FIXTURE = Path(__file__).parent / "fixtures" / "mnist" / "mnist_classifier.onnx"


def _mnist_quantized_closed_ir():
    graph = convert(run_structural_passes(load_model(MNIST_FIXTURE)))
    drop_final_requant(graph)
    rng = np.random.default_rng(0)
    calibration_data = [
        rng.uniform(0, 255, size=(1, 1, 28, 28)).astype(np.float32) for _ in range(8)
    ]
    quantize(graph, calibration_data)
    return graph


# --- real MNIST graph -------------------------------------------------


def test_plan_on_mnist_matches_hand_computed_sizes():
    graph = _mnist_quantized_closed_ir()
    result = plan(graph)

    assert result.buffer_size == 1176
    assert result.num_buffers == 2
    assert result.total_sram_bytes == 2352
    assert result.warning is None
    assert set(result.assignments) == {"conv2d", "conv2d_1", "linear"}


def test_plan_on_mnist_alternates_buffers_and_uses_only_two():
    graph = _mnist_quantized_closed_ir()
    result = plan(graph)

    assert result.assignments["conv2d"] != result.assignments["conv2d_1"]
    assert all(v in (0, 1) for v in result.assignments.values())


def test_plan_sets_tensor_buffer_field():
    graph = _mnist_quantized_closed_ir()
    result = plan(graph)

    conv1, conv2, gemm, _argmax = graph.ops
    assert conv1.outputs[0].buffer == result.assignments["conv2d"]
    assert conv2.outputs[0].buffer == result.assignments["conv2d_1"]
    assert gemm.outputs[0].buffer == result.assignments["linear"]


# --- synthetic edge cases (plain closed_ir dataclasses, no onnx needed) --


def _chain_graph(shapes_dtypes: list[tuple[tuple[int, ...], str]]) -> closed_ir.Graph:
    """Build a strict N-op chain: op[i] consumes tensor[i], produces
    tensor[i+1]. tensor[0] is the graph input, tensor[-1] the graph output."""
    tensors = [
        closed_ir.Tensor(name=f"t{i}", shape=shape, dtype=dtype)
        for i, (shape, dtype) in enumerate(shapes_dtypes)
    ]
    ops = [
        closed_ir.Op(op_type="Op", name=f"op{i}", inputs=[tensors[i]], outputs=[tensors[i + 1]])
        for i in range(len(tensors) - 1)
    ]
    return closed_ir.Graph(ops=ops, inputs=[tensors[0]], outputs=[tensors[-1]])


def test_raises_when_third_buffer_needed():
    t0 = closed_ir.Tensor(name="t0", shape=(4,), dtype="INT8")
    t1 = closed_ir.Tensor(name="t1", shape=(4,), dtype="INT8")
    t2 = closed_ir.Tensor(name="t2", shape=(4,), dtype="INT8")
    t3 = closed_ir.Tensor(name="t3", shape=(4,), dtype="INT8")

    op0 = closed_ir.Op(op_type="Op", name="op0", inputs=[], outputs=[t0])
    op1 = closed_ir.Op(op_type="Op", name="op1", inputs=[t0], outputs=[t1])
    # op2 fans in both t0 and t1 -- both must still be live here, alongside
    # the newly-defined t2, forcing a 3rd simultaneously-live buffer. t2
    # must itself be consumed by a later op (not the graph's final output,
    # which is excluded from planning entirely) for the conflict to bite.
    op2 = closed_ir.Op(op_type="Op", name="op2", inputs=[t0, t1], outputs=[t2])
    op3 = closed_ir.Op(op_type="Op", name="op3", inputs=[t2], outputs=[t3])

    graph = closed_ir.Graph(ops=[op0, op1, op2, op3], inputs=[], outputs=[t3])

    with pytest.raises(ClosedIRPassError, match="3rd simultaneously-live buffer"):
        plan(graph)


def test_hard_fails_over_sram_budget():
    oversized = BUFFER_BUDGET_BYTES + 1
    graph = _chain_graph([((1,), "INT8"), ((oversized,), "INT8"), ((1,), "INT8")])

    with pytest.raises(ClosedIRPassError, match="8 KB"):
        plan(graph)


def test_warns_between_soft_and_hard_budget():
    between = BUFFER_WARN_BYTES // 2 + 100  # so 2x lands between warn and budget
    assert BUFFER_WARN_BYTES < 2 * between <= BUFFER_BUDGET_BYTES
    graph = _chain_graph([((1,), "INT8"), ((between,), "INT8"), ((1,), "INT8")])

    result = plan(graph)

    assert result.warning is not None
    assert result.total_sram_bytes == 2 * between
