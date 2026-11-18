"""ONNX frontend — load a model via `onnx_ir`, validate it's something
this compiler can handle: fully static shapes and a supported op set.

Pipeline: `onnx.load` (resolves external-data sidecars automatically) ->
`onnx.shape_inference.infer_shapes` (onnx_ir doesn't perform shape
inference itself) -> `onnx_ir.from_proto`.
"""

from __future__ import annotations

import os

import onnx
import onnx.shape_inference
import onnx_ir

# The only ops this compiler's backend understands, per
# context/implementation_plan.md Phase 1.
#
# Input contract: the graph input is a float in [0, 1] (a leading Div(255)
# is *not* supported -- the fixed INT8 input encoding in passes/quantize.py
# already is that /255). Any other input scaling must be folded into the
# first layer's weights before the model reaches this compiler.
SUPPORTED_OPS = {"Conv", "Relu", "Reshape", "Gemm", "ArgMax"}


class FrontendError(ValueError):
    """Raised when a model can't be accepted by the CTinyML frontend."""


def load_model(path: str | os.PathLike) -> onnx_ir.Model:
    """Load and validate an ONNX model for the CTinyML pipeline.

    Raises FrontendError if any tensor has a non-static shape (including
    a dynamic batch dim) or if the graph contains an op outside
    SUPPORTED_OPS.
    """
    proto = onnx.load(str(path))
    inferred = onnx.shape_inference.infer_shapes(proto)
    model = onnx_ir.from_proto(inferred)

    _validate_static_shapes(model.graph)
    _validate_op_set(model.graph)

    return model


def _iter_all_values(graph: onnx_ir.Graph):
    """Yield every Value reachable in the graph: inputs, outputs, and
    every node's inputs/outputs."""
    seen = set()
    for values in (graph.inputs, graph.outputs):
        for value in values:
            if id(value) not in seen:
                seen.add(id(value))
                yield value
    for node in graph.all_nodes():
        for values in (node.inputs, node.outputs):
            for value in values:
                if value is None or id(value) in seen:
                    continue
                seen.add(id(value))
                yield value


def _validate_static_shapes(graph: onnx_ir.Graph) -> None:
    for value in _iter_all_values(graph):
        shape = value.shape
        if shape is None or not shape.is_static():
            raise FrontendError(
                f"Tensor {value.name!r} has a non-static shape ({shape!r}); "
                "CTinyML requires every intermediate tensor (including the "
                "batch dimension) to be fully static."
            )


def _validate_op_set(graph: onnx_ir.Graph) -> None:
    for node in graph.all_nodes():
        if node.op_type not in SUPPORTED_OPS:
            raise FrontendError(
                f"Unsupported op {node.op_type!r} (node {node.name!r}); "
                f"CTinyML only supports {sorted(SUPPORTED_OPS)}."
            )
