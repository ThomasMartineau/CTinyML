"""Mark the last Gemm (feeding ArgMax) as `skip_requant=True` — its int32
accumulator is emitted directly, no output scale/zero-point.
"""

from __future__ import annotations

from ctinyml import closed_ir
from ctinyml.passes.errors import ClosedIRPassError


def apply(graph: closed_ir.Graph) -> closed_ir.Graph:
    argmax_ops = [op for op in graph.ops if op.op_type == "ArgMax"]
    if len(argmax_ops) != 1:
        raise ClosedIRPassError(
            f"Expected exactly one ArgMax op, found {len(argmax_ops)}."
        )
    (argmax,) = argmax_ops
    (argmax_input,) = argmax.inputs

    producer = next((op for op in graph.ops if argmax_input in op.outputs), None)
    if producer is None:
        raise ClosedIRPassError(
            f"No op in the graph produces ArgMax's input {argmax_input.name!r}."
        )
    if producer.op_type != "Gemm":
        raise ClosedIRPassError(
            f"ArgMax's producer is {producer.op_type!r} (node {producer.name!r}), "
            "expected Gemm — this compiler only supports a final Gemm feeding "
            "ArgMax directly."
        )

    producer.attrs["skip_requant"] = True
    return graph
