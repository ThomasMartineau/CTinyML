"""Remove Reshape nodes that are row-major-compatible flatten/unflatten
no-ops; rewire producer->consumer directly. Assert (don't silently skip)
on an ineligible Reshape.

Per ONNX semantics, Reshape never physically moves data — it's always a
row-major reinterpretation of the same flat buffer — so the element-count
check below can't actually fail for a valid graph. It exists as a
defensive invariant guard for the "aliasing, no data movement" assumption
the rest of the pipeline depends on; the loud-failure path is exercised in
tests against a deliberately hand-built invalid graph.

Note: rewiring "producer -> consumer directly" means the consumer's input
Value keeps the *pre-reshape* shape (e.g. a Gemm fed by a flattened Conv
output ends up with a (1,6,7,7)-shaped input Tensor, not (1,294)). This is
intentional: nothing downstream cares about rank, only total element
count — Phase 5 codegen derives Gemm's K dimension from prod(shape).
"""

from __future__ import annotations

import math

import onnx_ir as ir

from ctinyml.passes.errors import StructuralPassError


class DropNoopReshapePass(ir.passes.InPlacePass):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        reshape_nodes = [n for n in graph.all_nodes() if n.op_type == "Reshape"]
        modified = False

        for node in reshape_nodes:
            data, shape_tensor = node.inputs[0], node.inputs[1]
            output = node.outputs[0]

            in_count = math.prod(data.shape)
            out_count = math.prod(output.shape)
            if in_count != out_count:
                raise StructuralPassError(
                    f"Reshape node {node.name!r} is not a no-op-eligible "
                    f"flatten/unflatten: input {tuple(data.shape)} has "
                    f"{in_count} elements, output {tuple(output.shape)} "
                    f"has {out_count}."
                )

            ir.convenience.replace_all_uses_with(output, data, replace_graph_outputs=True)
            graph.remove(node, safe=True)
            _drop_if_unused(graph, shape_tensor)
            modified = True

        remaining = [n for n in graph.all_nodes() if n.op_type == "Reshape"]
        if remaining:
            raise StructuralPassError(
                f"Reshape node(s) left unhandled: {[n.name for n in remaining]}"
            )

        return ir.passes.PassResult(model, modified=modified)


def _drop_if_unused(graph: ir.Graph, value: ir.Value | None) -> None:
    if value is None or not value.is_initializer() or value.name is None:
        return
    if value.uses() or value in graph.outputs:
        return
    del graph.initializers[value.name]
