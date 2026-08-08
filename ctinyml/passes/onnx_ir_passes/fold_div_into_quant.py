"""Detect Div(input, scalar) at the graph input, record as `input_prescale`
graph metadata, remove the node, rewire.

This pass only *records* the value; actual folding into layer-1's
quantization scale happens in Phase 3.

This compiler currently targets one fixed architecture (input
normalization is always a Div-by-scalar at the graph input), so a missing
Div pattern is a hard error rather than a silent no-op, matching the
other two structural passes' fail-loudly stance.
"""

from __future__ import annotations

import onnx_ir as ir

from ctinyml.passes.errors import StructuralPassError


class FoldDivIntoQuantPass(ir.passes.InPlacePass):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        div_node = _find_input_div(graph)
        if div_node is None:
            raise StructuralPassError(
                "No Div(graph_input, scalar) node found at the graph input; "
                "this compiler expects input normalization to be a scalar "
                "Div on the graph's first input."
            )

        data, scalar = div_node.inputs
        graph.meta["input_prescale"] = float(scalar.const_value.numpy())

        ir.convenience.replace_all_uses_with(
            div_node.outputs[0], data, replace_graph_outputs=True
        )
        graph.remove(div_node, safe=True)
        _drop_if_unused(graph, scalar)

        return ir.passes.PassResult(model, modified=True)


def _find_input_div(graph: ir.Graph) -> ir.Node | None:
    graph_inputs = frozenset(graph.inputs)
    for node in graph.all_nodes():
        if node.op_type != "Div" or len(node.inputs) != 2:
            continue
        data, scalar = node.inputs
        if data not in graph_inputs:
            continue
        if scalar is None or not scalar.is_initializer():
            continue
        if scalar.const_value is None or tuple(scalar.shape) != ():
            continue
        return node
    return None


def _drop_if_unused(graph: ir.Graph, value: ir.Value | None) -> None:
    if value is None or not value.is_initializer() or value.name is None:
        return
    if value.uses() or value in graph.outputs:
        return
    del graph.initializers[value.name]
