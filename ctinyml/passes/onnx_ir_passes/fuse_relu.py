"""Fuse Relu into the preceding Conv/Gemm as a `clamp_min0=1` attribute;
delete the Relu node. Assert every Relu gets fused (fail loudly on an
unfused one — that signals a graph shape this compiler doesn't support
yet, e.g. branching).
"""

from __future__ import annotations

import onnx_ir as ir

from ctinyml.passes.errors import StructuralPassError

_FUSABLE_PRODUCERS = {"Conv", "Gemm"}


def _label(node: ir.Node) -> str:
    return node.name or (node.outputs[0].name if node.outputs else "<unnamed>")


class FuseReluPass(ir.passes.InPlacePass):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        relu_nodes = [n for n in graph.all_nodes() if n.op_type == "Relu"]
        modified = False

        for node in relu_nodes:
            producer = node.inputs[0].producer()
            if producer is None or producer.op_type not in _FUSABLE_PRODUCERS:
                continue
            # Sole-consumer guard: fusing into a producer with other
            # consumers would silently clamp their view of the value too.
            if len(list(producer.outputs[0].uses())) != 1:
                continue

            producer.attributes.add(ir.AttrInt64("clamp_min0", 1))
            ir.convenience.replace_all_uses_with(
                node.outputs[0], producer.outputs[0], replace_graph_outputs=True
            )
            graph.remove(node, safe=True)
            modified = True

        remaining = [n for n in graph.all_nodes() if n.op_type == "Relu"]
        if remaining:
            labels = [_label(n) for n in remaining]
            raise StructuralPassError(
                f"Relu node(s) could not be fused: {labels} "
                "(producer is not a sole-consumer Conv/Gemm — unsupported graph "
                "shape, e.g. branching)."
            )

        return ir.passes.PassResult(model, modified=modified)
