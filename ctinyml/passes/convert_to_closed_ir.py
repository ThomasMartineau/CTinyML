"""onnx_ir.Graph -> closed_ir.Graph conversion — the Phase 2/3 boundary.

Takes an already-reduced model (post `run_structural_passes`) and walks it
once, building our own minimal `closed_ir.Graph`. Everything from Phase 3
onward operates on that IR, not `onnx_ir` — upstream stays general-purpose,
downstream stays narrow.
"""

from __future__ import annotations

import onnx_ir as ir

from ctinyml import closed_ir


def convert(model: ir.Model) -> closed_ir.Graph:
    graph = model.graph
    tensor_cache: dict[int, closed_ir.Tensor] = {}

    def to_tensor(value: ir.Value) -> closed_ir.Tensor:
        cached = tensor_cache.get(id(value))
        if cached is not None:
            return cached
        data = None
        if value.const_value is not None:
            data = value.const_value.numpy()
        tensor = closed_ir.Tensor(
            name=value.name or "",
            shape=tuple(value.shape),
            dtype=str(value.dtype),
            data=data,
            is_initializer=value.is_initializer(),
        )
        tensor_cache[id(value)] = tensor
        return tensor

    ops = []
    for node in graph.all_nodes():
        attrs = {name: attr.value for name, attr in node.attributes.items()}
        ops.append(
            closed_ir.Op(
                op_type=node.op_type,
                name=node.name or "",
                inputs=[to_tensor(v) for v in node.inputs if v is not None],
                outputs=[to_tensor(v) for v in node.outputs if v is not None],
                attrs=attrs,
            )
        )

    return closed_ir.Graph(
        ops=ops,
        inputs=[to_tensor(v) for v in graph.inputs],
        outputs=[to_tensor(v) for v in graph.outputs],
    )
