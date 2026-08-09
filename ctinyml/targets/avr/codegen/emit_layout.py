"""Orchestrates `#define` shape/zero-point macros and the `buf_a`/`buf_b`
ping-pong buffer declarations (matching the reference file), delegating
each op's own macro block to `targets/avr/codegen/registry.py`'s
`EMITTERS`. Builds the `LayoutContext` every other emit module needs.
"""

from __future__ import annotations

from ctinyml import closed_ir, layout
from ctinyml.passes.buffer_planner import BufferPlan
from ctinyml.targets.avr.codegen import base
from ctinyml.targets.avr.codegen.errors import CodegenError
from ctinyml.targets.avr.codegen.registry import EMITTERS


def generate(graph: closed_ir.Graph, buffer_plan: BufferPlan) -> tuple[str, base.LayoutContext]:
    (input_tensor,) = graph.inputs
    ctx = base.LayoutContext(macro_names=layout.macro_names(graph))

    lines: list[str] = []
    _, _in_c, in_h, in_w = input_tensor.shape
    lines += [f"#define IMG_H {in_h}", f"#define IMG_W {in_w}"]

    zp_define = base.zero_point_define(input_tensor)
    if zp_define is not None:
        ctx.zero_point_macros[input_tensor.name] = base.zero_point_macro_name(input_tensor)
        lines.append(zp_define)
    lines.append("")

    for op in graph.ops:
        emitter = EMITTERS.get(op.op_type)
        if emitter is None:
            raise CodegenError(f"no codegen emitter registered for op_type {op.op_type!r}")
        macros = emitter.layout_macros(op, ctx)
        if macros:
            lines += [*macros, ""]

    lines += [
        f"static int8_t buf_a[{buffer_plan.buffer_size}];",
        f"static int8_t buf_b[{buffer_plan.buffer_size}];",
    ]

    # Identity-keyed (not `in`/`==` on closed_ir.Tensor -- see
    # buffer_planner.py's docstring on why: value-equality would compare
    # .data numpy arrays and can raise).
    graph_output_ids = {id(t) for t in graph.outputs}

    def tensor_ref(tensor: closed_ir.Tensor) -> str:
        # Graph-topology-driven, not op-type-driven: whichever op happens
        # to produce the graph's declared output writes to the
        # model_forward() output parameter instead of a ping-pong buffer
        # -- true for ArgMax in this architecture, but not hardcoded to
        # ArgMax specifically, so a differently-shaped future graph
        # doesn't need this function to change.
        if tensor is input_tensor:
            return "input"
        if id(tensor) in graph_output_ids:
            return "out_class"
        if tensor.buffer is None:
            raise CodegenError(f"tensor {tensor.name!r} has no buffer assignment.")
        name = "buf_a" if tensor.buffer == 0 else "buf_b"
        return f"(int32_t*){name}" if tensor.dtype == "INT32" else name

    ctx.tensor_ref = tensor_ref

    return "\n".join(lines) + "\n", ctx
