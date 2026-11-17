"""Stitch emit_layout + emit_weights + one call-site per op (in
graph.ops order, already topological) into a full model.c source string.
"""

from __future__ import annotations

from ctinyml import closed_ir
from ctinyml.passes.buffer_planner import BufferPlan
from ctinyml.targets.avr.codegen import emit_layout, emit_weights
from ctinyml.targets.avr.codegen.errors import CodegenError
from ctinyml.targets.avr.codegen.registry import EMITTERS

MODEL_C_PRELUDE = (
    "#include <stdint.h>\n"
    '#include "pgm_read_f32.h"\n'
    '#include "qmath.h"\n'
    '#include "conv2d_direct.h"\n'
    '#include "gemm.h"\n'
    '#include "argmax.h"\n\n'
)


def generate_model_c(graph: closed_ir.Graph, buffer_plan: BufferPlan) -> str:
    layout_text, ctx = emit_layout.generate(graph, buffer_plan)
    weights_text = emit_weights.generate(graph, ctx)

    body_lines: list[str] = []
    for op in graph.ops:
        emitter = EMITTERS.get(op.op_type)
        if emitter is None:
            raise CodegenError(f"no codegen emitter registered for op_type {op.op_type!r}")
        # Every op's call_site is a uniform void statement -- no
        # per-op-type special-casing here (see emit_argmax.py's
        # call_site docstring for how the "final op" case is handled
        # without assemble.py needing to know about it).
        body_lines.append(f"    {emitter.call_site(op, ctx)}")

    body = "\n".join(body_lines)
    function = f"void model_forward(const int8_t *input, uint8_t *out_class) {{\n{body}\n}}\n"

    return "\n\n".join([layout_text, weights_text, function])
