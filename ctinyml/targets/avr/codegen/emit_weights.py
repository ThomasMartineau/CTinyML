"""Emit int8 weight arrays, int32 bias arrays, and per-channel requant
multiplier tables as `const ... PROGMEM` declarations.

Plain function, not a per-op-type Emitter method: flattening an
already-quantized Tensor.data into a C array initializer is identical
logic regardless of whether the tensor came from a Conv or a Gemm, only
the size differs. Which ops have weights at all is looked up from the
EMITTERS registry's HAS_WEIGHTS marker, not a separately-hardcoded
op_type tuple.
"""

from __future__ import annotations

import numpy as np

from ctinyml import closed_ir
from ctinyml.targets.avr.codegen import base
from ctinyml.targets.avr.codegen.registry import EMITTERS


def generate(graph: closed_ir.Graph, ctx: base.LayoutContext) -> str:
    lines: list[str] = []
    for op in graph.ops:
        emitter = EMITTERS.get(op.op_type)
        if emitter is None or not emitter.HAS_WEIGHTS:
            continue

        prefix = ctx.macro_names[op.name]
        weight, bias = op.inputs[1], op.inputs[2]

        lines.append(_array_decl("int8_t", base.weight_array_name(prefix), weight.data, _int_fmt))
        lines.append(_array_decl("int32_t", base.bias_array_name(prefix), bias.data, _int_fmt))
        if not op.attrs.get("skip_requant"):
            lines.append(
                _array_decl(
                    "float", base.requant_array_name(prefix), op.attrs["requant_mult"], _float_fmt
                )
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def _array_decl(ctype: str, name: str, data: np.ndarray, fmt) -> str:
    flat = data.flatten()
    values = ", ".join(fmt(v) for v in flat.tolist())
    return f"static const {ctype} {name}[{flat.size}] PROGMEM = {{ {values} }};"


def _int_fmt(v) -> str:
    return str(int(v))


def _float_fmt(v) -> str:
    return f"{float(v):.9g}f"
