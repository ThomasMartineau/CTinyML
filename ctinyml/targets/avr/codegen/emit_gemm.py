"""GemmEmitter: `#define` shape macros + one call-site line into the
shared `dense_int8()` runtime kernel, with the `skip_requant` flag wired
through for the final layer (writes to an `int32_t*` output instead of
`int8_t*`, per the parent plan's Phase 5 gap-filling note).
"""

from __future__ import annotations

from ctinyml import closed_ir
from ctinyml.targets.avr.codegen import base
from ctinyml.targets.avr.codegen.errors import CodegenError


class GemmEmitter:
    op_type = "Gemm"
    HAS_WEIGHTS = True

    def layout_macros(self, op: closed_ir.Op, ctx: base.LayoutContext) -> list[str]:
        prefix = ctx.macro_names[op.name]
        _, weight, _bias = op.inputs
        (output_tensor,) = op.outputs

        trans_a = op.attrs.get("transA", 0)
        trans_b = op.attrs.get("transB", 0)
        if trans_a != 0 or trans_b != 1:
            raise CodegenError(
                f"{op.name}: only transA=0, transB=1 Gemm orientation is supported "
                f"(got transA={trans_a}, transB={trans_b})."
            )

        out_dim, in_dim = weight.shape
        lines = [f"#define {prefix}_IN {in_dim}", f"#define {prefix}_OUT {out_dim}"]

        zp_define = base.zero_point_define(output_tensor)
        if zp_define is not None:
            ctx.zero_point_macros[output_tensor.name] = base.zero_point_macro_name(output_tensor)
            lines.append(zp_define)

        return lines

    def call_site(self, op: closed_ir.Op, ctx: base.LayoutContext) -> str:
        prefix = ctx.macro_names[op.name]
        input_tensor, _weight, _bias = op.inputs
        (output_tensor,) = op.outputs
        skip_requant = bool(op.attrs.get("skip_requant"))

        input_zp = ctx.zero_point_macros[input_tensor.name]
        relu = 1 if op.attrs.get("clamp_min0") else 0

        if skip_requant:
            requant_arg = "NULL"
            out_zp = "0"
        else:
            requant_arg = base.requant_array_name(prefix)
            out_zp = ctx.zero_point_macros[output_tensor.name]

        return (
            f"dense_int8({ctx.tensor_ref(input_tensor)}, {prefix}_IN, {input_zp}, "
            f"{base.weight_array_name(prefix)}, {base.bias_array_name(prefix)}, {requant_arg}, "
            f"{prefix}_OUT, {ctx.tensor_ref(output_tensor)}, {out_zp}, {relu}, "
            f"{1 if skip_requant else 0});"
        )
