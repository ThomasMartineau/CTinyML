"""ConvEmitter: `#define` shape macros + one call-site line per Conv
layer into the shared `conv2d_int8()` runtime kernel.
"""

from __future__ import annotations

from ctinyml import closed_ir
from ctinyml.ops import KERNELS
from ctinyml.targets.avr.codegen import base
from ctinyml.targets.avr.codegen.errors import CodegenError


class ConvEmitter:
    op_type = "Conv"
    HAS_WEIGHTS = True

    def layout_macros(self, op: closed_ir.Op, ctx: base.LayoutContext) -> list[str]:
        prefix = ctx.macro_names[op.name]
        input_tensor, weight, _bias = op.inputs
        (output_tensor,) = op.outputs

        _, in_ch, in_h, in_w = input_tensor.shape
        out_ch, weight_in_ch, kh, kw = weight.shape
        if weight_in_ch != in_ch:
            raise CodegenError(
                f"{op.name}: weight expects {weight_in_ch} input channels, "
                f"input tensor {input_tensor.name!r} has {in_ch}."
            )
        if kh != kw:
            raise CodegenError(f"{op.name}: non-square kernel {kh}x{kw} not supported.")

        strides = op.attrs["strides"]
        if strides[0] != strides[1]:
            raise CodegenError(f"{op.name}: asymmetric stride {strides} not supported.")
        pads = op.attrs["pads"]
        if not (pads[0] == pads[1] == pads[2] == pads[3]):
            raise CodegenError(f"{op.name}: asymmetric padding {pads} not supported.")
        stride, pad, k = strides[0], pads[0], kh

        _, _, out_h, out_w = KERNELS["Conv"].output_shape(op)
        _, _, stored_h, stored_w = output_tensor.shape
        if (out_h, out_w) != (stored_h, stored_w):
            raise CodegenError(
                f"{op.name}: computed output {out_h}x{out_w} doesn't match the "
                f"graph's stored shape {stored_h}x{stored_w}."
            )

        lines = [
            f"#define {prefix}_IN_CH {in_ch}",
            f"#define {prefix}_OUT_CH {out_ch}",
            f"#define {prefix}_K {k}",
            f"#define {prefix}_STRIDE {stride}",
            f"#define {prefix}_PAD {pad}",
            f"#define {prefix}_IN_H {in_h}",
            f"#define {prefix}_IN_W {in_w}",
            f"#define {prefix}_OUT_H {out_h}",
            f"#define {prefix}_OUT_W {out_w}",
        ]

        zp_define = base.zero_point_define(output_tensor)
        if zp_define is not None:
            ctx.zero_point_macros[output_tensor.name] = base.zero_point_macro_name(output_tensor)
            lines.append(zp_define)

        return lines

    def call_site(self, op: closed_ir.Op, ctx: base.LayoutContext) -> str:
        prefix = ctx.macro_names[op.name]
        input_tensor, _weight, _bias = op.inputs
        (output_tensor,) = op.outputs

        input_zp = ctx.zero_point_macros[input_tensor.name]
        out_zp = ctx.zero_point_macros[output_tensor.name]
        relu = 1 if op.attrs.get("clamp_min0") else 0

        return (
            f"conv2d_int8({ctx.tensor_ref(input_tensor)}, {prefix}_IN_CH, {prefix}_IN_H, "
            f"{prefix}_IN_W, {input_zp}, "
            f"{base.weight_array_name(prefix)}, {base.bias_array_name(prefix)}, "
            f"{base.requant_array_name(prefix)}, "
            f"{prefix}_OUT_CH, {prefix}_K, {prefix}_STRIDE, {prefix}_PAD, "
            f"{ctx.tensor_ref(output_tensor)}, {prefix}_OUT_H, {prefix}_OUT_W, {out_zp}, {relu});"
        )
