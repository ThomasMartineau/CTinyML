"""ArgMaxEmitter: call-site into the shared `argmax_i32()` runtime
kernel, operating on the final Gemm's raw int32 accumulator output.
"""

from __future__ import annotations

import math

from ctinyml import closed_ir
from ctinyml.targets.avr.codegen import base


class ArgMaxEmitter:
    op_type = "ArgMax"
    HAS_WEIGHTS = False

    def layout_macros(self, op: closed_ir.Op, ctx: base.LayoutContext) -> list[str]:
        # No shape macros of its own -- its class count comes from its
        # own input tensor's shape, used directly below.
        return []

    def call_site(self, op: closed_ir.Op, ctx: base.LayoutContext) -> str:
        """A void assignment statement, structurally uniform with
        Conv/Gemm's call sites (no special 'return' handling needed in
        assemble.py). `ctx.tensor_ref(output_tensor)` resolves to the
        model_forward() output parameter here because this op's output
        happens to be the graph's declared output in this architecture
        -- a graph-topology fact `tensor_ref` already knows, not
        something this op type hardcodes about itself.
        """
        (input_tensor,) = op.inputs
        (output_tensor,) = op.outputs
        n = math.prod(input_tensor.shape)
        return f"*{ctx.tensor_ref(output_tensor)} = argmax_i32({ctx.tensor_ref(input_tensor)}, {n});"
