"""Target-agnostic graph-rewrite passes.

`STRUCTURAL_PASSES` / `run_structural_passes` orchestrate the Phase 2
onnx_ir passes in the required order (see
context/implementation_plan.md, Phase 2: "Order matters"). Kept here
rather than inside `convert_to_closed_ir.py` so that module stays a pure
conversion function taking an already-reduced model.
"""

from __future__ import annotations

import onnx_ir

from ctinyml.passes.errors import StructuralPassError
from ctinyml.passes.onnx_ir_passes.drop_noop_reshape import DropNoopReshapePass
from ctinyml.passes.onnx_ir_passes.fuse_relu import FuseReluPass

__all__ = [
    "StructuralPassError",
    "STRUCTURAL_PASSES",
    "run_structural_passes",
]

STRUCTURAL_PASSES = onnx_ir.passes.Sequential(
    DropNoopReshapePass(),
    FuseReluPass(),
)


def run_structural_passes(model: onnx_ir.Model) -> onnx_ir.Model:
    return STRUCTURAL_PASSES(model).model
