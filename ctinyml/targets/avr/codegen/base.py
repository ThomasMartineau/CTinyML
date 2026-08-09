"""Shared types for the AVR codegen `Emitter` registry.

Each op type's `Emitter` (targets/avr/codegen/emit_conv2d.py etc.) owns
both the `#define` layout macros and the C call-site for that op --
colocated per operator, looked up via `targets/avr/codegen/registry.py`'s
`EMITTERS` dict instead of a branch re-derived in every codegen module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from ctinyml import closed_ir


@dataclass
class LayoutContext:
    macro_names: dict[str, str] = field(default_factory=dict)  # op.name -> "CONV1" / "GEMM1"
    zero_point_macros: dict[str, str] = field(default_factory=dict)  # tensor.name -> "INPUT_ZP" / ...
    tensor_ref: Callable[[closed_ir.Tensor], str] | None = None  # tensor -> C expr


def macro_safe(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).upper()


def zero_point_macro_name(tensor: closed_ir.Tensor) -> str:
    return f"{macro_safe(tensor.name)}_ZP"


def zero_point_define(tensor: closed_ir.Tensor) -> str | None:
    """The '#define {NAME}_ZP {value}' line for this tensor if it's a
    calibrated int8 activation, else None (e.g. a skip_requant int32
    output, or a not-yet-quantized/non-activation tensor)."""
    if tensor.dtype != "INT8" or tensor.zero_point is None:
        return None
    return f"#define {zero_point_macro_name(tensor)} {tensor.zero_point}"


def weight_array_name(prefix: str) -> str:
    return f"{prefix.lower()}_weight"


def bias_array_name(prefix: str) -> str:
    return f"{prefix.lower()}_bias"


def requant_array_name(prefix: str) -> str:
    return f"{prefix.lower()}_requant"


class Emitter(Protocol):
    op_type: str
    # Whether this op type owns a (weight, bias[, requant]) input triple
    # that emit_weights.py needs to emit PROGMEM arrays for. The single
    # source of truth for "does this op type have weights" -- checked via
    # EMITTERS, not a separately-hardcoded op_type tuple.
    HAS_WEIGHTS: bool

    def layout_macros(self, op: closed_ir.Op, ctx: LayoutContext) -> list[str]:
        """The `#define {PREFIX}_*` block (and any zero-point #define) for
        this op. Returns [] if this op type has no shape macros of its
        own (e.g. ArgMax)."""
        ...

    def call_site(self, op: closed_ir.Op, ctx: LayoutContext) -> str:
        """One C call-site line/expression into this op's runtime kernel."""
        ...
