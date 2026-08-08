"""Target-agnostic narrow post-fusion IR.

This is the Phase 2/3 boundary: `ctinyml.passes.convert_to_closed_ir` walks
a reduced `onnx_ir.Graph` once and builds one of these. Everything from
Phase 3 onward (quantization, buffer planning, codegen) operates on this
IR, not `onnx_ir`, deliberately kept minimal — flat `attrs` dict, no
generic tensor struct, quantization metadata living directly on `Tensor`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Tensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    data: np.ndarray | None = None
    is_initializer: bool = False
    # Quantization metadata (Phase 3 onward); unset until quantize.py runs.
    scale: float | None = None
    zero_point: int | None = None
    # Ping-pong SRAM buffer index (0 or 1); unset until buffer_planner.py
    # runs, and stays None for graph inputs/outputs and initializers,
    # which are caller-supplied/PROGMEM rather than planner-owned.
    buffer: int | None = None


@dataclass
class Op:
    op_type: str
    name: str
    inputs: list[Tensor]
    outputs: list[Tensor]
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Graph:
    ops: list[Op]
    inputs: list[Tensor]
    outputs: list[Tensor]
    # Folded from onnx_ir Graph.meta["input_prescale"] by fold_div_into_quant.
    input_prescale: float | None = None

    def to_text(self) -> str:
        """Deterministic, human-readable dump for golden-snapshot tests.

        Intentionally omits initializer `.data` array contents (shapes and
        dtypes only) so the output stays small and reviewable.
        """
        lines = [f"Graph(input_prescale={_fmt(self.input_prescale)})"]

        lines.append("inputs:")
        for t in self.inputs:
            lines.append(f"  {_tensor_line(t)}")

        lines.append("outputs:")
        for t in self.outputs:
            lines.append(f"  {_tensor_line(t)}")

        lines.append("ops:")
        for i, op in enumerate(self.ops):
            lines.append(f"  {i}: {op.op_type} name={op.name!r}")
            attrs = {k: op.attrs[k] for k in sorted(op.attrs)}
            lines.append(f"     attrs={_fmt_attrs(attrs)}")
            for t in op.inputs:
                lines.append(f"     in:  {_tensor_line(t)}")
            for t in op.outputs:
                lines.append(f"     out: {_tensor_line(t)}")

        return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return repr(value)


def _fmt_attrs(attrs: dict[str, Any]) -> str:
    parts = [f"{k}={_fmt(v)}" for k, v in attrs.items()]
    return "{" + ", ".join(parts) + "}"


def _tensor_line(t: Tensor) -> str:
    return f"{t.name}: shape={tuple(t.shape)} dtype={t.dtype} initializer={t.is_initializer}"
