"""Target-agnostic shape/stride macro derivation.

Pure-Python helpers reusable by any future codegen target, not just AVR
(see the "Naming & target extensibility" section of
context/implementation_plan.md) — no C text emission here. The AVR-specific
`#define`/buffer-declaration text emission lives in
`targets/avr/codegen/emit_layout.py`, which uses these helpers.
"""

from __future__ import annotations

from ctinyml import closed_ir

# Macro-name prefix per op_type this compiler supports giving shape macros
# to. ArgMax has no shape macros of its own (its input dims come from the
# preceding Gemm's GEMM{i}_OUT), so it's deliberately absent here.
_PREFIX = {"Conv": "CONV", "Gemm": "GEMM"}


def conv_output_dim(in_dim: int, k: int, stride: int, pad: int) -> int:
    """Standard conv output-size formula: floor((in + 2*pad - k) / stride) + 1."""
    return (in_dim + 2 * pad - k) // stride + 1


def macro_names(graph: closed_ir.Graph) -> dict[str, str]:
    """Op name -> macro-friendly prefix, e.g. the first Conv op -> "CONV1",
    the first Gemm op -> "GEMM1" -- a positional counter per op_type,
    matching the reference file's CONV1/CONV2/DENSE-style naming. Only
    covers op types that get shape macros (Conv, Gemm); other op types
    (e.g. ArgMax) are omitted.
    """
    counters: dict[str, int] = {}
    names: dict[str, str] = {}
    for op in graph.ops:
        prefix = _PREFIX.get(op.op_type)
        if prefix is None:
            continue
        counters[prefix] = counters.get(prefix, 0) + 1
        names[op.name] = f"{prefix}{counters[prefix]}"
    return names
