"""Liveness/refcount-based buffer planner — ping-pong static buffers sized
to the largest intermediate activation.

Call after quantize.quantize(graph) -- sizing is dtype-dependent (int8 vs
int32), so this must see final quantized dtypes, not the pre-quantization
float32 ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ctinyml import closed_ir
from ctinyml.passes.errors import ClosedIRPassError

BUFFER_BUDGET_BYTES = 8192  # 8 KB SRAM, ATmega2560 hard limit
BUFFER_WARN_BYTES = 6144  # 6 KB soft threshold -- leaves headroom for stack/globals
NUM_BUFFERS = 2

_ITEMSIZE = {"INT8": 1, "INT32": 4, "FLOAT": 4, "INT64": 8}


@dataclass
class BufferPlan:
    buffer_size: int  # bytes, uniform across both buffers
    num_buffers: int  # always 2 for this ping-pong strategy
    total_sram_bytes: int  # num_buffers * buffer_size
    assignments: dict[str, int]  # tensor name -> buffer index (0 or 1)
    warning: str | None  # set if over BUFFER_WARN_BYTES, still under budget


def plan(graph: closed_ir.Graph) -> BufferPlan:
    internal = _internal_tensors(graph)

    if not internal:
        return BufferPlan(
            buffer_size=0, num_buffers=NUM_BUFFERS, total_sram_bytes=0, assignments={}, warning=None
        )

    internal_ids = {id(t) for t in internal}
    last_use = {id(t): _last_use_index(graph, t) for t in internal}
    buffer_size = max(_nbytes(t) for t in internal)

    # slots[i] = (tensor_name, last_use_index) of the current occupant, or None if free.
    slots: list[tuple[str, int] | None] = [None] * NUM_BUFFERS
    assignments: dict[str, int] = {}

    for op_index, op in enumerate(graph.ops):
        for tensor in op.outputs:
            if id(tensor) not in internal_ids:
                continue

            for i in range(NUM_BUFFERS):
                if slots[i] is not None and slots[i][1] < op_index:
                    slots[i] = None

            free_index = next((i for i in range(NUM_BUFFERS) if slots[i] is None), None)
            if free_index is None:
                raise ClosedIRPassError(
                    f"buffer planner: tensor {tensor.name!r} (defined at op {op_index}) "
                    f"needs a 3rd simultaneously-live buffer -- this graph shape "
                    "(fan-out/branching) isn't supported by the 2-buffer ping-pong strategy."
                )

            slots[free_index] = (tensor.name, last_use[id(tensor)])
            assignments[tensor.name] = free_index
            tensor.buffer = free_index

    total_sram_bytes = NUM_BUFFERS * buffer_size
    if total_sram_bytes > BUFFER_BUDGET_BYTES:
        raise ClosedIRPassError(
            f"buffer plan needs {total_sram_bytes} bytes of SRAM, exceeds the "
            f"{BUFFER_BUDGET_BYTES}-byte (8 KB) budget."
        )

    warning = None
    if total_sram_bytes > BUFFER_WARN_BYTES:
        warning = (
            f"buffer plan uses {total_sram_bytes} bytes, over the {BUFFER_WARN_BYTES}-byte "
            "(6 KB) soft warning threshold (still under the 8 KB hard budget)."
        )

    return BufferPlan(
        buffer_size=buffer_size,
        num_buffers=NUM_BUFFERS,
        total_sram_bytes=total_sram_bytes,
        assignments=assignments,
        warning=warning,
    )


def _internal_tensors(graph: closed_ir.Graph) -> list[closed_ir.Tensor]:
    # Identity-keyed throughout (not `in`/`==`): closed_ir.Tensor is a plain
    # @dataclass, so value-equality would compare `.data` numpy arrays too
    # -- unsafe (ValueError: truth value of an array...) unless an earlier
    # field happens to differ first. Tensors are already deduplicated
    # object-per-value by convert_to_closed_ir, so identity is correct here.
    graph_output_ids = {id(t) for t in graph.outputs}
    seen: dict[int, closed_ir.Tensor] = {}
    for op in graph.ops:
        for tensor in op.outputs:
            if id(tensor) in graph_output_ids or id(tensor) in seen:
                continue
            seen[id(tensor)] = tensor
    return list(seen.values())


def _last_use_index(graph: closed_ir.Graph, tensor: closed_ir.Tensor) -> int:
    last = -1
    for op_index, op in enumerate(graph.ops):
        if any(inp is tensor for inp in op.inputs):
            last = op_index
    return last


def _nbytes(tensor: closed_ir.Tensor) -> int:
    itemsize = _ITEMSIZE.get(tensor.dtype)
    if itemsize is None:
        raise ClosedIRPassError(f"buffer planner: unknown dtype {tensor.dtype!r} on {tensor.name!r}")
    return math.prod(tensor.shape) * itemsize
