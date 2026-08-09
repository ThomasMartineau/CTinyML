"""Float32 reference interpreter for `closed_ir.Graph`.

Target-agnostic. Used by Phase 3 to calibrate activation ranges for
quantization, and reused unchanged by Phase 7 as the golden baseline the
compiled C implementation gets compared against.

The graph-walking loop lives here; the actual per-op-type numeric bodies
live in `ctinyml/ops.py`'s `KERNELS` registry (see that module's
docstring for why the dependency runs this direction).
"""

from __future__ import annotations

import numpy as np

from ctinyml import closed_ir
from ctinyml.ops import KERNELS


def run(graph: closed_ir.Graph, input_array: np.ndarray) -> dict[str, np.ndarray]:
    """Forward-evaluate `graph` in float32 for one input.

    Returns every activation tensor's computed value keyed by name: the
    graph input plus each op's output(s) -- exactly the tensor set
    quantize.py's calibration step needs.
    """
    (input_tensor,) = graph.inputs
    values: dict[str, np.ndarray] = {input_tensor.name: np.asarray(input_array, dtype=np.float32)}

    for op in graph.ops:
        inputs = [values[t.name] if t.data is None else t.data for t in op.inputs]

        kernel = KERNELS.get(op.op_type)
        if kernel is None:
            raise ValueError(f"interpreter: unsupported op_type {op.op_type!r}")
        out = kernel.forward(op, inputs)

        (output_tensor,) = op.outputs
        values[output_tensor.name] = out

    return values
