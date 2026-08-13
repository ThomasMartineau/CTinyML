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
    raw_input = np.asarray(input_array, dtype=np.float32)
    values: dict[str, np.ndarray] = {input_tensor.name: raw_input}

    # graph.inputs[0] is in the *raw* domain (e.g. camera-sensor byte
    # values) -- fold_div_into_quant.py (Phase 2) recorded input_prescale
    # but structurally removed the division that used to happen before
    # the first op, and nothing applied it since, silently feeding the
    # trained weights un-normalized input (see known_issues.md's Phase 7
    # entry -- this was a real, previously-undetected bug). Applying it
    # here, once, at the point where the input value actually flows into
    # computation -- while still recording the *raw* value under
    # input_tensor.name above, since that's the domain quantize.py needs
    # to calibrate the input tensor's own scale/zero_point against (it's
    # what the real device's int8 input actually represents).
    scaled_input = raw_input / graph.input_prescale if graph.input_prescale else raw_input

    for op in graph.ops:
        inputs = []
        for t in op.inputs:
            if t.data is not None:
                inputs.append(t.data)
            elif t is input_tensor:
                inputs.append(scaled_input)
            else:
                inputs.append(values[t.name])

        kernel = KERNELS.get(op.op_type)
        if kernel is None:
            raise ValueError(f"interpreter: unsupported op_type {op.op_type!r}")
        out = kernel.forward(op, inputs)

        (output_tensor,) = op.outputs
        values[output_tensor.name] = out

    return values
