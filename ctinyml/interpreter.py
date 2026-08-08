"""Float32 reference interpreter for `closed_ir.Graph`.

Target-agnostic, pure numpy. Used by Phase 3 to calibrate activation
ranges for quantization, and reused unchanged by Phase 7 as the golden
baseline the compiled C implementation gets compared against.

Only handles the op types that survive Phase 2's structural passes:
Conv, Gemm, ArgMax.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ctinyml import closed_ir


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

        match op.op_type:
            case "Conv":
                out = _conv2d(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    strides=op.attrs["strides"],
                    pads=op.attrs["pads"],
                    clamp_min0=bool(op.attrs.get("clamp_min0")),
                )
            case "Gemm":
                # Gemm's declared input may still carry its pre-flatten
                # rank (Phase 2's drop_noop_reshape rewires consumers to
                # the pre-reshape tensor -- see closed_ir's design note);
                # only total element count matters, so flatten to 2D here.
                gemm_input = inputs[0].reshape(inputs[0].shape[0], -1)
                out = _gemm(
                    gemm_input,
                    inputs[1],
                    inputs[2],
                    alpha=op.attrs.get("alpha", 1.0),
                    beta=op.attrs.get("beta", 1.0),
                    transA=bool(op.attrs.get("transA", 0)),
                    transB=bool(op.attrs.get("transB", 0)),
                    clamp_min0=bool(op.attrs.get("clamp_min0")),
                )
            case "ArgMax":
                # Unlike Gemm's input, ArgMax's `axis` attribute is
                # rank-sensitive: it was authored against the tensor's
                # *original* (pre-drop_noop_reshape) shape. A dropped
                # Reshape can leave a leading size-1 batch dim in place
                # (e.g. (1,10) instead of (10,)), which shifts what
                # "axis 0" means without changing any data. Squeezing
                # size-1 dims first undoes exactly that -- safe here
                # because batch=1 is a hard-locked invariant (Phase 1
                # rejects anything else), so no real information is
                # ever discarded.
                out = _argmax(
                    np.squeeze(inputs[0]),
                    axis=op.attrs.get("axis", 0),
                    keepdims=bool(op.attrs.get("keepdims", 1)),
                    select_last_index=bool(op.attrs.get("select_last_index", 0)),
                )
            case _:
                raise ValueError(f"interpreter: unsupported op_type {op.op_type!r}")

        (output_tensor,) = op.outputs
        values[output_tensor.name] = out

    return values


def _conv2d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    strides: tuple[int, int],
    pads: tuple[int, int, int, int],
    clamp_min0: bool,
) -> np.ndarray:
    sy, sx = strides
    pad_top, pad_left, pad_bottom, pad_right = pads

    padded = np.pad(x, [(0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)])
    kh, kw = weight.shape[-2:]
    # windows: (N, C, out_h_full, out_w_full, kh, kw); slice by stride.
    windows = sliding_window_view(padded, (kh, kw), axis=(2, 3))[:, :, ::sy, ::sx, :, :]

    out = np.einsum("ncyxkl,ockl->noyx", windows, weight) + bias[None, :, None, None]
    if clamp_min0:
        out = np.maximum(out, 0.0)
    return out.astype(np.float32)


def _gemm(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    alpha: float,
    beta: float,
    transA: bool,
    transB: bool,
    clamp_min0: bool,
) -> np.ndarray:
    lhs = a.T if transA else a
    rhs = b.T if transB else b
    out = alpha * (lhs @ rhs) + beta * c
    if clamp_min0:
        out = np.maximum(out, 0.0)
    return out.astype(np.float32)


def _argmax(x: np.ndarray, *, axis: int, keepdims: bool, select_last_index: bool) -> np.ndarray:
    if select_last_index:
        flipped = np.flip(x, axis=axis)
        idx = x.shape[axis] - 1 - np.argmax(flipped, axis=axis)
    else:
        idx = np.argmax(x, axis=axis)
    idx = idx.astype(np.int64)
    if keepdims:
        idx = np.expand_dims(idx, axis=axis)
    return idx
