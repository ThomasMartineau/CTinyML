"""Target-agnostic per-op-type kernel registry.

Each Kernel owns shape derivation and float32 forward evaluation for one
op_type. `interpreter.py` (the graph-walking loop) and `layout.py`/the AVR
codegen's shape cross-checks both dispatch through this registry instead
of scattering `if op.op_type == ...` branches across each module --
mirrors onnxruntime's kernel-registry pattern. Deliberately lightweight:
this module has no dependency on `interpreter.py` (the numeric bodies
that used to live there as `_conv2d`/`_gemm`/`_argmax` now live here,
and `interpreter.py` imports *from* this module, not the other way
around, to avoid a circular import between "the registry" and "the thing
that used to own the functions").

Only handles the op types that survive Phase 2's structural passes:
Conv, Gemm, ArgMax.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ctinyml import closed_ir, layout


class Kernel(Protocol):
    op_type: str

    def forward(self, op: closed_ir.Op, inputs: list[np.ndarray]) -> np.ndarray: ...


class ConvKernel:
    op_type = "Conv"

    def output_shape(self, op: closed_ir.Op) -> tuple[int, int, int, int]:
        """Recomputes (n, out_ch, out_h, out_w) from op.inputs/attrs via
        layout.conv_output_dim -- the target-agnostic half of the
        "stride-2/pad-1 boundary math" cross-check. Assumes pad-before ==
        pad-after on each axis independently (inherent to
        conv_output_dim's formula); the *AVR* codegen's ConvEmitter
        additionally requires full stride/pad uniformity across both
        axes, which is a limitation of the real C kernel's signature
        (one stride/pad scalar total), not a target-agnostic one.
        """
        data, weight = op.inputs[0], op.inputs[1]
        n, _in_c, in_h, in_w = data.shape
        out_ch, _, kh, kw = weight.shape
        stride_h, stride_w = op.attrs["strides"]
        pad_top, pad_left, pad_bottom, pad_right = op.attrs["pads"]
        out_h = layout.conv_output_dim(in_h, kh, stride_h, pad_top)
        out_w = layout.conv_output_dim(in_w, kw, stride_w, pad_left)
        return (n, out_ch, out_h, out_w)

    def forward(self, op: closed_ir.Op, inputs: list[np.ndarray]) -> np.ndarray:
        return _conv2d(
            inputs[0],
            inputs[1],
            inputs[2],
            strides=op.attrs["strides"],
            pads=op.attrs["pads"],
            clamp_min0=bool(op.attrs.get("clamp_min0")),
        )


class GemmKernel:
    op_type = "Gemm"

    def forward(self, op: closed_ir.Op, inputs: list[np.ndarray]) -> np.ndarray:
        # Gemm's declared input may still carry its pre-flatten rank
        # (Phase 2's drop_noop_reshape rewires consumers to the
        # pre-reshape tensor -- see closed_ir's design note); only total
        # element count matters, so flatten to 2D here.
        gemm_input = inputs[0].reshape(inputs[0].shape[0], -1)
        return _gemm(
            gemm_input,
            inputs[1],
            inputs[2],
            alpha=op.attrs.get("alpha", 1.0),
            beta=op.attrs.get("beta", 1.0),
            transA=bool(op.attrs.get("transA", 0)),
            transB=bool(op.attrs.get("transB", 0)),
            clamp_min0=bool(op.attrs.get("clamp_min0")),
        )


class ArgMaxKernel:
    op_type = "ArgMax"

    def forward(self, op: closed_ir.Op, inputs: list[np.ndarray]) -> np.ndarray:
        # Unlike Gemm's input, ArgMax's `axis` attribute is rank-sensitive:
        # it was authored against the tensor's *original*
        # (pre-drop_noop_reshape) shape. A dropped Reshape can leave a
        # leading size-1 batch dim in place (e.g. (1,10) instead of
        # (10,)), which shifts what "axis 0" means without changing any
        # data. Squeezing size-1 dims first undoes exactly that -- safe
        # here because batch=1 is a hard-locked invariant (Phase 1
        # rejects anything else), so no real information is ever
        # discarded.
        return _argmax(
            np.squeeze(inputs[0]),
            axis=op.attrs.get("axis", 0),
            keepdims=bool(op.attrs.get("keepdims", 1)),
            select_last_index=bool(op.attrs.get("select_last_index", 0)),
        )


KERNELS: dict[str, Kernel] = {"Conv": ConvKernel(), "Gemm": GemmKernel(), "ArgMax": ArgMaxKernel()}


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
