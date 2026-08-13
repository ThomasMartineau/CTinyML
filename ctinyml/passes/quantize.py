"""Weight + activation int8 quantization (calibration-driven) over
closed_ir.Graph.

Call `drop_final_requant.apply(graph)` before this — `quantize()` relies
on `skip_requant` already being set on the op that feeds ArgMax.

Calibration data is a plain iterable of preprocessed input arrays (e.g.
individual arrays of shape [1, 1, 28, 28], or any generator yielding
them) -- just `for x in calibration_data: ...` over whatever a caller
has in-process; no reader-class protocol. It's expected in the same
*raw*, pre-Div domain as the closed IR's declared graph input (see the
fold_div_into_quant docstring) -- e.g. actual [0,255] pixel bytes, not
pre-normalized [0,1] values.

`input_prescale` (folded by Phase 2, recorded on `graph.input_prescale`)
is *not* applied as a runtime division anywhere -- the whole point of
folding it was to avoid one. It still has to be accounted for
mathematically, though: the graph input tensor's own scale/zero_point
are (correctly) calibrated against the raw domain (matching what the
real device's int8 input represents), but the trained weights expect
the *normalized* domain. `_effective_input_scale` below divides the
raw-calibrated scale by `input_prescale` specifically when quantizing
the op that consumes the graph's input directly, reconstructing the
value the weights actually expect without ever running a division on
device. `interpreter.py` has the matching fix on the float-reference
side (applies the division only to what flows into computation, not to
what gets calibrated).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ctinyml import closed_ir, interpreter

# Signed int8, matching requantize()'s clamp and the int8_t zero-points in
# context/mnist_conv_dense_int8_avr.c (the reference implementation).
QMIN, QMAX = -128, 127
QRANGE = QMAX - QMIN  # 255

# Op types whose *output* is never a ranged float activation, so it must
# never be calibrated / assigned a scale+zero_point: ArgMax's output is a
# discrete class index (int64), not a quantity with a meaningful min/max
# range to fit an affine int8 mapping to. Named and centralized here
# (rather than an inline op_type check) so a future op type that also
# shouldn't be calibrated is a one-line addition, not a re-derivation.
NON_CALIBRATED_OPS = {"ArgMax"}


def quantize(graph: closed_ir.Graph, calibration_data: Iterable[np.ndarray]) -> closed_ir.Graph:
    """Mutate `graph` in place: calibrate + quantize every activation
    tensor, quantize weights/biases per-output-channel, and compute
    per-channel requant multipliers. Any op already marked
    `skip_requant` gets an unquantized int32 output instead. Returns
    `graph` for chaining.
    """
    _calibrate_activations(graph, calibration_data)
    (graph_input,) = graph.inputs

    for op in graph.ops:
        if op.op_type not in ("Conv", "Gemm"):
            continue

        input_scale = _effective_input_scale(graph, graph_input, op.inputs[0])
        weight_scale = _quantize_weight_and_bias(op, input_scale)
        (output_tensor,) = op.outputs

        if op.attrs.get("skip_requant"):
            output_tensor.scale = None
            output_tensor.zero_point = None
            output_tensor.dtype = "INT32"
        else:
            op.attrs["requant_mult"] = _requant_multiplier(
                input_scale, weight_scale, output_tensor.scale
            )

    return graph


def _effective_input_scale(
    graph: closed_ir.Graph, graph_input: closed_ir.Tensor, input_tensor: closed_ir.Tensor
) -> float:
    """The scale to use when reconstructing `input_tensor`'s float value
    for quantization math. Ordinarily just `input_tensor.scale` --
    except when `input_tensor` *is* the graph's own declared input,
    which is calibrated against the raw, pre-Div domain (matching what
    the real device's int8 input represents), not the normalized domain
    the trained weights expect. Dividing by `graph.input_prescale`
    reconstructs the normalized value without ever running a division
    on device -- see the module docstring.
    """
    if input_tensor is graph_input and graph.input_prescale:
        return input_tensor.scale / graph.input_prescale
    return input_tensor.scale


def quantize_input(image: np.ndarray, tensor: closed_ir.Tensor) -> np.ndarray:
    """Quantize a raw float image/array to int8 using `tensor`'s
    already-calibrated scale/zero_point (e.g. `graph.inputs[0]` after
    `quantize()` has run). Reusable beyond testing -- any future
    inference/deployment path needs exactly this to prepare a real input
    for the compiled model.
    """
    if tensor.scale is None or tensor.zero_point is None:
        raise ValueError(f"tensor {tensor.name!r} has no calibrated scale/zero_point.")
    q = _round_half_away_from_zero(image / tensor.scale) + tensor.zero_point
    return np.clip(q, QMIN, QMAX).astype(np.int8)


def _calibrate_activations(graph: closed_ir.Graph, calibration_data: Iterable[np.ndarray]) -> None:
    tensors_by_name: dict[str, closed_ir.Tensor] = {}
    for t in graph.inputs:
        tensors_by_name[t.name] = t
    for op in graph.ops:
        if op.op_type in NON_CALIBRATED_OPS:
            continue
        for t in op.outputs:
            tensors_by_name[t.name] = t

    lo = dict.fromkeys(tensors_by_name, np.inf)
    hi = dict.fromkeys(tensors_by_name, -np.inf)

    for x in calibration_data:
        values = interpreter.run(graph, x)
        for name in tensors_by_name:
            arr = values[name]
            lo[name] = min(lo[name], float(arr.min()))
            hi[name] = max(hi[name], float(arr.max()))

    for name, tensor in tensors_by_name.items():
        scale, zero_point = _asymmetric_qparams(lo[name], hi[name])
        tensor.scale = scale
        tensor.zero_point = zero_point
        tensor.dtype = "INT8"


def _asymmetric_qparams(lo: float, hi: float) -> tuple[float, int]:
    """Derive an asymmetric int8 (scale, zero_point) pair from a
    calibrated [lo, hi] float range: `lo` maps to QMIN, `hi` maps to
    QMAX. Widens the interval to include 0 (so the additive identity is
    exactly representable) and guards against a zero-width interval.
    """
    lo = min(lo, 0.0)
    hi = max(hi, 0.0)
    scale = max(hi - lo, 1e-8) / QRANGE
    zero_point = int(np.clip(_round_half_away_from_zero(QMIN - lo / scale), QMIN, QMAX))
    return scale, zero_point


def _quantize_weight_and_bias(op: closed_ir.Op, input_scale: float) -> np.ndarray:
    """Quantize op.inputs[1] (weight) and op.inputs[2] (bias) in place,
    per-output-channel (axis 0), symmetric (zero_point 0) for the
    weight. `input_scale` is the *effective* input scale (see
    `_effective_input_scale`), not necessarily `op.inputs[0].scale`
    directly. Returns the weight's per-channel scale array.
    """
    weight, bias = op.inputs[1], op.inputs[2]
    reduce_axes = tuple(range(1, weight.data.ndim))

    amax = np.abs(weight.data).max(axis=reduce_axes)
    weight_scale = np.where(amax == 0, 1.0, amax / 127.0)

    broadcast_shape = (-1,) + (1,) * len(reduce_axes)
    weight_int8 = np.clip(
        _round_half_away_from_zero(weight.data / weight_scale.reshape(broadcast_shape)), -127, 127
    ).astype(np.int8)

    weight.data = weight_int8
    weight.scale = weight_scale
    weight.zero_point = 0
    weight.dtype = "INT8"

    bias_scale = input_scale * weight_scale
    bias_int32 = _round_half_away_from_zero(bias.data / bias_scale).astype(np.int32)

    bias.data = bias_int32
    bias.scale = bias_scale
    bias.zero_point = 0
    bias.dtype = "INT32"

    return weight_scale


def _requant_multiplier(
    input_scale: float, weight_scale: np.ndarray, output_scale: float
) -> np.ndarray:
    """Per-channel float requant multiplier: input_scale * weight_scale
    / output_scale -- applied as a plain float multiply at runtime, not
    a fixed-point (M0, shift) pair (see the parent plan's Phase 3
    rationale)."""
    return (input_scale * weight_scale / output_scale).astype(np.float32)


def _round_half_away_from_zero(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.floor(np.abs(x) + 0.5)
