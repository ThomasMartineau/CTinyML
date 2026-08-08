"""Weight + activation int8 quantization (calibration-driven) over
closed_ir.Graph.

Call `drop_final_requant.apply(graph)` before this — `quantize()` relies
on `skip_requant` already being set on the op that feeds ArgMax.

Calibration data is a plain iterable of preprocessed input arrays (e.g.
a numpy array of shape [N, 1, 28, 28], or any generator yielding
individual/batched inputs) -- just `for x in calibration_data: ...` over
whatever a caller has in-process; no reader-class protocol. It's expected
in the same domain as the closed IR's declared graph input, i.e. the
*pre-Div* domain (see the fold_div_into_quant docstring) -- Phase 2
already removed the runtime division, so calibrating directly on that
domain is what "folds input_prescale into layer-1's input scale" in
practice: there's no separate multiply-by-input_prescale step here.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ctinyml import closed_ir, interpreter

# Signed int8, matching requantize()'s clamp and the int8_t zero-points in
# context/mnist_conv_dense_int8_avr.c (the reference implementation).
QMIN, QMAX = -128, 127
QRANGE = QMAX - QMIN  # 255


def quantize(graph: closed_ir.Graph, calibration_data: Iterable[np.ndarray]) -> closed_ir.Graph:
    """Mutate `graph` in place: calibrate + quantize every activation
    tensor, quantize weights/biases per-output-channel, and compute
    per-channel requant multipliers. Any op already marked
    `skip_requant` gets an unquantized int32 output instead. Returns
    `graph` for chaining.
    """
    _calibrate_activations(graph, calibration_data)

    for op in graph.ops:
        if op.op_type not in ("Conv", "Gemm"):
            continue

        input_tensor = op.inputs[0]
        weight_scale = _quantize_weight_and_bias(op, input_tensor)
        (output_tensor,) = op.outputs

        if op.attrs.get("skip_requant"):
            output_tensor.scale = None
            output_tensor.zero_point = None
            output_tensor.dtype = "INT32"
        else:
            op.attrs["requant_mult"] = _requant_multiplier(
                input_tensor.scale, weight_scale, output_tensor.scale
            )

    return graph


def _calibrate_activations(graph: closed_ir.Graph, calibration_data: Iterable[np.ndarray]) -> None:
    tensors_by_name: dict[str, closed_ir.Tensor] = {}
    for t in graph.inputs:
        tensors_by_name[t.name] = t
    for op in graph.ops:
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


def _quantize_weight_and_bias(op: closed_ir.Op, input_tensor: closed_ir.Tensor) -> np.ndarray:
    """Quantize op.inputs[1] (weight) and op.inputs[2] (bias) in place,
    per-output-channel (axis 0), symmetric (zero_point 0) for the
    weight. Returns the weight's per-channel scale array.
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

    bias_scale = input_tensor.scale * weight_scale
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
