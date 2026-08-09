"""Phase 6 acceptance tests — AVR device runtime library.

Standalone host-side (x86, not AVR) unit tests for the real runtime
kernels: hand-computed small examples, a byte-for-byte regression check
against a verbatim copy of the reference file's kernels, and a bonus
end-to-end sanity check linking Phase 5's real codegen output against
these real kernels.
"""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

RUNTIME_DIR = Path(__file__).parent.parent / "ctinyml" / "targets" / "avr" / "runtime"
FIXTURES_DIR = Path(__file__).parent / "fixtures"

GCC = shutil.which("gcc")

_CONV_ARGTYPES = [
    ctypes.POINTER(ctypes.c_int8),
    ctypes.c_uint8,
    ctypes.c_uint8,
    ctypes.c_uint8,
    ctypes.c_int8,
    ctypes.POINTER(ctypes.c_int8),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint8,
    ctypes.c_uint8,
    ctypes.c_uint8,
    ctypes.c_uint8,
    ctypes.POINTER(ctypes.c_int8),
    ctypes.c_uint8,
    ctypes.c_uint8,
    ctypes.c_int8,
    ctypes.c_uint8,
]

_DENSE_ARGTYPES_PORTED = [
    ctypes.POINTER(ctypes.c_int8),
    ctypes.c_uint16,
    ctypes.c_int8,
    ctypes.POINTER(ctypes.c_int8),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint16,
    ctypes.c_void_p,
    ctypes.c_int8,
    ctypes.c_uint8,
    ctypes.c_uint8,
]

_DENSE_ARGTYPES_REFERENCE = [
    ctypes.POINTER(ctypes.c_int8),
    ctypes.c_uint16,
    ctypes.c_int8,
    ctypes.POINTER(ctypes.c_int8),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint16,
    ctypes.POINTER(ctypes.c_int8),
    ctypes.c_int8,
    ctypes.c_uint8,
]


def _i8p(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))


def _i32p(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _f32p(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _compile_dll(build_dir: Path, source: Path, include_dirs: list[Path], dll_name: str) -> Path:
    dll_path = build_dir / dll_name
    cmd = [GCC, "-shared", "-std=c11", "-Wall", "-Wextra", "-Wl,--export-all-symbols"]
    for inc in include_dirs:
        cmd += ["-I", str(inc)]
    cmd += [str(source), "-o", str(dll_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stderr == "", result.stderr
    return dll_path


@pytest.fixture(scope="module")
def build_dir(tmp_path_factory):
    if GCC is None:
        pytest.skip("host gcc not available")
    return tmp_path_factory.mktemp("runtime_build")


@pytest.fixture(scope="module")
def ported_lib(build_dir):
    dll = _compile_dll(
        build_dir,
        FIXTURES_DIR / "runtime_shim_ported.c",
        [RUNTIME_DIR],
        "ported.dll",
    )
    lib = ctypes.CDLL(str(dll))
    lib.test_conv2d_int8.argtypes = _CONV_ARGTYPES
    lib.test_conv2d_int8.restype = None
    lib.test_dense_int8.argtypes = _DENSE_ARGTYPES_PORTED
    lib.test_dense_int8.restype = None
    lib.test_argmax_i32.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_uint16]
    lib.test_argmax_i32.restype = ctypes.c_uint8
    lib.test_requantize.argtypes = [ctypes.c_float, ctypes.c_int8]
    lib.test_requantize.restype = ctypes.c_int8
    lib.test_pgm_read_i32.argtypes = [ctypes.POINTER(ctypes.c_int32)]
    lib.test_pgm_read_i32.restype = ctypes.c_int32
    lib.test_pgm_read_f32.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.test_pgm_read_f32.restype = ctypes.c_float
    return lib


@pytest.fixture(scope="module")
def reference_lib(build_dir):
    dll = _compile_dll(
        build_dir,
        FIXTURES_DIR / "runtime_shim_reference.c",
        [FIXTURES_DIR],
        "reference.dll",
    )
    lib = ctypes.CDLL(str(dll))
    lib.test_conv2d_int8.argtypes = _CONV_ARGTYPES
    lib.test_conv2d_int8.restype = None
    lib.test_dense_int8.argtypes = _DENSE_ARGTYPES_REFERENCE
    lib.test_dense_int8.restype = None
    return lib


# --- hand-computed conv2d_int8 -----------------------------------------

# Case A: stride=1, pad=0 (no boundary math). 5x5 input 1..25, 3x3
# all-ones kernel, input_zp=0, requant_mult=0.5, out_zp=0. Exercises
# plain rounding (several exact .5 cases) without saturation.
#   out[0,0] = sum(rows0-2,cols0-2) = 1+2+3+6+7+8+11+12+13 = 63 -> *0.5=31.5 -> 32
#   out[0,1] = 72 -> 36.0 -> 36 ; out[0,2] = 81 -> 40.5 -> 41
#   out[1,0] = 108 -> 54.0 -> 54 ; out[1,1] = 117 -> 58.5 -> 59 ; out[1,2] = 126 -> 63.0 -> 63
#   out[2,0] = 153 -> 76.5 -> 77 ; out[2,1] = 162 -> 81.0 -> 81 ; out[2,2] = 171 -> 85.5 -> 86
_CASE_A_EXPECTED = np.array([[32, 36, 41], [54, 59, 63], [77, 81, 86]], dtype=np.int8)

# Case B: stride=2, pad=1 (the real model's actual config). 4x4 input
# 1..16, 3x3 all-ones kernel, input_zp=2, out_zp=3, requant_mult=1.0.
# Padding taps contribute zero (bounds-check skip), not (0 - input_zp).
#   out[0,0]: taps {1,2,5,6}, sum(v-2) = -1+0+3+4 = 6 -> +3 = 9
#   out[0,1]: taps {2,3,4,6,7,8}, sum(v-2) = 0+1+2+4+5+6 = 18 -> +3 = 21
#   out[1,0]: taps {5,6,9,10,13,14}, sum(v-2) = 3+4+7+8+11+12 = 45 -> +3 = 48
#   out[1,1]: full 3x3 window {6,7,8,10,11,12,14,15,16}, sum(v-2) = 81 -> +3 = 84
_CASE_B_EXPECTED = np.array([[9, 21], [48, 84]], dtype=np.int8)


def test_conv2d_int8_case_a_no_padding_rounding(ported_lib):
    input_arr = np.arange(1, 26, dtype=np.int8)
    weight_arr = np.ones(9, dtype=np.int8)
    bias_arr = np.zeros(1, dtype=np.int32)
    requant_arr = np.array([0.5], dtype=np.float32)
    output_arr = np.zeros(9, dtype=np.int8)

    ported_lib.test_conv2d_int8(
        _i8p(input_arr), 1, 5, 5, 0,
        _i8p(weight_arr), _i32p(bias_arr), _f32p(requant_arr),
        1, 3, 1, 0,
        _i8p(output_arr), 3, 3, 0, 0,
    )

    np.testing.assert_array_equal(output_arr.reshape(3, 3), _CASE_A_EXPECTED)


def test_conv2d_int8_case_b_stride2_pad1_zero_points(ported_lib):
    input_arr = np.arange(1, 17, dtype=np.int8)
    weight_arr = np.ones(9, dtype=np.int8)
    bias_arr = np.zeros(1, dtype=np.int32)
    requant_arr = np.array([1.0], dtype=np.float32)
    output_arr = np.zeros(4, dtype=np.int8)

    ported_lib.test_conv2d_int8(
        _i8p(input_arr), 1, 4, 4, 2,
        _i8p(weight_arr), _i32p(bias_arr), _f32p(requant_arr),
        1, 3, 2, 1,
        _i8p(output_arr), 2, 2, 3, 0,
    )

    np.testing.assert_array_equal(output_arr.reshape(2, 2), _CASE_B_EXPECTED)


def test_conv2d_int8_relu_zeroes_negative(ported_lib):
    input_arr = np.arange(1, 26, dtype=np.int8)
    weight_arr = np.ones(9, dtype=np.int8)
    bias_arr = np.array([-1000], dtype=np.int32)
    requant_arr = np.array([1.0], dtype=np.float32)
    output_arr = np.full(9, 99, dtype=np.int8)

    ported_lib.test_conv2d_int8(
        _i8p(input_arr), 1, 5, 5, 0,
        _i8p(weight_arr), _i32p(bias_arr), _f32p(requant_arr),
        1, 3, 1, 0,
        _i8p(output_arr), 3, 3, 0, 1,
    )

    np.testing.assert_array_equal(output_arr, np.zeros(9, dtype=np.int8))


# --- hand-computed dense_int8 -------------------------------------------

_DENSE_INPUT = np.array([1, 2, 3, 4], dtype=np.int8)
_DENSE_WEIGHT = np.array([1, 1, 1, 1, 2, 0, 0, 0], dtype=np.int8)
_DENSE_BIAS = np.array([0, -100], dtype=np.int32)
_DENSE_REQUANT = np.array([1.0, 1.0], dtype=np.float32)


def test_dense_int8_relu_path(ported_lib):
    # input_zp=1 -> effective (val-1) = [0,1,2,3]
    # out0 = 1*0+1*1+1*2+1*3 + 0 = 6 ; out1 = 2*0+0+0+0 + (-100) = -100 -> relu clamps to 0
    output_arr = np.zeros(2, dtype=np.int8)
    ported_lib.test_dense_int8(
        _i8p(_DENSE_INPUT), 4, 1,
        _i8p(_DENSE_WEIGHT), _i32p(_DENSE_BIAS), _f32p(_DENSE_REQUANT),
        2, ctypes.cast(_i8p(output_arr), ctypes.c_void_p), 0, 1, 0,
    )
    np.testing.assert_array_equal(output_arr, [6, 0])


def test_dense_int8_skip_requant_path(ported_lib):
    # Same accumulation, but raw int32 output, no clamp/round/requant.
    output_arr = np.zeros(2, dtype=np.int32)
    ported_lib.test_dense_int8(
        _i8p(_DENSE_INPUT), 4, 1,
        _i8p(_DENSE_WEIGHT), _i32p(_DENSE_BIAS), _f32p(_DENSE_REQUANT),
        2, ctypes.cast(_i32p(output_arr), ctypes.c_void_p), 0, 0, 1,
    )
    np.testing.assert_array_equal(output_arr, [6, -100])


# --- argmax_i32 / requantize / pgm_read_* --------------------------------


def test_argmax_i32_first_occurrence(ported_lib):
    arr = np.array([10, 50, -3, 7, 50], dtype=np.int32)
    assert ported_lib.test_argmax_i32(_i32p(arr), 5) == 1


@pytest.mark.parametrize(
    "acc_scaled,zero_point,expected",
    [
        (2.5, 0, 3),  # round half away from zero
        (-2.5, 0, -3),
        (200.0, 0, 127),  # saturate high
        (-200.0, 0, -128),  # saturate low
        (10.4, 5, 15),  # zero-point offset
        (-10.4, -5, -15),
    ],
)
def test_requantize(ported_lib, acc_scaled, zero_point, expected):
    assert ported_lib.test_requantize(acc_scaled, zero_point) == expected


def test_pgm_read_roundtrip(ported_lib):
    i = np.array([123456], dtype=np.int32)
    f = np.array([3.5], dtype=np.float32)
    assert ported_lib.test_pgm_read_i32(_i32p(i)) == 123456
    assert ported_lib.test_pgm_read_f32(_f32p(f)) == pytest.approx(3.5)


# --- regression: ported vs. verbatim reference-file copy ------------------


def _random_conv_inputs(rng, in_ch, out_ch, k, in_h, in_w):
    return (
        rng.integers(-128, 128, size=in_ch * in_h * in_w, dtype=np.int8),
        rng.integers(-127, 128, size=out_ch * in_ch * k * k, dtype=np.int8),
        rng.integers(-1000, 1000, size=out_ch, dtype=np.int32),
        rng.uniform(0.001, 0.1, size=out_ch).astype(np.float32),
    )


@pytest.mark.parametrize(
    "in_ch,out_ch,k,stride,pad,in_h,in_w,out_h,out_w",
    [
        (1, 8, 3, 2, 1, 28, 28, 14, 14),  # reference file's own CONV1 shape
        (8, 8, 3, 2, 1, 14, 14, 7, 7),  # reference file's own CONV2 shape
    ],
)
def test_conv2d_int8_matches_reference_verbatim(
    ported_lib, reference_lib, in_ch, out_ch, k, stride, pad, in_h, in_w, out_h, out_w
):
    rng = np.random.default_rng(0)
    input_arr, weight_arr, bias_arr, requant_arr = _random_conv_inputs(
        rng, in_ch, out_ch, k, in_h, in_w
    )

    outputs = {}
    for name, lib in [("ported", ported_lib), ("reference", reference_lib)]:
        output_arr = np.zeros(out_ch * out_h * out_w, dtype=np.int8)
        lib.test_conv2d_int8(
            _i8p(input_arr), in_ch, in_h, in_w, 5,
            _i8p(weight_arr), _i32p(bias_arr), _f32p(requant_arr),
            out_ch, k, stride, pad,
            _i8p(output_arr), out_h, out_w, 7, 1,
        )
        outputs[name] = output_arr

    np.testing.assert_array_equal(outputs["ported"], outputs["reference"])


def test_dense_int8_matches_reference_verbatim(ported_lib, reference_lib):
    rng = np.random.default_rng(1)
    in_dim, out_dim = 392, 128
    input_arr = rng.integers(-128, 128, size=in_dim, dtype=np.int8)
    weight_arr = rng.integers(-127, 128, size=out_dim * in_dim, dtype=np.int8)
    bias_arr = rng.integers(-1000, 1000, size=out_dim, dtype=np.int32)
    requant_arr = rng.uniform(0.001, 0.1, size=out_dim).astype(np.float32)

    out_ported = np.zeros(out_dim, dtype=np.int8)
    ported_lib.test_dense_int8(
        _i8p(input_arr), in_dim, 4,
        _i8p(weight_arr), _i32p(bias_arr), _f32p(requant_arr),
        out_dim, ctypes.cast(_i8p(out_ported), ctypes.c_void_p), 6, 1, 0,
    )

    out_reference = np.zeros(out_dim, dtype=np.int8)
    reference_lib.test_dense_int8(
        _i8p(input_arr), in_dim, 4,
        _i8p(weight_arr), _i32p(bias_arr), _f32p(requant_arr),
        out_dim, _i8p(out_reference), 6, 1,
    )

    np.testing.assert_array_equal(out_ported, out_reference)


# --- bonus: real Phase 5 codegen + real Phase 6 kernels, end to end ------


def test_generated_model_runs_against_real_kernels(build_dir):
    from ctinyml.frontend.load import load_model
    from ctinyml.passes import run_structural_passes
    from ctinyml.passes.buffer_planner import plan
    from ctinyml.passes.convert_to_closed_ir import convert
    from ctinyml.passes.drop_final_requant import apply as drop_final_requant
    from ctinyml.passes.quantize import quantize
    from ctinyml.targets.avr.codegen.assemble import generate_model_c

    mnist_fixture = FIXTURES_DIR / "mnist" / "mnist_classifier.onnx"
    graph = convert(run_structural_passes(load_model(mnist_fixture)))
    drop_final_requant(graph)
    rng = np.random.default_rng(0)
    calibration_data = [
        rng.uniform(0, 255, size=(1, 1, 28, 28)).astype(np.float32) for _ in range(8)
    ]
    calibration_data.append(np.zeros((1, 1, 28, 28), dtype=np.float32))
    calibration_data.append(np.full((1, 1, 28, 28), 255.0, dtype=np.float32))
    quantize(graph, calibration_data)
    buffer_plan = plan(graph)
    model_c = generate_model_c(graph, buffer_plan)

    header_includes = (
        "#include <stdint.h>\n"
        '#include "pgm_read_f32.h"\n'
        '#include "qmath.h"\n'
        '#include "conv2d_direct.h"\n'
        '#include "gemm.h"\n'
        '#include "argmax.h"\n\n'
    )
    model_c_path = build_dir / "model_e2e.c"
    model_c_path.write_text(header_includes + model_c)

    dll = _compile_dll(build_dir, model_c_path, [RUNTIME_DIR], "model_e2e.dll")
    lib = ctypes.CDLL(str(dll))
    lib.model_forward.argtypes = [ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_uint8)]
    lib.model_forward.restype = None

    input_arr = np.full(28 * 28, 100, dtype=np.int8)
    out_class = np.zeros(1, dtype=np.uint8)
    lib.model_forward(_i8p(input_arr), out_class.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))

    assert 0 <= int(out_class[0]) <= 9
