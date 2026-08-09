"""Phase 5 acceptance tests — AVR codegen backend."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ctinyml import closed_ir
from ctinyml.frontend.load import load_model
from ctinyml.passes import run_structural_passes
from ctinyml.passes.buffer_planner import plan
from ctinyml.passes.convert_to_closed_ir import convert
from ctinyml.passes.drop_final_requant import apply as drop_final_requant
from ctinyml.passes.quantize import quantize
from ctinyml.targets.avr.codegen import base, emit_layout, emit_weights
from ctinyml.targets.avr.codegen.assemble import generate_model_c
from ctinyml.targets.avr.codegen.emit_conv2d import ConvEmitter
from ctinyml.targets.avr.codegen.emit_gemm import GemmEmitter
from ctinyml.targets.avr.codegen.errors import CodegenError

MNIST_FIXTURE = Path(__file__).parent / "fixtures" / "mnist" / "mnist_classifier.onnx"
RUNTIME_HEADERS_DIR = Path(__file__).parent.parent / "ctinyml" / "targets" / "avr" / "runtime"


def _mnist_quantized():
    graph = convert(run_structural_passes(load_model(MNIST_FIXTURE)))
    drop_final_requant(graph)
    rng = np.random.default_rng(0)
    calibration_data = [
        rng.uniform(0, 255, size=(1, 1, 28, 28)).astype(np.float32) for _ in range(8)
    ]
    calibration_data.append(np.zeros((1, 1, 28, 28), dtype=np.float32))
    calibration_data.append(np.full((1, 1, 28, 28), 255.0, dtype=np.float32))
    quantize(graph, calibration_data)
    buffer_plan = plan(graph)
    return graph, buffer_plan


def test_emit_layout_macros_on_mnist():
    graph, buffer_plan = _mnist_quantized()
    text, _ctx = emit_layout.generate(graph, buffer_plan)

    assert "#define CONV1_OUT_H 14" in text
    assert "#define CONV2_OUT_H 7" in text
    assert "#define GEMM1_IN 294" in text
    assert "#define GEMM1_OUT 10" in text
    assert "static int8_t buf_a[1176];" in text
    assert "static int8_t buf_b[1176];" in text


def test_conv_emitter_rejects_asymmetric_stride():
    input_tensor = closed_ir.Tensor(
        name="x", shape=(1, 1, 4, 4), dtype="INT8", scale=1.0, zero_point=0
    )
    weight = closed_ir.Tensor(name="w", shape=(1, 1, 3, 3), dtype="INT8", is_initializer=True)
    bias = closed_ir.Tensor(name="b", shape=(1,), dtype="INT32", is_initializer=True)
    output = closed_ir.Tensor(
        name="y", shape=(1, 1, 2, 2), dtype="INT8", scale=1.0, zero_point=0
    )
    op = closed_ir.Op(
        op_type="Conv",
        name="conv",
        inputs=[input_tensor, weight, bias],
        outputs=[output],
        attrs={"strides": (2, 1), "pads": (1, 1, 1, 1)},
    )
    ctx = base.LayoutContext(macro_names={"conv": "CONV1"})

    with pytest.raises(CodegenError, match="asymmetric stride"):
        ConvEmitter().layout_macros(op, ctx)


def test_emit_weights_values_match_quantized_tensor():
    graph, buffer_plan = _mnist_quantized()
    _text, ctx = emit_layout.generate(graph, buffer_plan)
    weights_text = emit_weights.generate(graph, ctx)

    conv1 = graph.ops[0]
    expected = conv1.inputs[1].data.flatten().tolist()

    start = weights_text.index("conv1_weight[")
    brace_start = weights_text.index("{", start)
    brace_end = weights_text.index("}", brace_start)
    actual = [int(v.strip()) for v in weights_text[brace_start + 1 : brace_end].split(",")]

    assert actual == expected


def test_gemm_emitter_skip_requant_call_site():
    graph, buffer_plan = _mnist_quantized()
    _text, ctx = emit_layout.generate(graph, buffer_plan)
    _conv1, _conv2, gemm, _argmax = graph.ops

    call = GemmEmitter().call_site(gemm, ctx)

    assert "NULL" in call
    assert "(int32_t*)" in call
    assert call.rstrip().endswith(", 1);")  # skip_requant flag is the last arg


def test_generate_model_c_structure():
    graph, buffer_plan = _mnist_quantized()
    model_c = generate_model_c(graph, buffer_plan)

    assert model_c.count("conv2d_int8(") == 2
    assert model_c.count("dense_int8(") == 1
    assert model_c.count("argmax_i32(") == 1
    assert "void model_forward(const int8_t *input, uint8_t *out_class)" in model_c

    body = model_c[model_c.index("void model_forward") :]
    i_conv1 = body.index("conv2d_int8(")
    i_conv2 = body.index("conv2d_int8(", i_conv1 + 1)
    i_dense = body.index("dense_int8(")
    i_argmax = body.index("argmax_i32(")
    assert i_conv1 < i_conv2 < i_dense < i_argmax


def test_generate_model_c_compiles_against_stub_headers(tmp_path):
    gcc = shutil.which("gcc")
    if gcc is None:
        pytest.skip("host gcc not available")

    graph, buffer_plan = _mnist_quantized()
    model_c = generate_model_c(graph, buffer_plan)

    header_includes = (
        "#include <stdint.h>\n"
        '#include "pgm_read_f32.h"\n'
        '#include "qmath.h"\n'
        '#include "conv2d_direct.h"\n'
        '#include "gemm.h"\n'
        '#include "argmax.h"\n\n'
    )
    (tmp_path / "model.c").write_text(header_includes + model_c)

    for header in RUNTIME_HEADERS_DIR.glob("*.h"):
        (tmp_path / header.name).write_text(header.read_text())

    result = subprocess.run(
        [gcc, "-c", "-std=c11", "-Wall", "-Wextra", "model.c", "-o", "model.o"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
