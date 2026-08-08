"""Phase 0 acceptance tests — validate the repo/package scaffold itself,
not any real compiler logic (that starts with the frontend, tested in
test_frontend.py).
"""

import subprocess
import sys

import pytest

STUB_MODULES = [
    "ctinyml.closed_ir",
    "ctinyml.layout",
    "ctinyml.frontend",
    "ctinyml.frontend.load",
    "ctinyml.passes",
    "ctinyml.passes.onnx_ir_passes.fold_div_into_quant",
    "ctinyml.passes.onnx_ir_passes.drop_noop_reshape",
    "ctinyml.passes.onnx_ir_passes.fuse_relu",
    "ctinyml.passes.convert_to_closed_ir",
    "ctinyml.passes.quantize",
    "ctinyml.passes.drop_final_requant",
    "ctinyml.passes.buffer_planner",
    "ctinyml.passes.sparsity_analysis",
    "ctinyml.targets",
    "ctinyml.targets.avr",
    "ctinyml.targets.avr.codegen",
    "ctinyml.targets.avr.codegen.emit_weights",
    "ctinyml.targets.avr.codegen.emit_conv2d",
    "ctinyml.targets.avr.codegen.emit_gemm",
    "ctinyml.targets.avr.codegen.emit_argmax",
    "ctinyml.targets.avr.codegen.assemble",
    "tools.cli",
]


def test_package_importable():
    import ctinyml  # noqa: F401


@pytest.mark.parametrize("module_name", STUB_MODULES)
def test_submodules_importable(module_name):
    __import__(module_name)


def test_onnx_ir_importable():
    import onnx_ir  # noqa: F401


def test_cli_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "ctinyml", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_cli_requires_target(tmp_path):
    fake_model = tmp_path / "model.onnx"
    fake_model.write_bytes(b"")
    result = subprocess.run(
        [sys.executable, "-m", "ctinyml", "compile", str(fake_model), "-o", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--target" in result.stderr
