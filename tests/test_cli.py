"""CLI end-to-end: `ctinyml compile` wires the full pipeline and writes model.c + headers."""

import subprocess
import sys
from pathlib import Path

import numpy as np

from tests.fixtures.dll_compile import compile_dll

FIXTURES = Path(__file__).parent / "fixtures"
MNIST_FIXTURE = FIXTURES / "mnist" / "mnist_classifier.onnx"
HOLDOUT_FIXTURE = FIXTURES / "mnist_holdout.npz"


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "ctinyml", "compile", *args], capture_output=True, text=True
    )


def test_compile_writes_model_and_headers(tmp_path):
    out = tmp_path / "build"
    result = _run(
        str(MNIST_FIXTURE), "--target", "avr", "--calibration", str(HOLDOUT_FIXTURE), "-o", str(out)
    )
    assert result.returncode == 0, result.stderr
    assert (out / "model.c").read_text().startswith("#include <stdint.h>")
    for header in ("pgm_read_f32.h", "qmath.h", "conv2d_direct.h", "gemm.h", "argmax.h"):
        assert (out / header).is_file()
    compile_dll(out, out / "model.c", [out], "model_cli.dll")


def test_compile_bad_model_fails(tmp_path):
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not an onnx file")
    result = _run(
        str(bad), "--target", "avr", "--calibration", str(HOLDOUT_FIXTURE), "-o", str(tmp_path / "o")
    )
    assert result.returncode == 1
    assert "error" in result.stderr


def test_compile_bad_calibration_fails(tmp_path):
    bad = tmp_path / "cal.npz"
    np.savez(bad, other=np.zeros((1, 28, 28)))
    result = _run(
        str(MNIST_FIXTURE), "--target", "avr", "--calibration", str(bad), "-o", str(tmp_path / "o")
    )
    assert result.returncode == 1
    assert "calibration_images" in result.stderr


def test_compile_rejects_raw_0_255_calibration(tmp_path):
    raw = tmp_path / "raw.npz"
    np.savez(raw, calibration_images=np.full((2, 1, 28, 28), 255.0, dtype=np.float32))
    result = _run(
        str(MNIST_FIXTURE), "--target", "avr", "--calibration", str(raw), "-o", str(tmp_path / "o")
    )
    assert result.returncode == 1
    assert "[0, 1]" in result.stderr


def test_compile_requires_calibration(tmp_path):
    result = _run(str(MNIST_FIXTURE), "--target", "avr", "-o", str(tmp_path))
    assert result.returncode != 0
    assert "--calibration" in result.stderr
