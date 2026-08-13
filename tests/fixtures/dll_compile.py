"""Shared host-gcc compile + ctypes-marshaling helpers for the C-runtime
test suites (Phase 6's test_runtime.py, Phase 7's test_e2e_validation.py).
Factored out once a second test file needed the exact same DLL-compile
plumbing and per-kernel ctypes argtypes, rather than re-deriving/
duplicating them.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

GCC = shutil.which("gcc")

RUNTIME_DIR = Path(__file__).parent.parent.parent / "ctinyml" / "targets" / "avr" / "runtime"
FIXTURES_DIR = Path(__file__).parent


def compile_dll(build_dir: Path, source: Path, include_dirs: list[Path], dll_name: str) -> Path:
    """Compile `source` (+ include_dirs) into a shared lib with
    -Wl,--export-all-symbols -- needed because the real kernels are
    declared `static` (correct for a single-translation-unit embedded
    build) and MinGW doesn't auto-export DLL symbols the way ELF .so's
    do; see context/known_issues.md's Phase 6 entry."""
    dll_path = build_dir / dll_name
    cmd = [GCC, "-shared", "-std=c11", "-Wall", "-Wextra", "-Wl,--export-all-symbols"]
    for inc in include_dirs:
        cmd += ["-I", str(inc)]
    cmd += [str(source), "-o", str(dll_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stderr == "", result.stderr
    return dll_path


def i8p(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))


def i32p(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def f32p(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


# Pinned per targets/avr/runtime/*.h's real signatures (Phase 5/6) --
# matches runtime_shim_ported.c's/runtime_shim_reference.c's wrappers.
CONV_ARGTYPES = [
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

DENSE_ARGTYPES_PORTED = [
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

DENSE_ARGTYPES_REFERENCE = [
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
