"""Shared exception type for AVR codegen."""


class CodegenError(ValueError):
    """Raised when the closed_ir.Graph has a shape/attribute combination
    this codegen backend doesn't support (e.g. asymmetric stride/pad, a
    non-square kernel, a Gemm orientation other than transB=1)."""
