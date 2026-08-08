"""Shared exception types for passes.

Kept in their own module (rather than directly in `ctinyml/passes/__init__.py`)
so the individual pass modules can import them without a circular import back
through the package `__init__`, which is where the onnx_ir-level passes get
assembled into `STRUCTURAL_PASSES`.
"""


class StructuralPassError(ValueError):
    """Raised when an onnx_ir-level structural pass can't safely rewrite
    the graph."""


class ClosedIRPassError(ValueError):
    """Raised when a closed_ir-level pass (drop_final_requant, quantize)
    can't safely process the graph."""
