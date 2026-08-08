"""Shared exception type for structural passes.

Kept in its own module (rather than directly in `ctinyml/passes/__init__.py`)
so the individual pass modules can import it without a circular import back
through the package `__init__`, which is where they themselves get
assembled into `STRUCTURAL_PASSES`.
"""


class StructuralPassError(ValueError):
    """Raised when a structural pass can't safely rewrite the graph."""
