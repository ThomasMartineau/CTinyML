"""Fuse Relu into the preceding Conv/Gemm as a `clamp_min0=True` attribute;
delete the Relu node. Assert every Relu gets fused.

Stub — implemented in Phase 2 (structural passes).
"""
