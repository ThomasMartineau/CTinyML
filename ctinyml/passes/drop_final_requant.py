"""Mark the last Gemm (feeding ArgMax) as `skip_requant=True` — its int32
accumulator is emitted directly, no output scale/zero-point.

Stub — implemented in Phase 3 (Quantization).
"""
