"""Remove Reshape nodes that are row-major-compatible flatten/unflatten
no-ops; rewire producer->consumer directly. Assert (don't silently skip)
on an ineligible Reshape.

Stub — implemented in Phase 2 (structural passes).
"""
