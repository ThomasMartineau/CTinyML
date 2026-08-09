"""EMITTERS: op_type -> Emitter, used by both emit_layout.py and
assemble.py so the per-op-type dispatch has exactly one lookup site.
"""

from __future__ import annotations

from ctinyml.targets.avr.codegen.base import Emitter
from ctinyml.targets.avr.codegen.emit_argmax import ArgMaxEmitter
from ctinyml.targets.avr.codegen.emit_conv2d import ConvEmitter
from ctinyml.targets.avr.codegen.emit_gemm import GemmEmitter

EMITTERS: dict[str, Emitter] = {
    "Conv": ConvEmitter(),
    "Gemm": GemmEmitter(),
    "ArgMax": ArgMaxEmitter(),
}
