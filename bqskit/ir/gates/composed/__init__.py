"""This package contains composed gates."""
from __future__ import annotations

from bqskit.ir.gates.composed.controlled import ControlledGate
from bqskit.ir.gates.composed.daggergate import DaggerGate
from bqskit.ir.gates.composed.frozenparam import FrozenParameterGate
from bqskit.ir.gates.composed.tagged import TaggedGate
from bqskit.ir.gates.composed.vlg import VariableLocationGate
from bqskit.ir.gates.composed.embedded import EmbeddedGate

__all__ = [
    'ControlledGate',
    'DaggerGate',
    'FrozenParameterGate',
    'VariableLocationGate',
    'TaggedGate',
    'EmbeddedGate'
]
