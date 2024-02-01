"""This package contains composed gates."""
from __future__ import annotations

from bqskit.ir.gates.composed.controlled import ControlledGate
from bqskit.ir.gates.composed.daggergate import DaggerGate
from bqskit.ir.gates.composed.powergate import PowerGate
from bqskit.ir.gates.composed.embedded import EmbeddedGate
from bqskit.ir.gates.composed.frozenparam import FrozenParameterGate
from bqskit.ir.gates.composed.tagged import TaggedGate
from bqskit.ir.gates.composed.vlg import VariableLocationGate

__all__ = [
    'ControlledGate',
    'PowerGate',
    'DaggerGate',
    'EmbeddedGate',
    'FrozenParameterGate',
    'TaggedGate',
    'VariableLocationGate',
]
