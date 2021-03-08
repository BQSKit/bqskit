"""This module implements the PauliGate."""
from __future__ import annotations

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary


class PauliGate(Gate, DifferentiableUnitary, LocallyOptimizableUnitary):
    """A gate representing an arbitrary rotation."""
    pass  # TODO
