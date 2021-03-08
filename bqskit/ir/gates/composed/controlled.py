from __future__ import annotations

from bqskit.ir.gates.composedgate import ComposedGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary


class ControlledGate(
    ComposedGate, LocallyOptimizableUnitary,
    DifferentiableUnitary,
):
    pass  # TODO
