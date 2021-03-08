"""This module implements the ComposedGate class."""
from __future__ import annotations

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary


class ComposedGate(Gate):
    """
    The ComposedGate class.

    A ComposedGate provides methods for determining if the gate is
    differentiable or locally optimizable.

    A ComposedGate that is differentiable/locally optimizable if
    it inherits from the appropriate base class and all of its subgates
    (in either self.gate or self.gates) inherit from the base class.

    For more complex behaviors, one can override is_differentiable
    or is_locally_optimizable.
    """

    def is_differentiable(self) -> bool:
        """Check is all sub gates are differentiable."""
        if hasattr(self, 'gate'):
            return isinstance(self.gate, DifferentiableUnitary)  # type: ignore
        if hasattr(self, 'gates'):
            return all(
                isinstance(gate, DifferentiableUnitary)
                for gate in self.gates  # type: ignore
            )

        raise AttributeError(
            'Expected gate or gates field for composed gate %s.'
            % self.get_name(),
        )

    def is_locally_optimizable(self) -> bool:
        """Check is all sub gates are locally optimizable."""
        if hasattr(self, 'gate'):
            return isinstance(
                self.gate, LocallyOptimizableUnitary,  # type: ignore
            )
        if hasattr(self, 'gates'):
            return all(
                isinstance(gate, LocallyOptimizableUnitary)
                for gate in self.gates  # type: ignore
            )

        raise AttributeError(
            'Expected gate or gates field for composed gate %s.'
            % self.get_name(),
        )
