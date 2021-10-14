"""This module implements the ComposedGate class."""
from __future__ import annotations

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary


class ComposedGate(Gate):
    """
    A gate composed of other gates.

    A ComposedGate provides methods for determining if the gate is
    differentiable or locally optimizable.

    A ComposedGate is differentiable/locally optimizable if
    it inherits from the appropriate base class and all of its subgates
    (in either self.gate or self.gates) inherit from the base class.

    For more complex behaviors, one can override :func:`is_differentiable`
    or :func:`is_locally_optimizable`.
    """

    def is_differentiable(self) -> bool:
        """Check if all sub gates are differentiable."""
        if hasattr(self, 'gate'):
            return isinstance(self.gate, DifferentiableUnitary)  # type: ignore
        if hasattr(self, 'gates'):
            return all(
                isinstance(gate, DifferentiableUnitary)
                for gate in self.gates  # type: ignore
            )

        raise AttributeError(
            'Expected gate or gates field for composed gate %s.'
            % self.name,
        )

    def is_locally_optimizable(self) -> bool:
        """Check if all sub gates are locally optimizable."""
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
            % self.name,
        )

    def __hash__(self) -> int:
        if hasattr(self, 'gate'):
            return hash((self.name, self.gate))  # type: ignore
        if hasattr(self, 'gates'):
            return hash((self.name, tuple(self.gates)))  # type: ignore
        raise RuntimeError(
            f"Composed gate '{self.name}' has no attribute 'gate' or 'gates'.",
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        return self.__dict__ == other.__dict__
