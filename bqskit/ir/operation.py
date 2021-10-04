"""This module implements the Operation class."""
from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np

from bqskit.ir.gate import Gate
from bqskit.ir.gates import FrozenParameterGate
from bqskit.ir.location import CircuitLocation
from bqskit.ir.location import CircuitLocationLike
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class Operation(DifferentiableUnitary):
    """
    The Operation class.

    A Operation groups together a gate, its parameters and location.
    """

    def __init__(
        self, gate: Gate,
        location: CircuitLocationLike,
        params: Sequence[float] = [],
    ) -> None:
        """
        Operation Constructor.
s
        Args:
            gate (Gate): The cell's gate.

            location (CircuitLocationLike):  The set of qudits this gate
                affects.

            params (Sequence[float]): The parameters for the gate.

        Raises:
            ValueError: If `gate`'s size doesn't match `location`'s length.

            ValueError: If `gate`'s size doesn't match `params`'s length.
        """

        if not isinstance(gate, Gate):
            raise TypeError('Expected gate, got %s.' % type(gate))

        if not CircuitLocation.is_location(location):
            raise TypeError('Invalid location.')

        location = CircuitLocation(location)

        if len(location) != gate.num_qudits:
            raise ValueError('Gate and location size mismatch.')

        self._num_params = gate.num_params
        self._radixes = gate.radixes
        self._num_qudits = gate.num_qudits

        if len(params) == 0 and self.num_params != 0:
            params = [0.0] * self.num_params

        self.check_parameters(params)

        self._gate = gate
        self._location = location
        self._params = list(params)

    @property
    def gate(self) -> Gate:
        return self._gate

    @property
    def location(self) -> CircuitLocation:
        return self._location

    @property
    def params(self) -> list[float]:
        return self._params

    @params.setter
    def params(self, params: list[float]) -> None:
        self.check_parameters(params)
        self._params = params

    def get_qasm(self) -> str:
        """Returns the qasm string for this operation."""

        if isinstance(self.gate, FrozenParameterGate):
            full_params = self.gate.get_full_params(self.params)
        else:
            full_params = self.params

        return '{}({}) q[{}];\n'.format(
            self.gate.qasm_name,
            ', '.join([str(p) for p in full_params]),
            '], q['.join([str(q) for q in self.location]),
        ).replace('()', '')

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """Return the op's unitary, see Unitary for more info."""
        if len(params) != 0:
            self.check_parameters(params)
            return self.gate.get_unitary(params)
        return self.gate.get_unitary(self.params)

    def get_grad(self, params: Sequence[float] = []) -> np.ndarray:
        """Return the op's gradient, see Unitary for more info."""
        if len(params) != 0:
            self.check_parameters(params)
            return self.gate.get_grad(params)  # type: ignore
        return self.gate.get_grad(self.params)  # type: ignore

    def get_unitary_and_grad(
        self, params: Sequence[float] = [],
    ) -> tuple[UnitaryMatrix, np.ndarray]:
        """Return the op's unitary and gradient, see Unitary for more info."""
        if len(params) != 0:
            self.check_parameters(params)
            return self.gate.get_unitary_and_grad(params)  # type: ignore
        return self.gate.get_unitary_and_grad(self.params)  # type: ignore

    def __eq__(self, rhs: Any) -> bool:
        """Check for equality."""
        if self is rhs:
            return True

        if not isinstance(rhs, Operation):
            return NotImplemented

        return (
            self.gate == rhs.gate
            and all(x == y for x, y in zip(self.params, rhs.params))
            and all(x == y for x, y in zip(self.location, rhs.location))
        )

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def __str__(self) -> str:
        return str(self.gate) + '@' + str(self.location) + str(self.params)

    def __repr__(self) -> str:
        return str(self.gate) + '@' + str(self.location)

    def is_differentiable(self) -> bool:
        """Check if operation is differentiable."""
        return isinstance(self.gate, DifferentiableUnitary)

    def is_locally_optimizable(self) -> bool:
        """Check if operation is locally optimizable."""
        return isinstance(self.gate, LocallyOptimizableUnitary)
