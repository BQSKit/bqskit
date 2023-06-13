"""This module implements the Operation class."""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from bqskit.ir.gate import Gate
from bqskit.ir.gates import FrozenParameterGate
from bqskit.ir.location import CircuitLocation
from bqskit.ir.location import CircuitLocationLike
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_sequence


class Operation(DifferentiableUnitary):
    """An Operation groups together a gate, its parameters and location."""

    def __init__(
        self,
        gate: Gate,
        location: CircuitLocationLike,
        params: RealVector = [],
    ) -> None:
        """
        Construct an operation.

        Args:
            gate (Gate): The operation's gate.

            location (CircuitLocationLike):  The set of qudits this gate
                is applied to.

            params (RealVector): The parameters for the gate.

        Raises:
            ValueError: If `gate`'s size doesn't match `location`'s length.

            ValueError: If `gate`'s num_params doesn't match `params`'s
                length.
        """

        if not isinstance(gate, Gate):
            raise TypeError('Expected gate, got %s.' % type(gate))

        if not CircuitLocation.is_location(location):
            raise TypeError('Invalid location.')

        if is_sequence(params) and len(params) == 0 and gate.num_params != 0:
            params = [0.0] * gate.num_params

        gate.check_parameters(params)

        location = CircuitLocation(location)

        if len(location) != gate.num_qudits:
            raise ValueError('Gate and location size mismatch.')

        self._num_params = gate.num_params
        self._radixes = gate.radixes
        self._num_qudits = gate.num_qudits
        self._gate = gate
        self._location = location
        self._params = list(params)

    @property
    def gate(self) -> Gate:
        """The operation's gate."""
        return self._gate

    @property
    def location(self) -> CircuitLocation:
        """The qudit this operation is applied to."""
        return self._location

    @property
    def params(self) -> list[float]:
        """The operation's parameters for its gate."""
        return self._params

    @params.setter
    def params(self, params: list[float]) -> None:
        self.check_parameters(params)
        self._params = params

    def get_qasm(self) -> str:
        """
        Return the qasm string for this operation.

        Returns:
            str: The operation as a qasm line.
        """
        if isinstance(self.gate, FrozenParameterGate):
            full_params = self.gate.get_full_params(self.params)
        else:
            full_params = self.params

        return self.gate.get_qasm(full_params, self.location)

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        if len(params) != 0:
            return self.gate.get_unitary(params)

        return self.gate.get_unitary(self.params)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this operation.

        See :class:`DifferentiableUnitary` for more info.
        """
        if len(params) != 0:
            return self.gate.get_grad(params)  # type: ignore

        return self.gate.get_grad(self.params)  # type: ignore

    def get_unitary_and_grad(
        self,
        params: RealVector = [],
    ) -> tuple[UnitaryMatrix, npt.NDArray[np.complex128]]:
        """
        Return the unitary and gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        if len(params) != 0:
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
            and self.params == rhs.params
            and self.location == rhs.location
        )

    def __hash__(self) -> int:
        return hash((self.gate, self.location))

    def __str__(self) -> str:
        return f'{self.gate}@{self.location}'

    def __repr__(self) -> str:
        return f'{self.gate}({self.params})@{self.location}'

    def is_differentiable(self) -> bool:
        """Check if operation is differentiable."""
        return isinstance(self.gate, DifferentiableUnitary)
