"""
This module implements the Gate base class.

A gate is a potentially-parameterized, immutable, unitary operation that can be
applied to a circuit.
"""
from __future__ import annotations

from typing import Callable
from typing import ClassVar
from typing import TYPE_CHECKING

import numpy as np

from bqskit.ir.location import CircuitLocation
from bqskit.qis.unitary.unitary import Unitary

if TYPE_CHECKING:
    from bqskit.qis.unitary.unitary import RealVector
    from bqskit.ir.gates.composed.frozenparam import FrozenParameterGate


class Gate(Unitary):
    """Gate Base Class."""

    _name: str
    _qasm_name: str

    @property
    def name(self) -> str:
        """The name of this gate."""
        return getattr(self, '_name', self.__class__.__name__)

    @property
    def qasm_name(self) -> str:
        """The qasm identifier for this gate."""
        if not self.is_qubit_only():
            raise AttributeError('QASM only supports qubit gates.')

        return getattr(self, '_qasm_name')

    def get_qasm_gate_def(self) -> str:
        """Returns a qasm gate definition block for this gate."""
        if not self.is_qubit_only():
            raise AttributeError('QASM only supports qubit gates.')

        return ''

    def get_qasm(self, params: RealVector, location: CircuitLocation) -> str:
        """Returns the qasm string for this gate."""
        return '{}({}) q[{}];\n'.format(
            self.qasm_name,
            ', '.join([str(p) for p in params]),
            '], q['.join([str(q) for q in location]),
        ).replace('()', '')

    def is_self_inverse(self) -> bool:
        """
        Checks whether the gate is its own inverse.

        A gate is its own inverse if its unitary matrix is equal to
        its Hermitian conjugate.

        Returns:
            bool: True if the gate is self-inverse, False otherwise.
        """
        # Get the unitary matrix of the gate
        unitary_matrix = self.get_unitary()

        # Calculate the Hermitian conjugate (adjoint) of the unitary matrix
        hermitian_conjugate = unitary_matrix.conj().T

        # Check if the unitary matrix is equal to its Hermitian conjugate
        return np.allclose(unitary_matrix, hermitian_conjugate)

    def get_inverse_params(self, params: RealVector = []):
        if params:
            # Negate the parameters and normalize to the 0-2pi range
            inverse_params = [-param % (2 * np.pi) for param in params]
            # Create a new gate of the same type with the inverse parameters
            self.check_parameters(inverse_params)
            return self, inverse_params

    def get_inverse(self) -> Gate:
        """Return the gate's inverse as a gate."""
        if self.is_self_inverse():
            return self
        else:
            from bqskit.ir.gates.composed import DaggerGate
            return DaggerGate(self)

    with_frozen_params: ClassVar[
        Callable[[Gate, dict[int, float]], FrozenParameterGate]
    ]
    with_all_frozen_params: ClassVar[
        Callable[[Gate, list[float]], FrozenParameterGate]
    ]

    def __repr__(self) -> str:
        return self.name
