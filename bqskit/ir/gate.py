"""
This module implements the Gate base class.

A gate is a potentially-parameterized, immutable, unitary operation that can be
applied to a circuit.
"""
from __future__ import annotations

from typing import Callable
from typing import TYPE_CHECKING

from bqskit.ir.location import CircuitLocation
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitary import Unitary

if TYPE_CHECKING:
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

    with_frozen_params: Callable[[Gate, dict[int, float]], FrozenParameterGate]
    with_all_frozen_params: Callable[[Gate, list[float]], FrozenParameterGate]

    def __repr__(self) -> str:
        return self.name
