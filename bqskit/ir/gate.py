"""
This module implements the Gate base class.

A gate is a potentially-parameterized, immutable, unitary operation that can be
applied to a circuit.
"""
from __future__ import annotations

from typing import Callable
from typing import TYPE_CHECKING

from bqskit.qis.unitary.unitary import Unitary
from bqskit.utils.cachedclass import CachedClass

if TYPE_CHECKING:
    from bqskit.ir.gates.composed.frozenparam import FrozenParameterGate


class Gate(Unitary, CachedClass):
    """Gate Base Class."""

    name: str
    qasm_name: str

    def get_name(self) -> str:
        """Returns the name of the gate, defaults to the class name."""
        if hasattr(self, 'name'):
            return self.name

        return self.__class__.__name__

    def get_qasm_name(self) -> str:
        """Returns the qasm name for this gate."""
        if not self.is_qubit_only():
            raise AttributeError('QASM only supports qubit gates.')

        if hasattr(self, 'qasm_name'):
            return self.qasm_name

        raise AttributeError(
            'Expected qasm_name field for gate %s.'
            % self.get_name(),
        )

    def get_qasm_gate_def(self) -> str:
        """Returns a qasm gate definition block for this gate."""
        if not self.is_qubit_only():
            raise AttributeError('QASM only supports qubit gates.')

        return ''

    with_frozen_params: Callable[[Gate, dict[int, float]], FrozenParameterGate]
    with_all_frozen_params: Callable[[Gate, list[float]], FrozenParameterGate]

    def __repr__(self) -> str:
        return self.get_name()
