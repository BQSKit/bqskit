"""
This module implements the Gate base class.

A gate is a potentially-parameterized, immutable, unitary operation that can be
applied to a circuit.
"""
from __future__ import annotations

from typing import Callable
from typing import ClassVar
from typing import TYPE_CHECKING

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

    def get_inverse_params(self, params: RealVector = []) -> RealVector:
        """
        Return the parameters that invert the gate.

        Args:
            params (RealVector): The parameters of the gate to invert.

        Note:
            - The default implementation returns the same paramters because
              the default implementation of `Gate.get_inverse` returns a
              :class:`DaggerGate` wrapper of the gate. The wrapper will
              correctly handle the inversion. When overriding `get_inverse`,
              on a parameterized gate this method should be overridden
              as well.
        """
        return params

    def get_inverse(self) -> Gate:
        """Return the gate's inverse as a gate."""
        if self.is_constant() and self.is_self_inverse():
            return self

        from bqskit.ir.gates.composed import DaggerGate
        return getattr(self, '_inverse', DaggerGate(self))
        # TODO: Fill out inverse definitions throughout the gate library

    with_frozen_params: ClassVar[
        Callable[[Gate, dict[int, float]], FrozenParameterGate]
    ]
    with_all_frozen_params: ClassVar[
        Callable[[Gate, list[float]], FrozenParameterGate]
    ]

    def __repr__(self) -> str:
        return self.name
