"""This module implements the CircuitGate class."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit

from bqskit.ir.gate import Gate


class CircuitGate(Gate):
    """
    The CircuitGate class.

    A CircuitGate is a immutable circuit represented as a gate.
    """

    def __init__(self, circuit: Circuit, move: bool = False) -> None:
        """
        CircuitGate Constructor.

        Args:
            circuit (Circuit): The circuit to copy into gate format.

            move (bool): If true, the constructor will not copy the circuit.
                This should only be used when you are sure `circuit` will no
                longer be used on caller side. If unsure use the default.
                (Default: False)
        """

        self._circuit = circuit if move else circuit.copy()
        self._num_qudits = self._circuit.num_qudits
        self._radixes = self._circuit.radixes
        self._num_params = self._circuit.num_params
        self._name = 'CircuitGate(%s)' % str(self._circuit)

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        return self._circuit.get_unitary(params)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        return self._circuit.get_grad(params)

    def get_unitary_and_grad(
        self,
        params: RealVector = [],
    ) -> tuple[UnitaryMatrix, npt.NDArray[np.complex128]]:
        """
        Return the unitary and gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        return self._circuit.get_unitary_and_grad(params)

    def is_differentiable(self) -> bool:
        """Return true if the circuit is differentiable."""
        return self._circuit.is_differentiable()

    def __hash__(self) -> int:
        hashes: list[int] = [hash(self.name)]
        for op in self._circuit:
            hashes.append(hash(op))

            # Don't let the hash list grow too large.
            if len(hashes) >= 100:
                hashes = [hash(tuple(hashes))]

        hash_val = hash(tuple(hashes)) if len(hashes) > 1 else hashes[0]
        return hash_val

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CircuitGate):
            return NotImplemented

        if self._circuit.num_qudits != other._circuit.num_qudits:
            return False

        if self._circuit.radixes != other._circuit.radixes:
            return False

        return all(
            op1.gate == op2.gate and op1.location == op2.location
            for op1, op2 in zip(self._circuit, other._circuit)
        )
