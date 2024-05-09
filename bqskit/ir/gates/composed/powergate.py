"""This module implements the DaggerGate Class."""
from __future__ import annotations

import re

import numpy as np
import numpy.typing as npt

from bqskit.ir.gate import Gate
from bqskit.ir.gates.composed.daggergate import DaggerGate
from bqskit.ir.gates.composedgate import ComposedGate
from bqskit.ir.gates.constant.identity import IdentityGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.docs import building_docs
from bqskit.utils.typing import is_integer


class PowerGate(
    ComposedGate,
    DifferentiableUnitary,
):
    """
    An arbitrary inverted gate.

    The PowerGate is a composed gate that equivalent to the
    integer power of the input gate.

    For example:
        >>> from bqskit.ir.gates import TGate, TdgGate
        >>> PowerGate(TGate(),2).get_unitary() == TdgGate().get_unitary()*TdgGate().get_unitary()
        True
    """

    def __init__(self, gate: Gate, power: int = 1) -> None:
        """
        Create a gate which is the integer power of the input gate.

        Args:
            gate     (Gate): The Gate to conjugate transpose.
            power (integer): The power index for the PowerGate
        """
        if not isinstance(gate, Gate):
            raise TypeError('Expected gate object, got %s' % type(gate))

        if not is_integer(power):
            raise TypeError(
                f'Expected integer for num_controls, got {type(power)}.',
            )

        self.gate = gate
        self.power = power
        self._name = 'Power(%s)' % gate.name
        self._num_params = gate.num_params
        self._num_qudits = gate.num_qudits
        self._radixes = gate.radixes

        # If input is a constant gate, we can cache the unitary.
        if self.num_params == 0 and not building_docs():
            self.utry = np.linalg.matrix_power(
                self.gate.get_unitary(), self.power,
            )

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        if hasattr(self, 'utry'):
            return self.utry

        return np.linalg.matrix_power(self.gate.get_unitary(params), self.power)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.

        Notes:
            The derivative of the integer power of matrix is equal
            to the derivative of the matrix multiplied by the integer-1 power of the matrix
            and by the integer power.
        """
        if hasattr(self, 'utry'):
            return np.array([])

        _, grad = self.get_unitary_and_grad(params)
        return grad

    def get_unitary_and_grad(
        self,
        params: RealVector = [],
    ) -> tuple[UnitaryMatrix, npt.NDArray[np.complex128]]:
        """
        Return the unitary and gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        if hasattr(self, 'utry'):
            return self.utry, np.array([])

        if self.power == 0:
            return IdentityGate(radixes=self.gate.radixes).get_unitary(), 0 * IdentityGate(radixes=self.gate.radixes).get_unitary()

        # powers = {0: IdentityGate(radixes=self.gate.radixes).get_unitary()}
        # grads = {0: 0*IdentityGate(radixes=self.gate.radixes).get_unitary()}

        powers = {}
        grads = {}

        # decompose the power as sum of powers of 2
        indexbin = bin(abs(self.power))[2:]
        indices = [
            len(indexbin) - 1 - xb.start()
            for xb in re.finditer('1', indexbin)
        ][::-1]

        powers[0], grads[0] = self.gate.get_unitary_and_grad(params)

        # avoid doing computations if not needed
        if self.power == 1:
            return powers[0], grads[0]

        # check if the power is negative, and
        if np.sign(self.power) == -1:
            gate = DaggerGate(self.gate)
            powers[0], grads[0] = gate.get_unitary_and_grad(params)

        # avoid doing computations if not needed
        if abs(self.power) == 1:
            return powers[0], grads[0]

        grads[1] = grads[0] @ powers[0] + powers[0] @ grads[0]
        powers[1] = powers[0] @ powers[0]

        # avoid doing more computations if not needed
        if abs(self.power) == 2:
            return powers[1], grads[1]

        # loop over powers of 2
        for i in range(2, indices[-1] + 1):
            powers[i] = powers[i - 1] @ powers[i - 1]
            grads[i] = grads[i - 1] @ powers[i - 1] + \
                powers[i - 1] @ grads[i - 1]

        unitary = powers[indices[0]]
        for i in indices[1:]:
            unitary = unitary @ powers[indices[i]]

        grad = 0 * IdentityGate(radixes=self.gate.radixes).get_unitary()
        for i in indices:
            grad_tmp = grads[i]
            for j in indices:
                if j < i:
                    grad_tmp = powers[j] @ grad_tmp
                elif j > i:
                    grad_tmp = grad_tmp @ powers[j]
            grad = grad + grad_tmp

        return unitary, grad

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PowerGate)
            and self.gate == other.gate
        )

    def __hash__(self) -> int:
        return hash(self.gate)

    def get_inverse(self) -> Gate:
        """Return the gate's inverse as a gate."""
        return DaggerGate(self.gate)
