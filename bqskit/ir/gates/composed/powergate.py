"""This module implements the DaggerGate Class."""
from __future__ import annotations

import re

import numpy as np
import numpy.typing as npt

from bqskit.ir.gate import Gate
from bqskit.ir.gates.composed.daggergate import DaggerGate
from bqskit.ir.gates.composedgate import ComposedGate
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

    Examples:
        >>> from bqskit.ir.gates import TGate, TdgGate
        >>> PowerGate(TGate(),2).get_unitary() ==
           TdgGate().get_unitary()*TdgGate().get_unitary()
        True
    """

    def __init__(self, gate: Gate, power: int = 1) -> None:
        """
        Create a gate which is the integer power of the input gate.

        Args:
            gate (Gate): The Gate to conjugate transpose.
            power (int): The power index for the PowerGate.
        """
        if not isinstance(gate, Gate):
            raise TypeError('Expected gate object, got %s' % type(gate))

        if not is_integer(power):
            raise TypeError(f'Expected integer power, got {type(power)}.')

        self.gate = gate
        self.power = power
        self._name = f'[{gate.name}^{power}]'
        self._num_params = gate.num_params
        self._num_qudits = gate.num_qudits
        self._radixes = gate.radixes

        # If input is a constant gate, we can cache the unitary.
        if self.num_params == 0 and not building_docs():
            self.utry = self.gate.get_unitary([]).ipower(power)

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        if hasattr(self, 'utry'):
            return self.utry

        return self.gate.get_unitary(params).ipower(self.power)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.

        Notes:
            The derivative of the integer power of matrix is equal
            to the derivative of the matrix multiplied by
            the integer-1 power of the matrix
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
        # Constant gate case
        if hasattr(self, 'utry'):
            return self.utry, np.array([])

        grad_shape = (self.num_params, self.dim, self.dim)

        # Identity gate case
        if self.power == 0:
            utry = UnitaryMatrix.identity(self.dim)
            grad = np.zeros(grad_shape, dtype=np.complex128)
            return utry, grad

        # Invert the gate if the power is negative
        gate = self.gate if self.power > 0 else DaggerGate(self.gate)
        power = abs(self.power)

        # Parallel Dicts for unitary and gradient powers
        utrys = {}  # utrys[i] = gate^(2^i)
        grads = {}  # grads[i] = d(gate^(2^i))/d(params)

        # decompose the power as sum of powers of 2
        power_bin = bin(abs(power))[2:]
        binary_decomp = [
            len(power_bin) - 1 - xb.start()
            for xb in re.finditer('1', power_bin)
        ][::-1]
        max_power_of_2 = max(binary_decomp)

        # Base Case: 2^0
        utrys[0], grads[0] = gate.get_unitary_and_grad(params)  # type: ignore

        # Loop over powers of 2
        for i in range(1, max_power_of_2 + 1):
            # u^(2^i) = u^(2^(i-1)) @ u^(2^(i-1))
            utrys[i] = utrys[i - 1] @ utrys[i - 1]

            # d[u^(2^i)] = d[u^(2^(i-1)) @ u^(2^(i-1))] =
            grads[i] = grads[i - 1] @ utrys[i - 1] + utrys[i - 1] @ grads[i - 1]

        # Calculate binary composition of the unitary and gradient
        utry = utrys[binary_decomp[0]]
        grad = grads[binary_decomp[0]]
        for i in sorted(binary_decomp[1:]):
            grad = grad @ utrys[i] + utry @ grads[i]
            utry = utry @ utrys[i]

        return utry, grad

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PowerGate)
            and self.gate == other.gate
            and self.power == other.power
        )

    def __hash__(self) -> int:
        return hash((self.power, self.gate))

    def get_inverse(self) -> Gate:
        """Return the gate's inverse as a gate."""
        return PowerGate(self.gate, -self.power)
