"""
This module implements the DaggerGate Class.

The DaggerGate is a composed gate that equivalent to the
conjugate transpose of the input gate.

For example:
    >>> DaggerGate(TGate()).get_unitary() == TdgGate().get_unitary()
    True
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from bqskit.ir import Gate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class DaggerGate(Gate):
    """The DaggerGate Class."""

    def __init__(self, gate: Gate) -> None:
        """
        Create a gate which is the conjugate transpose of another.

        Args:
            gate (Gate): The Gate to conjugate transpose.
        """
        if not isinstance(gate, Gate):
            raise TypeError('Expected gate object, got %s' % type(gate))

        self.gate = gate
        self.name = 'Dagger(%s)' % gate.name
        self.num_params = gate.num_params
        self.size = gate.size
        self.radixes = gate.radixes

        # If input is a constant gate, we can cache the unitary.
        if self.num_params == 0:
            self.utry = gate.get_unitary().dagger()

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """Returns the unitary for this gate, see Unitary for more info."""
        self.check_parameters(params)
        if self.utry:
            return self.utry

        return self.gate.get_unitary(params).dagger()

    def get_grad(self, params: Sequence[float] = []) -> np.ndarray:
        """
        Returns the gradient for this gate, see Gate for more info.

        Notes:
            The derivative of the conjugate transpose of matrix is equal
            to the conjugate transpose of the derivative.
        """
        return np.transpose(self.gate.get_grad(params).conj(), (0, 2, 1))

    def optimize(self, env_matrix: np.ndarray) -> list[float]:
        """Returns optimal parameters with respect to an environment matrix."""
        return self.gate.optimize(env_matrix.conj().T)

    def get_name(self) -> str:
        """Returns the name of the gate, see Gate for more info."""
        return '%s(%s)' % (self.__class__.__name__, self.gate.get_name())
