"""
This module implements the DaggerGate Class.

The DaggerGate is a composed gate that equivalent to the
conjugate transpose of the input gate.

For example:
    >>> DaggerGate(TGate()).get_unitary() == TdgGate().get_unitary()
    ... True
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from bqskit.ir.gate import Gate
from bqskit.ir.gates.composedgate import ComposedGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class DaggerGate(
    ComposedGate,
    LocallyOptimizableUnitary,
    DifferentiableUnitary,
):
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
        self._name = 'Dagger(%s)' % gate.name
        self._num_params = gate.num_params
        self._num_qudits = gate.num_qudits
        self._radixes = gate.radixes

        # If input is a constant gate, we can cache the unitary.
        if self.num_params == 0:
            self.utry = gate.get_unitary().get_dagger()

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """Returns the unitary for this gate, see Unitary for more info."""
        self.check_parameters(params)
        if hasattr(self, 'utry'):
            return self.utry

        return self.gate.get_unitary(params).get_dagger()

    def get_grad(self, params: Sequence[float] = []) -> np.ndarray:
        """
        Returns the gradient for this gate, see Gate for more info.

        Notes:
            The derivative of the conjugate transpose of matrix is equal
            to the conjugate transpose of the derivative.
        """
        self.check_parameters(params)
        if hasattr(self, 'utry'):
            return np.array([])

        grads = self.gate.get_grad(params)  # type: ignore
        return np.transpose(grads.conj(), (0, 2, 1))

    def optimize(self, env_matrix: np.ndarray) -> list[float]:
        """Returns optimal parameters with respect to an environment matrix."""
        if hasattr(self, 'utry'):
            return []
        self.check_env_matrix(env_matrix)
        return self.gate.optimize(env_matrix.conj().T)  # type: ignore
