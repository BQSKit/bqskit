"""This module implements the PauliGate."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy as sp

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.pauli import PauliMatrices
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.math import dexpmv


class PauliGate(QubitGate, DifferentiableUnitary, LocallyOptimizableUnitary):
    """A gate representing an arbitrary rotation."""

    def __init__(self, size: int) -> None:
        """Create a PauliGate acting on `size` qubits."""

        if size <= 0:
            raise ValueError('Expected positive integer, got %d' % size)

        self.size = size
        self.paulis = PauliMatrices(self.size)
        self.num_params = len(self.paulis)

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """Returns the unitary for this gate, see Unitary for more info."""
        self.check_parameters(params)

        eiH = sp.linalg.expm(-1j * self.paulis.dot_product(params))
        return UnitaryMatrix(eiH)

    def get_grad(self, params: Sequence[float] = []) -> np.ndarray:
        """Returns the gradient for this gate, see Gate for more info."""
        self.check_parameters(params)

        H = -1j * self.paulis.dot_product(params)
        _, dU = dexpmv(H, -1j * self.paulis.get_numpy())
        return dU

    def get_unitary_and_grad(
        self,
        params: Sequence[float],
    ) -> tuple[UnitaryMatrix, np.ndarray]:
        """Returns the unitary and gradient, see Gate for more info."""
        self.check_parameters(params)

        H = -1j * self.paulis.dot_product(params)
        U, dU = dexpmv(H, -1j * self.paulis.get_numpy())
        return UnitaryMatrix(U), dU

    def optimize(self, env_matrix: np.ndarray) -> list[float]:
        """Returns optimal parameters with respect to an environment matrix."""
        self.check_env_matrix(env_matrix)
        U, _, Vh = sp.linalg.svd(env_matrix)
        # return pauli_expansion(unitary_log_no_i(Vh.conj().T @ U.conj().T))
