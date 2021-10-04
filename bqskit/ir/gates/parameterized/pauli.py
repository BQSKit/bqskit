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
from bqskit.utils.math import dot_product
from bqskit.utils.math import pauli_expansion
from bqskit.utils.math import unitary_log_no_i


class PauliGate(QubitGate, DifferentiableUnitary, LocallyOptimizableUnitary):
    """A gate representing an arbitrary rotation."""

    def __init__(self, num_qudits: int) -> None:
        """Create a PauliGate acting on `num_qudits` qubits."""

        if num_qudits <= 0:
            raise ValueError('Expected positive integer, got %d' % num_qudits)

        self._num_qudits = num_qudits
        self.paulis = PauliMatrices(self.num_qudits)
        self._num_params = len(self.paulis)
        self.sigmav = (-1j / self.dim) * self.paulis.get_numpy()

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """Returns the unitary for this gate, see Unitary for more info."""
        self.check_parameters(params)

        H = dot_product(params, self.sigmav)
        eiH = sp.linalg.expm(H)
        return UnitaryMatrix(eiH, check_arguments=False)

    def get_grad(self, params: Sequence[float] = []) -> np.ndarray:
        """Returns the gradient for this gate, see Gate for more info."""
        self.check_parameters(params)

        H = dot_product(params, self.sigmav)
        _, dU = dexpmv(H, self.sigmav)
        return dU

    def get_unitary_and_grad(
        self,
        params: Sequence[float] = [],
    ) -> tuple[UnitaryMatrix, np.ndarray]:
        """Returns the unitary and gradient, see Gate for more info."""
        self.check_parameters(params)

        H = dot_product(params, self.sigmav)
        U, dU = dexpmv(H, self.sigmav)
        return UnitaryMatrix(U, check_arguments=False), dU

    def optimize(self, env_matrix: np.ndarray) -> list[float]:
        """Returns optimal parameters with respect to an environment matrix."""
        self.check_env_matrix(env_matrix)
        U, _, Vh = sp.linalg.svd(env_matrix)
        return list(pauli_expansion(unitary_log_no_i(Vh.conj().T @ U.conj().T)))
