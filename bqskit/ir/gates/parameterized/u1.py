"""This module implements the U1Gate."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class U1Gate(QubitGate, DifferentiableUnitary, LocallyOptimizableUnitary):
    """The U1 single qubit gate."""

    size = 1
    num_params = 1
    qasm_name = 'u1'

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """Returns the unitary for this gate, see Unitary for more info."""
        self.check_parameters(params)

        exp = np.exp(1j * params[0])

        return UnitaryMatrix(
            [
                [1, 0],
                [0, exp],
            ],
        )

    def get_grad(self, params: Sequence[float] = []) -> np.ndarray:
        """Returns the gradient for this gate, see Gate for more info."""
        self.check_parameters(params)

        dexp = 1j * np.exp(1j * params[0])

        return np.array(
            [
                [
                    [0, 0],
                    [0, dexp],
                ],
            ], dtype=np.complex128,
        )

    def optimize(self, env_matrix: np.ndarray) -> list[float]:
        """Returns optimal parameters with respect to an environment matrix."""
        self.check_env_matrix(env_matrix)
        a = np.real(env_matrix[1, 1])
        b = np.imag(env_matrix[1, 1])
        arctan = np.arctan(b / a)

        if a < 0 and b > 0:
            arctan += np.pi
        elif a < 0 and b < 0:
            arctan -= np.pi

        return [-arctan]
