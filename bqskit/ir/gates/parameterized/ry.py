"""This module implements the RYGate."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class RYGate(QubitGate):
    """A gate representing an arbitrary rotation around the Y axis."""

    size = 1
    num_params = 1
    qasm_name = 'ry'

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """Returns the unitary for this gate, see Unitary for more info."""
        self.check_parameters(params)

        cos = np.cos(params[0] / 2)
        sin = np.sin(params[0] / 2)

        return UnitaryMatrix(
            [
                [cos, -sin],
                [sin, cos],
            ],
        )

    def get_grad(self, params: Sequence[float] = []) -> np.ndarray:
        """Returns the gradient for this gate, see Gate for more info."""
        self.check_parameters(params)

        dcos = -np.sin(params[0] / 2) / 2
        dsin = np.cos(params[0] / 2) / 2

        return np.array(
            [
                [
                    [dcos, -dsin],
                    [dsin, dcos],
                ],
            ], dtype=np.complex128,
        )

    def optimize(self, env_matrix: np.ndarray) -> list[float]:
        """Returns optimal parameters with respect to an environment matrix."""
        self.check_env_matrix(env_matrix)
        a = np.real(env_matrix[0, 0] + env_matrix[1, 1])
        b = np.real(env_matrix[1, 0] - env_matrix[0, 1])
        theta = 2 * np.arccos(a / np.sqrt(a ** 2 + b ** 2))
        theta *= -1 if b > 0 else 1
        return [theta]
