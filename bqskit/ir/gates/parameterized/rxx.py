"""This module implements the RXXGate."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class RXXGate(QubitGate):
    """A gate representing an arbitrary rotation around the XX axis."""

    size = 2
    num_params = 1
    qasm_name = 'rxx'

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """Returns the unitary for this gate, see Unitary for more info."""
        self.check_parameters(params)

        cos = np.cos(params[0] / 2)
        sin = -1j * np.sin(params[0] / 2)

        return UnitaryMatrix(
            [
                [cos, 0, 0, sin],
                [0, cos, sin, 0],
                [0, sin, cos, 0],
                [sin, 0, 0, cos],
            ],
        )

    def get_grad(self, params: Sequence[float] = []) -> np.ndarray:
        """Returns the gradient for this gate, see Gate for more info."""
        self.check_parameters(params)

        dcos = -np.sin(params[0] / 2) / 2
        dsin = -1j * np.cos(params[0] / 2) / 2

        return np.array(
            [
                [
                    [dcos, 0, 0, dsin],
                    [0, dcos, dsin, 0],
                    [0, dsin, dcos, 0],
                    [dsin, 0, 0, dcos],
                ],
            ], dtype=np.complex128,
        )

    def optimize(self, env_matrix: np.ndarray) -> list[float]:
        """Returns optimal parameters with respect to an environment matrix."""
        self.check_env_matrix(env_matrix)
        a = np.real(
            env_matrix[0, 0] + env_matrix[1, 1]
            + env_matrix[2, 2] + env_matrix[3, 3],
        )
        b = np.imag(
            env_matrix[0, 3] + env_matrix[1, 2]
            + env_matrix[2, 1] + env_matrix[3, 0],
        )
        theta = np.arccos(a / np.sqrt(a ** 2 + b ** 2))
        theta *= -2 if b < 0 else 2
        return [theta]
