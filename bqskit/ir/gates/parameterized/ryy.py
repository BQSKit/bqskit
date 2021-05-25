"""This module implements the RYYGate."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class RYYGate(QubitGate, DifferentiableUnitary, LocallyOptimizableUnitary):
    """A gate representing an arbitrary rotation around the YY axis."""

    size = 2
    num_params = 1
    qasm_name = 'ryy'

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """Returns the unitary for this gate, see Unitary for more info."""
        self.check_parameters(params)

        cos = np.cos(params[0] / 2)
        nsin = -1j * np.sin(params[0] / 2)
        psin = 1j * np.sin(params[0] / 2)

        return UnitaryMatrix(
            [
                [cos, 0, 0, psin],
                [0, cos, nsin, 0],
                [0, nsin, cos, 0],
                [psin, 0, 0, cos],
            ],
        )

    def get_grad(self, params: Sequence[float] = []) -> np.ndarray:
        """Returns the gradient for this gate, see Gate for more info."""
        self.check_parameters(params)

        dcos = -np.sin(params[0] / 2) / 2
        dnsin = -1j * np.cos(params[0] / 2) / 2
        dpsin = 1j * np.cos(params[0] / 2) / 2

        return np.array(
            [
                [
                    [dcos, 0, 0, dpsin],
                    [0, dcos, dnsin, 0],
                    [0, dnsin, dcos, 0],
                    [dpsin, 0, 0, dcos],
                ],
            ], dtype=np.complex128,
        )

    def optimize(self, env_matrix: np.ndarray) -> list[float]:
        """Returns optimal parameters with respect to an environment matrix."""
        self.check_env_matrix(env_matrix)
        raise NotImplementedError()  # TODO
