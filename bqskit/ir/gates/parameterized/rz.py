"""This module implements the RZGate."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class RZGate(QubitGate):
    """A gate representing an arbitrary rotation around the Z axis."""

    size = 1
    num_params = 1
    qasm_name = 'rz'

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """Returns the unitary for this gate, see Unitary for more info."""
        self.check_parameters(params)

        pexp = np.exp(1j * params[0] / 2)
        nexp = np.exp(-1j * params[0] / 2)

        return UnitaryMatrix(
            [
                [nexp, 0],
                [0, pexp],
            ],
        )

    def get_grad(self, params: Sequence[float] = []) -> np.ndarray:
        """Returns the gradient for this gate, see Gate for more info."""
        self.check_parameters(params)

        dpexp = 1j * np.exp(1j * params[0] / 2) / 2
        dnexp = -1j * np.exp(-1j * params[0] / 2) / 2

        return np.array(
            [
                [
                    [dnexp, 0],
                    [0, dpexp],
                ],
            ], dtype=np.complex128,
        )
