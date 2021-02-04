from __future__ import annotations
from typing import Sequence

import numpy as np

from bqskit.ir.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class U3Gate(QubitGate):
    """IBM's U3 single qubit gate."""

    num_params = 3
    size = 1

    def get_unitary(self, params: Sequence[float] | None = None) -> UnitaryMatrix:
        if params is None or len(params) != self.num_params:
            raise ValueError(f"{self.name} takes {self.num_params} parameters.")

        theta = params[0]
        phi = params[1]
        lamda = params[2]

        return UnitaryMatrix(np.array(
            [
                [np.cos(theta/2), -np.exp(1j*lamda)*np.sin(theta)],
                [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(lamda + phi))*np.cos(theta)]
            ],
        dtype=np.complex128))
