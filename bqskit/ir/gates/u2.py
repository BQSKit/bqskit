"""This module implements the IBM U2 Gate."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from bqskit.ir.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix
from bqskit.utils.rotation import rot_z

class U2Gate(QubitGate):
    """IBM's U2 single qubit gate."""

    num_params = 2
    size = 1

    def get_unitary(self, params: Sequence[float] | None = None) -> UnitaryMatrix:
        if params is None or len(params) != self.num_params:
            raise ValueError(f"{self.name} takes {self.num_params} parameters.")

        return UnitaryMatrix(1/np.sqrt(2) * np.array(
            [
                [1, -np.exp(1j * params[1])],
                [np.exp(1j * params[0]), np.exp(1j * (params[0] + params[1]))]
            ]
        ))