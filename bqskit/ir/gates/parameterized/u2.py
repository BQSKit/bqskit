"""This module implements the U2Gate."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix
from bqskit.utils.rotation import rot_z

class U2Gate(QubitGate):
    """The U2 single qubit gate."""

    num_params = 2
    size = 1
    qasm_name = "u2"

    def get_unitary(self, params: Optional[Sequence[float]] = None) -> UnitaryMatrix:
        if params is None or len(params) != self.num_params:
            raise ValueError(f"{self.name} takes {self.num_params} parameters.")

        return UnitaryMatrix(1/np.sqrt(2) * np.array(
            [
                [1, -np.exp(1j * params[1])],
                [np.exp(1j * params[0]), np.exp(1j * (params[0] + params[1]))]
            ]
        ))