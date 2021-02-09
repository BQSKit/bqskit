"""This module implements the U1Gate."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix
from bqskit.utils.rotation import rot_z

class U1Gate(QubitGate):
    """The U1 single qubit gate."""

    num_params = 1
    size = 1
    qasm_name = "u1"

    def get_unitary(self, params: Optional[Sequence[float]] = None) -> UnitaryMatrix:
        if params is None or len(params) != self.num_params:
            raise ValueError(f"{self.name} takes {self.num_params} parameters.")

        return UnitaryMatrix(np.exp(1j*params[0]/2) * rot_z(params[0]))
