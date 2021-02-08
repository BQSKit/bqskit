"""This module implements the IBM U1 Gate."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from bqskit.ir.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix
from bqskit.utils.rotation import rot_z

class U1Gate(QubitGate):
    """IBM's U1 single qubit gate."""

    num_params = 1
    size = 1

    def get_unitary(self, params: Sequence[float] | None = None) -> UnitaryMatrix:
        if params is None or len(params) != self.num_params:
            raise ValueError(f"{self.name} takes {self.num_params} parameters.")

        return UnitaryMatrix(np.exp(1j*params[0]/2) * rot_z(params[0]))
