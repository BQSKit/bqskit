"""This module implements the RZ Gate."""
from __future__ import annotations
from typing import Sequence

import numpy as np

from bqskit.ir.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix
from bqskit.utils.rotation import rot_z

class RZGate(QubitGate):
    """A gate representing an arbitrary rotation around the Z axis."""

    size = 1
    num_params = 1
    
    def get_unitary(self, params: Sequence[float] | None = None) -> UnitaryMatrix:
        if params is None or len(params) != self.num_params:
            raise ValueError(f"{self.name} takes {self.num_params} parameters.")

        return UnitaryMatrix(rot_z(params[0]))
