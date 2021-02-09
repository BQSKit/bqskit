"""This module implements the RYGate."""
from __future__ import annotations
from typing import Optional, Sequence

import numpy as np

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix
from bqskit.utils.rotation import rot_y

class RYGate(QubitGate):
    """A gate representing an arbitrary rotation around the Y axis."""

    size = 1
    num_params = 1
    qasm_name = "ry"
    
    def get_unitary(self, params: Optional[Sequence[float]] = None) -> UnitaryMatrix:
        if params is None or len(params) != self.num_params:
            raise ValueError(f"{self.name} takes {self.num_params} parameters.")

        return UnitaryMatrix(rot_y(params[0]))
