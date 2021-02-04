"""This module implements the XX Gate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.fixedgate import FixedGate
from bqskit.ir.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class XXGate(FixedGate, QubitGate):
    """The Ising XX coupling gate."""

    utry = UnitaryMatrix(
        np.array(
            [
                [1,0,0,-1j],
                [0,1,-1j,0],
                [0,-1j,1,0],
                [-1j,0,0,1],
            ], dtype=np.complex128,
        ),
    )
