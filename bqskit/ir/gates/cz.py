"""This module implements the CZ Gate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.fixedgate import FixedGate
from bqskit.ir.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class CZGate(FixedGate, QubitGate):
    """The controlled Z gate."""

    utry = UnitaryMatrix(
        np.array(
            [
                [1, 0, 0,  0],
                [0, 1, 0,  0],
                [0, 0, 1,  0],
                [0, 0, 0, -1],
            ], dtype=np.complex128,
        ),
    )
