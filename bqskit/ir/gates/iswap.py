"""This module implements the ISWAP Gate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.fixedgate import FixedGate
from bqskit.ir.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class ISWAPGate(FixedGate, QubitGate):
    """The two qubit swap and phase iSWAP gate."""

    utry = UnitaryMatrix(
        np.array(
            [
                [1,0,0,0],
                [0,0,1j,0],
                [0,1j,0,0],
                [0,0,0,1],
            ], dtype=np.complex128,
        ),
    )
