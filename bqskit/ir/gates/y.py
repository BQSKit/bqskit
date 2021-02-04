"""This module implements the Y Gate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.fixedgate import FixedGate
from bqskit.ir.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class YGate(FixedGate, QubitGate):
    """The Pauli Y gate."""

    utry = UnitaryMatrix(
        np.array(
            [
                [0, -1j],
                [1j, 0],
            ], dtype=np.complex128,
        ),
    )
