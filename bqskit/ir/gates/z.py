"""This module implements the Z Gate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.fixedgate import FixedGate
from bqskit.ir.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class ZGate(FixedGate, QubitGate):
    """The Pauli Z gate."""

    utry = UnitaryMatrix(
        np.array(
            [
                [1, 0],
                [0, -1],
            ], dtype=np.complex128,
        ),
    )
