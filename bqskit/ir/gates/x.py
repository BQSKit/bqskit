"""This module implements the X Gate."""

from __future__ import annotations

import numpy as np

from bqskit.ir.fixedgate import FixedGate
from bqskit.ir.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class XGate(FixedGate, QubitGate):

    size = 1
    utry = UnitaryMatrix(
        np.array(
            [
                [0, 1],
                [1, 0],
            ], dtype=np.complex128,
        ),
    )
