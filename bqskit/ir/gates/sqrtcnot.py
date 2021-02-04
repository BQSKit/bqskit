"""This module implements the SQRTCNOT Gate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.fixedgate import FixedGate
from bqskit.ir.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class SQRTCNOTGate(FixedGate, QubitGate):

    utry = UnitaryMatrix(
        np.array(
            [
                [1,0,0,0],
                [0,1,0,0],
                [0,0,0.5+0.5j,0.5-0.5j],
                [0,0,0.5-0.5j,0.5+0.5j],
            ], dtype=np.complex128,
        ),
    )
