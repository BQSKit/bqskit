"""This module implements the SqrtCNOTGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class SqrtCNOTGate(ConstantGate, QubitGate):

    size = 2
    qasm_name = "csx"
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
