"""This module implements the SGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class SGate(ConstantGate, QubitGate):
    """The S gate."""

    size = 1
    qasm_name = "s"
    utry = UnitaryMatrix(
        np.array(
            [
                [1, 0],
                [0, 1j],
            ], dtype=np.complex128,
        ),
    )
