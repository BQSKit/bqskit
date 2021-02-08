"""This module implements the SwapGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class SwapGate(ConstantGate, QubitGate):
    """The swap gate."""

    size = 2
    qasm_name = "swap"
    utry = UnitaryMatrix(
        np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ], dtype=np.complex128,
        ),
    )
