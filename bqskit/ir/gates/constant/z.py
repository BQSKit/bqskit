"""This module implements the ZGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class ZGate(ConstantGate, QubitGate):
    """The Pauli Z gate."""

    size = 1
    qasm_name = "z"
    utry = UnitaryMatrix(
        np.array(
            [
                [1, 0],
                [0, -1],
            ], dtype=np.complex128,
        ),
    )
