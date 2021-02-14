"""This module implements the CHGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class CHGate(ConstantGate, QubitGate):
    """The controlled-H gate."""

    size = 2
    qasm_name = 'ch'
    utry = UnitaryMatrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2],
            [0, 0, np.sqrt(2) / 2, -np.sqrt(2) / 2],
        ],
    )
