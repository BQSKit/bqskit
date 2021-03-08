"""This module implements the HGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class HGate(ConstantGate, QubitGate):
    """The Hadamard gate."""

    size = 1
    qasm_name = 'h'
    utry = UnitaryMatrix(
        [
            [np.sqrt(2) / 2, np.sqrt(2) / 2],
            [np.sqrt(2) / 2, -np.sqrt(2) / 2],
        ],
    )
