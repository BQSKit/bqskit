"""This module implements the XXGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class XXGate(ConstantGate, QubitGate):
    """The Ising XX coupling gate."""

    size = 2
    qasm_name = 'rxx(pi/2)'
    utry = UnitaryMatrix(
        [
            [np.sqrt(2) / 2, 0, 0, -1j * np.sqrt(2) / 2],
            [0, np.sqrt(2) / 2, -1j * np.sqrt(2) / 2, 0],
            [0, -1j * np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
            [-1j * np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2],
        ],
    )
