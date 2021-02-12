"""This module implements the XXGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class XXGate(ConstantGate, QubitGate):
    """The Ising XX coupling gate."""

    size = 1
    qasm_name = "rxx(pi/2)"
    utry = UnitaryMatrix(
        [
            [1,0,0,-1j],
            [0,1,-1j,0],
            [0,-1j,1,0],
            [-1j,0,0,1],
        ]
    )
