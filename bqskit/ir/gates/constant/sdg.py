"""This module implements the SdgGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class SdgGate(ConstantGate, QubitGate):
    """The S Dagger gate."""

    size = 1
    qasm_name = "sdg"
    utry = UnitaryMatrix(
        [
            [1, 0],
            [0, -1j],
        ]
    )
