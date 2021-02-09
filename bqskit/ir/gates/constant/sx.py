"""This module implements the SqrtXGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class SqrtXGate(ConstantGate, QubitGate):
    """The Sqrt(X) gate."""

    size = 1
    qasm_name = "sx"
    utry = UnitaryMatrix(
        np.array(
            [
                [np.sqrt(2)/2, -1j*np.sqrt(2)/2],
                [-1j*np.sqrt(2)/2, np.sqrt(2)/2],
            ], dtype=np.complex128,
        ),
    )

SXGate = SqrtXGate