"""This module implements the SqrtXGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class SqrtXGate(ConstantGate, QubitGate):
    """The Sqrt(X) gate."""

    _num_qudits = 1
    _qasm_name = 'sx'
    _utry = UnitaryMatrix(
        [
            [np.sqrt(2) / 2, -1j * np.sqrt(2) / 2],
            [-1j * np.sqrt(2) / 2, np.sqrt(2) / 2],
        ],
    )


SXGate = SqrtXGate
