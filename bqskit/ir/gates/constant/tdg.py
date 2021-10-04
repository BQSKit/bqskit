"""This module implements the TdgGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class TdgGate(ConstantGate, QubitGate):
    """The T Dagger gate."""

    _num_qudits = 1
    _qasm_name = 'tdg'
    _utry = UnitaryMatrix(
        [
            [1, 0],
            [0, np.exp(-1j * np.pi / 4)],
        ],
    )
