"""This module implements the CZGate."""
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class CZGate(ConstantGate, QubitGate):
    """The controlled-Z gate."""

    _num_qudits = 2
    _qasm_name = 'cz'
    _utry = UnitaryMatrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1],
        ],
    )
