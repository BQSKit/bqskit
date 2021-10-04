"""This module implements the CNOTGate/CXGate."""
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class CNOTGate(ConstantGate, QubitGate):
    """The controlled-not or controlled-X gate."""

    _num_qudits = 2
    _qasm_name = 'cx'
    _utry = UnitaryMatrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
    )


CXGate = CNOTGate
