"""This module implements the SqrtCNOTGate."""
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class SqrtCNOTGate(ConstantGate, QubitGate):

    _num_qudits = 2
    _qasm_name = 'csx'
    _utry = UnitaryMatrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0.5 + 0.5j, 0.5 - 0.5j],
            [0, 0, 0.5 - 0.5j, 0.5 + 0.5j],
        ],
    )
