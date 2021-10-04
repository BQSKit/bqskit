"""This module implements the SGate."""
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class SGate(ConstantGate, QubitGate):
    """The S gate."""

    _num_qudits = 1
    _qasm_name = 's'
    _utry = UnitaryMatrix(
        [
            [1, 0],
            [0, 1j],
        ],
    )
