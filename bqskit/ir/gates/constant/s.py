"""This module implements the SGate."""
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class SGate(ConstantGate, QubitGate):
    """The S gate."""

    size = 1
    qasm_name = 's'
    utry = UnitaryMatrix(
        [
            [1, 0],
            [0, 1j],
        ],
    )
