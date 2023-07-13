"""This module implements the CHGate."""
from __future__ import annotations

from bqskit.ir.gates.composed import ControlledGate
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary import IntegerVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class CHGate(QuditGate):
    """
    The controlled-Hadamard gate for qudits
    num_levels: (int) = number of levels os single qudit (greater or equal to 2)
    controls: list(int) = list of control levels

    """

    _num_qudits = 2
    _qasm_name = 'ch'

    def __init__(self, num_levels: int = 2, controls: IntegerVector = [1]):
        super().__init__(
            HGate(num_levels=num_levels, level_1=level_1, level_2=level_2),
            num_levels=num_levels, controls=controls,
        )
