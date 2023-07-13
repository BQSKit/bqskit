"""This module implements the CHGate."""
from __future__ import annotations


from bqskit.ir.gates.composed import ControlledGate
from bqskit.ir.gates.constant.s import SGate
from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary import IntegerVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class CSGate(QuditGate):
    """
    The controlled-Hadamard gate for qudits
    num_levels: (int) = number of levels os single qudit (greater or equal to 2)
    controls: list(int) = list of control levels

    """

    _num_qudits = 2
    _qasm_name = 'cs'

    def __init__(self, num_levels: int = 2, controls: IntegerVector = [1]):
        super().__init__(
            SGate(num_levels=num_levels, level_1=level_1, level_2=level_2),
            num_levels=num_levels, controls=controls,
        )
