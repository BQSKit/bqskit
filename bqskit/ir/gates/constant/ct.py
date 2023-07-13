"""This module implements the CTGate."""
from __future__ import annotations

from bqskit.ir.gates.composed import ControlledGate
from bqskit.ir.gates.constant.t import TGate
from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary import IntegerVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class CTGate(ControlledGate):
    """
    The Controlled-T gate for qudits.

    num_levels: (int) = number of levels os single qudit (greater or equal to 2)
    controls: list(int) = list of control levels
    """

    _num_qudits = 2
    _qasm_name = 'ct'

    def __init__(self, num_levels: int = 2, controls: IntegerVector = [1]):
        super().__init__(
            TGate(num_levels=num_levels, level_1=level_1, level_2=level_2),
            num_levels=num_levels, controls=controls,
        )
