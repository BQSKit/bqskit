"""This module implements the CHGate."""
from __future__ import annotations

from bqskit.ir.gates.composed import ControlledGate
from bqskit.ir.gates.constant.shift import ShiftGate
from bqskit.ir.gates.quditgate import QuditGate
from typing import Sequence
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class CShiftGate(QuditGate):
    """
    The controlled-Shift gate for qudits
    """

    _num_qudits = 2
    _qasm_name = 'ch'

    def __init__(self, num_levels: int, controls: Sequence[int]):
        super().__init__(
            ShiftGate(num_levels=num_levels),
            num_levels=num_levels, controls=controls,
        )
