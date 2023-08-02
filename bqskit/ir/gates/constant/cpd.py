"""This module implements the CPDGate."""
from __future__ import annotations

from bqskit.ir.gates.composed import ControlledGate
from bqskit.ir.gates.constant.pd import PDGate
from typing import Sequence
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class CPDGate(ControlledGate):
    """
    The Controlled-S gate for qudits.

    num_levels: (int) = number of levels os single qudit (greater or equal to 2)
    controls: list(int) = list of control levels
    ind: int = The level on which to apply rotation (see above equation).
    """

    _num_qudits = 2

    def __init__(self, num_levels: int, controls: Sequence[int], ind: int):
        super().__init__(
            PDGate(num_levels=num_levels, ind=ind),
            num_levels=num_levels, controls=controls,
        )
